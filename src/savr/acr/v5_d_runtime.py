"""Fail-closed orchestration primitives for the frozen V5-D GPU gate.

This module is dependency-free and intentionally performs no GPU inspection,
model loading, compilation, capture, timing, filesystem writes, or outcome
access.  It makes the protocol state machine mechanically testable before the
one-GPU phase is authorized.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import threading
import time
from copy import deepcopy
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any

from savr.acr.reuse_executor import (
    EagerReuseExecutor,
    ExecutorLifecycle,
    ReuseCompatibilityKey,
    ReuseExecutionInputs,
    ReuseExecutorUnavailable,
    StaticBufferReuseExecutor,
)


V5_D_V01_RUN_ID = "acr-v5d-real-tensor-feasibility-v01"
V5_D_RUN_ID = "acr-v5d-real-tensor-feasibility-v02"
V5_D_FREEZE_SCHEMA = "acr.v5d-gpu-feasibility-freeze.v1"
V5_D_RESOLVED_SCHEMA = "acr.v5d-gpu-feasibility-resolved.v2"
V5_D_RECOVERY_SCHEMA = "acr.v5d-gpu-feasibility-recovery.v2"
V5_D_FREEZE_RELATIVE = Path("configs/acr/v5_d_gpu_feasibility_freeze.json")
V5_D_RECOVERY_RELATIVE = Path("configs/acr/v5_d_gpu_feasibility_recovery_v02.json")
V5_D_BACKEND_VERSION = "acr-v5d-static-backend-v1"
V5_D_COMPILE_BACKEND = "torch-compile"
V5_D_RAW_BACKEND = "raw-cudagraph"


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def semantic_sha256(value: Mapping[str, Any], *, field: str = "semantic_sha256") -> str:
    payload = dict(value)
    payload.pop(field, None)
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def validate_v5_d_freeze(config: Mapping[str, Any]) -> None:
    """Reject any drift from the pre-output V5-D machine contract."""

    schema = config.get("schema_version")
    if schema not in (V5_D_FREEZE_SCHEMA, V5_D_RESOLVED_SCHEMA):
        raise ValueError("V5-D freeze schema changed")
    if config.get("semantic_sha256") != semantic_sha256(config):
        raise ValueError("V5-D freeze semantic hash mismatch")
    expected_run_id = V5_D_V01_RUN_ID if schema == V5_D_FREEZE_SCHEMA else V5_D_RUN_ID
    if config.get("run_id") != expected_run_id:
        raise ValueError("V5-D run identity changed")
    if schema == V5_D_RESOLVED_SCHEMA:
        recovery = config.get("recovery_v02", {})
        if recovery.get("base_freeze_semantic_sha256") != (
            "f445cf5d1a5ec6877ebea46ccc3883a11a676b38cb33a711ee4b74baf22f53f8"
        ) or recovery.get("v01_technical_stop_semantic_sha256") != (
            "edf5872fa818f5806601f52143cb17cec7dd4974e03cc4e2ed43c3d042fb4412"
        ):
            raise ValueError("V5-D v02 recovery provenance changed")
    waterfall = config["backend_waterfall"]
    if waterfall["order"] != [V5_D_COMPILE_BACKEND, V5_D_RAW_BACKEND]:
        raise ValueError("V5-D backend waterfall changed")
    if waterfall["compiler"] != {
        "backend": "inductor",
        "fullgraph": True,
        "dynamic": False,
        "mode": "reduce-overhead",
    }:
        raise ValueError("V5-D compiler arguments changed")
    paths = tuple(config["timing"]["paths"])
    expected_permutations = [list(items) for items in itertools.permutations(paths)]
    if config["timing"]["permutations"] != expected_permutations:
        raise ValueError("V5-D timing permutations changed")
    correctness = int(config["correctness"]["query_count"])
    warmups = int(config["timing"]["warmup_query_count"])
    timed = int(config["timing"]["timed_query_count"])
    cap = int(config["resource_caps"]["full_model_query_hard_cap"])
    if (correctness, warmups, timed, correctness + warmups + timed, cap) != (
        7,
        8,
        96,
        111,
        111,
    ):
        raise ValueError("V5-D query budget changed")
    if config["resource_caps"]["backend_preparation_core_launch_hard_cap"] != 24:
        raise ValueError("V5-D preparation launch cap changed")
    if any(
        config["resource_caps"][key] != 0
        for key in ("simulator_episodes", "simulator_resets", "downloads", "new_task_outcomes")
    ):
        raise ValueError("V5-D protected resource boundary changed")


def validate_v5_d_recovery_overlay(recovery: Mapping[str, Any], base: Mapping[str, Any]) -> None:
    if recovery.get("schema_version") != V5_D_RECOVERY_SCHEMA:
        raise ValueError("V5-D v02 recovery schema changed")
    if recovery.get("semantic_sha256") != semantic_sha256(recovery):
        raise ValueError("V5-D v02 recovery semantic hash mismatch")
    if recovery.get("base_freeze") != V5_D_FREEZE_RELATIVE.as_posix():
        raise ValueError("V5-D v02 base-freeze path changed")
    if recovery.get("base_freeze_semantic_sha256") != base.get("semantic_sha256"):
        raise ValueError("V5-D v02 base-freeze identity changed")
    if recovery.get("v01_run_id") != V5_D_V01_RUN_ID or recovery.get("run_id") != V5_D_RUN_ID:
        raise ValueError("V5-D v02 run provenance changed")
    if recovery.get("v01_technical_stop_semantic_sha256") != (
        "edf5872fa818f5806601f52143cb17cec7dd4974e03cc4e2ed43c3d042fb4412"
    ):
        raise ValueError("V5-D v01 technical-stop identity changed")
    if recovery.get("permitted_changes") != [
        "canonical-run-local-libero-config-before-upstream-import",
        "pre-model-technical-stop-envelope",
    ]:
        raise ValueError("V5-D v02 recovery scope changed")
    libero = recovery.get("libero_config", {})
    if libero.get("keys") != [
        "assets",
        "bddl_files",
        "benchmark_root",
        "datasets",
        "init_states",
    ] or libero.get("paths_relative_to_project") != {
        "assets": "third_party/LIBERO/libero/libero/assets",
        "bddl_files": "third_party/LIBERO/libero/libero/bddl_files",
        "benchmark_root": "third_party/LIBERO/libero/libero",
        "datasets": "third_party/LIBERO/libero/datasets",
        "init_states": "third_party/LIBERO/libero/libero/init_files",
    }:
        raise ValueError("V5-D v02 LIBERO mapping changed")
    if libero.get("config_relative_to_run") != "cache/libero/config.yaml":
        raise ValueError("V5-D v02 LIBERO config location changed")
    if recovery.get("current_authorization") != {
        "recovery_implementation": True,
        "gpu_inspection": False,
        "gpu_selection": False,
        "model_loading": False,
        "model_queries": False,
        "cuda_compile_capture_or_timing": False,
        "simulator_use": False,
        "protected_outcome_access": False,
        "manuscript_changes": False,
    }:
        raise ValueError("V5-D v02 authorization boundary changed")


def resolve_v5_d_recovery(base: Mapping[str, Any], recovery: Mapping[str, Any]) -> dict[str, Any]:
    validate_v5_d_freeze(base)
    validate_v5_d_recovery_overlay(recovery, base)
    resolved = deepcopy(dict(base))
    resolved.update(
        {
            "schema_version": V5_D_RESOLVED_SCHEMA,
            "status": recovery["status"],
            "authorized_at": recovery["authorized_at"],
            "authorized_scope": recovery["authorized_scope"],
            "protocol": recovery["recovery_plan"],
            "run_id": recovery["run_id"],
            "recovery_v02": {
                "base_freeze_semantic_sha256": recovery["base_freeze_semantic_sha256"],
                "v01_run_id": recovery["v01_run_id"],
                "v01_technical_stop_semantic_sha256": recovery[
                    "v01_technical_stop_semantic_sha256"
                ],
                "permitted_changes": recovery["permitted_changes"],
                "libero_config": recovery["libero_config"],
            },
            "current_authorization": recovery["current_authorization"],
            "advance_only_to": recovery["advance_only_to"],
            "semantic_sha256": recovery["resolved_configuration_semantic_sha256"],
        }
    )
    validate_v5_d_freeze(resolved)
    return resolved


def load_v5_d_freeze(project_root: Path) -> dict[str, Any]:
    base = json.loads((project_root / V5_D_FREEZE_RELATIVE).read_text(encoding="utf-8"))
    recovery = json.loads((project_root / V5_D_RECOVERY_RELATIVE).read_text(encoding="utf-8"))
    return resolve_v5_d_recovery(base, recovery)


class BackendKind(str, Enum):
    TORCH_COMPILE = V5_D_COMPILE_BACKEND
    RAW_CUDAGRAPH = V5_D_RAW_BACKEND


class ExecutionStage(str, Enum):
    UNSTARTED = "unstarted"
    PREPARATION = "preparation"
    CORRECTNESS = "correctness"
    WARMUP = "warmup"
    TIMING = "timing"
    TERMINAL = "terminal"


class TechnicalReason(str, Enum):
    COMPILER_CONSTRUCTION_OR_FIRST_CALL_ERROR = "compiler-construction-or-first-call-error"
    FULL_GRAPH_CAPTURE_ERROR = "full-graph-capture-error"
    STATIC_KEY_RECOMPILE = "static-key-recompile"
    VERIFIED_EAGER_FALLBACK = "verified-eager-fallback"
    PREPARATION_OOM = "preparation-oom"


class V5DRuntimeError(RuntimeError):
    """Base fail-closed V5-D runtime error."""


class V5DProtocolViolation(V5DRuntimeError):
    """The requested operation differs from the frozen protocol."""


class V5DResourceExceeded(V5DRuntimeError):
    """A frozen wall, artifact, query, launch, or memory cap was exceeded."""


@dataclass(frozen=True)
class BackendAttempt:
    backend: str
    process_token: str
    status: str
    preparation_launches: int
    correctness_records: int
    timing_records: int
    technical_reason: str | None = None
    message: str | None = None


class BackendWaterfall:
    """Enforce compile-first selection without statistical backend shopping."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        validate_v5_d_freeze(config)
        self._config = config
        self._current: BackendKind | None = None
        self._process_token: str | None = None
        self._stage = ExecutionStage.UNSTARTED
        self._preparation_labels: list[str] = []
        self._correctness_records = 0
        self._timing_records = 0
        self._attempts: list[BackendAttempt] = []
        self._raw_permit = False
        self._compile_process_token: str | None = None

    @property
    def stage(self) -> ExecutionStage:
        return self._stage

    @property
    def current_backend(self) -> BackendKind | None:
        return self._current

    @property
    def attempts(self) -> tuple[BackendAttempt, ...]:
        return tuple(self._attempts)

    @property
    def preparation_launches(self) -> int:
        return len(self._preparation_labels) + sum(
            attempt.preparation_launches for attempt in self._attempts
        )

    def begin(self, backend: BackendKind, *, process_token: str) -> None:
        if not process_token:
            raise V5DProtocolViolation("V5-D backend process token is required")
        if self._stage is ExecutionStage.TERMINAL:
            raise V5DProtocolViolation("Another V5-D backend attempt is not permitted")
        if self._current is not None or self._stage is not ExecutionStage.UNSTARTED:
            raise V5DProtocolViolation("A V5-D backend attempt is already active")
        if not self._attempts:
            if backend is not BackendKind.TORCH_COMPILE:
                raise V5DProtocolViolation("V5-D must attempt torch.compile first")
            self._compile_process_token = process_token
        else:
            if backend is not BackendKind.RAW_CUDAGRAPH or not self._raw_permit:
                raise V5DProtocolViolation("Raw CUDA graph attempt is not permitted")
            if process_token == self._compile_process_token:
                raise V5DProtocolViolation("Raw fallback requires a fresh process")
            self._raw_permit = False
        self._current = backend
        self._process_token = process_token
        self._stage = ExecutionStage.PREPARATION
        self._preparation_labels.clear()
        self._correctness_records = 0
        self._timing_records = 0

    def record_preparation_launch(self, label: str) -> int:
        if self._stage is not ExecutionStage.PREPARATION or self._current is None:
            raise V5DProtocolViolation("Preparation launch outside backend preparation")
        if not label or label in self._preparation_labels:
            raise V5DProtocolViolation("Preparation launch labels must be unique")
        cap = int(self._config["resource_caps"]["backend_preparation_core_launch_hard_cap"])
        if self.preparation_launches >= cap:
            raise V5DResourceExceeded("V5-D preparation core-launch cap exhausted")
        self._preparation_labels.append(label)
        return self.preparation_launches - 1

    def begin_correctness(self) -> None:
        if self._stage is not ExecutionStage.PREPARATION:
            raise V5DProtocolViolation("Correctness can begin only after preparation")
        self._stage = ExecutionStage.CORRECTNESS

    def record_correctness(self) -> None:
        if self._stage is not ExecutionStage.CORRECTNESS:
            raise V5DProtocolViolation("Correctness record outside correctness stage")
        self._correctness_records += 1

    def begin_warmup(self) -> None:
        required = int(self._config["correctness"]["query_count"])
        if self._stage is not ExecutionStage.CORRECTNESS or self._correctness_records != required:
            raise V5DProtocolViolation("Warm-up requires the complete correctness matrix")
        self._stage = ExecutionStage.WARMUP

    def begin_timing(self) -> None:
        if self._stage is not ExecutionStage.WARMUP:
            raise V5DProtocolViolation("Timing requires the complete warm-up stage")
        self._stage = ExecutionStage.TIMING

    def record_timing(self) -> None:
        if self._stage is not ExecutionStage.TIMING:
            raise V5DProtocolViolation("Timing record outside timing stage")
        self._timing_records += 1

    def technical_failure(self, reason: TechnicalReason, message: str) -> bool:
        """End an attempt and return whether the frozen raw transition is allowed."""

        if self._current is None or self._process_token is None:
            raise V5DProtocolViolation("No active backend attempt to fail")
        if not message:
            raise V5DProtocolViolation("Technical failure requires a message")
        raw_allowed = (
            self._current is BackendKind.TORCH_COMPILE
            and self._stage is ExecutionStage.PREPARATION
            and self._correctness_records == 0
            and self._timing_records == 0
            and reason.value in self._config["backend_waterfall"]["raw_technical_reasons"]
        )
        self._attempts.append(
            BackendAttempt(
                backend=self._current.value,
                process_token=self._process_token,
                status="technical-failure",
                preparation_launches=len(self._preparation_labels),
                correctness_records=self._correctness_records,
                timing_records=self._timing_records,
                technical_reason=reason.value,
                message=message,
            )
        )
        self._current = None
        self._process_token = None
        self._stage = ExecutionStage.UNSTARTED if raw_allowed else ExecutionStage.TERMINAL
        self._raw_permit = raw_allowed
        self._preparation_labels.clear()
        return raw_allowed

    def fail_after_output(self, message: str) -> None:
        if self._current is None or self._process_token is None:
            raise V5DProtocolViolation("No active backend attempt to stop")
        self._attempts.append(
            BackendAttempt(
                backend=self._current.value,
                process_token=self._process_token,
                status="failed-after-output",
                preparation_launches=len(self._preparation_labels),
                correctness_records=self._correctness_records,
                timing_records=self._timing_records,
                message=message,
            )
        )
        self._current = None
        self._process_token = None
        self._raw_permit = False
        self._stage = ExecutionStage.TERMINAL
        self._preparation_labels.clear()

    def complete(self) -> None:
        required = int(self._config["timing"]["timed_query_count"])
        if self._stage is not ExecutionStage.TIMING or self._timing_records != required:
            raise V5DProtocolViolation("V5-D completion requires all timed records")
        assert self._current is not None and self._process_token is not None
        self._attempts.append(
            BackendAttempt(
                backend=self._current.value,
                process_token=self._process_token,
                status="completed",
                preparation_launches=len(self._preparation_labels),
                correctness_records=self._correctness_records,
                timing_records=self._timing_records,
            )
        )
        self._current = None
        self._process_token = None
        self._stage = ExecutionStage.TERMINAL
        self._preparation_labels.clear()


@dataclass(frozen=True)
class QueryIdentity:
    label: str
    kind: str
    path: str
    block: int | None
    position: int | None
    input_label: str


def frozen_query_schedule(config: Mapping[str, Any]) -> tuple[QueryIdentity, ...]:
    validate_v5_d_freeze(config)
    identities: list[QueryIdentity] = []
    for label in config["correctness"]["labels"]:
        input_label = "input-b" if label.startswith("b-") else "input-a"
        path = label.removeprefix("a-").removeprefix("b-")
        identities.append(QueryIdentity(label, "correctness", path, None, None, input_label))
    for path in config["timing"]["paths"]:
        for repetition in range(int(config["timing"]["warmups_per_path"])):
            identities.append(
                QueryIdentity(
                    f"warmup-{path}-{repetition:02d}",
                    "warmup",
                    path,
                    None,
                    None,
                    "input-a" if repetition % 2 == 0 else "input-b",
                )
            )
    for block, order in enumerate(config["timing"]["permutations"]):
        input_label = "input-a" if block % 2 == 0 else "input-b"
        for position, path in enumerate(order):
            identities.append(
                QueryIdentity(
                    f"timed-{block:02d}-{position}-{path}",
                    "timed",
                    path,
                    block,
                    position,
                    input_label,
                )
            )
    if len(identities) != int(config["resource_caps"]["full_model_query_hard_cap"]):
        raise V5DProtocolViolation("Frozen V5-D query schedule does not equal its hard cap")
    return tuple(identities)


class FrozenQueryLedger:
    """Consume the exact query schedule before every full model call."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        self._schedule = frozen_query_schedule(config)
        self._consumed: list[QueryIdentity] = []

    @property
    def consumed(self) -> int:
        return len(self._consumed)

    @property
    def next_identity(self) -> QueryIdentity | None:
        return None if self.consumed == len(self._schedule) else self._schedule[self.consumed]

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(item.label for item in self._consumed)

    @property
    def schedule(self) -> tuple[QueryIdentity, ...]:
        return self._schedule

    def consume(self, label: str) -> QueryIdentity:
        expected = self.next_identity
        if expected is None:
            raise V5DResourceExceeded("V5-D full-query cap exhausted")
        if label != expected.label:
            raise V5DProtocolViolation(
                f"V5-D query schedule expected {expected.label}, received {label}"
            )
        self._consumed.append(expected)
        return expected

    def require_complete(self) -> None:
        if self.consumed != len(self._schedule):
            raise V5DProtocolViolation("V5-D query ledger is incomplete")


@dataclass(frozen=True)
class MemorySnapshot:
    allocated_bytes: int
    reserved_bytes: int


class ResourceEnvelope:
    """Check cumulative wall, artifact, and GPU-memory limits outside hot cores."""

    GIB = 1024**3

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        clock: Callable[[], float] = time.monotonic,
        artifact_bytes: Callable[[], int] = lambda: 0,
        elapsed_before: float = 0.0,
    ) -> None:
        validate_v5_d_freeze(config)
        if elapsed_before < 0:
            raise V5DProtocolViolation("Prior V5-D wall time cannot be negative")
        self._config = config
        self._clock = clock
        self._artifact_bytes = artifact_bytes
        self._started = clock() - elapsed_before
        self._eager_reserved: int | None = None
        self._peak = MemorySnapshot(0, 0)

    @property
    def peak(self) -> MemorySnapshot:
        return self._peak

    @property
    def elapsed_seconds(self) -> float:
        return self._clock() - self._started

    def set_eager_baseline(self, snapshot: MemorySnapshot) -> None:
        if self._eager_reserved is not None:
            raise V5DProtocolViolation("V5-D eager memory baseline is immutable")
        self._validate_memory_values(snapshot)
        self._eager_reserved = snapshot.reserved_bytes
        self.observe_memory(snapshot)

    @staticmethod
    def _validate_memory_values(snapshot: MemorySnapshot) -> None:
        if snapshot.allocated_bytes < 0 or snapshot.reserved_bytes < 0:
            raise V5DProtocolViolation("GPU memory counters must be non-negative")
        if snapshot.allocated_bytes > snapshot.reserved_bytes:
            raise V5DProtocolViolation("Allocated GPU memory cannot exceed reserved memory")

    def observe_memory(self, snapshot: MemorySnapshot) -> None:
        self._validate_memory_values(snapshot)
        self._peak = MemorySnapshot(
            max(self._peak.allocated_bytes, snapshot.allocated_bytes),
            max(self._peak.reserved_bytes, snapshot.reserved_bytes),
        )
        peak_cap = int(self._config["memory"]["peak_reserved_gib_max"]) * self.GIB
        if snapshot.reserved_bytes > peak_cap:
            raise V5DResourceExceeded("V5-D peak reserved-memory cap exceeded")
        if self._eager_reserved is not None:
            incremental_cap = (
                int(self._config["memory"]["incremental_reserved_gib_over_eager_max"]) * self.GIB
            )
            if snapshot.reserved_bytes - self._eager_reserved > incremental_cap:
                raise V5DResourceExceeded("V5-D incremental reserved-memory cap exceeded")

    def check_host_resources(self) -> None:
        wall_cap = float(self._config["resource_caps"]["wall_seconds"])
        if self.elapsed_seconds > wall_cap:
            raise V5DResourceExceeded("V5-D cumulative wall-time cap exceeded")
        artifact_cap = int(self._config["resource_caps"]["artifact_bytes"])
        size = self._artifact_bytes()
        if size < 0 or size > artifact_cap:
            raise V5DResourceExceeded("V5-D artifact cap exceeded")


@dataclass(frozen=True)
class MethodPatch:
    target: Any
    name: str
    replacement: Any


class MethodRestorationGuard(AbstractContextManager["MethodRestorationGuard"]):
    """Install instance-local methods and restore exact prior bindings on exit."""

    def __init__(self, patches: Sequence[MethodPatch]) -> None:
        self._patches = tuple(patches)
        self._records: list[tuple[Any, str, bool, Any]] = []
        self._entered = False

    def __enter__(self) -> "MethodRestorationGuard":
        if self._entered:
            raise V5DProtocolViolation("Method restoration guard is non-reentrant")
        try:
            for patch in self._patches:
                if not patch.name or not callable(patch.replacement):
                    raise TypeError("Method patches require a named callable")
                instance = vars(patch.target)
                had_override = patch.name in instance
                previous = instance.get(patch.name)
                self._records.append((patch.target, patch.name, had_override, previous))
                setattr(patch.target, patch.name, patch.replacement)
        except Exception:
            self._restore()
            raise
        self._entered = True
        return self

    def _restore(self) -> None:
        for target, name, had_override, previous in reversed(self._records):
            if had_override:
                setattr(target, name, previous)
            else:
                try:
                    delattr(target, name)
                except AttributeError:
                    pass
        self._records.clear()

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self._restore()
        self._entered = False


class V5DStaticBufferReuseExecutor(StaticBufferReuseExecutor):
    """V5-C execution order with the frozen real mixed-dtype tensor contract."""

    def __init__(
        self,
        *,
        prompt_input_dtype: str,
        attention_mask_dtype: str,
        **kwargs: Any,
    ) -> None:
        if prompt_input_dtype != "torch.int64":
            raise ValueError("V5-D prepared prompt IDs require torch.int64")
        if attention_mask_dtype not in ("torch.int64", "torch.bool"):
            raise ValueError("V5-D attention mask requires pinned torch.int64 or torch.bool")
        self.prompt_input_dtype = prompt_input_dtype
        self.attention_mask_dtype = attention_mask_dtype
        super().__init__(**kwargs)

    def _prepare_buffers(self, key: ReuseCompatibilityKey) -> dict[str, Any]:
        typed_shapes = {
            "wrist_pixels": (key.wrist_shape, key.dtype),
            "cached_scene_tokens": (key.cached_scene_shape, key.dtype),
            "prompt_input": (key.prompt_input_shape, self.prompt_input_dtype),
            "prompt_embeddings": (key.embedding_shape, key.dtype),
            "attention_mask": (key.attention_mask_shape, self.attention_mask_dtype),
            "wrist_tokens": (key.cached_scene_shape, key.dtype),
            "combined_tokens": (
                (1, key.patch_count * 2, key.projected_dimension),
                key.dtype,
            ),
            "normalized_actions": (key.action_shape, key.dtype),
        }
        if key.proprioception_shape:
            typed_shapes["proprioception"] = (key.proprioception_shape, key.dtype)
        return {
            name: self.tensor_ops.allocate(shape, dtype=dtype, device=key.device)
            for name, (shape, dtype) in typed_shapes.items()
        }

    @staticmethod
    def _mixed_metadata(
        value: Any,
        *,
        shape: tuple[int, ...],
        dtype: str,
        device: str,
        name: str,
    ) -> None:
        if value is None:
            raise ReuseExecutorUnavailable(f"{name} is missing")
        try:
            actual_shape = tuple(int(item) for item in value.shape)
        except Exception as error:
            raise ReuseExecutorUnavailable(f"{name} lacks valid shape metadata") from error
        if (
            actual_shape != shape
            or str(getattr(value, "dtype", "")) != dtype
            or str(getattr(value, "device", "")) != device
        ):
            raise ReuseExecutorUnavailable(f"{name} metadata differs from V5-D contract")

    def _typed_contract(self, key: ReuseCompatibilityKey) -> dict[str, tuple[tuple[int, ...], str]]:
        contract = {
            "wrist_pixels": (key.wrist_shape, key.dtype),
            "cached_scene_tokens": (key.cached_scene_shape, key.dtype),
            "prompt_input": (key.prompt_input_shape, self.prompt_input_dtype),
            "prompt_embeddings": (key.embedding_shape, key.dtype),
            "attention_mask": (key.attention_mask_shape, self.attention_mask_dtype),
            "wrist_tokens": (key.cached_scene_shape, key.dtype),
            "combined_tokens": (
                (1, key.patch_count * 2, key.projected_dimension),
                key.dtype,
            ),
            "normalized_actions": (key.action_shape, key.dtype),
        }
        if key.proprioception_shape:
            contract["proprioception"] = (key.proprioception_shape, key.dtype)
        return contract

    def _validate_owned_buffers(self, key: ReuseCompatibilityKey) -> None:
        contract = self._typed_contract(key)
        if set(self._buffers) != set(contract):
            raise ReuseExecutorUnavailable("V5-D static executor buffer set is incomplete")
        for name, (shape, dtype) in contract.items():
            self._mixed_metadata(
                self._buffers[name],
                shape=shape,
                dtype=dtype,
                device=key.device,
                name=name,
            )

    def _validate_prelaunch(self, inputs: ReuseExecutionInputs) -> None:
        if self.lifecycle is not ExecutorLifecycle.PREPARED:
            raise ReuseExecutorUnavailable(f"Executor is {self.lifecycle.value}, not PREPARED")
        if threading.get_ident() != self._owner_thread:
            raise ReuseExecutorUnavailable("Executor query belongs to another thread")
        if inputs.compatibility_key != self._key:
            raise ReuseExecutorUnavailable("Executor compatibility key mismatch")
        key = inputs.compatibility_key
        self._validate_owned_buffers(key)
        supplied = {
            "wrist_pixels": inputs.wrist_pixels,
            "cached_scene_tokens": inputs.cached_scene_tokens,
            "prompt_input": inputs.prompt_input,
            "prompt_embeddings": inputs.prompt_embeddings,
            "attention_mask": inputs.attention_mask,
        }
        if key.proprioception_shape:
            supplied["proprioception"] = inputs.proprioception
        elif inputs.proprioception is not None:
            raise ReuseExecutorUnavailable("Unexpected proprioception for disabled key")
        contract = self._typed_contract(key)
        for name, value in supplied.items():
            shape, dtype = contract[name]
            self._mixed_metadata(value, shape=shape, dtype=dtype, device=key.device, name=name)

    def owned_buffers_for_backend_preparation(self) -> Mapping[str, Any]:
        if self.lifecycle is not ExecutorLifecycle.PREPARED:
            raise V5DProtocolViolation("Backend buffers require a prepared V5-C executor")
        return MappingProxyType(dict(self._buffers))


class V5DEagerReuseExecutor(EagerReuseExecutor):
    """Allocation-based V5-C oracle with the frozen real mixed dtypes."""

    def __init__(
        self,
        *,
        prompt_input_dtype: str,
        attention_mask_dtype: str,
        **kwargs: Any,
    ) -> None:
        if prompt_input_dtype != "torch.int64":
            raise ValueError("V5-D prepared prompt IDs require torch.int64")
        if attention_mask_dtype not in ("torch.int64", "torch.bool"):
            raise ValueError("V5-D attention mask requires pinned torch.int64 or torch.bool")
        self.prompt_input_dtype = prompt_input_dtype
        self.attention_mask_dtype = attention_mask_dtype
        super().__init__(**kwargs)

    def _validate_prelaunch(self, inputs: ReuseExecutionInputs) -> None:
        if self.lifecycle is not ExecutorLifecycle.PREPARED:
            raise ReuseExecutorUnavailable(f"Executor is {self.lifecycle.value}, not PREPARED")
        if threading.get_ident() != self._owner_thread:
            raise ReuseExecutorUnavailable("Executor query belongs to another thread")
        if inputs.compatibility_key != self._key:
            raise ReuseExecutorUnavailable("Executor compatibility key mismatch")
        key = inputs.compatibility_key
        contract = {
            "wrist_pixels": (inputs.wrist_pixels, key.wrist_shape, key.dtype),
            "cached_scene_tokens": (
                inputs.cached_scene_tokens,
                key.cached_scene_shape,
                key.dtype,
            ),
            "prompt_input": (
                inputs.prompt_input,
                key.prompt_input_shape,
                self.prompt_input_dtype,
            ),
            "prompt_embeddings": (
                inputs.prompt_embeddings,
                key.embedding_shape,
                key.dtype,
            ),
            "attention_mask": (
                inputs.attention_mask,
                key.attention_mask_shape,
                self.attention_mask_dtype,
            ),
        }
        if key.proprioception_shape:
            contract["proprioception"] = (
                inputs.proprioception,
                key.proprioception_shape,
                key.dtype,
            )
        elif inputs.proprioception is not None:
            raise ReuseExecutorUnavailable("Unexpected proprioception for disabled key")
        for name, (value, shape, dtype) in contract.items():
            V5DStaticBufferReuseExecutor._mixed_metadata(
                value, shape=shape, dtype=dtype, device=key.device, name=name
            )
