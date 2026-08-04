#!/usr/bin/env python3
"""Run the frozen paired ACR V3-D LIBERO-Object development matrix."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
from collections import deque
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_ROOT = Path("/home/ved/SAVR")
PRIMARY_RUN_ID = "acr-v3d-paired-object-dev03-09-v01"
RECOVERY_RUN_ID = "acr-v3d-paired-object-dev03-09-recovery-v01"
RECOVERY_2_RUN_ID = "acr-v3d-paired-object-dev03-09-recovery-02-v01"
RUN_ID = PRIMARY_RUN_ID
PHASE = "V3-D"
SUITE = "libero_object"
TASK_IDS = tuple(range(10))
STATE_IDS = tuple(range(3, 10))
POLICIES = ("batched-fr", "sa-bdp-acr-t25-h2-b30-v01")
SEED = 0
ATTEMPT_CAP = 140
CUMULATIVE_RECOVERY_ATTEMPT_CAP = 141
CUMULATIVE_RECOVERY_2_ATTEMPT_CAP = 143
WALL_CAP_SECONDS = 43_200
ARTIFACT_CAP_BYTES = 2 * 1024**3
STEADY_EXCLUSIONS = frozenset({0, 1, 2})
CHECKPOINT_REVISION = "638918f3d1c2e43a39a8a20772bdb8b91835e4b7"
OPENVLA_REVISION = "e4287e94541f459edc4feabc4e181f537cd569a8"
LIBERO_REVISION = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
CHECKPOINT_RELATIVE = Path("checkpoints/openvla-7b-oft-libero-four-suite")
CONFIG_RELATIVE = Path("configs/acr/v3_d_development.json")
RECOVERY_CONFIG_RELATIVE = Path("configs/acr/v3_d_recovery.json")
RECOVERY_2_CONFIG_RELATIVE = Path("configs/acr/v3_d_recovery_2.json")


class Interrupted(RuntimeError):
    """Raised when an operator terminates the bounded runner."""


class ResourceCap(RuntimeError):
    """Raised before a frozen V3-D resource cap is crossed."""


class InvariantFailure(RuntimeError):
    """Raised when work, cache, action, record, or restoration truth fails."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def value_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: Any, np: Any) -> str:
    return hashlib.sha256(np.asarray(value).tobytes()).hexdigest()


def directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def git_revision(path: Path, expected: str | None = None) -> str:
    revision = subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "-C", str(path), "status", "--porcelain"], text=True
    ).strip()
    if expected is not None and revision != expected:
        raise RuntimeError(f"Expected {expected} at {path}, found {revision}")
    if status:
        raise RuntimeError(f"Refusing to use dirty source tree: {path}")
    return revision


def selected_gpu_snapshot(physical_gpu_id: str) -> dict[str, Any]:
    fields = "index,uuid,name,memory.total,memory.used,utilization.gpu"
    output = subprocess.check_output(
        [
            "nvidia-smi",
            f"--id={physical_gpu_id}",
            f"--query-gpu={fields}",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    rows = [row.strip() for row in output.splitlines() if row.strip()]
    if len(rows) != 1:
        raise RuntimeError("Selected GPU did not resolve exactly one aggregate row")
    values = [value.strip() for value in rows[0].split(",")]
    if len(values) != 6 or values[0] != physical_gpu_id:
        raise RuntimeError("Selected GPU snapshot identity is inconsistent")
    return {
        "index": int(values[0]),
        "uuid": values[1],
        "name": values[2],
        "memory_total_mib": int(values[3]),
        "memory_used_mib": int(values[4]),
        "utilization_percent": int(values[5]),
        "recorded_at_utc": utc_now(),
    }


def pair_index(task_id: int, state_id: int) -> int:
    if task_id not in TASK_IDS or state_id not in STATE_IDS:
        raise ValueError("V3-D pair lies outside the frozen population")
    return task_id * len(STATE_IDS) + state_id - STATE_IDS[0]


def policy_order(task_id: int, state_id: int) -> tuple[str, str]:
    return POLICIES if pair_index(task_id, state_id) % 2 == 0 else (POLICIES[1], POLICIES[0])


def schedule() -> tuple[tuple[int, int, str], ...]:
    return tuple(
        (task_id, state_id, policy)
        for task_id in TASK_IDS
        for state_id in STATE_IDS
        for policy in policy_order(task_id, state_id)
    )


def validate_frozen_config(config: dict[str, Any]) -> None:
    if (
        config.get("schema_version") != "acr.v3d-development.v1"
        or config.get("phase") != PHASE
        or config.get("run_id") != PRIMARY_RUN_ID
        or config.get("suite") != SUITE
        or config.get("task_ids") != list(TASK_IDS)
        or config.get("initial_state_ids") != list(STATE_IDS)
        or config.get("seed") != SEED
        or config.get("policies") != list(POLICIES)
    ):
        raise ValueError("Frozen V3-D identity changed")
    if config.get("counterbalance") != {
        "unit": ["task_id", "initial_state_id", "seed"],
        "rule": "even_pair_index_batched_fr_first_odd_pair_index_v3_first",
        "pair_index": "task_id_times_7_plus_initial_state_id_minus_3",
        "pairs": 70,
        "episodes_per_policy": 70,
        "total_episode_attempts": ATTEMPT_CAP,
    }:
        raise ValueError("Frozen V3-D counterbalance changed")
    if config.get("controller") != {
        "configuration_id": "acr-t25-h2-b30",
        "controller_version": "acr-controller-v1",
        "scene_threshold": 0.2476380718954248,
        "translation_threshold": 0.5479944908411765,
        "horizon": 2,
        "hard_reuse_cap": 0.3,
        "wrist_always_fresh": True,
    }:
        raise ValueError("Frozen V3-D controller changed")
    if config.get("resource_caps") != {
        "gpu_count": 1,
        "model_processes": 1,
        "episode_attempts": ATTEMPT_CAP,
        "wall_seconds": WALL_CAP_SECONDS,
        "artifact_bytes": ARTIFACT_CAP_BYTES,
        "downloads_allowed": False,
    }:
        raise ValueError("Frozen V3-D resource caps changed")
    if config.get("analysis") != {
        "success_pairing": ["suite", "task_id", "initial_state_id", "seed"],
        "steady_state_exclusions_per_policy": sorted(STEADY_EXCLUSIONS),
        "point_estimate": "sum_all_retained_query_measurements_divided_by_retained_query_count",
        "sequential_fr_source_run_id": "acr-a4-upstream-fr-object-dev00-09-v01",
        "sequential_fr_source_states": list(STATE_IDS),
        "sequential_fr_rerun": False,
        "outlier_deletion": False,
    }:
        raise ValueError("Frozen V3-D analysis semantics changed")
    expected_model = {
        "checkpoint_revision": CHECKPOINT_REVISION,
        "openvla_oft_revision": OPENVLA_REVISION,
        "libero_revision": LIBERO_REVISION,
        "num_open_loop_steps": 8,
        "num_images_in_input": 2,
        "use_proprio": True,
        "use_l1_regression": True,
        "use_diffusion": False,
        "use_film": False,
        "center_crop": True,
    }
    if config.get("model") != expected_model:
        raise ValueError("Frozen V3-D model configuration changed")
    frozen = config.get("provenance_freeze", {})
    required = {
        "v3_freeze_sha256": "configs/acr/v3_freeze.json",
        "v3_c_gate_sha256": "configs/acr/v3_c_gate.json",
        "v3_c_result_sha256": "reports/runtime/acr_v3_c.json",
        "a4_checkpoint_sha256": "reports/runtime/acr_a4_analysis.json",
        "batched_adapter_sha256": "src/savr/acr/batched_dual_path.py",
    }
    root = Path(__file__).resolve().parents[1]
    for key, relative in required.items():
        if frozen.get(key) != file_sha256(root / relative):
            raise ValueError(f"Frozen V3-D provenance changed: {key}")
    observed = schedule()
    if len(observed) != ATTEMPT_CAP or len(set(observed)) != ATTEMPT_CAP:
        raise ValueError("Frozen V3-D schedule is not exactly 140 unique attempts")
    first_counts = {
        policy: sum(
            order[0] == policy
            for order in map(
                lambda x: policy_order(*x), ((t, s) for t in TASK_IDS for s in STATE_IDS)
            )
        )
        for policy in POLICIES
    }
    if first_counts != {POLICIES[0]: 35, POLICIES[1]: 35}:
        raise ValueError("Frozen V3-D first-position balance changed")


def validate_recovery_config(project_root: Path, recovery: dict[str, Any]) -> None:
    expected = {
        "schema_version": "acr.v3d-recovery.v1",
        "phase": PHASE,
        "source_run_id": PRIMARY_RUN_ID,
        "recovery_run_id": RECOVERY_RUN_ID,
        "technical_attempts_preserved": 1,
        "scientific_terminal_episodes_preserved": 0,
        "recovery_episode_attempts": ATTEMPT_CAP,
        "cumulative_episode_attempt_cap": CUMULATIVE_RECOVERY_ATTEMPT_CAP,
        "cumulative_wall_seconds": WALL_CAP_SECONDS,
        "cumulative_artifact_bytes": ARTIFACT_CAP_BYTES,
        "automatic_episode_retry": False,
        "method_population_schedule_and_gates_changed": False,
    }
    for key, value in expected.items():
        if recovery.get(key) != value:
            raise ValueError(f"Frozen V3-D recovery changed: {key}")
    source = recovery.get("source_records", {})
    source_root = project_root / "results" / PRIMARY_RUN_ID
    for name, expected_hash in {
        "manifest": "a521e6d677405958896a567c2e777b9908ee728fb7e96a795a8ba5f309ec0afb",
        "completion": "d16854b2bd2d8e6a2ac4d065f868d43e78a9ee925859463c976e1a4b8a027887",
        "summary": "d15cdbb5c16eda020e35033369add77e78b25d9cde89a52987d015a6ddcde316",
    }.items():
        path = source_root / name / "record.json"
        if source.get(f"{name}_sha256") != expected_hash or file_sha256(path) != expected_hash:
            raise ValueError(f"Preserved V3-D technical {name} record changed")
    summary = json.loads((source_root / "summary/record.json").read_text(encoding="utf-8"))
    if (
        summary.get("status") != "failed"
        or summary.get("attempts_started") != 1
        or summary.get("terminal_records") != 1
        or sum(summary.get("query_counts_per_policy", {}).values()) != 0
        or summary.get("error_type") != "TypeError"
        or summary.get("error")
        != "isfinite(): argument 'input' (position 1) must be Tensor, not list"
        or summary.get("restoration_error") is not None
        or summary.get("checkpoint_before") != summary.get("checkpoint_after")
    ):
        raise ValueError("Preserved V3-D technical attempt is not recovery-eligible")


def validate_recovery_2_config(project_root: Path, recovery: dict[str, Any]) -> None:
    expected = {
        "schema_version": "acr.v3d-recovery-2.v1",
        "phase": PHASE,
        "source_run_id": RECOVERY_RUN_ID,
        "recovery_run_id": RECOVERY_2_RUN_ID,
        "cumulative_attempts_preserved": 3,
        "official_scientific_episodes_preserved": 0,
        "excluded_completed_bfr_episodes": 1,
        "recovery_episode_attempts": ATTEMPT_CAP,
        "cumulative_episode_attempt_cap": CUMULATIVE_RECOVERY_2_ATTEMPT_CAP,
        "cumulative_wall_seconds": WALL_CAP_SECONDS,
        "cumulative_artifact_bytes": ARTIFACT_CAP_BYTES,
        "automatic_episode_retry": False,
        "method_population_schedule_and_gates_changed": False,
    }
    for key, value in expected.items():
        if recovery.get(key) != value:
            raise ValueError(f"Frozen V3-D recovery 2 changed: {key}")
    source = recovery.get("source_records", {})
    source_root = project_root / "results" / RECOVERY_RUN_ID
    for name, expected_hash in {
        "manifest": "25e607947d07a97ab4c3f198826cceae552ab9a1cfe31aa67c59bba7acfb78eb",
        "completion": "731e1508cce06f52e332af2e61d3c48d2e4fafa803c0409c63fe1e2be15f5dec",
        "summary": "b8853c72664f0ec65e6aa30c85c9ac2d565d24cccac47c9c2d251350c23b61ce",
    }.items():
        path = source_root / name / "record.json"
        if source.get(f"{name}_sha256") != expected_hash or file_sha256(path) != expected_hash:
            raise ValueError(f"Preserved V3-D recovery-1 {name} record changed")
    summary = json.loads((source_root / "summary/record.json").read_text(encoding="utf-8"))
    if (
        summary.get("status") != "failed"
        or summary.get("attempts_started") != 2
        or summary.get("cumulative_attempts_started") != 3
        or summary.get("terminal_records") != 2
        or summary.get("query_counts_per_policy", {}).get(POLICIES[0]) != 35
        or summary.get("query_counts_per_policy", {}).get(POLICIES[1]) != 0
        or summary.get("error_type") != "ValueError"
        or summary.get("error") != "Controller and context configuration identities differ"
        or summary.get("restoration_error") is not None
        or summary.get("checkpoint_before") != summary.get("checkpoint_after")
    ):
        raise ValueError("Preserved V3-D recovery-1 stop is not recovery-2 eligible")


def context_configuration_id(policy: str) -> str:
    if policy == POLICIES[0]:
        return "batched-full-refresh"
    if policy == POLICIES[1]:
        return "acr-t25-h2-b30"
    raise ValueError("Unsupported V3-D policy context")


def visual_cuda_ms(result: Any) -> float:
    timing = result.device_timing
    if timing is None:
        raise InvariantFailure("V3-D requires synchronized CUDA timing")
    names = (
        "batched.siglip",
        "batched.dinov2",
        "batched.projector",
        "wrist.siglip",
        "wrist.dinov2",
        "wrist.projector",
    )
    return sum(float(timing.component_device_ms.get(name, 0.0)) for name in names)


def action_is_finite(value: Any, np: Any) -> bool:
    """Accept the pinned evaluator's list/NumPy action representation."""

    try:
        return bool(np.isfinite(np.asarray(value)).all())
    except (TypeError, ValueError):
        return False


def query_record(
    *,
    identity: Any,
    policy: str,
    query_index: int,
    global_policy_query_index: int,
    environment_step: int,
    result: Any,
    context: Any,
    scene: Any,
    wrist: Any,
    raw_state: tuple[float, ...],
    normalized_state: tuple[float, ...],
    actions: Any,
    configuration_sha256: str,
    savr_revision: str,
    model_dtype: str,
    model_device: str,
    visual_token_count: int,
    np: Any,
) -> dict[str, Any]:
    is_v3 = policy == POLICIES[1]
    decision = result.decision if is_v3 else None
    refresh = bool(decision.refresh) if decision is not None else True
    result.work.validate()
    work = result.work
    record = {
        "schema_version": "acr.v3d-query.v1",
        "run_id": RUN_ID,
        "attempt_id": identity.value,
        "query_id": identity.query_id(query_index),
        "phase": PHASE,
        "policy": policy,
        "suite": SUITE,
        "task_id": identity.task_id,
        "initial_state_id": identity.initial_state_id,
        "seed": identity.seed,
        "query_index": query_index,
        "global_policy_query_index": global_policy_query_index,
        "steady_state": global_policy_query_index not in STEADY_EXCLUSIONS,
        "environment_step": environment_step,
        "status": "completed",
        "decision": {
            "scene_refresh": refresh,
            "reasons": list(decision.reasons) if decision is not None else ["policy"],
            "cache_age_before": decision.cache_age_before if decision is not None else None,
            "reference_query_index": decision.reference_query_index
            if decision is not None
            else None,
            "scene_score": decision.scene_score if decision is not None else None,
            "translation_score": decision.translation_score if decision is not None else None,
            "gripper_transition_veto": decision.gripper_transition_veto
            if decision is not None
            else None,
        },
        "camera_work": {
            "physical_siglip_calls": work.physical_siglip_calls,
            "physical_dinov2_calls": work.physical_dinov2_calls,
            "physical_projector_calls": work.physical_projector_calls,
            "logical_scene_backbone_calls": work.logical_scene_backbone_calls,
            "logical_wrist_backbone_calls": work.logical_wrist_backbone_calls,
            "logical_scene_projector_calls": work.logical_scene_projector_calls,
            "logical_wrist_projector_calls": work.logical_wrist_projector_calls,
            "downstream_calls": work.downstream_calls,
            "visual_token_count": visual_token_count,
            "token_order": "scene-wrist",
            "dtype": model_dtype,
            "device": model_device,
        },
        "timing": {
            "query_wall_ms": float(result.query_wall_ms),
            "query_cuda_ms": float(result.device_timing.total_device_ms),
            "total_visual_cuda_ms": visual_cuda_ms(result),
            "controller_wall_ms": float(result.controller_wall_ms) if is_v3 else 0.0,
        },
        "inputs": {
            "scene_image_sha256": array_sha256(scene, np),
            "wrist_image_sha256": array_sha256(wrist, np),
            "proprio_sha256": array_sha256(raw_state, np),
            "action_sha256": array_sha256(actions, np),
            "context_sha256": value_sha256(asdict(context)),
            "proprio_normalized": list(normalized_state),
        },
        "provenance": {
            "configuration_sha256": configuration_sha256,
            "savr_revision": savr_revision,
            "openvla_oft_revision": OPENVLA_REVISION,
            "checkpoint_revision": CHECKPOINT_REVISION,
            "recorded_at_utc": utc_now(),
        },
    }
    record["semantic_sha256"] = value_sha256(record)
    return record


def query_counts(record: dict[str, Any]) -> dict[str, int]:
    refresh = int(record["decision"]["scene_refresh"])
    work = record["camera_work"]
    counts = {
        "queries": 1,
        "scene_refreshes": refresh,
        "scene_reuses": 1 - refresh,
        "wrist_refreshes": 1,
        "scene_siglip_calls": int(work["logical_scene_backbone_calls"]),
        "scene_dinov2_calls": int(work["logical_scene_backbone_calls"]),
        "scene_projector_calls": int(work["logical_scene_projector_calls"]),
        "wrist_siglip_calls": int(work["logical_wrist_backbone_calls"]),
        "wrist_dinov2_calls": int(work["logical_wrist_backbone_calls"]),
        "wrist_projector_calls": int(work["logical_wrist_projector_calls"]),
        "downstream_calls": int(work["downstream_calls"]),
    }
    return counts


def add_counts(target: dict[str, int], source: dict[str, int]) -> None:
    for key, value in source.items():
        target[key] += value


def main() -> int:
    global RUN_ID
    parser = argparse.ArgumentParser()
    parser.add_argument("--recovery", action="store_true")
    parser.add_argument("--recovery-2", action="store_true")
    arguments = parser.parse_args()
    if arguments.recovery and arguments.recovery_2:
        raise SystemExit("Select exactly one V3-D recovery mode")
    project_root = Path(__file__).resolve().parents[1]
    if project_root != EXPECTED_ROOT:
        raise SystemExit(f"Refusing to run outside {EXPECTED_ROOT}: {project_root}")
    physical_gpu_id = os.environ.get("SAVR_PHYSICAL_GPU_ID")
    selected_uuid = os.environ.get("SAVR_SELECTED_GPU_UUID")
    if (
        not physical_gpu_id
        or os.environ.get("CUDA_VISIBLE_DEVICES") != physical_gpu_id
        or not selected_uuid
    ):
        raise SystemExit("Selected GPU ID/UUID variables are incomplete or inconsistent")

    def handle_signal(_signum: int, _frame: Any) -> None:
        raise Interrupted("V3-D runner received a termination signal")

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    for key, relative in {
        "HF_HOME": "cache/huggingface",
        "HF_HUB_CACHE": "cache/huggingface/hub",
        "LIBERO_CONFIG_PATH": "cache/libero",
        "TORCH_HOME": "cache/torch",
    }.items():
        os.environ[key] = str(project_root / relative)
    os.environ.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "MUJOCO_GL": "osmesa",
            "PYOPENGL_PLATFORM": "osmesa",
            "PYTHONNOUSERSITE": "1",
            "WANDB_MODE": "disabled",
            "TOKENIZERS_PARALLELISM": "false",
            "TF_CPP_MIN_LOG_LEVEL": "2",
        }
    )

    config_path = project_root / CONFIG_RELATIVE
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_frozen_config(config)
    recovery: dict[str, Any] | None = None
    prior_attempts = 0
    prior_wall_seconds = 0.0
    prior_artifact_bytes = 0
    if arguments.recovery:
        recovery_path = project_root / RECOVERY_CONFIG_RELATIVE
        recovery = json.loads(recovery_path.read_text(encoding="utf-8"))
        validate_recovery_config(project_root, recovery)
        RUN_ID = RECOVERY_RUN_ID
        prior_summary = json.loads(
            (project_root / "results" / PRIMARY_RUN_ID / "summary" / "record.json").read_text(
                encoding="utf-8"
            )
        )
        prior_attempts = int(prior_summary["attempts_started"])
        prior_wall_seconds = float(prior_summary["elapsed_seconds"])
        prior_artifact_bytes = directory_size(project_root / "results" / PRIMARY_RUN_ID)
    elif arguments.recovery_2:
        recovery_path = project_root / RECOVERY_2_CONFIG_RELATIVE
        recovery = json.loads(recovery_path.read_text(encoding="utf-8"))
        validate_recovery_2_config(project_root, recovery)
        RUN_ID = RECOVERY_2_RUN_ID
        prior_summary = json.loads(
            (project_root / "results" / RECOVERY_RUN_ID / "summary" / "record.json").read_text(
                encoding="utf-8"
            )
        )
        prior_attempts = int(prior_summary["cumulative_attempts_started"])
        prior_wall_seconds = float(prior_summary["cumulative_wall_seconds"])
        prior_artifact_bytes = int(prior_summary["cumulative_artifact_bytes"])
    run_configuration_sha256 = value_sha256(
        {
            "config_sha256": file_sha256(config_path),
            "recovery_config_sha256": (
                file_sha256(
                    project_root
                    / (
                        RECOVERY_2_CONFIG_RELATIVE
                        if arguments.recovery_2
                        else RECOVERY_CONFIG_RELATIVE
                    )
                )
                if recovery is not None
                else None
            ),
            "schedule": schedule(),
        }
    )
    upstream_root = project_root / "third_party/openvla-oft"
    libero_root = project_root / "third_party/LIBERO"
    checkpoint = project_root / CHECKPOINT_RELATIVE
    run_dir = project_root / "results" / RUN_ID
    if run_dir.exists():
        raise SystemExit(f"Immutable V3-D run already exists: {run_dir}")

    sys.path.insert(0, str(project_root / "src"))
    sys.path.insert(0, str(project_root / "scripts"))
    sys.path.insert(0, str(upstream_root))
    from run_acr_correctness import validate_checkpoint

    import numpy as np
    import torch  # type: ignore[import-not-found]
    from libero.libero import benchmark  # type: ignore[import-not-found]
    from experiments.robot.libero import run_libero_eval as upstream_eval  # type: ignore[import-not-found]
    from experiments.robot.libero.libero_utils import (  # type: ignore[import-not-found]
        get_libero_dummy_action,
        get_libero_env,
        quat2axisangle,
    )
    from experiments.robot.robot_utils import (  # type: ignore[import-not-found]
        get_image_resize_size,
        set_seed_everywhere,
    )
    from run_phase5_core_smoke import raw_robot_state
    from savr.acr.batched_dual_path import (
        BatchedDualPathOpenVLAAdapter,
        BatchedFullRefreshAdapter,
    )
    from savr.acr.controller import ACRController
    from savr.acr.instrumentation import CameraInstrumentation
    from savr.acr.openvla_oft import TorchTensorOperations
    from savr.acr.records import AttemptIdentity, ImmutableRecordStore, reconcile_episode_counts
    from savr.acr.types import ACRConfiguration, ACRContext, ACRPolicy
    from savr.signals import normalize_bounds
    from savr.timing import SynchronizedQueryTimer, TorchCudaEventBackend

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one selected CUDA device must be visible")
    savr_revision = git_revision(project_root)
    upstream_revision = git_revision(upstream_root, OPENVLA_REVISION)
    libero_revision = git_revision(libero_root, LIBERO_REVISION)
    checkpoint_before = validate_checkpoint(project_root, checkpoint)
    protected_names = ("config.json", "configuration_prismatic.py", "modeling_prismatic.py")
    protected_bytes = {name: (checkpoint / name).read_bytes() for name in protected_names}
    checkpoint_files_before = {item.name for item in checkpoint.iterdir()}
    gpu_before = selected_gpu_snapshot(physical_gpu_id)
    if gpu_before["uuid"] != selected_uuid:
        raise RuntimeError("Selected GPU UUID changed between selection and launch")

    run_dir.mkdir(parents=True, exist_ok=False)
    store = ImmutableRecordStore(run_dir)
    planned = [
        AttemptIdentity(RUN_ID, policy, "libero-object", task, state, SEED, 1)
        for task, state, policy in schedule()
    ]
    manifest = {
        "schema_version": "acr.v3d-run.v1",
        "run_id": RUN_ID,
        "phase": PHASE,
        "status": "running",
        "configuration_sha256": run_configuration_sha256,
        "population": {
            "task_ids": list(TASK_IDS),
            "initial_state_ids": list(STATE_IDS),
            "seed": SEED,
        },
        "counterbalance": config["counterbalance"],
        "planned_attempts": [identity.value for identity in planned],
        "resource_caps": recovery if recovery is not None else config["resource_caps"],
        "recovery": arguments.recovery or arguments.recovery_2,
        "recovery_index": 2 if arguments.recovery_2 else 1 if arguments.recovery else None,
        "source_technical_run_id": (
            RECOVERY_RUN_ID
            if arguments.recovery_2
            else PRIMARY_RUN_ID
            if arguments.recovery
            else None
        ),
        "outcome_blind_until_complete": True,
        "revisions": {
            "savr": savr_revision,
            "openvla_oft": upstream_revision,
            "libero": libero_revision,
            "checkpoint": CHECKPOINT_REVISION,
        },
        "host": socket.gethostname(),
        "selected_gpu_id": int(physical_gpu_id),
        "selected_gpu_uuid": selected_uuid,
        "started_at_utc": utc_now(),
        "finished_at_utc": None,
        "terminal_record_ids_sha256": None,
    }
    store.write_once("manifest", manifest)

    cfg = upstream_eval.GenerateConfig(
        pretrained_checkpoint=str(checkpoint),
        task_suite_name=SUITE,
        num_trials_per_task=len(STATE_IDS),
        seed=SEED,
        local_log_dir=str(run_dir / "logs"),
        use_wandb=False,
        center_crop=True,
        num_open_loop_steps=8,
        num_images_in_input=2,
        use_proprio=True,
        use_l1_regression=True,
        use_diffusion=False,
        use_film=False,
    )
    upstream_eval.validate_config(cfg)
    controller_configuration = ACRConfiguration(
        "acr-t25-h2-b30",
        ACRPolicy.SA_ACR,
        scene_threshold=0.2476380718954248,
        translation_threshold=0.5479944908411765,
        horizon=2,
        hard_reuse_cap=0.3,
    )
    set_seed_everywhere(SEED)
    model = action_head = proprio_projector = noisy_action_projector = processor = None
    current_env = None
    caught: BaseException | None = None
    restoration_error: BaseException | None = None
    terminal_status = "failed"
    attempts_started = 0
    terminal_record_ids: list[str] = []
    global_policy_queries = {policy: 0 for policy in POLICIES}
    aggregate_counts = {
        policy: {
            name: 0
            for name in (
                "queries",
                "scene_refreshes",
                "scene_reuses",
                "wrist_refreshes",
                "scene_siglip_calls",
                "scene_dinov2_calls",
                "scene_projector_calls",
                "wrist_siglip_calls",
                "wrist_dinov2_calls",
                "wrist_projector_calls",
                "downstream_calls",
            )
        }
        for policy in POLICIES
    }
    run_started = time.monotonic()
    try:
        os.chdir(upstream_root)
        model, action_head, proprio_projector, noisy_action_projector, processor = (
            upstream_eval.initialize_model(cfg)
        )
        torch.cuda.synchronize()
        if action_head is None or proprio_projector is None or processor is None:
            raise RuntimeError("Pinned L1/proprio/processor modules were not loaded")
        statistics = model.norm_stats[cfg.unnorm_key]["proprio"]
        q01, q99 = statistics["q01"], statistics["q99"]
        parameter = next(model.parameters())
        model_dtype, model_device = str(parameter.dtype), str(parameter.device)
        patch_count = int(model.vision_backbone.get_num_patches())
        visual_token_count = patch_count * 2
        resize_size = get_image_resize_size(cfg)
        task_suite = benchmark.get_benchmark_dict()[SUITE]()
        for task_id in TASK_IDS:
            task = task_suite.get_task(task_id)
            initial_states = task_suite.get_task_init_states(task_id)
            current_env, task_description = get_libero_env(
                task, cfg.model_family, resolution=cfg.env_img_res
            )
            try:
                for state_id in STATE_IDS:
                    for policy in policy_order(task_id, state_id):
                        cumulative_attempt_cap = (
                            CUMULATIVE_RECOVERY_2_ATTEMPT_CAP
                            if arguments.recovery_2
                            else CUMULATIVE_RECOVERY_ATTEMPT_CAP
                            if arguments.recovery
                            else ATTEMPT_CAP
                        )
                        if prior_attempts + attempts_started >= cumulative_attempt_cap:
                            raise ResourceCap("V3-D episode-attempt cap exhausted")
                        if prior_wall_seconds + time.monotonic() - run_started >= WALL_CAP_SECONDS:
                            raise ResourceCap("V3-D wall-time cap reached before scheduling")
                        if prior_artifact_bytes + directory_size(run_dir) >= ARTIFACT_CAP_BYTES:
                            raise ResourceCap("V3-D artifact cap reached before scheduling")
                        identity = AttemptIdentity(
                            RUN_ID, policy, "libero-object", task_id, state_id, SEED, 1
                        )
                        attempts_started += 1
                        episode_started = time.perf_counter()
                        episode_started_at = utc_now()
                        torch.cuda.reset_peak_memory_stats()
                        current_env.reset()
                        observation = current_env.set_init_state(initial_states[state_id])
                        action_queue: deque[Any] = deque(maxlen=cfg.num_open_loop_steps)
                        environment_step = query_index = control_steps = 0
                        success = False
                        query_records: list[dict[str, Any]] = []
                        query_wall: list[float] = []
                        visual_cuda: list[float] = []
                        steady_query_wall: list[float] = []
                        steady_visual_cuda: list[float] = []
                        trajectory = hashlib.sha256()
                        episode_counts = {name: 0 for name in aggregate_counts[policy]}
                        peak_scene_cache_bytes = 0
                        episode_error: BaseException | None = None
                        context = ACRContext(
                            episode_id=identity.episode_id,
                            attempt_id=identity.value,
                            task_id=f"{task_id:02d}",
                            instruction_sha256=hashlib.sha256(
                                task_description.encode()
                            ).hexdigest(),
                            checkpoint_id=CHECKPOINT_REVISION,
                            upstream_revision=OPENVLA_REVISION,
                            configuration_id=context_configuration_id(policy),
                            controller_version=controller_configuration.controller_version
                            if policy == POLICIES[1]
                            else "none",
                            preprocessing_id="openvla-center-crop-v1",
                            action_head_id="l1-regression-8x7",
                            dtype=model_dtype,
                            device=model_device,
                            patch_count=patch_count,
                        )
                        instrumentation = CameraInstrumentation(
                            timer=SynchronizedQueryTimer(TorchCudaEventBackend(torch))
                        )
                        tensor_ops = TorchTensorOperations(torch)
                        adapter = (
                            BatchedFullRefreshAdapter(
                                model=model,
                                tensor_ops=tensor_ops,
                                instrumentation=instrumentation,
                                action_finite_checker=lambda value: action_is_finite(value, np),
                            )
                            if policy == POLICIES[0]
                            else BatchedDualPathOpenVLAAdapter(
                                model=model,
                                controller=ACRController(controller_configuration),
                                tensor_ops=tensor_ops,
                                instrumentation=instrumentation,
                                action_finite_checker=lambda value: action_is_finite(value, np),
                            )
                        )
                        try:
                            with adapter.episode(context):
                                max_steps = upstream_eval.TASK_MAX_STEPS[cfg.task_suite_name]
                                while environment_step < max_steps + cfg.num_steps_wait:
                                    if (
                                        prior_wall_seconds + time.monotonic() - run_started
                                        >= WALL_CAP_SECONDS
                                    ):
                                        raise ResourceCap(
                                            "V3-D wall-time cap reached during episode"
                                        )
                                    if environment_step < cfg.num_steps_wait:
                                        observation, _, _, _ = current_env.step(
                                            get_libero_dummy_action(cfg.model_family)
                                        )
                                        environment_step += 1
                                        continue
                                    if not action_queue:
                                        policy_observation, _ = upstream_eval.prepare_observation(
                                            observation, resize_size
                                        )
                                        raw_state = tuple(
                                            float(value)
                                            for value in np.asarray(
                                                policy_observation["state"]
                                            ).reshape(-1)
                                        )
                                        normalized_state = normalize_bounds(raw_state, q01, q99)
                                        if len(raw_state) != 8 or len(normalized_state) != 8:
                                            raise InvariantFailure(
                                                "V3-D proprioception width changed"
                                            )
                                        scene = policy_observation["full_image"]
                                        wrist = policy_observation["wrist_image"]

                                        def invoke() -> Any:
                                            return upstream_eval.get_action(
                                                cfg,
                                                model,
                                                policy_observation,
                                                task_description,
                                                processor=processor,
                                                action_head=action_head,
                                                proprio_projector=proprio_projector,
                                                noisy_action_projector=noisy_action_projector,
                                                use_film=False,
                                            )

                                        if isinstance(adapter, BatchedFullRefreshAdapter):
                                            result = adapter.run_query(invoke)
                                        else:
                                            result = adapter.run_query(
                                                query=invoke,
                                                scene_image=scene,
                                                wrist_image=wrist,
                                                state=raw_state,
                                                state_q01=q01,
                                                state_q99=q99,
                                            )
                                        actions = np.asarray(result.value)
                                        if (
                                            actions.shape != (8, 7)
                                            or not np.isfinite(actions).all()
                                        ):
                                            raise InvariantFailure(
                                                f"Malformed V3-D action chunk: {actions.shape}"
                                            )
                                        global_index = global_policy_queries[policy]
                                        record = query_record(
                                            identity=identity,
                                            policy=policy,
                                            query_index=query_index,
                                            global_policy_query_index=global_index,
                                            environment_step=environment_step,
                                            result=result,
                                            context=context,
                                            scene=scene,
                                            wrist=wrist,
                                            raw_state=raw_state,
                                            normalized_state=normalized_state,
                                            actions=actions,
                                            configuration_sha256=run_configuration_sha256,
                                            savr_revision=savr_revision,
                                            model_dtype=model_dtype,
                                            model_device=model_device,
                                            visual_token_count=visual_token_count,
                                            np=np,
                                        )
                                        store.write_once(identity.query_id(query_index), record)
                                        query_records.append(record)
                                        query_wall.append(record["timing"]["query_wall_ms"])
                                        visual_cuda.append(record["timing"]["total_visual_cuda_ms"])
                                        if record["steady_state"]:
                                            steady_query_wall.append(query_wall[-1])
                                            steady_visual_cuda.append(visual_cuda[-1])
                                        add_counts(episode_counts, query_counts(record))
                                        reconcile_episode_counts(episode_counts)
                                        if (
                                            isinstance(adapter, BatchedDualPathOpenVLAAdapter)
                                            and adapter.cache.entry is not None
                                        ):
                                            tokens = adapter.cache.entry.tokens
                                            peak_scene_cache_bytes = max(
                                                peak_scene_cache_bytes,
                                                int(tokens.numel() * tokens.element_size()),
                                            )
                                        query_index += 1
                                        global_policy_queries[policy] += 1
                                        action_queue.extend(actions)
                                    action = upstream_eval.process_action(
                                        action_queue.popleft(), cfg.model_family
                                    )
                                    if not np.isfinite(action).all():
                                        raise InvariantFailure(
                                            "Processed V3-D action is non-finite"
                                        )
                                    observation, _, done, _ = current_env.step(action.tolist())
                                    control_steps += 1
                                    trajectory.update(np.asarray(action, dtype="<f8").tobytes())
                                    trajectory.update(
                                        np.asarray(
                                            raw_robot_state(observation, np, quat2axisangle),
                                            dtype="<f8",
                                        ).tobytes()
                                    )
                                    if done:
                                        success = True
                                        break
                                    environment_step += 1
                            if adapter.installed:
                                raise InvariantFailure("V3-D episode adapter was not restored")
                        except BaseException as error:
                            episode_error = error
                        if adapter.installed:
                            episode_error = InvariantFailure(
                                "V3-D episode adapter remained installed after episode exit"
                            )
                        counts = {"environment_steps": control_steps, **episode_counts}
                        reconcile_episode_counts(counts)
                        classification = (
                            "operator"
                            if isinstance(episode_error, Interrupted)
                            else "resource"
                            if isinstance(episode_error, ResourceCap)
                            else "invariant"
                            if isinstance(episode_error, InvariantFailure)
                            else "technical"
                            if episode_error is not None
                            else "scientific"
                            if not success
                            else None
                        )
                        episode_record = {
                            "schema_version": "acr.v3d-episode.v1",
                            "run_id": RUN_ID,
                            "attempt_id": identity.value,
                            "episode_id": identity.episode_id,
                            "phase": PHASE,
                            "policy": policy,
                            "suite": SUITE,
                            "task_id": task_id,
                            "initial_state_id": state_id,
                            "seed": SEED,
                            "pair_index": pair_index(task_id, state_id),
                            "pair_position": policy_order(task_id, state_id).index(policy),
                            "status": "failed" if episode_error is not None else "completed",
                            "success": None if episode_error is not None else success,
                            "failure_classification": classification,
                            "failure_reason": str(episode_error)
                            if episode_error is not None
                            else "task_not_completed_within_horizon"
                            if not success
                            else None,
                            "counts": counts,
                            "timing": {
                                "episode_wall_ms": (time.perf_counter() - episode_started) * 1000,
                                "inclusive_query_wall_ms": query_wall,
                                "steady_query_wall_ms": steady_query_wall,
                                "inclusive_visual_cuda_ms": visual_cuda,
                                "steady_visual_cuda_ms": steady_visual_cuda,
                            },
                            "memory": {
                                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                                "scene_cache_bytes": peak_scene_cache_bytes,
                            },
                            "records_sha256": value_sha256({"queries": query_records}),
                            "trajectory_sha256": trajectory.hexdigest(),
                            "configuration_sha256": run_configuration_sha256,
                            "started_at_utc": episode_started_at,
                            "finished_at_utc": utc_now(),
                        }
                        episode_record["semantic_sha256"] = value_sha256(episode_record)
                        store.write_once(identity.episode_id, episode_record)
                        terminal_record_ids.append(identity.episode_id)
                        add_counts(aggregate_counts[policy], episode_counts)
                        if episode_error is not None:
                            raise episode_error
                        if prior_artifact_bytes + directory_size(run_dir) >= ARTIFACT_CAP_BYTES:
                            raise ResourceCap("V3-D artifact cap reached")
            finally:
                current_env.close()
                current_env = None
        if attempts_started != ATTEMPT_CAP or len(terminal_record_ids) != ATTEMPT_CAP:
            raise InvariantFailure("V3-D terminal matrix did not reconcile")
        for policy in POLICIES:
            reconcile_episode_counts(aggregate_counts[policy])
        if terminal_record_ids != [identity.episode_id for identity in planned]:
            raise InvariantFailure("V3-D execution order differs from the frozen schedule")
        terminal_status = "completed"
    except BaseException as error:
        caught = error
        terminal_status = "interrupted" if isinstance(error, Interrupted) else "failed"
    finally:
        if current_env is not None:
            current_env.close()
        model = action_head = proprio_projector = noisy_action_projector = processor = None
        gc.collect()
        torch.cuda.empty_cache()
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        checkpoint_after = gpu_after = None
        upstream_after = libero_after = None
        try:
            for name, content in protected_bytes.items():
                (checkpoint / name).write_bytes(content)
            new_files = {item.name for item in checkpoint.iterdir()} - checkpoint_files_before
            unexpected = sorted(name for name in new_files if ".back." not in name)
            for name in sorted(new_files):
                if ".back." in name:
                    (checkpoint / name).unlink()
            if unexpected:
                raise RuntimeError(f"Unexpected checkpoint files after V3-D: {unexpected}")
            if any(
                (checkpoint / name).read_bytes() != content
                for name, content in protected_bytes.items()
            ):
                raise RuntimeError("V3-D checkpoint protected bytes were not restored")
            checkpoint_after = validate_checkpoint(project_root, checkpoint)
            upstream_after = git_revision(upstream_root, OPENVLA_REVISION)
            libero_after = git_revision(libero_root, LIBERO_REVISION)
            gpu_after = selected_gpu_snapshot(physical_gpu_id)
            if gpu_after["uuid"] != selected_uuid:
                raise RuntimeError("Selected GPU UUID changed during V3-D")
        except BaseException as error:
            restoration_error = error
            caught = caught or error
            terminal_status = "failed"
        elapsed_seconds = time.monotonic() - run_started
        artifact_bytes = directory_size(run_dir)
        cumulative_wall_seconds = prior_wall_seconds + elapsed_seconds
        cumulative_artifact_bytes = prior_artifact_bytes + artifact_bytes
        if cumulative_wall_seconds > WALL_CAP_SECONDS:
            caught = caught or ResourceCap("V3-D wall cap exceeded")
            terminal_status = "failed"
        if cumulative_artifact_bytes > ARTIFACT_CAP_BYTES:
            caught = caught or ResourceCap("V3-D artifact cap exceeded")
            terminal_status = "failed"
        completion = {
            **manifest,
            "status": terminal_status,
            "finished_at_utc": utc_now(),
            "terminal_record_ids_sha256": value_sha256(terminal_record_ids),
        }
        store.write_once("completion", completion)
        summary = {
            "schema_version": "acr.v3d-run-summary.v1",
            "run_id": RUN_ID,
            "status": terminal_status,
            "outcomes_aggregated": False,
            "attempts_started": attempts_started,
            "cumulative_attempts_started": prior_attempts + attempts_started,
            "terminal_records": len(terminal_record_ids),
            "terminal_records_per_policy": {
                policy: sum(f"/{policy}/" in item for item in terminal_record_ids)
                for policy in POLICIES
            },
            "query_counts_per_policy": dict(global_policy_queries),
            "work_counts_per_policy": aggregate_counts,
            "elapsed_seconds": elapsed_seconds,
            "cumulative_wall_seconds": cumulative_wall_seconds,
            "artifact_bytes": artifact_bytes,
            "cumulative_artifact_bytes": cumulative_artifact_bytes,
            "gpu_before": gpu_before,
            "gpu_after": gpu_after,
            "checkpoint_before": checkpoint_before,
            "checkpoint_after": checkpoint_after,
            "upstream_revision_after": upstream_after,
            "libero_revision_after": libero_after,
            "configuration_sha256": run_configuration_sha256,
            "error_type": type(caught).__name__ if caught else None,
            "error": str(caught) if caught else None,
            "restoration_error": str(restoration_error) if restoration_error else None,
            "finished_at_utc": utc_now(),
        }
        summary["semantic_sha256"] = value_sha256(summary)
        store.write_once("summary", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 1 if caught is not None else 0


if __name__ == "__main__":
    raise SystemExit(main())
