"""Dependency-free split-core executors for IR-SA-ACR Version 5.

The static executor models the ownership, compatibility, and lifecycle contract
that a later CUDA implementation must preserve.  It deliberately contains no
PyTorch, compiler, timing, filesystem, serialization, or device-sync logic.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, fields
from enum import Enum
from typing import Any, Protocol


EAGER_REUSE_EXECUTOR_VERSION = "acr-reuse-executor-eager-v1"
STATIC_REUSE_EXECUTOR_VERSION = "acr-reuse-executor-static-v1"


class ExecutorLifecycle(str, Enum):
    UNPREPARED = "UNPREPARED"
    PREPARED = "PREPARED"
    ACTIVE = "ACTIVE"
    INVALIDATED = "INVALIDATED"


class ReuseExecutorError(RuntimeError):
    """Base executor failure."""


class ReuseExecutorUnavailable(ReuseExecutorError):
    """Execution was rejected before either core launched."""


class ReuseExecutorFailure(ReuseExecutorError):
    """Execution failed after at least one core launched."""


Shape = tuple[int, ...]


def _valid_shape(shape: Shape, *, allow_empty: bool = False) -> bool:
    return (
        bool(shape)
        and all(isinstance(value, int) and value > 0 for value in shape)
        or (allow_empty and shape == ())
    )


@dataclass(frozen=True)
class ReuseCompatibilityKey:
    """Complete, immutable identity and static-shape compatibility key."""

    checkpoint_id: str
    upstream_revision: str
    configuration_id: str
    controller_version: str
    executor_version: str
    preprocessing_id: str
    action_head_id: str
    instruction_sha256: str
    prompt_input_shape: Shape
    dtype: str
    device: str
    image_height: int
    image_width: int
    patch_count: int
    projected_dimension: int
    wrist_shape: Shape
    cached_scene_shape: Shape
    embedding_shape: Shape
    attention_mask_shape: Shape
    proprioception_shape: Shape
    action_shape: Shape
    model_training_state: bool
    use_film: bool
    use_diffusion: bool

    def __post_init__(self) -> None:
        text_fields = (
            self.checkpoint_id,
            self.upstream_revision,
            self.configuration_id,
            self.controller_version,
            self.executor_version,
            self.preprocessing_id,
            self.action_head_id,
            self.dtype,
            self.device,
        )
        if not all(isinstance(value, str) and value for value in text_fields):
            raise ValueError("Executor compatibility identities must be non-empty strings")
        if len(self.instruction_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.instruction_sha256
        ):
            raise ValueError("Instruction identity must be a lowercase SHA-256")
        if self.image_height < 1 or self.image_width < 1:
            raise ValueError("Image dimensions must be positive and static")
        if self.patch_count < 1 or self.projected_dimension < 1:
            raise ValueError("Patch count and projected dimension must be positive")
        shapes = (
            self.prompt_input_shape,
            self.wrist_shape,
            self.cached_scene_shape,
            self.embedding_shape,
            self.attention_mask_shape,
            self.action_shape,
        )
        if not all(_valid_shape(shape) for shape in shapes):
            raise ValueError("Executor shapes must contain only positive dimensions")
        if not _valid_shape(self.proprioception_shape, allow_empty=True):
            raise ValueError("Proprioception shape must be positive or empty when disabled")
        expected_scene = (1, self.patch_count, self.projected_dimension)
        if self.cached_scene_shape != expected_scene:
            raise ValueError("Cached scene shape differs from the frozen projected contract")
        if len(self.wrist_shape) != 4 or self.wrist_shape != (
            1,
            6,
            self.image_height,
            self.image_width,
        ):
            raise ValueError("Wrist pixels require static [1,6,H,W] shape")
        if len(self.embedding_shape) != 3 or self.embedding_shape[0] != 1:
            raise ValueError("Prompt embeddings require batch-one rank-three shape")
        if self.embedding_shape[2] != self.projected_dimension:
            raise ValueError("Embedding and projected dimensions must match")
        if self.prompt_input_shape != self.attention_mask_shape:
            raise ValueError("Prompt IDs and attention-mask shapes must match")
        if self.prompt_input_shape != self.embedding_shape[:2]:
            raise ValueError("Prompt IDs and embeddings must share batch/sequence dimensions")
        if self.proprioception_shape and self.proprioception_shape[0] != 1:
            raise ValueError("Proprioception requires batch one")
        if not self.action_shape or self.action_shape[0] != 1:
            raise ValueError("Normalized actions require batch one")
        if self.model_training_state:
            raise ValueError("V5-C executor requires model evaluation mode")
        if self.use_film or self.use_diffusion:
            raise ValueError("V5-C executor does not support FiLM or diffusion")

    @classmethod
    def field_names(cls) -> tuple[str, ...]:
        return tuple(field.name for field in fields(cls))


@dataclass(frozen=True)
class ReuseExecutionInputs:
    compatibility_key: ReuseCompatibilityKey
    wrist_pixels: Any
    cached_scene_tokens: Any
    prompt_input: Any
    prompt_embeddings: Any
    attention_mask: Any
    proprioception: Any | None


@dataclass(frozen=True)
class ReuseExecutorWork:
    completed_queries: int
    scene_core_calls: int
    wrist_core_calls: int
    downstream_core_calls: int
    prelaunch_rejections: int
    postlaunch_failures: int


@dataclass(frozen=True)
class ReuseExecutorSnapshot:
    executor_version: str
    lifecycle: ExecutorLifecycle
    owner_thread: int | None
    compatibility_key: ReuseCompatibilityKey | None
    buffer_identities: tuple[tuple[str, int], ...]
    work: ReuseExecutorWork


@dataclass(frozen=True)
class ReuseExecutionResult:
    wrist_tokens: Any
    combined_tokens: Any
    normalized_actions: Any
    snapshot: ReuseExecutorSnapshot


class StaticTensorOperations(Protocol):
    """Minimal in-place operations supplied by the eventual tensor backend."""

    def allocate(self, shape: Sequence[int], *, dtype: str, device: str) -> Any: ...

    def copy_(self, destination: Any, source: Any) -> None: ...

    def cat_into(self, destination: Any, values: Sequence[Any], *, dim: int) -> None: ...


WristVisualCore = Callable[[Any, Any], None]
DownstreamActionCore = Callable[[Any, Any, Any, Any | None, Any], None]


def _metadata(value: Any, *, name: str) -> tuple[Shape, str, str]:
    if value is None:
        raise ReuseExecutorUnavailable(f"{name} is missing")
    try:
        shape = tuple(int(item) for item in value.shape)
    except Exception as error:
        raise ReuseExecutorUnavailable(f"{name} lacks valid shape metadata") from error
    if not _valid_shape(shape):
        raise ReuseExecutorUnavailable(f"{name} has invalid or dynamic shape metadata")
    dtype = str(getattr(value, "dtype", ""))
    device = str(getattr(value, "device", ""))
    if not dtype or not device:
        raise ReuseExecutorUnavailable(f"{name} lacks dtype/device metadata")
    return shape, dtype, device


class _BaseReuseExecutor:
    def __init__(
        self,
        *,
        executor_version: str,
        tensor_ops: StaticTensorOperations,
        wrist_visual_core: WristVisualCore,
        downstream_action_core: DownstreamActionCore,
    ) -> None:
        if not callable(wrist_visual_core) or not callable(downstream_action_core):
            raise TypeError("Both frozen executor cores must be callable")
        self.executor_version = executor_version
        self.tensor_ops = tensor_ops
        self.wrist_visual_core = wrist_visual_core
        self.downstream_action_core = downstream_action_core
        self._lifecycle = ExecutorLifecycle.UNPREPARED
        self._owner_thread: int | None = None
        self._key: ReuseCompatibilityKey | None = None
        self._buffers: dict[str, Any] = {}
        self._completed_queries = 0
        self._scene_core_calls = 0
        self._wrist_core_calls = 0
        self._downstream_core_calls = 0
        self._prelaunch_rejections = 0
        self._postlaunch_failures = 0

    @property
    def lifecycle(self) -> ExecutorLifecycle:
        return self._lifecycle

    def _allocate(self, shape: Shape, key: ReuseCompatibilityKey) -> Any:
        return self.tensor_ops.allocate(shape, dtype=key.dtype, device=key.device)

    def prepare(self, key: ReuseCompatibilityKey) -> None:
        if self._lifecycle is ExecutorLifecycle.ACTIVE:
            raise ReuseExecutorUnavailable("Cannot prepare an active executor")
        if key.executor_version != self.executor_version:
            raise ReuseExecutorUnavailable("Compatibility key has a different executor identity")
        self._validate_thread_for_prepare()
        try:
            buffers = self._prepare_buffers(key)
        except Exception as error:
            self._buffers.clear()
            self._key = None
            self._owner_thread = None
            self._lifecycle = ExecutorLifecycle.INVALIDATED
            raise ReuseExecutorUnavailable(f"Executor preparation failed: {error}") from error
        self._key = key
        self._owner_thread = threading.get_ident()
        self._buffers = buffers
        self._zero_counters()
        self._lifecycle = ExecutorLifecycle.PREPARED

    def _validate_thread_for_prepare(self) -> None:
        if self._owner_thread is not None and threading.get_ident() != self._owner_thread:
            raise ReuseExecutorUnavailable("Executor belongs to another thread")

    def _prepare_buffers(self, key: ReuseCompatibilityKey) -> dict[str, Any]:
        del key
        return {}

    def _zero_counters(self) -> None:
        self._completed_queries = 0
        self._scene_core_calls = 0
        self._wrist_core_calls = 0
        self._downstream_core_calls = 0
        self._prelaunch_rejections = 0
        self._postlaunch_failures = 0

    def reset(self) -> None:
        if self._lifecycle is ExecutorLifecycle.ACTIVE:
            raise ReuseExecutorUnavailable("Cannot reset an active executor")
        if self._owner_thread is not None and threading.get_ident() != self._owner_thread:
            raise ReuseExecutorUnavailable("Executor belongs to another thread")
        self._buffers.clear()
        self._key = None
        self._owner_thread = None
        self._zero_counters()
        self._lifecycle = ExecutorLifecycle.UNPREPARED

    def invalidate(self) -> None:
        if self._lifecycle is ExecutorLifecycle.ACTIVE:
            raise ReuseExecutorUnavailable("Active execution invalidates itself on failure")
        self._buffers.clear()
        self._lifecycle = ExecutorLifecycle.INVALIDATED

    def snapshot(self) -> ReuseExecutorSnapshot:
        return ReuseExecutorSnapshot(
            executor_version=self.executor_version,
            lifecycle=self._lifecycle,
            owner_thread=self._owner_thread,
            compatibility_key=self._key,
            buffer_identities=tuple(
                sorted((name, id(value)) for name, value in self._buffers.items())
            ),
            work=ReuseExecutorWork(
                completed_queries=self._completed_queries,
                scene_core_calls=self._scene_core_calls,
                wrist_core_calls=self._wrist_core_calls,
                downstream_core_calls=self._downstream_core_calls,
                prelaunch_rejections=self._prelaunch_rejections,
                postlaunch_failures=self._postlaunch_failures,
            ),
        )

    def ready(self, inputs: ReuseExecutionInputs) -> bool:
        try:
            self._validate_prelaunch(inputs)
        except ReuseExecutorUnavailable:
            return False
        return True

    def _validate_prelaunch(self, inputs: ReuseExecutionInputs) -> None:
        if self._lifecycle is not ExecutorLifecycle.PREPARED:
            raise ReuseExecutorUnavailable(f"Executor is {self._lifecycle.value}, not PREPARED")
        if threading.get_ident() != self._owner_thread:
            raise ReuseExecutorUnavailable("Executor query belongs to another thread")
        if inputs.compatibility_key != self._key:
            raise ReuseExecutorUnavailable("Executor compatibility key mismatch")
        key = inputs.compatibility_key
        self._validate_owned_buffers(key)
        expected = (
            ("wrist_pixels", inputs.wrist_pixels, key.wrist_shape),
            ("cached_scene_tokens", inputs.cached_scene_tokens, key.cached_scene_shape),
            ("prompt_input", inputs.prompt_input, key.prompt_input_shape),
            ("prompt_embeddings", inputs.prompt_embeddings, key.embedding_shape),
            ("attention_mask", inputs.attention_mask, key.attention_mask_shape),
        )
        for name, value, shape in expected:
            actual_shape, dtype, device = _metadata(value, name=name)
            if actual_shape != shape or dtype != key.dtype or device != key.device:
                raise ReuseExecutorUnavailable(f"{name} metadata differs from compatibility key")
        if key.proprioception_shape:
            shape, dtype, device = _metadata(inputs.proprioception, name="proprioception")
            if shape != key.proprioception_shape or dtype != key.dtype or device != key.device:
                raise ReuseExecutorUnavailable(
                    "proprioception metadata differs from compatibility key"
                )
        elif inputs.proprioception is not None:
            raise ReuseExecutorUnavailable("Unexpected proprioception for disabled key")

    def _validate_owned_buffers(self, key: ReuseCompatibilityKey) -> None:
        del key

    @staticmethod
    def _require_metadata(
        value: Any, expected_shape: Shape, key: ReuseCompatibilityKey, *, name: str
    ) -> None:
        shape, dtype, device = _metadata(value, name=name)
        if shape != expected_shape or dtype != key.dtype or device != key.device:
            raise ValueError(f"{name} metadata changed during executor operation")

    def run(self, inputs: ReuseExecutionInputs) -> ReuseExecutionResult:
        try:
            self._validate_prelaunch(inputs)
        except ReuseExecutorUnavailable:
            self._prelaunch_rejections += 1
            raise
        self._lifecycle = ExecutorLifecycle.ACTIVE
        launched = False
        try:
            prepared = self._copy_inputs(inputs)
            launched = True
            wrist_tokens, combined, normalized_actions = self._execute(
                prepared, inputs.compatibility_key
            )
        except Exception as error:
            if launched:
                self._postlaunch_failures += 1
                self._buffers.clear()
                self._lifecycle = ExecutorLifecycle.INVALIDATED
                if isinstance(error, ReuseExecutorFailure):
                    raise
                raise ReuseExecutorFailure(str(error)) from error
            self._lifecycle = ExecutorLifecycle.PREPARED
            self._prelaunch_rejections += 1
            if isinstance(error, ReuseExecutorUnavailable):
                raise
            raise ReuseExecutorUnavailable(str(error)) from error
        self._completed_queries += 1
        self._lifecycle = ExecutorLifecycle.PREPARED
        return ReuseExecutionResult(
            wrist_tokens=wrist_tokens,
            combined_tokens=combined,
            normalized_actions=normalized_actions,
            snapshot=self.snapshot(),
        )

    def _copy_inputs(self, inputs: ReuseExecutionInputs) -> ReuseExecutionInputs:
        return inputs

    def _execute(
        self, inputs: ReuseExecutionInputs, key: ReuseCompatibilityKey
    ) -> tuple[Any, Any, Any]:
        raise NotImplementedError


class EagerReuseExecutor(_BaseReuseExecutor):
    """Allocation-based reference implementation for exact parity checks."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(executor_version=EAGER_REUSE_EXECUTOR_VERSION, **kwargs)

    def _execute(
        self, inputs: ReuseExecutionInputs, key: ReuseCompatibilityKey
    ) -> tuple[Any, Any, Any]:
        wrist_tokens = self._allocate(key.cached_scene_shape, key)
        self._wrist_core_calls += 1
        self.wrist_visual_core(inputs.wrist_pixels, wrist_tokens)
        self._require_metadata(wrist_tokens, key.cached_scene_shape, key, name="wrist output")
        combined = self._allocate((1, key.patch_count * 2, key.projected_dimension), key)
        self.tensor_ops.cat_into(combined, (inputs.cached_scene_tokens, wrist_tokens), dim=1)
        self._require_metadata(
            combined,
            (1, key.patch_count * 2, key.projected_dimension),
            key,
            name="combined output",
        )
        actions = self._allocate(key.action_shape, key)
        self._downstream_core_calls += 1
        self.downstream_action_core(
            combined,
            inputs.prompt_embeddings,
            inputs.attention_mask,
            inputs.proprioception,
            actions,
        )
        self._require_metadata(actions, key.action_shape, key, name="action output")
        return wrist_tokens, combined, actions


class StaticBufferReuseExecutor(_BaseReuseExecutor):
    """Owned-buffer, non-reentrant static execution plan."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(executor_version=STATIC_REUSE_EXECUTOR_VERSION, **kwargs)

    def _prepare_buffers(self, key: ReuseCompatibilityKey) -> dict[str, Any]:
        shapes = {
            "wrist_pixels": key.wrist_shape,
            "cached_scene_tokens": key.cached_scene_shape,
            "prompt_input": key.prompt_input_shape,
            "prompt_embeddings": key.embedding_shape,
            "attention_mask": key.attention_mask_shape,
            "wrist_tokens": key.cached_scene_shape,
            "combined_tokens": (1, key.patch_count * 2, key.projected_dimension),
            "normalized_actions": key.action_shape,
        }
        if key.proprioception_shape:
            shapes["proprioception"] = key.proprioception_shape
        return {name: self._allocate(shape, key) for name, shape in shapes.items()}

    def _copy_inputs(self, inputs: ReuseExecutionInputs) -> ReuseExecutionInputs:
        mappings = (
            ("wrist_pixels", inputs.wrist_pixels),
            ("cached_scene_tokens", inputs.cached_scene_tokens),
            ("prompt_input", inputs.prompt_input),
            ("prompt_embeddings", inputs.prompt_embeddings),
            ("attention_mask", inputs.attention_mask),
        )
        for name, source in mappings:
            self.tensor_ops.copy_(self._buffers[name], source)
        proprioception = None
        if inputs.proprioception is not None:
            proprioception = self._buffers["proprioception"]
            self.tensor_ops.copy_(proprioception, inputs.proprioception)
        return ReuseExecutionInputs(
            compatibility_key=inputs.compatibility_key,
            wrist_pixels=self._buffers["wrist_pixels"],
            cached_scene_tokens=self._buffers["cached_scene_tokens"],
            prompt_input=self._buffers["prompt_input"],
            prompt_embeddings=self._buffers["prompt_embeddings"],
            attention_mask=self._buffers["attention_mask"],
            proprioception=proprioception,
        )

    def _validate_owned_buffers(self, key: ReuseCompatibilityKey) -> None:
        expected = {
            "wrist_pixels": key.wrist_shape,
            "cached_scene_tokens": key.cached_scene_shape,
            "prompt_input": key.prompt_input_shape,
            "prompt_embeddings": key.embedding_shape,
            "attention_mask": key.attention_mask_shape,
            "wrist_tokens": key.cached_scene_shape,
            "combined_tokens": (1, key.patch_count * 2, key.projected_dimension),
            "normalized_actions": key.action_shape,
        }
        if key.proprioception_shape:
            expected["proprioception"] = key.proprioception_shape
        if set(self._buffers) != set(expected):
            raise ReuseExecutorUnavailable("Static executor buffer set is incomplete")
        for name, shape in expected.items():
            try:
                self._require_metadata(self._buffers[name], shape, key, name=name)
            except (ReuseExecutorUnavailable, ValueError) as error:
                raise ReuseExecutorUnavailable(str(error)) from error

    def _execute(
        self, inputs: ReuseExecutionInputs, key: ReuseCompatibilityKey
    ) -> tuple[Any, Any, Any]:
        del key
        wrist_tokens = self._buffers["wrist_tokens"]
        self._wrist_core_calls += 1
        self.wrist_visual_core(inputs.wrist_pixels, wrist_tokens)
        self._require_metadata(
            wrist_tokens,
            inputs.compatibility_key.cached_scene_shape,
            inputs.compatibility_key,
            name="wrist output",
        )
        combined = self._buffers["combined_tokens"]
        self.tensor_ops.cat_into(combined, (inputs.cached_scene_tokens, wrist_tokens), dim=1)
        self._require_metadata(
            combined,
            (
                1,
                inputs.compatibility_key.patch_count * 2,
                inputs.compatibility_key.projected_dimension,
            ),
            inputs.compatibility_key,
            name="combined output",
        )
        actions = self._buffers["normalized_actions"]
        self._downstream_core_calls += 1
        self.downstream_action_core(
            combined,
            inputs.prompt_embeddings,
            inputs.attention_mask,
            inputs.proprioception,
            actions,
        )
        self._require_metadata(
            actions,
            inputs.compatibility_key.action_shape,
            inputs.compatibility_key,
            name="action output",
        )
        return wrist_tokens, combined, actions
