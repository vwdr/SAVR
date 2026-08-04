"""CPU-verifiable batched visual paths for ACR Version 3."""

from __future__ import annotations

import math
import threading
import time
import types
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, replace
from typing import Any, Generic, Literal, Protocol, TypeVar

from savr.acr.cache import SceneCacheError
from savr.acr.dual_path import DualPathFailure, DualPathOpenVLAAdapter
from savr.acr.instrumentation import CameraInstrumentation
from savr.acr.signals import normalized_eef_position, prepare_scene_representation
from savr.acr.types import ACRContext, SceneDecision, SceneTensorMetadata
from savr.signals import SignalValidationError
from savr.timing import QueryTiming


T = TypeVar("T")


class BatchedTensorOperations(Protocol):
    def split(self, value: Any, sections: Sequence[int], *, dim: int) -> tuple[Any, ...]: ...

    def cat(self, values: Sequence[Any], *, dim: int) -> Any: ...

    def reshape(self, value: Any, shape: Sequence[int]) -> Any: ...

    def all_finite(self, value: Any) -> bool: ...


def _shape(value: Any) -> tuple[int, ...]:
    try:
        shape = tuple(int(item) for item in value.shape)
    except Exception as error:
        raise ValueError("Tensor lacks valid shape metadata") from error
    if not shape or any(item < 1 for item in shape):
        raise ValueError("Tensor dimensions must be positive")
    return shape


@dataclass(frozen=True)
class BatchedCameraWork:
    """Physical invocations and logical camera work for a V3 path."""

    mode: Literal["batched-fr", "v3-refresh", "v3-reuse"]
    intercepted_boundary_calls: int
    physical_siglip_calls: int
    physical_dinov2_calls: int
    physical_projector_calls: int
    logical_scene_backbone_calls: int
    logical_wrist_backbone_calls: int
    logical_scene_projector_calls: int
    logical_wrist_projector_calls: int
    downstream_calls: int
    component_wall_ms: dict[str, float]

    def validate(self) -> None:
        scene_refresh = self.mode != "v3-reuse"
        expected = (
            1,
            1,
            1,
            int(scene_refresh),
            1,
            int(scene_refresh),
            1,
        )
        observed = (
            self.physical_siglip_calls,
            self.physical_dinov2_calls,
            self.physical_projector_calls,
            self.logical_scene_backbone_calls,
            self.logical_wrist_backbone_calls,
            self.logical_scene_projector_calls,
            self.logical_wrist_projector_calls,
        )
        if observed != expected:
            raise ValueError("Batched physical/logical work differs from the selected path")
        if self.intercepted_boundary_calls != 1 or self.downstream_calls != 1:
            raise ValueError("Every completed V3 query requires one boundary and downstream call")
        if any(value < 0 or not math.isfinite(value) for value in self.component_wall_ms.values()):
            raise ValueError("Component wall times must be finite and non-negative")


@dataclass(frozen=True)
class BatchedQueryFailure:
    classification: Literal["invariant", "technical"]
    message: str
    query_index: int | None
    cache_invalidated: bool


@dataclass(frozen=True)
class BatchedFullRefreshResult(Generic[T]):
    value: T
    work: BatchedCameraWork
    query_wall_ms: float
    device_timing: QueryTiming | None


@dataclass(frozen=True)
class BatchedDualPathResult(Generic[T]):
    value: T
    decision: SceneDecision
    work: BatchedCameraWork
    cache_event: str
    controller_wall_ms: float
    query_wall_ms: float
    device_timing: QueryTiming | None


class ModelQueryBudget:
    """Consume a unique bounded query identity before every model call."""

    def __init__(self, maximum: int) -> None:
        if maximum < 1:
            raise ValueError("Model query maximum must be positive")
        self.maximum = maximum
        self._labels: list[str] = []

    @property
    def consumed(self) -> int:
        return len(self._labels)

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(self._labels)

    def consume(self, label: str) -> int:
        if not label or label in self._labels:
            raise ValueError("Model query labels must be non-empty and unique")
        if self.consumed >= self.maximum:
            raise RuntimeError("Model query budget exhausted")
        index = self.consumed
        self._labels.append(label)
        return index


class BatchedVisionPath:
    """Encode ordered scene/wrist cameras in one batch per vision tower."""

    def __init__(
        self,
        *,
        model: Any,
        tensor_ops: BatchedTensorOperations,
        instrumentation: CameraInstrumentation,
        correctness_mode: bool = False,
    ) -> None:
        self.model = model
        self.tensor_ops = tensor_ops
        self.instrumentation = instrumentation
        self.correctness_mode = correctness_mode

    def validate_contract(self, context: ACRContext) -> None:
        backbone = getattr(self.model, "vision_backbone", None)
        if backbone is None or not bool(getattr(backbone, "use_fused_vision_backbone", False)):
            raise RuntimeError("V3 requires the fused SigLIP/DINOv2 backbone")
        if int(backbone.get_num_images_in_input()) != 2:
            raise RuntimeError("V3 requires exactly two upstream images")
        if int(backbone.get_num_patches()) != context.patch_count:
            raise RuntimeError("Upstream patch count differs from the V3 context")
        if not callable(getattr(backbone, "featurizer", None)) or not callable(
            getattr(backbone, "fused_featurizer", None)
        ):
            raise RuntimeError("Pinned vision featurizers are unavailable")
        if not callable(getattr(self.model, "projector", None)):
            raise RuntimeError("Pinned visual projector is unavailable")

    @staticmethod
    def _validate_context(context: ACRContext) -> None:
        if context.number_of_images != 2 or context.image_order != (
            "full_image",
            "wrist_image",
        ):
            raise RuntimeError("V3 requires frozen scene-wrist image ordering")
        if not context.center_crop:
            raise RuntimeError("V3 requires frozen center-crop preprocessing")

    @staticmethod
    def _validate_tensor_context(value: Any, context: ACRContext, *, name: str) -> None:
        if (
            str(getattr(value, "dtype", "")) != context.dtype
            or str(getattr(value, "device", "")) != context.device
        ):
            raise RuntimeError(f"{name} dtype/device differs from the V3 context")

    def encode(
        self,
        *,
        pixel_values: Any,
        language_embeddings: Any,
        context: ACRContext,
        use_film: bool = False,
    ) -> Any:
        self._validate_context(context)
        self.validate_contract(context)
        if use_film:
            raise RuntimeError("SA-BDP-ACR does not support FiLM")
        pixel_shape = _shape(pixel_values)
        language_shape = _shape(language_embeddings)
        if len(pixel_shape) != 4 or pixel_shape[0] != 1 or pixel_shape[1] != 12:
            raise RuntimeError("V3 requires pixel tensor shape [1,12,H,W]")
        if len(language_shape) != 3 or language_shape[0] != 1:
            raise RuntimeError("V3 language embeddings require batch-one rank-three shape")
        self._validate_tensor_context(pixel_values, context, name="Pixel tensor")
        self._validate_tensor_context(language_embeddings, context, name="Language embeddings")

        scene, wrist = self.tensor_ops.split(pixel_values, (6, 6), dim=1)
        scene_regular, scene_fused = self.tensor_ops.split(scene, (3, 3), dim=1)
        wrist_regular, wrist_fused = self.tensor_ops.split(wrist, (3, 3), dim=1)
        regular_batch = self.instrumentation.measure_operation(
            "batched.input-siglip",
            lambda: self.tensor_ops.cat((scene_regular, wrist_regular), dim=0),
        )
        fused_batch = self.instrumentation.measure_operation(
            "batched.input-dinov2",
            lambda: self.tensor_ops.cat((scene_fused, wrist_fused), dim=0),
        )
        expected_input = (2, 3, pixel_shape[2], pixel_shape[3])
        if _shape(regular_batch) != expected_input or _shape(fused_batch) != expected_input:
            raise RuntimeError("V3 camera batching changed the frozen input shape")

        backbone = self.model.vision_backbone
        regular_features = self.instrumentation.measure_operation(
            "batched.siglip", lambda: backbone.featurizer(regular_batch)
        )
        fused_features = self.instrumentation.measure_operation(
            "batched.dinov2", lambda: backbone.fused_featurizer(fused_batch)
        )
        regular_shape = _shape(regular_features)
        fused_shape = _shape(fused_features)
        expected_prefix = (2, context.patch_count)
        if (
            len(regular_shape) != 3
            or len(fused_shape) != 3
            or regular_shape[:2] != expected_prefix
            or fused_shape[:2] != expected_prefix
        ):
            raise RuntimeError("V3 tower output differs from the ordered two-camera contract")

        per_camera = self.instrumentation.measure_operation(
            "batched.tower-concat",
            lambda: self.tensor_ops.cat((regular_features, fused_features), dim=2),
        )
        expected_feature_shape = (
            2,
            context.patch_count,
            regular_shape[2] + fused_shape[2],
        )
        if _shape(per_camera) != expected_feature_shape:
            raise RuntimeError("V3 fused feature shape is invalid")
        ordered_features = self.instrumentation.measure_operation(
            "batched.scene-wrist-reshape",
            lambda: self.tensor_ops.reshape(
                per_camera,
                (1, context.patch_count * 2, expected_feature_shape[2]),
            ),
        )
        projected = self.instrumentation.measure_operation(
            "batched.projector", lambda: self.model.projector(ordered_features)
        )
        expected_projected = SceneTensorMetadata(
            shape=(1, context.patch_count * 2, language_shape[-1]),
            dtype=context.dtype,
            device=context.device,
            patch_count=context.patch_count * 2,
        )
        if SceneTensorMetadata.from_value(
            projected, patch_count=context.patch_count * 2
        ) != expected_projected:
            raise RuntimeError("V3 projected output differs from the frozen combined contract")
        if self.correctness_mode and not self.tensor_ops.all_finite(projected):
            raise RuntimeError("V3 projected output contains non-finite values")
        return projected


@dataclass
class _BatchedActiveQuery:
    boundary_calls: int = 0
    work: BatchedCameraWork | None = None


class _BatchedFullRefreshEpisode(AbstractContextManager["BatchedFullRefreshAdapter"]):
    def __init__(self, adapter: "BatchedFullRefreshAdapter", context: ACRContext) -> None:
        self.adapter = adapter
        self.context = context
        self.entered = False

    def __enter__(self) -> "BatchedFullRefreshAdapter":
        self.adapter._install(self.context)
        self.entered = True
        return self.adapter

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if self.entered:
            self.adapter._restore()
            self.entered = False


class BatchedFullRefreshAdapter:
    """Episode-scoped Batched FR with no controller or scene cache."""

    METHOD_NAME = "_process_vision_features"

    def __init__(
        self,
        *,
        model: Any,
        tensor_ops: BatchedTensorOperations,
        instrumentation: CameraInstrumentation | None = None,
        action_chunk_getter: Callable[[Any], Any] | None = None,
        action_finite_checker: Callable[[Any], bool] | None = None,
        projected_tokens_observer: Callable[[Any], None] | None = None,
        correctness_mode: bool = False,
    ) -> None:
        if not callable(getattr(model, self.METHOD_NAME, None)):
            raise TypeError(f"Model does not expose {self.METHOD_NAME}")
        self.model = model
        self.tensor_ops = tensor_ops
        self.instrumentation = instrumentation or CameraInstrumentation()
        self.action_chunk_getter = action_chunk_getter or (lambda value: value)
        self.action_finite_checker = action_finite_checker or tensor_ops.all_finite
        self.projected_tokens_observer = projected_tokens_observer
        self.vision = BatchedVisionPath(
            model=model,
            tensor_ops=tensor_ops,
            instrumentation=self.instrumentation,
            correctness_mode=correctness_mode,
        )
        self._lock = threading.Lock()
        self._owner_thread: int | None = None
        self._context: ACRContext | None = None
        self._active: _BatchedActiveQuery | None = None
        self._had_override = False
        self._previous_override: Any = None
        self.last_failure: BatchedQueryFailure | None = None

    @property
    def installed(self) -> bool:
        return self._owner_thread is not None

    def episode(self, context: ACRContext) -> _BatchedFullRefreshEpisode:
        return _BatchedFullRefreshEpisode(self, context)

    def _install(self, context: ACRContext) -> None:
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Nested or concurrent Batched-FR episode use is prohibited")
        installed = False
        try:
            BatchedVisionPath._validate_context(context)
            self.vision.validate_contract(context)
            instance = vars(self.model)
            self._had_override = self.METHOD_NAME in instance
            self._previous_override = instance.get(self.METHOD_NAME)
            self._context = context
            self._owner_thread = threading.get_ident()
            setattr(
                self.model,
                self.METHOD_NAME,
                types.MethodType(self._intercepted_vision, self.model),
            )
            installed = True
        except Exception:
            if installed:
                if self._had_override:
                    setattr(self.model, self.METHOD_NAME, self._previous_override)
                else:
                    delattr(self.model, self.METHOD_NAME)
            self._context = None
            self._owner_thread = None
            self._previous_override = None
            self._lock.release()
            raise

    def _restore(self) -> None:
        if not self.installed:
            raise RuntimeError("Batched-FR adapter is not installed")
        active = self._active is not None
        try:
            if self._had_override:
                setattr(self.model, self.METHOD_NAME, self._previous_override)
            else:
                delattr(self.model, self.METHOD_NAME)
        finally:
            self._context = None
            self._owner_thread = None
            self._previous_override = None
            self._lock.release()
        if active:
            raise RuntimeError("Batched-FR episode exited during an active query")

    def _intercepted_vision(
        self,
        model_instance: Any,
        pixel_values: Any,
        language_embeddings: Any = None,
        use_film: bool = False,
    ) -> Any:
        del model_instance
        if self._active is None or self._context is None:
            raise RuntimeError("Batched visual boundary invoked outside an active query")
        if language_embeddings is None:
            raise RuntimeError("Language embeddings are required")
        self._active.boundary_calls += 1
        if self._active.boundary_calls != 1:
            raise RuntimeError("Expected exactly one Batched-FR visual-boundary invocation")
        projected = self.vision.encode(
            pixel_values=pixel_values,
            language_embeddings=language_embeddings,
            context=self._context,
            use_film=use_film,
        )
        snapshot = self.instrumentation.snapshot()
        self._active.work = BatchedCameraWork(
            mode="batched-fr",
            intercepted_boundary_calls=1,
            physical_siglip_calls=1,
            physical_dinov2_calls=1,
            physical_projector_calls=1,
            logical_scene_backbone_calls=1,
            logical_wrist_backbone_calls=1,
            logical_scene_projector_calls=1,
            logical_wrist_projector_calls=1,
            downstream_calls=0,
            component_wall_ms=dict(snapshot.component_wall_ms),
        )
        if self.projected_tokens_observer is not None:
            self.projected_tokens_observer(projected)
        return projected

    def run_query(self, query: Callable[[], T]) -> BatchedFullRefreshResult[T]:
        if not self.installed or self._context is None:
            raise RuntimeError("Use Batched FR inside an episode context")
        if threading.get_ident() != self._owner_thread:
            raise RuntimeError("Concurrent Batched-FR query use is prohibited")
        if self._active is not None:
            raise RuntimeError("Nested Batched-FR query use is prohibited")
        self.vision.validate_contract(self._context)
        self.instrumentation.reset()
        self._active = _BatchedActiveQuery()
        self.last_failure = None
        started = time.perf_counter()
        timing_started = False
        timing_finished = False
        try:
            self.instrumentation.start_query()
            timing_started = self.instrumentation.timer is not None
            value = query()
            device_timing = self.instrumentation.finish_query()
            timing_finished = True
            query_wall_ms = (time.perf_counter() - started) * 1000.0
            active = self._active
            assert active is not None
            if active.boundary_calls != 1 or active.work is None:
                raise RuntimeError("Batched-FR boundary did not produce complete accounting")
            self.instrumentation.record_downstream()
            work = replace(active.work, downstream_calls=1)
            work.validate()
            action_chunk = self.action_chunk_getter(value)
            if not self.action_finite_checker(action_chunk):
                raise RuntimeError("Returned action chunk contains non-finite values")
        except Exception as error:
            self.last_failure = BatchedQueryFailure(
                classification="invariant"
                if isinstance(error, (RuntimeError, ValueError))
                else "technical",
                message=str(error),
                query_index=None,
                cache_invalidated=False,
            )
            raise
        finally:
            try:
                if timing_started and not timing_finished:
                    self.instrumentation.finish_query()
            finally:
                self._active = None
        return BatchedFullRefreshResult(value, work, query_wall_ms, device_timing)


@dataclass
class _V3ActiveQuery:
    decision: SceneDecision
    scene_representation: tuple[float, ...] | None
    normalized_position: tuple[float, float, float] | None
    boundary_calls: int = 0
    effective_decision: SceneDecision | None = None
    cache_event: str = ""
    work: BatchedCameraWork | None = None


class BatchedDualPathOpenVLAAdapter(DualPathOpenVLAAdapter):
    """SA-BDP-ACR: batched refresh and cached-scene/fresh-wrist reuse."""

    tensor_ops: BatchedTensorOperations

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.vision = BatchedVisionPath(
            model=self.model,
            tensor_ops=self.tensor_ops,
            instrumentation=self.instrumentation,
            correctness_mode=self.correctness_mode,
        )

    def _validate_model_contract(self) -> None:
        if self._context is None:
            raise RuntimeError("V3 episode context is unavailable")
        self.vision.validate_contract(self._context)

    def _batched_refresh(
        self,
        pixel_values: Any,
        language_embeddings: Any,
        expected_scene: SceneTensorMetadata,
        *,
        refresh_query_index: int,
    ) -> Any:
        context = self._context
        assert context is not None
        combined = self.vision.encode(
            pixel_values=pixel_values,
            language_embeddings=language_embeddings,
            context=context,
        )
        scene_tokens, wrist_tokens = self.tensor_ops.split(
            combined, (context.patch_count, context.patch_count), dim=1
        )
        self._validate_projected(scene_tokens, expected=expected_scene, name="scene")
        self._validate_projected(wrist_tokens, expected=expected_scene, name="wrist")
        self.instrumentation.measure_operation(
            "scene.cache-store",
            lambda: self.cache.store(
                context=context,
                tokens=scene_tokens,
                refresh_query_index=refresh_query_index,
            ),
        )
        return combined

    def _work(self, *, scene_refresh: bool) -> BatchedCameraWork:
        snapshot = self.instrumentation.snapshot()
        return BatchedCameraWork(
            mode="v3-refresh" if scene_refresh else "v3-reuse",
            intercepted_boundary_calls=1,
            physical_siglip_calls=1,
            physical_dinov2_calls=1,
            physical_projector_calls=1,
            logical_scene_backbone_calls=int(scene_refresh),
            logical_wrist_backbone_calls=1,
            logical_scene_projector_calls=int(scene_refresh),
            logical_wrist_projector_calls=1,
            downstream_calls=0,
            component_wall_ms=dict(snapshot.component_wall_ms),
        )

    def _intercepted_vision(
        self,
        model_instance: Any,
        pixel_values: Any,
        language_embeddings: Any = None,
        use_film: bool = False,
    ) -> Any:
        del model_instance
        active = self._active_query
        context = self._context
        if not isinstance(active, _V3ActiveQuery) or context is None:
            raise RuntimeError("V3 visual boundary invoked outside an active query")
        active.boundary_calls += 1
        if active.boundary_calls != 1:
            raise RuntimeError("Expected exactly one V3 visual-boundary invocation")
        if use_film:
            raise RuntimeError("SA-BDP-ACR does not support FiLM")
        if language_embeddings is None:
            raise RuntimeError("Language embeddings are required")
        pixel_shape = _shape(pixel_values)
        language_shape = _shape(language_embeddings)
        if len(pixel_shape) != 4 or pixel_shape[0] != 1 or pixel_shape[1] != 12:
            raise RuntimeError("V3 requires pixel tensor shape [1,12,H,W]")
        if len(language_shape) != 3 or language_shape[0] != 1:
            raise RuntimeError("V3 language embeddings require batch-one rank-three shape")
        BatchedVisionPath._validate_tensor_context(pixel_values, context, name="Pixel tensor")
        BatchedVisionPath._validate_tensor_context(
            language_embeddings, context, name="Language embeddings"
        )
        expected_scene = SceneTensorMetadata(
            shape=(1, context.patch_count, language_shape[-1]),
            dtype=context.dtype,
            device=context.device,
            patch_count=context.patch_count,
        )
        effective = active.decision
        cache_event = "refresh" if effective.refresh else "reuse"
        if not effective.refresh and not self.cache.compatible(context, expected_scene):
            effective = effective.force_refresh("cache")
            cache_event = "forced-refresh"
            self.cache.invalidate()

        if effective.refresh:
            returned = self._batched_refresh(
                pixel_values,
                language_embeddings,
                expected_scene,
                refresh_query_index=effective.query_index,
            )
        else:
            _, wrist_pixels = self.tensor_ops.split(pixel_values, (6, 6), dim=1)
            try:
                scene_tokens = self.instrumentation.measure_operation(
                    "scene.cache-load", lambda: self.cache.load(context, expected_scene)
                )
            except SceneCacheError:
                effective = effective.force_refresh("cache")
                cache_event = "forced-refresh"
                self.cache.invalidate()
                returned = self._batched_refresh(
                    pixel_values,
                    language_embeddings,
                    expected_scene,
                    refresh_query_index=effective.query_index,
                )
            else:
                wrist_tokens = self._encode_wrist(wrist_pixels, expected_scene)
                returned = self.instrumentation.measure_operation(
                    "camera-block-concat",
                    lambda: self.tensor_ops.cat((scene_tokens, wrist_tokens), dim=1),
                )
                expected_combined = SceneTensorMetadata(
                    shape=(1, context.patch_count * 2, language_shape[-1]),
                    dtype=context.dtype,
                    device=context.device,
                    patch_count=context.patch_count * 2,
                )
                self._validate_projected(returned, expected=expected_combined, name="combined")

        active.effective_decision = effective
        active.cache_event = cache_event
        active.work = self._work(scene_refresh=effective.refresh)
        if self.projected_tokens_observer is not None:
            self.projected_tokens_observer(returned)
        return returned

    def run_query(
        self,
        *,
        query: Callable[[], T],
        scene_image: Any,
        wrist_image: Any,
        state: Any,
        state_q01: Sequence[float],
        state_q99: Sequence[float],
    ) -> BatchedDualPathResult[T]:  # type: ignore[override]
        del wrist_image
        if not self.installed or self._context is None:
            raise RuntimeError("Use the V3 adapter inside an episode context")
        if threading.get_ident() != self._owner_thread:
            raise RuntimeError("Concurrent V3 query use is prohibited")
        if self._active_query is not None:
            raise RuntimeError("Nested V3 query use is prohibited")
        self._validate_model_contract()
        query_started = time.perf_counter()
        decision_started = time.perf_counter()
        try:
            scene_representation = prepare_scene_representation(scene_image)
        except SignalValidationError:
            scene_representation = None
        try:
            position = normalized_eef_position(state, state_q01, state_q99)
        except SignalValidationError:
            position = None
        decision = self.controller.decide(
            scene_representation=scene_representation,
            normalized_eef_position=position,
            cache_available=self.cache.available(self._context),
            cache_age=self.cache.age,
        )
        controller_wall_ms = (time.perf_counter() - decision_started) * 1000.0
        self.instrumentation.reset()
        active = _V3ActiveQuery(decision, scene_representation, position)
        self._active_query = active  # type: ignore[assignment]
        self.last_failure = None
        timing_started = False
        timing_finished = False
        try:
            self.instrumentation.start_query()
            timing_started = self.instrumentation.timer is not None
            value = query()
            device_timing = self.instrumentation.finish_query()
            timing_finished = True
            pre_validation_wall_ms = (time.perf_counter() - query_started) * 1000.0
            if active.boundary_calls != 1 or active.work is None:
                raise RuntimeError("V3 visual boundary did not produce complete accounting")
            effective = active.effective_decision
            if effective is None:
                raise RuntimeError("V3 visual boundary did not preserve its effective decision")
            self.instrumentation.record_downstream()
            work = replace(active.work, downstream_calls=1)
            work.validate()
            action_chunk = self.action_chunk_getter(value)
            if not self.action_finite_checker(action_chunk):
                raise RuntimeError("Returned action chunk contains non-finite values")
            post_validation_started = time.perf_counter()
            if effective.refresh:
                if self.cache.entry is None:
                    raise RuntimeError("V3 refresh completed without a compatible scene cache")
            else:
                self.cache.mark_reused()
            self.controller.observe(
                decision=effective,
                scene_representation=scene_representation,
                normalized_eef_position=position,
                action_chunk=action_chunk,
            )
            query_wall_ms = pre_validation_wall_ms + (
                time.perf_counter() - post_validation_started
            ) * 1000.0
        except Exception as error:
            self.cache.invalidate()
            self.last_failure = DualPathFailure(
                classification="invariant"
                if isinstance(error, (RuntimeError, ValueError))
                else "technical",
                message=str(error),
                query_index=decision.query_index,
                cache_invalidated=True,
            )
            raise
        finally:
            try:
                if timing_started and not timing_finished:
                    self.instrumentation.finish_query()
            finally:
                self._active_query = None
        return BatchedDualPathResult(
            value,
            effective,
            work,
            active.cache_event,
            controller_wall_ms,
            query_wall_ms,
            device_timing,
        )
