"""Episode-scoped dual-path adapter for State-Aware ACR Version 2."""

from __future__ import annotations

import math
import threading
import time
import types
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, Generic, Literal, Protocol, TypeVar

from savr.acr.cache import SceneCacheError, SceneTokenCache
from savr.acr.controller import ACRController
from savr.acr.instrumentation import CameraInstrumentation
from savr.acr.signals import audit_sha256, normalized_eef_position, prepare_scene_representation
from savr.acr.types import ACRContext, SceneDecision, SceneTensorMetadata
from savr.signals import SignalValidationError
from savr.timing import QueryTiming


T = TypeVar("T")


class DualPathTensorOperations(Protocol):
    def split(self, value: Any, sections: Sequence[int], *, dim: int) -> tuple[Any, ...]: ...

    def cat(self, values: Sequence[Any], *, dim: int) -> Any: ...

    def all_finite(self, value: Any) -> bool: ...


@dataclass(frozen=True)
class DualPathWork:
    """Truthful physical invocation and logical camera-work accounting."""

    intercepted_boundary_calls: int
    upstream_two_view_refresh_calls: int
    physical_fused_backbone_calls: int
    physical_siglip_calls: int
    physical_dinov2_calls: int
    physical_projector_calls: int
    logical_scene_backbone_calls: int
    logical_wrist_backbone_calls: int
    logical_scene_projector_calls: int
    logical_wrist_projector_calls: int
    downstream_calls: int
    component_wall_ms: dict[str, float]

    def validate(self, *, scene_refresh: bool) -> None:
        expected = (1, 1, 0, 0, 1, 1, 1, 1) if scene_refresh else (0, 0, 1, 1, 1, 0, 1, 0)
        actual = (
            self.upstream_two_view_refresh_calls,
            self.physical_fused_backbone_calls,
            self.physical_siglip_calls,
            self.physical_dinov2_calls,
            self.physical_projector_calls,
            self.logical_scene_backbone_calls,
            self.logical_wrist_backbone_calls,
            self.logical_scene_projector_calls,
        )
        if actual != expected:
            raise ValueError("Dual-path physical/logical work differs from the selected path")
        if self.intercepted_boundary_calls != 1:
            raise ValueError("Every query must cross the intercepted visual boundary once")
        if self.logical_wrist_projector_calls != 1 or self.downstream_calls != 1:
            raise ValueError("Every query must execute fresh wrist projection and downstream work")
        if any(value < 0 or not math.isfinite(value) for value in self.component_wall_ms.values()):
            raise ValueError("Component wall times must be finite and non-negative")


@dataclass(frozen=True)
class DualPathFailure:
    query_index: int
    classification: Literal["invariant", "technical"]
    message: str
    cache_invalidated: bool


@dataclass(frozen=True)
class DualPathQueryResult(Generic[T]):
    value: T
    decision: SceneDecision
    work: DualPathWork
    cache_event: str
    controller_wall_ms: float
    query_wall_ms: float
    device_timing: QueryTiming | None
    scene_image_sha256: str
    wrist_image_sha256: str
    proprio_sha256: str


@dataclass
class _ActiveQuery:
    decision: SceneDecision
    scene_representation: tuple[float, ...] | None
    normalized_position: tuple[float, float, float] | None
    original_boundary_calls: int = 0
    effective_decision: SceneDecision | None = None
    cache_event: str = ""
    work: DualPathWork | None = None


def _shape(value: Any) -> tuple[int, ...]:
    try:
        shape = tuple(int(item) for item in value.shape)
    except Exception as error:
        raise ValueError("Tensor lacks valid shape metadata") from error
    if not shape or any(item < 1 for item in shape):
        raise ValueError("Tensor dimensions must be positive")
    return shape


class _EpisodeInstallation(AbstractContextManager["DualPathOpenVLAAdapter"]):
    def __init__(self, adapter: "DualPathOpenVLAAdapter", context: ACRContext) -> None:
        self.adapter = adapter
        self.context = context
        self.entered = False

    def __enter__(self) -> "DualPathOpenVLAAdapter":
        self.adapter._install(self.context)
        self.entered = True
        return self.adapter

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if self.entered:
            self.adapter._restore()
            self.entered = False


class DualPathOpenVLAAdapter:
    """Use upstream two-view refreshes and wrist-only scene-token reuse."""

    METHOD_NAME = "_process_vision_features"

    def __init__(
        self,
        *,
        model: Any,
        controller: ACRController,
        tensor_ops: DualPathTensorOperations,
        cache: SceneTokenCache | None = None,
        instrumentation: CameraInstrumentation | None = None,
        action_chunk_getter: Callable[[Any], Any] | None = None,
        projected_tokens_observer: Callable[[Any], None] | None = None,
        correctness_mode: bool = False,
    ) -> None:
        if not callable(getattr(model, self.METHOD_NAME, None)):
            raise TypeError(f"Model does not expose {self.METHOD_NAME}")
        self.model = model
        self.controller = controller
        self.tensor_ops = tensor_ops
        self.cache = cache or SceneTokenCache()
        self.instrumentation = instrumentation or CameraInstrumentation()
        self.action_chunk_getter = action_chunk_getter or (lambda result: result)
        self.projected_tokens_observer = projected_tokens_observer
        self.correctness_mode = correctness_mode
        self._episode_lock = threading.Lock()
        self._owner_thread: int | None = None
        self._context: ACRContext | None = None
        self._active_query: _ActiveQuery | None = None
        self._original_method: Callable[..., Any] | None = None
        self._had_override = False
        self._previous_override: Any = None
        self.last_failure: DualPathFailure | None = None

    @property
    def context(self) -> ACRContext | None:
        return self._context

    @property
    def installed(self) -> bool:
        return self._owner_thread is not None

    def episode(self, context: ACRContext) -> _EpisodeInstallation:
        return _EpisodeInstallation(self, context)

    def _install(self, context: ACRContext) -> None:
        if not self._episode_lock.acquire(blocking=False):
            raise RuntimeError("Nested or concurrent dual-path episode use is prohibited")
        override_installed = False
        try:
            if self.installed:
                raise RuntimeError("Dual-path adapter is already installed")
            self._validate_context(context)
            instance_dict = vars(self.model)
            self._had_override = self.METHOD_NAME in instance_dict
            self._previous_override = instance_dict.get(self.METHOD_NAME)
            self._original_method = getattr(self.model, self.METHOD_NAME)
            self.cache.invalidate()
            self.controller.reset(context)
            self._context = context
            self._owner_thread = threading.get_ident()
            setattr(
                self.model,
                self.METHOD_NAME,
                types.MethodType(self._intercepted_vision, self.model),
            )
            override_installed = True
        except Exception:
            if override_installed:
                if self._had_override:
                    setattr(self.model, self.METHOD_NAME, self._previous_override)
                else:
                    delattr(self.model, self.METHOD_NAME)
            self._owner_thread = None
            self._context = None
            self._original_method = None
            self._previous_override = None
            self._episode_lock.release()
            raise

    def _restore(self) -> None:
        if not self.installed:
            raise RuntimeError("Dual-path adapter is not installed")
        active_query = self._active_query is not None
        try:
            if self._had_override:
                setattr(self.model, self.METHOD_NAME, self._previous_override)
            else:
                delattr(self.model, self.METHOD_NAME)
        finally:
            self.cache.invalidate()
            self._context = None
            self._original_method = None
            self._previous_override = None
            self._owner_thread = None
            self._episode_lock.release()
        if active_query:
            raise RuntimeError("Dual-path episode exited during an active query")

    def _validate_context(self, context: ACRContext) -> None:
        if context.number_of_images != 2 or context.image_order != (
            "full_image",
            "wrist_image",
        ):
            raise RuntimeError("Dual-path ACR requires frozen scene-wrist image ordering")
        if not context.center_crop:
            raise RuntimeError("Dual-path ACR requires the frozen center-crop preprocessing")

    def _validate_model_contract(self) -> None:
        context = self._context
        if context is None:
            raise RuntimeError("Dual-path episode context is unavailable")
        backbone = self.model.vision_backbone
        if not bool(getattr(backbone, "use_fused_vision_backbone", False)):
            raise RuntimeError("Dual-path ACR requires the fused SigLIP/DINOv2 backbone")
        if int(backbone.get_num_images_in_input()) != 2:
            raise RuntimeError("Dual-path ACR requires exactly two upstream images")
        if int(backbone.get_num_patches()) != context.patch_count:
            raise RuntimeError("Upstream patch count differs from the frozen context")
        if not callable(getattr(backbone, "featurizer", None)) or not callable(
            getattr(backbone, "fused_featurizer", None)
        ):
            raise RuntimeError("Pinned wrist-camera featurizers are unavailable")
        if not callable(getattr(self.model, "projector", None)):
            raise RuntimeError("Pinned visual projector is unavailable")

    def _validate_projected(
        self,
        value: Any,
        *,
        expected: SceneTensorMetadata,
        name: str,
    ) -> None:
        actual = SceneTensorMetadata.from_value(value, patch_count=expected.patch_count)
        if actual != expected:
            raise RuntimeError(f"{name} projected metadata {actual} differs from {expected}")
        if self.correctness_mode and not self.tensor_ops.all_finite(value):
            raise RuntimeError(f"{name} projected values are non-finite")

    def _encode_wrist(self, wrist_pixels: Any, expected: SceneTensorMetadata) -> Any:
        regular, fused = self.tensor_ops.split(wrist_pixels, (3, 3), dim=1)
        backbone = self.model.vision_backbone
        regular_features = self.instrumentation.call("wrist.siglip", backbone.featurizer, regular)
        fused_features = self.instrumentation.call("wrist.dinov2", backbone.fused_featurizer, fused)
        combined = self.instrumentation.measure_operation(
            "wrist.tower-concat",
            lambda: self.tensor_ops.cat((regular_features, fused_features), dim=2),
        )
        projected = self.instrumentation.call("wrist.projector", self.model.projector, combined)
        self._validate_projected(projected, expected=expected, name="wrist")
        return projected

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
        original = self._original_method
        if active is None or context is None or original is None:
            raise RuntimeError("Visual boundary invoked outside an active dual-path query")
        active.original_boundary_calls += 1
        if active.original_boundary_calls != 1:
            raise RuntimeError("Expected exactly one visual-boundary invocation")
        if use_film:
            raise RuntimeError("SA-DP-ACR does not support FiLM")
        if language_embeddings is None:
            raise RuntimeError("Language embeddings are required")
        pixel_shape = _shape(pixel_values)
        language_shape = _shape(language_embeddings)
        if len(pixel_shape) != 4 or pixel_shape[0] != 1 or pixel_shape[1] != 12:
            raise RuntimeError("SA-DP-ACR requires pixel tensor shape [1,12,H,W]")
        if len(language_shape) != 3 or language_shape[0] != 1:
            raise RuntimeError("Language embeddings require batch-one rank-three shape")
        if (
            str(getattr(language_embeddings, "dtype", "")) != context.dtype
            or str(getattr(language_embeddings, "device", "")) != context.device
        ):
            raise RuntimeError("Language dtype/device differs from the frozen context")

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
            combined = self.instrumentation.measure_operation(
                "refresh.upstream-two-view",
                lambda: original(pixel_values, language_embeddings, use_film),
            )
            expected_combined = SceneTensorMetadata(
                shape=(1, context.patch_count * 2, language_shape[-1]),
                dtype=context.dtype,
                device=context.device,
                patch_count=context.patch_count * 2,
            )
            self._validate_projected(combined, expected=expected_combined, name="combined")
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
                    refresh_query_index=effective.query_index,
                ),
            )
            returned = combined
        else:
            _, wrist_pixels = self.tensor_ops.split(pixel_values, (6, 6), dim=1)
            try:
                scene_tokens = self.instrumentation.measure_operation(
                    "scene.cache-load", lambda: self.cache.load(context, expected_scene)
                )
            except SceneCacheError:
                effective = effective.force_refresh("cache")
                active.effective_decision = effective
                active.cache_event = "forced-refresh"
                return self._intercepted_forced_refresh(
                    original, pixel_values, language_embeddings, expected_scene
                )
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
        active.work = self._make_work(scene_refresh=effective.refresh)
        if self.projected_tokens_observer is not None:
            self.projected_tokens_observer(returned)
        return returned

    def _intercepted_forced_refresh(
        self,
        original: Callable[..., Any],
        pixel_values: Any,
        language_embeddings: Any,
        expected_scene: SceneTensorMetadata,
    ) -> Any:
        context = self._context
        active = self._active_query
        assert context is not None and active is not None and active.effective_decision is not None
        combined = self.instrumentation.measure_operation(
            "refresh.upstream-two-view",
            lambda: original(pixel_values, language_embeddings, False),
        )
        expected_combined = SceneTensorMetadata(
            shape=(1, context.patch_count * 2, expected_scene.shape[-1]),
            dtype=context.dtype,
            device=context.device,
            patch_count=context.patch_count * 2,
        )
        self._validate_projected(combined, expected=expected_combined, name="combined")
        scene_tokens, wrist_tokens = self.tensor_ops.split(
            combined, (context.patch_count, context.patch_count), dim=1
        )
        self._validate_projected(scene_tokens, expected=expected_scene, name="scene")
        self._validate_projected(wrist_tokens, expected=expected_scene, name="wrist")
        self.cache.store(
            context=context,
            tokens=scene_tokens,
            refresh_query_index=active.effective_decision.query_index,
        )
        active.work = self._make_work(scene_refresh=True)
        if self.projected_tokens_observer is not None:
            self.projected_tokens_observer(combined)
        return combined

    def _make_work(self, *, scene_refresh: bool) -> DualPathWork:
        snapshot = self.instrumentation.snapshot()
        return DualPathWork(
            intercepted_boundary_calls=1,
            upstream_two_view_refresh_calls=int(scene_refresh),
            physical_fused_backbone_calls=int(scene_refresh),
            physical_siglip_calls=int(not scene_refresh),
            physical_dinov2_calls=int(not scene_refresh),
            physical_projector_calls=1,
            logical_scene_backbone_calls=int(scene_refresh),
            logical_wrist_backbone_calls=1,
            logical_scene_projector_calls=int(scene_refresh),
            logical_wrist_projector_calls=1,
            downstream_calls=1,
            component_wall_ms=dict(snapshot.component_wall_ms),
        )

    def run_query(
        self,
        *,
        query: Callable[[], T],
        scene_image: Any,
        wrist_image: Any,
        state: Any,
        state_q01: Sequence[float],
        state_q99: Sequence[float],
    ) -> DualPathQueryResult[T]:
        if not self.installed or self._context is None:
            raise RuntimeError("Use the dual-path adapter inside an episode context")
        if threading.get_ident() != self._owner_thread:
            raise RuntimeError("Concurrent dual-path query use is prohibited")
        if self._active_query is not None:
            raise RuntimeError("Nested dual-path query use is prohibited")
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
        scene_sha = audit_sha256(scene_image)
        wrist_sha = audit_sha256(wrist_image)
        proprio_sha = audit_sha256(state)
        self.instrumentation.reset()
        active = _ActiveQuery(decision, scene_representation, position)
        self._active_query = active
        timing_started = False
        timing_finished = False
        self.last_failure = None
        try:
            self.instrumentation.start_query()
            timing_started = self.instrumentation.timer is not None
            value = query()
            query_wall_ms = (time.perf_counter() - query_started) * 1000.0
            if active.original_boundary_calls != 1:
                raise RuntimeError(
                    f"Expected one visual-boundary invocation, observed {active.original_boundary_calls}"
                )
            effective = active.effective_decision
            work = active.work
            if effective is None or work is None:
                raise RuntimeError("Dual-path visual boundary did not produce accounting")
            self.instrumentation.record_downstream()
            work = DualPathWork(**{**work.__dict__, "downstream_calls": 1})
            work.validate(scene_refresh=effective.refresh)
            action_chunk = self.action_chunk_getter(value)
            if not self.tensor_ops.all_finite(action_chunk):
                raise RuntimeError("Returned action chunk contains non-finite values")
            if effective.refresh:
                if self.cache.entry is None:
                    raise RuntimeError("Refresh completed without a compatible scene cache")
            else:
                self.cache.mark_reused()
            self.controller.observe(
                decision=effective,
                scene_representation=scene_representation,
                normalized_eef_position=position,
                action_chunk=action_chunk,
            )
            timing_finished = True
            device_timing = self.instrumentation.finish_query()
        except Exception as error:
            self.cache.invalidate()
            self.last_failure = DualPathFailure(
                query_index=decision.query_index,
                classification="invariant"
                if isinstance(error, (RuntimeError, ValueError))
                else "technical",
                message=str(error),
                cache_invalidated=True,
            )
            raise
        finally:
            try:
                if timing_started and not timing_finished:
                    self.instrumentation.finish_query()
            finally:
                self._active_query = None

        return DualPathQueryResult(
            value=value,
            decision=effective,
            work=work,
            cache_event=active.cache_event,
            controller_wall_ms=controller_wall_ms,
            query_wall_ms=query_wall_ms,
            device_timing=device_timing,
            scene_image_sha256=scene_sha,
            wrist_image_sha256=wrist_sha,
            proprio_sha256=proprio_sha,
        )
