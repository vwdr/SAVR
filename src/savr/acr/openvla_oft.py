"""Exception-safe camera-factorized adapter for pinned OpenVLA-OFT."""

from __future__ import annotations

import threading
import time
import types
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeVar

from savr.acr.cache import SceneCacheError, SceneTokenCache
from savr.acr.controller import ACRController
from savr.acr.instrumentation import CameraInstrumentation
from savr.acr.signals import (
    audit_sha256,
    normalized_eef_position,
    prepare_scene_representation,
)
from savr.acr.types import ACRContext, CameraWork, SceneDecision, SceneTensorMetadata
from savr.signals import SignalValidationError
from savr.timing import QueryTiming


T = TypeVar("T")


class TensorOperations(Protocol):
    def split(self, value: Any, sections: Sequence[int], *, dim: int) -> tuple[Any, ...]: ...

    def cat(self, values: Sequence[Any], *, dim: int) -> Any: ...

    def all_finite(self, value: Any) -> bool: ...


class TorchTensorOperations:
    def __init__(self, torch_module: Any) -> None:
        self.torch = torch_module

    def split(self, value: Any, sections: Sequence[int], *, dim: int) -> tuple[Any, ...]:
        return tuple(self.torch.split(value, list(sections), dim=dim))

    def cat(self, values: Sequence[Any], *, dim: int) -> Any:
        return self.torch.cat(tuple(values), dim=dim)

    def all_finite(self, value: Any) -> bool:
        return bool(self.torch.isfinite(value).all().item())


@dataclass(frozen=True)
class ACRQueryResult(Generic[T]):
    value: T
    decision: SceneDecision
    work: CameraWork
    cache_event: str
    controller_wall_ms: float
    query_wall_ms: float
    device_timing: QueryTiming | None
    scene_image_sha256: str
    wrist_image_sha256: str
    proprio_sha256: str


def _shape(value: Any) -> tuple[int, ...]:
    try:
        shape = tuple(int(item) for item in value.shape)
    except Exception as error:
        raise ValueError("Tensor lacks valid shape metadata") from error
    if not shape or any(item < 1 for item in shape):
        raise ValueError("Tensor dimensions must be positive")
    return shape


class OpenVLAAsymmetricCameraAdapter:
    """Replace one visual boundary while leaving the downstream query unchanged."""

    METHOD_NAME = "_process_vision_features"

    def __init__(
        self,
        *,
        model: Any,
        controller: ACRController,
        tensor_ops: TensorOperations,
        cache: SceneTokenCache | None = None,
        instrumentation: CameraInstrumentation | None = None,
        action_chunk_getter: Callable[[Any], Any] | None = None,
    ) -> None:
        if not callable(getattr(model, self.METHOD_NAME, None)):
            raise TypeError(f"Model does not expose {self.METHOD_NAME}")
        self.model = model
        self.controller = controller
        self.tensor_ops = tensor_ops
        self.cache = cache or SceneTokenCache()
        self.instrumentation = instrumentation or CameraInstrumentation()
        self.action_chunk_getter = action_chunk_getter or (lambda result: result)
        self._context: ACRContext | None = None
        self._lock = threading.Lock()

    @property
    def context(self) -> ACRContext | None:
        return self._context

    def begin_context(self, context: ACRContext) -> bool:
        changed = context != self._context
        if changed:
            self.cache.invalidate()
            self.controller.reset(context)
            self._context = context
        return changed

    def _validate_model_contract(self) -> None:
        context = self._context
        assert context is not None
        backbone = self.model.vision_backbone
        if not bool(getattr(backbone, "use_fused_vision_backbone", False)):
            raise RuntimeError("ACR requires the fused SigLIP/DINOv2 backbone")
        if int(backbone.get_num_images_in_input()) != 2:
            raise RuntimeError("ACR requires exactly two upstream images")
        if int(backbone.get_num_patches()) != context.patch_count:
            raise RuntimeError("Upstream patch count differs from the frozen context")
        if not callable(getattr(backbone, "featurizer", None)) or not callable(
            getattr(backbone, "fused_featurizer", None)
        ):
            raise RuntimeError("Pinned per-camera featurizers are unavailable")
        if not callable(getattr(self.model, "projector", None)):
            raise RuntimeError("Pinned visual projector is unavailable")

    def _encode_camera(
        self,
        *,
        name: str,
        pixels: Any,
        expected: SceneTensorMetadata,
    ) -> Any:
        regular, fused = self.tensor_ops.split(pixels, (3, 3), dim=1)
        backbone = self.model.vision_backbone
        regular_features = self.instrumentation.call(
            f"{name}.siglip", backbone.featurizer, regular
        )
        fused_features = self.instrumentation.call(
            f"{name}.dinov2", backbone.fused_featurizer, fused
        )
        combined = self.instrumentation.measure_operation(
            f"{name}.tower-concat",
            lambda: self.tensor_ops.cat((regular_features, fused_features), dim=2),
        )
        projected = self.instrumentation.call(
            f"{name}.projector", self.model.projector, combined
        )
        actual = SceneTensorMetadata.from_value(
            projected, patch_count=expected.patch_count
        )
        if actual != expected:
            raise RuntimeError(
                f"{name} projected block metadata {actual} differs from {expected}"
            )
        if not self.tensor_ops.all_finite(projected):
            raise RuntimeError(f"{name} projected block contains non-finite values")
        return projected

    def run_query(
        self,
        *,
        query: Callable[[], T],
        scene_image: Any,
        wrist_image: Any,
        state: Any,
        state_q01: Sequence[float],
        state_q99: Sequence[float],
    ) -> ACRQueryResult[T]:
        context = self._context
        if context is None:
            raise RuntimeError("begin_context must be called before run_query")
        self._validate_model_contract()
        query_started = time.perf_counter()

        decision_started = time.perf_counter()
        scene_representation: tuple[float, ...] | None
        position: tuple[float, float, float] | None
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
            cache_available=self.cache.available(context),
            cache_age=self.cache.age,
        )
        controller_wall_ms = (time.perf_counter() - decision_started) * 1000.0
        scene_image_sha256 = audit_sha256(scene_image)
        wrist_image_sha256 = audit_sha256(wrist_image)
        proprio_sha256 = audit_sha256(state)

        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Nested or concurrent ACR model use is prohibited")
        instance_dict = vars(self.model)
        had_override = self.METHOD_NAME in instance_dict
        previous_override = instance_dict.get(self.METHOD_NAME)
        invocation_count = 0
        effective = decision
        cache_event = "refresh" if decision.refresh else "reuse"
        self.instrumentation.reset()
        timing_started = False
        timing_finished = False
        override_installed = False
        device_timing: QueryTiming | None = None

        def intercepted(
            model_instance: Any,
            pixel_values: Any,
            language_embeddings: Any = None,
            use_film: bool = False,
        ) -> Any:
            nonlocal invocation_count, effective, cache_event
            invocation_count += 1
            if invocation_count != 1:
                raise RuntimeError("Expected exactly one visual-boundary invocation")
            if use_film:
                raise RuntimeError("ACR Version 1 does not support FiLM")
            if language_embeddings is None:
                raise RuntimeError("Language embeddings are required")
            pixel_shape = _shape(pixel_values)
            language_shape = _shape(language_embeddings)
            if len(pixel_shape) != 4 or pixel_shape[0] != 1 or pixel_shape[1] != 12:
                raise RuntimeError("ACR requires pixel tensor shape [1,12,H,W]")
            if len(language_shape) != 3 or language_shape[0] != 1:
                raise RuntimeError("Language embeddings require batch-one rank-three shape")

            expected = SceneTensorMetadata(
                shape=(1, context.patch_count, language_shape[-1]),
                dtype=str(getattr(language_embeddings, "dtype", "")),
                device=str(getattr(language_embeddings, "device", "")),
                patch_count=context.patch_count,
            )
            cacheable = expected.dtype == context.dtype and expected.device == context.device
            if not cacheable:
                effective = effective.force_refresh("cache")
                cache_event = "forced-refresh"
                self.cache.invalidate()

            scene_pixels, wrist_pixels = self.tensor_ops.split(
                pixel_values, (6, 6), dim=1
            )
            scene_tokens: Any
            if effective.refresh:
                scene_tokens = self._encode_camera(
                    name="scene", pixels=scene_pixels, expected=expected
                )
                if cacheable:
                    self.cache.store(
                        context=context,
                        tokens=scene_tokens,
                        refresh_query_index=effective.query_index,
                    )
            else:
                try:
                    scene_tokens = self.cache.load(context, expected)
                except SceneCacheError:
                    effective = effective.force_refresh("cache")
                    cache_event = "forced-refresh"
                    scene_tokens = self._encode_camera(
                        name="scene", pixels=scene_pixels, expected=expected
                    )
                    self.cache.store(
                        context=context,
                        tokens=scene_tokens,
                        refresh_query_index=effective.query_index,
                    )

            wrist_tokens = self._encode_camera(
                name="wrist", pixels=wrist_pixels, expected=expected
            )
            combined = self.instrumentation.measure_operation(
                "camera-block-concat",
                lambda: self.tensor_ops.cat((scene_tokens, wrist_tokens), dim=1),
            )
            combined_shape = _shape(combined)
            if combined_shape != (1, context.patch_count * 2, language_shape[-1]):
                raise RuntimeError("Combined visual block has an unexpected shape/order")
            if str(getattr(combined, "dtype", "")) != expected.dtype or str(
                getattr(combined, "device", "")
            ) != expected.device:
                raise RuntimeError("Combined visual block dtype/device changed")
            if not self.tensor_ops.all_finite(combined):
                raise RuntimeError("Combined visual block contains non-finite values")
            return combined

        try:
            setattr(
                self.model,
                self.METHOD_NAME,
                types.MethodType(intercepted, self.model),
            )
            override_installed = True
            self.instrumentation.start_query()
            timing_started = self.instrumentation.timer is not None
            value = query()
            query_wall_ms = (time.perf_counter() - query_started) * 1000.0
            if invocation_count != 1:
                raise RuntimeError(
                    f"Expected one visual-boundary invocation, observed {invocation_count}"
                )
            self.instrumentation.record_downstream()
            work = self.instrumentation.snapshot()
            work.validate(scene_refresh=effective.refresh)
            action_chunk = self.action_chunk_getter(value)
            if effective.refresh:
                if self.cache.entry is None:
                    self.cache.invalidate()
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
        except Exception:
            self.cache.invalidate()
            raise
        finally:
            try:
                if timing_started and not timing_finished:
                    self.instrumentation.finish_query()
            finally:
                if override_installed:
                    if had_override:
                        setattr(self.model, self.METHOD_NAME, previous_override)
                    else:
                        delattr(self.model, self.METHOD_NAME)
                self._lock.release()

        return ACRQueryResult(
            value=value,
            decision=effective,
            work=work,
            cache_event=cache_event,
            controller_wall_ms=controller_wall_ms,
            query_wall_ms=query_wall_ms,
            device_timing=device_timing,
            scene_image_sha256=scene_image_sha256,
            wrist_image_sha256=wrist_image_sha256,
            proprio_sha256=proprio_sha256,
        )
