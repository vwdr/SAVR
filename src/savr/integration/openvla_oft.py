"""Exception-safe projected-feature interception for pinned OpenVLA-OFT."""

from __future__ import annotations

import threading
import time
import types
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from typing import Any, Generic, TypeVar

from savr.cache import (
    CacheCompatibilityError,
    CacheContext,
    CacheError,
    ProjectedFeatureCache,
    TensorMetadata,
)
from savr.controllers import RefreshController, RefreshDecision
from savr.logging import ImmutableRecordStore


T = TypeVar("T")


@dataclass(frozen=True)
class QueryResult(Generic[T]):
    value: T
    decision: RefreshDecision
    decision_seconds: float
    query_seconds: float
    cache_event: str


def _shape(value: Any) -> tuple[int, ...]:
    try:
        return tuple(int(dimension) for dimension in value.shape)
    except Exception as error:
        raise CacheCompatibilityError("Current model input lacks valid shape metadata") from error


class OpenVLAProjectedFeatureAdapter:
    """Wrap unmodified upstream inference while intercepting one visual method."""

    METHOD_NAME = "_process_vision_features"

    def __init__(
        self,
        *,
        model: Any,
        controller: RefreshController,
        cache: ProjectedFeatureCache | None = None,
        record_store: ImmutableRecordStore | None = None,
        action_chunk_getter: Callable[[Any], Any] | None = None,
    ) -> None:
        if not callable(getattr(model, self.METHOD_NAME, None)):
            raise TypeError(
                f"Model does not expose required {self.METHOD_NAME} boundary"
            )
        self.model = model
        self.controller = controller
        self.cache = cache or ProjectedFeatureCache()
        self.record_store = record_store
        self.action_chunk_getter = action_chunk_getter or (lambda value: value)
        self._context: CacheContext | None = None
        self._lock = threading.RLock()

    @property
    def context(self) -> CacheContext | None:
        return self._context

    def begin_context(self, context: CacheContext) -> bool:
        """Set context, resetting all state only when identity changes."""

        changed = context != self._context
        if changed:
            self.cache.invalidate()
            self.controller.reset(context)
            self._context = context
        return changed

    def _expected_metadata(
        self,
        *,
        pixel_values: Any,
        language_embeddings: Any,
    ) -> TensorMetadata:
        pixel_shape = _shape(pixel_values)
        language_shape = _shape(language_embeddings)
        if not pixel_shape or not language_shape:
            raise CacheCompatibilityError("Model inputs must be non-empty tensors")
        vision_backbone = self.model.vision_backbone
        patch_count = int(vision_backbone.get_num_patches()) * int(
            vision_backbone.get_num_images_in_input()
        )
        return TensorMetadata(
            shape=(pixel_shape[0], patch_count, language_shape[-1]),
            dtype=str(getattr(language_embeddings, "dtype", "unknown")),
            device=str(getattr(language_embeddings, "device", "unknown")),
        )

    def run_query(
        self,
        *,
        query: Callable[[], T],
        images: Mapping[str, Any],
        state: Any,
        environment_step: int,
    ) -> QueryResult[T]:
        """Execute one otherwise-unmodified upstream policy query."""

        context = self._context
        if context is None:
            raise RuntimeError("begin_context must be called before run_query")
        if environment_step < 0:
            raise ValueError("Environment step cannot be negative")

        decision_started = time.perf_counter()
        decision = self.controller.decide(
            images=images,
            state=state,
            cache_available=self.cache.available(context),
            cache_age=self.cache.age,
        )
        decision_seconds = time.perf_counter() - decision_started

        with self._lock:
            original = getattr(self.model, self.METHOD_NAME)
            instance_dict = vars(self.model)
            had_instance_override = self.METHOD_NAME in instance_dict
            previous_instance_value = instance_dict.get(self.METHOD_NAME)
            invocation_count = 0
            cache_event = "refresh" if decision.refresh else "reuse"
            effective_decision = decision

            def intercepted(
                model_instance: Any,
                pixel_values: Any,
                language_embeddings: Any = None,
                use_film: bool = False,
            ) -> Any:
                nonlocal invocation_count, cache_event, effective_decision
                invocation_count += 1
                if invocation_count != 1:
                    raise RuntimeError(
                        "Expected exactly one visual-feature request per policy query"
                    )
                if use_film:
                    raise RuntimeError("Phase 3 adapter does not support FiLM")
                if language_embeddings is None:
                    raise CacheCompatibilityError(
                        "Language embeddings are required for compatibility checks"
                    )

                expected = self._expected_metadata(
                    pixel_values=pixel_values,
                    language_embeddings=language_embeddings,
                )
                if not effective_decision.refresh:
                    try:
                        return self.cache.load(context, expected)
                    except CacheError:
                        effective_decision = effective_decision.force_refresh(
                            "cache_incompatible"
                        )
                        cache_event = "forced_refresh"

                feature = original(pixel_values, language_embeddings, use_film)
                stored = self.cache.store(context, feature)
                if stored != expected:
                    self.cache.invalidate()
                    raise CacheCompatibilityError(
                        f"Produced feature metadata {stored} does not match {expected}"
                    )
                return feature

            setattr(
                self.model,
                self.METHOD_NAME,
                types.MethodType(intercepted, self.model),
            )
            query_started = time.perf_counter()
            try:
                value = query()
                query_seconds = time.perf_counter() - query_started
                if invocation_count != 1:
                    raise RuntimeError(
                        f"Expected one visual-feature request, observed {invocation_count}"
                    )
                action_chunk = self.action_chunk_getter(value)
                if not effective_decision.refresh:
                    self.cache.mark_reused()
                self.controller.observe(
                    decision=effective_decision,
                    images=images,
                    state=state,
                    action_chunk=action_chunk,
                )
            except Exception:
                self.cache.invalidate()
                raise
            finally:
                if had_instance_override:
                    setattr(self.model, self.METHOD_NAME, previous_instance_value)
                else:
                    delattr(self.model, self.METHOD_NAME)

        result = QueryResult(
            value=value,
            decision=effective_decision,
            decision_seconds=decision_seconds,
            query_seconds=query_seconds,
            cache_event=cache_event,
        )
        if self.record_store is not None:
            self.record_store.write_query(
                effective_decision.query_index,
                {
                    "context": asdict(context),
                    "environment_step": environment_step,
                    "query_index": effective_decision.query_index,
                    "decision": asdict(effective_decision),
                    "decision_seconds": decision_seconds,
                    "query_seconds": query_seconds,
                    "cache_event": cache_event,
                },
            )
        return result
