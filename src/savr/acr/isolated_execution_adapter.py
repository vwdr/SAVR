"""Fail-closed integration of IR-SA-ACR with the V5 static executor."""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, replace
from typing import Any, Generic, TypeVar

from savr.acr.cache import SceneCacheError, SceneTokenCache
from savr.acr.isolated_controller import IsolatedACRController
from savr.acr.reuse_executor import (
    ReuseCompatibilityKey,
    ReuseExecutionInputs,
    ReuseExecutionResult,
    ReuseExecutorFailure,
    ReuseExecutorSnapshot,
    ReuseExecutorUnavailable,
    StaticBufferReuseExecutor,
)
from savr.acr.signals import normalized_eef_position, prepare_scene_representation
from savr.acr.types import ACRContext, SceneDecision, SceneTensorMetadata
from savr.signals import SignalValidationError


ISOLATED_EXECUTION_ADAPTER_VERSION = "ir-sa-acr-static-executor-v1"
SELECTED_V5_CONFIGURATION_ID = "v5-a100-b40"
_EXECUTOR_REASON_ORDER = ("executor-unavailable", "executor-failure")
T = TypeVar("T")


@dataclass(frozen=True)
class EpisodeMethodPatch:
    """One temporary instance-level method replacement."""

    target: Any
    name: str
    replacement: Any


@dataclass(frozen=True)
class RefreshExecutionResult(Generic[T]):
    value: T
    scene_tokens: Any
    action_chunk: Any


@dataclass(frozen=True)
class IsolatedExecutionResult(Generic[T]):
    value: T | Any
    action_chunk: Any
    decision: SceneDecision
    cache_event: str
    executor_snapshot: ReuseExecutorSnapshot
    reuse_execution: ReuseExecutionResult | None


@dataclass(frozen=True)
class IsolatedExecutionFailure:
    classification: str
    reason: str
    message: str
    query_index: int | None
    cache_invalidated: bool
    controller_observed: bool
    retry_attempted: bool


class _Episode(AbstractContextManager["IsolatedReuseExecutionAdapter"]):
    def __init__(
        self,
        adapter: "IsolatedReuseExecutionAdapter",
        context: ACRContext,
        key: ReuseCompatibilityKey,
        patches: Sequence[EpisodeMethodPatch],
    ) -> None:
        self.adapter = adapter
        self.context = context
        self.key = key
        self.patches = tuple(patches)
        self.entered = False

    def __enter__(self) -> "IsolatedReuseExecutionAdapter":
        self.adapter._install(self.context, self.key, self.patches)
        self.entered = True
        return self.adapter

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if self.entered:
            self.adapter._restore()
            self.entered = False


class IsolatedReuseExecutionAdapter:
    """Episode-scoped controller/cache/executor coordinator.

    Refresh execution remains an injected eager callback.  Reuse execution is
    delegated only to the prepared static executor.  A prelaunch rejection is
    converted to an explicit refresh; a postlaunch failure is never retried.
    """

    version = ISOLATED_EXECUTION_ADAPTER_VERSION

    def __init__(
        self,
        *,
        controller: IsolatedACRController,
        cache: SceneTokenCache,
        executor: StaticBufferReuseExecutor,
        reuse_value_builder: Callable[[ReuseExecutionResult], Any] | None = None,
        reuse_action_getter: Callable[[ReuseExecutionResult], Any] | None = None,
    ) -> None:
        if not isinstance(controller, IsolatedACRController):
            raise TypeError("V5 static integration requires IsolatedACRController")
        if not isinstance(cache, SceneTokenCache):
            raise TypeError("V5 static integration requires SceneTokenCache")
        if not isinstance(executor, StaticBufferReuseExecutor):
            raise TypeError("V5 static integration requires StaticBufferReuseExecutor")
        configuration = controller.configuration
        selected = (
            configuration.configuration_id == SELECTED_V5_CONFIGURATION_ID
            and configuration.scene_threshold == 0.30046895424836606
            and configuration.translation_threshold == 0.685919037527938
            and configuration.horizon == 1
            and configuration.hard_reuse_cap == 0.40
            and configuration.minimum_query_index == 2
        )
        if not selected:
            raise ValueError("V5 static integration requires the exact V5-B selected controller")
        self.controller = controller
        self.cache = cache
        self.executor = executor
        self.reuse_value_builder = reuse_value_builder or (lambda result: result.normalized_actions)
        self.reuse_action_getter = reuse_action_getter or (lambda result: result.normalized_actions)
        self._lock = threading.Lock()
        self._owner_thread: int | None = None
        self._context: ACRContext | None = None
        self._key: ReuseCompatibilityKey | None = None
        self._active = False
        self._patch_records: list[tuple[Any, str, bool, Any]] = []
        self.last_failure: IsolatedExecutionFailure | None = None

    @property
    def installed(self) -> bool:
        return self._owner_thread is not None

    def episode(
        self,
        context: ACRContext,
        key: ReuseCompatibilityKey,
        *,
        patches: Sequence[EpisodeMethodPatch] = (),
    ) -> _Episode:
        return _Episode(self, context, key, patches)

    @staticmethod
    def _validate_context_key(context: ACRContext, key: ReuseCompatibilityKey) -> None:
        shared = (
            "checkpoint_id",
            "upstream_revision",
            "configuration_id",
            "controller_version",
            "preprocessing_id",
            "action_head_id",
            "instruction_sha256",
            "dtype",
            "device",
            "patch_count",
        )
        if any(getattr(context, name) != getattr(key, name) for name in shared):
            raise ReuseExecutorUnavailable("Episode context and executor key are incompatible")

    @staticmethod
    def _force_executor_refresh(decision: SceneDecision, reason: str) -> SceneDecision:
        if reason not in _EXECUTOR_REASON_ORDER:
            raise ValueError(f"Unsupported V5 executor refresh reason: {reason}")
        reasons = list(decision.reasons)
        if reason not in reasons:
            policy_index = reasons.index("policy") if "policy" in reasons else len(reasons)
            reasons.insert(policy_index, reason)
        return replace(decision, refresh=True, reasons=tuple(reasons))

    def _install(
        self,
        context: ACRContext,
        key: ReuseCompatibilityKey,
        patches: Sequence[EpisodeMethodPatch],
    ) -> None:
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Nested or concurrent V5 executor episodes are prohibited")
        try:
            self._validate_context_key(context, key)
            self.controller.reset(context)
            self.cache.invalidate()
            self.executor.reset()
            self.executor.prepare(key)
            self._context = context
            self._key = key
            self._owner_thread = threading.get_ident()
            for patch in patches:
                if not patch.name or not callable(patch.replacement):
                    raise TypeError("Episode method patches require a callable replacement")
                instance = vars(patch.target)
                had_override = patch.name in instance
                previous = instance.get(patch.name)
                self._patch_records.append((patch.target, patch.name, had_override, previous))
                setattr(patch.target, patch.name, patch.replacement)
        except Exception:
            self._restore_partial_install()
            raise

    def _restore_partial_install(self) -> None:
        self._restore_patches()
        self.cache.invalidate()
        if self.executor.lifecycle.value != "ACTIVE":
            self.executor.reset()
        self._context = None
        self._key = None
        self._owner_thread = None
        if self._lock.locked():
            self._lock.release()

    def _restore_patches(self) -> None:
        for target, name, had_override, previous in reversed(self._patch_records):
            if had_override:
                setattr(target, name, previous)
            else:
                try:
                    delattr(target, name)
                except AttributeError:
                    pass
        self._patch_records.clear()

    def _restore(self) -> None:
        if self._active:
            raise RuntimeError("Cannot close a V5 executor episode during a query")
        self._restore_patches()
        self.cache.invalidate()
        self.executor.reset()
        self._context = None
        self._key = None
        self._owner_thread = None
        self.last_failure = None
        self._lock.release()

    def _require_query_scope(self) -> tuple[ACRContext, ReuseCompatibilityKey]:
        if not self.installed or self._context is None or self._key is None:
            raise RuntimeError("Use the V5 executor adapter inside an episode context")
        if threading.get_ident() != self._owner_thread:
            raise RuntimeError("Concurrent V5 executor query use is prohibited")
        if self._active:
            raise RuntimeError("Nested V5 executor query use is prohibited")
        return self._context, self._key

    def run_query(
        self,
        *,
        scene_image: Any,
        state: Any,
        state_q01: Sequence[float],
        state_q99: Sequence[float],
        reuse_inputs: ReuseExecutionInputs,
        refresh_query: Callable[[SceneDecision], RefreshExecutionResult[T]],
    ) -> IsolatedExecutionResult[T]:
        context, key = self._require_query_scope()
        self._active = True
        self.last_failure = None
        decision: SceneDecision | None = None
        controller_observed = False
        try:
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

            reuse_execution: ReuseExecutionResult | None = None
            if decision.refresh:
                effective = decision
                refresh = refresh_query(effective)
                self.cache.store(
                    context=context,
                    tokens=refresh.scene_tokens,
                    refresh_query_index=effective.query_index,
                )
                value = refresh.value
                action_chunk = refresh.action_chunk
                cache_event = "refresh"
            else:
                expected_scene = SceneTensorMetadata(
                    shape=key.cached_scene_shape,
                    dtype=key.dtype,
                    device=key.device,
                    patch_count=key.patch_count,
                )
                try:
                    cached_scene_tokens = self.cache.load(context, expected_scene)
                except SceneCacheError:
                    effective = decision.force_refresh("cache")
                    self.cache.invalidate()
                    refresh = refresh_query(effective)
                    self.cache.store(
                        context=context,
                        tokens=refresh.scene_tokens,
                        refresh_query_index=effective.query_index,
                    )
                    value = refresh.value
                    action_chunk = refresh.action_chunk
                    cache_event = "forced-refresh"
                    reuse_execution = None
                    self.controller.observe(
                        decision=effective,
                        scene_representation=scene_representation,
                        normalized_eef_position=position,
                        action_chunk=action_chunk,
                    )
                    controller_observed = True
                    return IsolatedExecutionResult(
                        value=value,
                        action_chunk=action_chunk,
                        decision=effective,
                        cache_event=cache_event,
                        executor_snapshot=self.executor.snapshot(),
                        reuse_execution=reuse_execution,
                    )
                expected_inputs = ReuseExecutionInputs(
                    compatibility_key=reuse_inputs.compatibility_key,
                    wrist_pixels=reuse_inputs.wrist_pixels,
                    cached_scene_tokens=cached_scene_tokens,
                    prompt_input=reuse_inputs.prompt_input,
                    prompt_embeddings=reuse_inputs.prompt_embeddings,
                    attention_mask=reuse_inputs.attention_mask,
                    proprioception=reuse_inputs.proprioception,
                )
                try:
                    reuse_execution = self.executor.run(expected_inputs)
                except ReuseExecutorUnavailable:
                    effective = self._force_executor_refresh(decision, "executor-unavailable")
                    refresh = refresh_query(effective)
                    self.cache.store(
                        context=context,
                        tokens=refresh.scene_tokens,
                        refresh_query_index=effective.query_index,
                    )
                    value = refresh.value
                    action_chunk = refresh.action_chunk
                    cache_event = "forced-refresh"
                except ReuseExecutorFailure as error:
                    self.cache.invalidate()
                    self.last_failure = IsolatedExecutionFailure(
                        classification="technical",
                        reason="executor-failure",
                        message=str(error),
                        query_index=decision.query_index,
                        cache_invalidated=True,
                        controller_observed=False,
                        retry_attempted=False,
                    )
                    raise
                else:
                    effective = decision
                    self.cache.mark_reused()
                    value = self.reuse_value_builder(reuse_execution)
                    action_chunk = self.reuse_action_getter(reuse_execution)
                    cache_event = "reuse"

            self.controller.observe(
                decision=effective,
                scene_representation=scene_representation,
                normalized_eef_position=position,
                action_chunk=action_chunk,
            )
            controller_observed = True
            return IsolatedExecutionResult(
                value=value,
                action_chunk=action_chunk,
                decision=effective,
                cache_event=cache_event,
                executor_snapshot=self.executor.snapshot(),
                reuse_execution=reuse_execution,
            )
        except Exception as error:
            if self.last_failure is None:
                self.cache.invalidate()
                self.last_failure = IsolatedExecutionFailure(
                    classification=(
                        "invariant"
                        if isinstance(error, (RuntimeError, ValueError))
                        else "technical"
                    ),
                    reason=(
                        "executor-failure"
                        if isinstance(error, ReuseExecutorFailure)
                        else "query-failure"
                    ),
                    message=str(error),
                    query_index=None if decision is None else decision.query_index,
                    cache_invalidated=True,
                    controller_observed=controller_observed,
                    retry_attempted=False,
                )
            raise
        finally:
            self._active = False
