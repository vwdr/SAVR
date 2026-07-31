"""Common refresh-controller interface for FR, PR, VOR, and SAVR."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from savr.cache import CacheContext
from savr.signals import (
    SignalValidationError,
    action_change,
    freeze_numeric,
    image_change,
    prepare_image_representations,
    state_change,
)


class Policy(str, Enum):
    FR = "FR"
    PR = "PR"
    VOR = "VOR"
    SAVR = "SAVR"
    SAVR2 = "SAVR2"
    SAVR3 = "SAVR3"


@dataclass(frozen=True)
class SignalStatistics:
    state_q01: tuple[float, ...] = ()
    state_q99: tuple[float, ...] = ()
    action_q01: tuple[float, ...] = ()
    action_q99: tuple[float, ...] = ()


@dataclass(frozen=True)
class RefreshDecision:
    policy: Policy
    query_index: int
    refresh: bool
    cache_age_before: int
    triggers: tuple[str, ...]
    image_score: float | None = None
    per_camera_image_scores: dict[str, float] = field(default_factory=dict)
    state_score: float | None = None
    action_score: float | None = None
    thresholds: dict[str, float | int] = field(default_factory=dict)
    camera_patch_scores: dict[str, tuple[float, ...]] = field(default_factory=dict)
    camera_global_scores: dict[str, float] = field(default_factory=dict)
    state_group_scores: dict[str, float] = field(default_factory=dict)
    action_group_scores: dict[str, float] = field(default_factory=dict)
    gripper_transition_veto: bool = False
    translation_direction_reversals: tuple[bool, ...] = ()
    stable_fresh_before: int = 0
    completed_reuses_before: int = 0
    signals_stable: bool = False

    def force_refresh(self, trigger: str) -> "RefreshDecision":
        triggers = tuple(dict.fromkeys((*self.triggers, trigger)))
        return RefreshDecision(
            policy=self.policy,
            query_index=self.query_index,
            refresh=True,
            cache_age_before=self.cache_age_before,
            triggers=triggers,
            image_score=self.image_score,
            per_camera_image_scores=dict(self.per_camera_image_scores),
            state_score=self.state_score,
            action_score=self.action_score,
            thresholds=dict(self.thresholds),
            camera_patch_scores={
                name: tuple(values) for name, values in self.camera_patch_scores.items()
            },
            camera_global_scores=dict(self.camera_global_scores),
            state_group_scores=dict(self.state_group_scores),
            action_group_scores=dict(self.action_group_scores),
            gripper_transition_veto=self.gripper_transition_veto,
            translation_direction_reversals=tuple(
                self.translation_direction_reversals
            ),
            stable_fresh_before=self.stable_fresh_before,
            completed_reuses_before=self.completed_reuses_before,
            signals_stable=self.signals_stable,
        )


class RefreshController(ABC):
    """Stateful controller contract at policy-query granularity."""

    @abstractmethod
    def reset(self, context: CacheContext) -> None:
        """Reset all episode/context-scoped controller history."""

    @abstractmethod
    def decide(
        self,
        *,
        images: Mapping[str, Any],
        state: Any,
        cache_available: bool,
        cache_age: int,
    ) -> RefreshDecision:
        """Return a refresh/reuse decision before the policy query."""

    @abstractmethod
    def observe(
        self,
        *,
        decision: RefreshDecision,
        images: Mapping[str, Any],
        state: Any,
        action_chunk: Any,
    ) -> None:
        """Record a successfully completed policy query."""


class _PolicyController(RefreshController):
    def __init__(
        self,
        *,
        policy: Policy,
        statistics: SignalStatistics = SignalStatistics(),
        period: int | None = None,
        image_threshold: float | None = None,
        state_threshold: float | None = None,
        action_threshold: float | None = None,
        max_reuse_horizon: int | None = None,
    ) -> None:
        self.policy = policy
        self.statistics = statistics
        self.period = period
        self.image_threshold = image_threshold
        self.state_threshold = state_threshold
        self.action_threshold = action_threshold
        self.max_reuse_horizon = max_reuse_horizon
        self._validate_configuration()
        self._context: CacheContext | None = None
        self._query_index = 0
        self._reference_images: dict[str, tuple[float, ...]] | None = None
        self._previous_state: tuple[float, ...] | None = None
        self._action_history: deque[tuple[float, ...]] = deque(maxlen=2)

    def _validate_configuration(self) -> None:
        if self.policy is Policy.PR and (self.period is None or self.period < 1):
            raise ValueError("PR requires period >= 1")
        if self.policy in (Policy.VOR, Policy.SAVR):
            if self.image_threshold is None:
                raise ValueError(f"{self.policy.value} requires an image threshold")
            if self.max_reuse_horizon is None or self.max_reuse_horizon < 1:
                raise ValueError(f"{self.policy.value} requires maximum horizon >= 1")
        if self.policy is Policy.SAVR:
            if self.state_threshold is None or self.action_threshold is None:
                raise ValueError("SAVR requires state and action thresholds")
            if len(self.statistics.state_q01) != 8 or len(self.statistics.state_q99) != 8:
                raise ValueError("SAVR requires eight-dimensional state statistics")
            if not self.statistics.action_q01 or not self.statistics.action_q99:
                raise ValueError("SAVR requires action statistics")
            for lows, highs in (
                (self.statistics.state_q01, self.statistics.state_q99),
                (self.statistics.action_q01, self.statistics.action_q99),
            ):
                if len(lows) != len(highs) or any(
                    not math.isfinite(low)
                    or not math.isfinite(high)
                    or high <= low
                    for low, high in zip(lows, highs)
                ):
                    raise ValueError("Every q99 statistic must finitely exceed q01")

        for threshold in (
            self.image_threshold,
            self.state_threshold,
            self.action_threshold,
        ):
            if threshold is not None and (
                not math.isfinite(threshold) or threshold < 0
            ):
                raise ValueError("Signal thresholds must be finite and non-negative")

    @property
    def context(self) -> CacheContext | None:
        return self._context

    @property
    def query_index(self) -> int:
        return self._query_index

    def reset(self, context: CacheContext) -> None:
        self._context = context
        self._query_index = 0
        self._reference_images = None
        self._previous_state = None
        self._action_history.clear()

    def _thresholds(self) -> dict[str, float | int]:
        values: dict[str, float | int] = {}
        if self.period is not None:
            values["period"] = self.period
        if self.image_threshold is not None:
            values["image"] = self.image_threshold
        if self.state_threshold is not None:
            values["state"] = self.state_threshold
        if self.action_threshold is not None:
            values["action"] = self.action_threshold
        if self.max_reuse_horizon is not None:
            values["max_reuse_horizon"] = self.max_reuse_horizon
        return values

    def decide(
        self,
        *,
        images: Mapping[str, Any],
        state: Any,
        cache_available: bool,
        cache_age: int,
    ) -> RefreshDecision:
        if self._context is None:
            raise RuntimeError("Controller context must be reset before deciding")
        if cache_age < 0:
            raise ValueError("Cache age cannot be negative")

        triggers: list[str] = []
        image_score_value: float | None = None
        camera_scores: dict[str, float] = {}
        state_score_value: float | None = None
        action_score_value: float | None = None

        if not cache_available:
            triggers.append("empty_cache")

        if self.policy is Policy.FR:
            triggers.append("full_refresh")

        elif self.policy is Policy.PR:
            assert self.period is not None
            if self._query_index % self.period == 0:
                triggers.append("periodic")

        else:
            if self._reference_images is not None:
                try:
                    image_result = image_change(images, self._reference_images)
                    image_score_value = image_result.mean
                    camera_scores = image_result.per_camera
                    assert self.image_threshold is not None
                    if image_score_value > self.image_threshold:
                        triggers.append("image_change")
                except SignalValidationError:
                    triggers.append("invalid_image")
            elif cache_available:
                triggers.append("invalid_image_reference")

            assert self.max_reuse_horizon is not None
            if cache_available and cache_age >= self.max_reuse_horizon:
                triggers.append("max_reuse_horizon")

            if self.policy is Policy.SAVR:
                if self._previous_state is None:
                    if cache_available:
                        triggers.append("invalid_state_history")
                else:
                    try:
                        state_score_value = state_change(
                            state,
                            self._previous_state,
                            self.statistics.state_q01,
                            self.statistics.state_q99,
                        )
                        assert self.state_threshold is not None
                        if state_score_value > self.state_threshold:
                            triggers.append("state_change")
                    except SignalValidationError:
                        triggers.append("invalid_state")

                if len(self._action_history) < 2:
                    triggers.append("action_history_warmup")
                else:
                    try:
                        action_score_value = action_change(
                            self._action_history[-1],
                            self._action_history[-2],
                            self.statistics.action_q01,
                            self.statistics.action_q99,
                        )
                        assert self.action_threshold is not None
                        if action_score_value > self.action_threshold:
                            triggers.append("action_change")
                    except SignalValidationError:
                        triggers.append("invalid_action_history")

        triggers = list(dict.fromkeys(triggers))
        return RefreshDecision(
            policy=self.policy,
            query_index=self._query_index,
            refresh=bool(triggers),
            cache_age_before=cache_age,
            triggers=tuple(triggers),
            image_score=image_score_value,
            per_camera_image_scores=camera_scores,
            state_score=state_score_value,
            action_score=action_score_value,
            thresholds=self._thresholds(),
        )

    def observe(
        self,
        *,
        decision: RefreshDecision,
        images: Mapping[str, Any],
        state: Any,
        action_chunk: Any,
    ) -> None:
        if decision.query_index != self._query_index:
            raise RuntimeError("Decision/query index does not match controller state")
        if decision.policy is not self.policy:
            raise RuntimeError("Decision policy does not match controller")

        if self.policy in (Policy.VOR, Policy.SAVR) and decision.refresh:
            try:
                self._reference_images = prepare_image_representations(images)
            except SignalValidationError:
                self._reference_images = None

        if self.policy is Policy.SAVR:
            try:
                frozen_state = freeze_numeric(state)
                self._previous_state = (
                    frozen_state if len(frozen_state) == len(self.statistics.state_q01) else None
                )
            except SignalValidationError:
                self._previous_state = None
            try:
                self._action_history.append(freeze_numeric(action_chunk))
            except SignalValidationError:
                self._action_history.clear()

        self._query_index += 1


class FullRefreshController(_PolicyController):
    def __init__(self) -> None:
        super().__init__(policy=Policy.FR)


class PeriodicRefreshController(_PolicyController):
    def __init__(self, period: int) -> None:
        super().__init__(policy=Policy.PR, period=period)


class VisualOnlyRefreshController(_PolicyController):
    def __init__(self, *, image_threshold: float, max_reuse_horizon: int) -> None:
        super().__init__(
            policy=Policy.VOR,
            image_threshold=image_threshold,
            max_reuse_horizon=max_reuse_horizon,
        )


class StateAwareVisualRefreshController(_PolicyController):
    def __init__(
        self,
        *,
        image_threshold: float,
        state_threshold: float,
        action_threshold: float,
        max_reuse_horizon: int,
        state_q01: Sequence[float],
        state_q99: Sequence[float],
        action_q01: Sequence[float],
        action_q99: Sequence[float],
    ) -> None:
        super().__init__(
            policy=Policy.SAVR,
            statistics=SignalStatistics(
                state_q01=tuple(float(value) for value in state_q01),
                state_q99=tuple(float(value) for value in state_q99),
                action_q01=tuple(float(value) for value in action_q01),
                action_q99=tuple(float(value) for value in action_q99),
            ),
            image_threshold=image_threshold,
            state_threshold=state_threshold,
            action_threshold=action_threshold,
            max_reuse_horizon=max_reuse_horizon,
        )
