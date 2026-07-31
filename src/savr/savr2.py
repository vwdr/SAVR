"""Frozen SAVR 2.0 safety-constrained refresh controller."""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from savr.cache import CacheContext
from savr.controllers import Policy, RefreshController, RefreshDecision, SignalStatistics
from savr.signals import (
    SignalValidationError,
    action_transition,
    freeze_numeric,
    grouped_action_change,
    grouped_state_change,
    patch_image_change,
    prepare_image_representations,
)


EXPECTED_CAMERAS = frozenset({"full_image", "wrist_image"})
STATE_GROUPS = frozenset({"translation", "orientation", "gripper"})
ACTION_GROUPS = frozenset({"translation", "rotation", "gripper"})


@dataclass(frozen=True)
class SAVR2Configuration:
    """Frozen controller values for one candidate."""

    configuration_id: str
    image_thresholds: Mapping[str, float]
    state_thresholds: Mapping[str, float]
    action_thresholds: Mapping[str, float]
    skip_budget: float
    minimum_query_index: int = 5
    required_stable_fresh: int = 2
    maximum_consecutive_reuses: int = 1

    def __post_init__(self) -> None:
        if not self.configuration_id:
            raise ValueError("SAVR 2.0 configuration ID must be non-empty")
        expected = (
            (set(self.image_thresholds), EXPECTED_CAMERAS, "image"),
            (set(self.state_thresholds), STATE_GROUPS, "state"),
            (set(self.action_thresholds), ACTION_GROUPS, "action"),
        )
        for actual, required, name in expected:
            if actual != required:
                raise ValueError(f"SAVR 2.0 {name} threshold keys differ: {sorted(actual)}")
        for thresholds in (
            self.image_thresholds,
            self.state_thresholds,
            self.action_thresholds,
        ):
            if any(
                not math.isfinite(float(value)) or float(value) < 0
                for value in thresholds.values()
            ):
                raise ValueError("SAVR 2.0 thresholds must be finite and non-negative")
        if not math.isfinite(self.skip_budget) or self.skip_budget not in {0.05, 0.10, 0.15}:
            raise ValueError("SAVR 2.0 skip budget must be one of 0.05, 0.10, or 0.15")
        if self.minimum_query_index < 0:
            raise ValueError("Minimum query index cannot be negative")
        if self.required_stable_fresh < 1:
            raise ValueError("At least one stable fresh query is required")
        if self.maximum_consecutive_reuses != 1:
            raise ValueError("Protocol Version 1 requires isolated reuse")


@dataclass(frozen=True)
class SAVR2State:
    """Auditable controller counters for checkpoint/resume validation."""

    context: CacheContext
    query_index: int
    stable_fresh: int
    completed_reuses: int
    action_history_count: int
    has_image_reference: bool
    has_previous_state: bool


class StateAwareVisualRefresh2Controller(RefreshController):
    """Training-free SAVR 2.0 controller frozen in Phase 6R-B."""

    def __init__(
        self,
        *,
        configuration: SAVR2Configuration,
        state_q01: Sequence[float],
        state_q99: Sequence[float],
        action_q01: Sequence[float],
        action_q99: Sequence[float],
    ) -> None:
        self.configuration = configuration
        self.statistics = SignalStatistics(
            state_q01=tuple(float(value) for value in state_q01),
            state_q99=tuple(float(value) for value in state_q99),
            action_q01=tuple(float(value) for value in action_q01),
            action_q99=tuple(float(value) for value in action_q99),
        )
        self._validate_statistics()
        self._context: CacheContext | None = None
        self._query_index = 0
        self._reference_images: dict[str, tuple[float, ...]] | None = None
        self._previous_state: tuple[float, ...] | None = None
        self._action_history: deque[tuple[float, ...]] = deque(maxlen=2)
        self._stable_fresh = 0
        self._completed_reuses = 0

    def _validate_statistics(self) -> None:
        if len(self.statistics.state_q01) != 8 or len(self.statistics.state_q99) != 8:
            raise ValueError("SAVR 2.0 requires eight-dimensional state statistics")
        if len(self.statistics.action_q01) != 7 or len(self.statistics.action_q99) != 7:
            raise ValueError("SAVR 2.0 requires seven-dimensional action statistics")
        for lows, highs in (
            (self.statistics.state_q01, self.statistics.state_q99),
            (self.statistics.action_q01, self.statistics.action_q99),
        ):
            if any(
                not math.isfinite(low)
                or not math.isfinite(high)
                or high <= low
                for low, high in zip(lows, highs)
            ):
                raise ValueError("Every q99 statistic must finitely exceed q01")

    @property
    def context(self) -> CacheContext | None:
        return self._context

    @property
    def query_index(self) -> int:
        return self._query_index

    def snapshot(self) -> SAVR2State:
        if self._context is None:
            raise RuntimeError("Controller context has not been initialized")
        return SAVR2State(
            context=self._context,
            query_index=self._query_index,
            stable_fresh=self._stable_fresh,
            completed_reuses=self._completed_reuses,
            action_history_count=len(self._action_history),
            has_image_reference=self._reference_images is not None,
            has_previous_state=self._previous_state is not None,
        )

    def reset(self, context: CacheContext) -> None:
        if context.configuration_id != self.configuration.configuration_id:
            raise ValueError("Cache context/configuration identity mismatch")
        self._context = context
        self._query_index = 0
        self._reference_images = None
        self._previous_state = None
        self._action_history.clear()
        self._stable_fresh = 0
        self._completed_reuses = 0

    def _threshold_record(self) -> dict[str, float | int]:
        values: dict[str, float | int] = {
            "skip_budget": self.configuration.skip_budget,
            "minimum_query_index": self.configuration.minimum_query_index,
            "required_stable_fresh": self.configuration.required_stable_fresh,
            "maximum_consecutive_reuses": (
                self.configuration.maximum_consecutive_reuses
            ),
        }
        for name, threshold in self.configuration.image_thresholds.items():
            values[f"image.{name}"] = float(threshold)
        for name, threshold in self.configuration.state_thresholds.items():
            values[f"state.{name}"] = float(threshold)
        for name, threshold in self.configuration.action_thresholds.items():
            values[f"action.{name}"] = float(threshold)
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

        signal_triggers: list[str] = []
        camera_scores: dict[str, float] = {}
        patch_scores: dict[str, tuple[float, ...]] = {}
        global_scores: dict[str, float] = {}
        state_scores: dict[str, float] = {}
        action_scores: dict[str, float] = {}
        gripper_veto = False
        direction_reversals: tuple[bool, ...] = ()

        if self._reference_images is None:
            signal_triggers.append("invalid_image_reference")
        else:
            try:
                if set(images) != EXPECTED_CAMERAS:
                    raise SignalValidationError("SAVR 2.0 requires both frozen cameras")
                image_result = patch_image_change(images, self._reference_images)
                for name, result in image_result.per_camera.items():
                    camera_scores[name] = result.top_k_mean
                    patch_scores[name] = result.patch_scores
                    global_scores[name] = result.global_mean
                    if result.top_k_mean > self.configuration.image_thresholds[name]:
                        signal_triggers.append(f"image_change.{name}")
            except SignalValidationError:
                signal_triggers.append("invalid_image")

        if self._previous_state is None:
            signal_triggers.append("invalid_state_history")
        else:
            try:
                state_result = grouped_state_change(
                    state,
                    self._previous_state,
                    self.statistics.state_q01,
                    self.statistics.state_q99,
                )
                state_scores = state_result.scores
                for name, score in state_scores.items():
                    if score > self.configuration.state_thresholds[name]:
                        signal_triggers.append(f"state_change.{name}")
            except SignalValidationError:
                signal_triggers.append("invalid_state")

        if len(self._action_history) < 2:
            signal_triggers.append("invalid_action_history")
        else:
            try:
                newer, older = self._action_history[-1], self._action_history[-2]
                action_result = grouped_action_change(
                    newer,
                    older,
                    self.statistics.action_q01,
                    self.statistics.action_q99,
                )
                action_scores = action_result.scores
                for name, score in action_scores.items():
                    if score > self.configuration.action_thresholds[name]:
                        signal_triggers.append(f"action_change.{name}")
                transition = action_transition(newer, older)
                gripper_veto = transition.gripper_veto
                direction_reversals = transition.translation_direction_reversals
                if transition.mixed_latest_gripper:
                    signal_triggers.append("gripper_transition.mixed_latest")
                if transition.final_gripper_changed:
                    signal_triggers.append("gripper_transition.final_changed")
            except SignalValidationError:
                signal_triggers.append("invalid_action_history")

        signal_triggers = list(dict.fromkeys(signal_triggers))
        signals_stable = not signal_triggers
        triggers = list(signal_triggers)
        if not cache_available:
            triggers.append("empty_cache")
        if self._query_index < self.configuration.minimum_query_index:
            triggers.append("minimum_query_warmup")
        if self._stable_fresh < self.configuration.required_stable_fresh:
            triggers.append("insufficient_stable_fresh")
        if cache_age >= self.configuration.maximum_consecutive_reuses:
            triggers.append("maximum_consecutive_reuse")
        proposed_fraction = (self._completed_reuses + 1) / (self._query_index + 1)
        if proposed_fraction > self.configuration.skip_budget:
            triggers.append("skip_budget_prefix_cap")

        return RefreshDecision(
            policy=Policy.SAVR2,
            query_index=self._query_index,
            refresh=bool(triggers),
            cache_age_before=cache_age,
            triggers=tuple(dict.fromkeys(triggers)),
            per_camera_image_scores=camera_scores,
            thresholds=self._threshold_record(),
            camera_patch_scores=patch_scores,
            camera_global_scores=global_scores,
            state_group_scores=state_scores,
            action_group_scores=action_scores,
            gripper_transition_veto=gripper_veto,
            translation_direction_reversals=direction_reversals,
            stable_fresh_before=self._stable_fresh,
            completed_reuses_before=self._completed_reuses,
            signals_stable=signals_stable,
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
            raise RuntimeError("Decision/query index does not match SAVR 2.0 state")
        if decision.policy is not Policy.SAVR2:
            raise RuntimeError("Decision policy does not match SAVR 2.0")

        if decision.refresh:
            try:
                if set(images) != EXPECTED_CAMERAS:
                    raise SignalValidationError("SAVR 2.0 requires both frozen cameras")
                self._reference_images = prepare_image_representations(images)
            except SignalValidationError:
                self._reference_images = None
            self._stable_fresh = self._stable_fresh + 1 if decision.signals_stable else 0
        else:
            if not decision.signals_stable:
                raise RuntimeError("SAVR 2.0 cannot reuse with unstable decision signals")
            self._completed_reuses += 1
            self._stable_fresh = 0

        try:
            frozen_state = freeze_numeric(state)
            self._previous_state = frozen_state if len(frozen_state) == 8 else None
        except SignalValidationError:
            self._previous_state = None
        try:
            frozen_action = freeze_numeric(action_chunk)
            if len(frozen_action) != 8 * 7:
                raise SignalValidationError("Action chunk must have shape 8x7")
            self._action_history.append(frozen_action)
        except SignalValidationError:
            self._action_history.clear()
        self._query_index += 1
