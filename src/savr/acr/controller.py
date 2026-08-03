"""Deterministic fail-closed ACR Version 1 controllers."""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from savr.acr.signals import (
    scene_change_from_representations,
    scene_relative_translation,
    transition_signal,
)
from savr.acr.types import (
    REFRESH_REASON_ORDER,
    ACRConfiguration,
    ACRContext,
    ACRPolicy,
    SceneDecision,
)
from savr.signals import SignalValidationError, freeze_numeric


@dataclass(frozen=True)
class ACRControllerState:
    context: ACRContext
    query_index: int
    completed_queries: int
    completed_reuses: int
    reference_query_index: int | None
    has_scene_reference: bool
    has_position_reference: bool
    action_history_count: int


class ACRController:
    """One stateful scene decision stream; the wrist decision is always fresh."""

    def __init__(self, configuration: ACRConfiguration) -> None:
        self.configuration = configuration
        self._context: ACRContext | None = None
        self._query_index = 0
        self._completed_queries = 0
        self._completed_reuses = 0
        self._reference_scene: tuple[float, ...] | None = None
        self._reference_position: tuple[float, float, float] | None = None
        self._reference_query_index: int | None = None
        self._action_history: deque[tuple[float, ...]] = deque(maxlen=2)

    @property
    def context(self) -> ACRContext | None:
        return self._context

    @property
    def query_index(self) -> int:
        return self._query_index

    def reset(self, context: ACRContext) -> None:
        if context.configuration_id != self.configuration.configuration_id:
            raise ValueError("Controller and context configuration identities differ")
        if context.controller_version != self.configuration.controller_version:
            raise ValueError("Controller version differs from the cache context")
        self._context = context
        self._query_index = 0
        self._completed_queries = 0
        self._completed_reuses = 0
        self._reference_scene = None
        self._reference_position = None
        self._reference_query_index = None
        self._action_history.clear()

    def snapshot(self) -> ACRControllerState:
        if self._context is None:
            raise RuntimeError("ACR context has not begun")
        return ACRControllerState(
            context=self._context,
            query_index=self._query_index,
            completed_queries=self._completed_queries,
            completed_reuses=self._completed_reuses,
            reference_query_index=self._reference_query_index,
            has_scene_reference=self._reference_scene is not None,
            has_position_reference=self._reference_position is not None,
            action_history_count=len(self._action_history),
        )

    @staticmethod
    def _finite_tuple(value: Sequence[float] | None, width: int | None = None) -> tuple[float, ...]:
        if value is None:
            raise SignalValidationError("Required signal is missing")
        values = freeze_numeric(value)
        if width is not None and len(values) != width:
            raise SignalValidationError(f"Required signal must contain {width} values")
        if not values or any(not math.isfinite(item) for item in values):
            raise SignalValidationError("Required signal must be finite and non-empty")
        return values

    def decide(
        self,
        *,
        scene_representation: Sequence[float] | None,
        normalized_eef_position: Sequence[float] | None,
        cache_available: bool,
        cache_age: int,
    ) -> SceneDecision:
        if self._context is None:
            raise RuntimeError("reset must be called before decide")
        if cache_age < 0:
            raise ValueError("Cache age cannot be negative")

        reasons: list[str] = []
        scene_score: float | None = None
        translation_score: float | None = None
        gripper_veto: bool | None = None
        reversals = (False, False, False)
        patch_scores: tuple[float, ...] = ()

        if not cache_available:
            reasons.append("cache")

        scene_values: tuple[float, ...] | None = None
        position_values: tuple[float, float, float] | None = None
        signals_valid = True
        try:
            scene_values = self._finite_tuple(scene_representation)
            if self.configuration.policy is ACRPolicy.SA_ACR:
                position = self._finite_tuple(normalized_eef_position, 3)
                position_values = (position[0], position[1], position[2])

            if self._reference_scene is not None:
                change = scene_change_from_representations(
                    scene_values, self._reference_scene
                )
                scene_score = change.top_four_mean
                patch_scores = change.patch_scores
            elif cache_available and self.configuration.policy in {
                ACRPolicy.SA_ACR,
                ACRPolicy.SCENE_VISUAL,
            }:
                raise SignalValidationError("Scene cache lacks its reference image")

            if self.configuration.policy is ACRPolicy.SA_ACR:
                if self._reference_position is not None:
                    assert position_values is not None
                    translation_score = scene_relative_translation(
                        position_values, self._reference_position
                    )
                elif cache_available:
                    raise SignalValidationError("Scene cache lacks its reference position")

                if len(self._action_history) >= 2:
                    transition = transition_signal(
                        self._action_history[-1], self._action_history[-2]
                    )
                    gripper_veto = transition.gripper_veto
                    reversals = transition.translation_direction_reversals
                elif self._query_index >= self.configuration.minimum_query_index:
                    raise SignalValidationError("Required action history is incomplete")
        except (SignalValidationError, TypeError, ValueError, OverflowError):
            signals_valid = False
            reasons.append("invalid-signal")

        if self._query_index < self.configuration.minimum_query_index:
            reasons.append("warm-up")

        if self.configuration.policy is ACRPolicy.FACTORIZED_FR:
            reasons.append("policy")
        elif self.configuration.policy is ACRPolicy.SCENE_PERIODIC:
            assert self.configuration.period is not None
            if self._query_index % self.configuration.period == 0:
                reasons.append("policy")
        else:
            assert self.configuration.scene_threshold is not None
            if scene_score is not None and scene_score > self.configuration.scene_threshold:
                reasons.append("scene-change")
            if self.configuration.policy is ACRPolicy.SA_ACR:
                assert self.configuration.translation_threshold is not None
                if (
                    translation_score is not None
                    and translation_score > self.configuration.translation_threshold
                ):
                    reasons.append("translation")
                if gripper_veto:
                    reasons.append("gripper-transition")
            assert self.configuration.horizon is not None
            if cache_available and cache_age >= self.configuration.horizon:
                reasons.append("horizon")
            assert self.configuration.hard_reuse_cap is not None
            prospective_rate = (self._completed_reuses + 1) / (
                self._completed_queries + 1
            )
            if prospective_rate > self.configuration.hard_reuse_cap:
                reasons.append("hard-cap")

        if not signals_valid and "invalid-signal" not in reasons:
            reasons.insert(1 if reasons and reasons[0] == "cache" else 0, "invalid-signal")
        reason_set = set(reasons)
        reasons = [reason for reason in REFRESH_REASON_ORDER if reason in reason_set]
        return SceneDecision(
            policy=self.configuration.policy,
            query_index=self._query_index,
            refresh=bool(reasons),
            reasons=tuple(reasons),
            cache_age_before=cache_age if cache_available else None,
            scene_score=scene_score,
            translation_score=translation_score,
            gripper_transition_veto=gripper_veto,
            translation_direction_reversals=reversals,
            reference_query_index=self._reference_query_index,
            completed_queries_before=self._completed_queries,
            completed_reuses_before=self._completed_reuses,
            patch_scores=patch_scores,
        )

    def observe(
        self,
        *,
        decision: SceneDecision,
        scene_representation: Sequence[float] | None,
        normalized_eef_position: Sequence[float] | None,
        action_chunk: Any,
    ) -> None:
        if self._context is None:
            raise RuntimeError("reset must be called before observe")
        if decision.query_index != self._query_index:
            raise ValueError("Decision/query sequence is not contiguous")
        if decision.policy is not self.configuration.policy:
            raise ValueError("Decision policy differs from controller policy")

        if decision.refresh:
            try:
                self._reference_scene = self._finite_tuple(scene_representation)
            except SignalValidationError:
                self._reference_scene = None
            if self.configuration.policy is ACRPolicy.SA_ACR:
                try:
                    position = self._finite_tuple(normalized_eef_position, 3)
                    self._reference_position = (position[0], position[1], position[2])
                except SignalValidationError:
                    self._reference_position = None
            self._reference_query_index = self._query_index
        else:
            self._completed_reuses += 1

        try:
            self._action_history.append(freeze_numeric(action_chunk))
        except SignalValidationError:
            self._action_history.clear()
        self._completed_queries += 1
        self._query_index += 1
