"""Fail-closed isolated-reuse controller for ACR Version 5."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from savr.acr.controller import ACRController, ACRControllerState
from savr.acr.types import ACRConfiguration, ACRContext, ACRPolicy, SceneDecision


ISOLATED_CONTROLLER_VERSION = "acr-isolated-controller-v1"


@dataclass(frozen=True)
class IsolatedACRControllerState(ACRControllerState):
    """Auditable Version 5 state in addition to the legacy controller state."""

    refresh_required_after_reuse: bool
    completed_refreshes: int


class IsolatedACRController(ACRController):
    """State-aware ACR with a mandatory completed refresh after every reuse."""

    def __init__(self, configuration: ACRConfiguration) -> None:
        if configuration.policy is not ACRPolicy.SA_ACR:
            raise ValueError("IR-SA-ACR requires the state-aware ACR policy")
        if configuration.controller_version != ISOLATED_CONTROLLER_VERSION:
            raise ValueError("IR-SA-ACR requires its isolated controller identity")
        if configuration.horizon != 1:
            raise ValueError("IR-SA-ACR requires horizon 1 as defense in depth")
        super().__init__(configuration)
        self._refresh_required_after_reuse = False
        self._completed_refreshes = 0

    @property
    def refresh_required_after_reuse(self) -> bool:
        return self._refresh_required_after_reuse

    def reset(self, context: ACRContext) -> None:
        super().reset(context)
        self._refresh_required_after_reuse = False
        self._completed_refreshes = 0

    def snapshot(self) -> IsolatedACRControllerState:
        base = super().snapshot()
        return IsolatedACRControllerState(
            context=base.context,
            query_index=base.query_index,
            completed_queries=base.completed_queries,
            completed_reuses=base.completed_reuses,
            reference_query_index=base.reference_query_index,
            has_scene_reference=base.has_scene_reference,
            has_position_reference=base.has_position_reference,
            action_history_count=base.action_history_count,
            refresh_required_after_reuse=self._refresh_required_after_reuse,
            completed_refreshes=self._completed_refreshes,
        )

    def decide(
        self,
        *,
        scene_representation: Any,
        normalized_eef_position: Any,
        cache_available: bool,
        cache_age: int,
    ) -> SceneDecision:
        decision = super().decide(
            scene_representation=scene_representation,
            normalized_eef_position=normalized_eef_position,
            cache_available=cache_available,
            cache_age=cache_age,
        )
        if self._refresh_required_after_reuse:
            decision = decision.force_refresh("post-reuse-refresh")

        if cache_available:
            expected_age = 1 if self._refresh_required_after_reuse else 0
            if cache_age != expected_age:
                decision = decision.force_refresh("isolation-state-mismatch")
        return decision

    def observe(
        self,
        *,
        decision: SceneDecision,
        scene_representation: Any,
        normalized_eef_position: Any,
        action_chunk: Any,
    ) -> None:
        if self._refresh_required_after_reuse and not decision.refresh:
            raise RuntimeError("IR-SA-ACR cannot complete consecutive reuse decisions")
        super().observe(
            decision=decision,
            scene_representation=scene_representation,
            normalized_eef_position=normalized_eef_position,
            action_chunk=action_chunk,
        )
        if decision.refresh:
            self._refresh_required_after_reuse = False
            self._completed_refreshes += 1
        else:
            self._refresh_required_after_reuse = True
