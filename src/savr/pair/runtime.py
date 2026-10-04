"""Fail-closed PAIR contract lifecycle and calibrated route decisions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from savr.pair.calibration import EmpiricalCalibrator
from savr.pair.features import RouterFeatures
from savr.pair.router import PairRouter
from savr.pair.types import HORIZONS, PairValidationError, ProfileSpec


class RuntimeMode(str, Enum):
    DENSE = "dense"
    CONTRACT = "contract"


@dataclass(frozen=True)
class RouteDecision:
    execute_dense: bool
    profile_id: str | None
    calibrated_risk: float | None
    fallback_reason: str | None


@dataclass(frozen=True)
class PairRuntime:
    episode_id: str
    mode: RuntimeMode = RuntimeMode.DENSE
    last_query: int = -1
    profile_id: str | None = None
    horizon: int = 0
    remaining: int = 0
    abort_history: tuple[str, ...] = ()

    @classmethod
    def reset(cls, episode_id: str) -> "PairRuntime":
        if not episode_id:
            raise PairValidationError("runtime reset requires an episode identity")
        return cls(episode_id=episode_id)

    def start(self, *, query: int, profile: ProfileSpec, horizon: int) -> "PairRuntime":
        profile.validate()
        if self.mode is not RuntimeMode.DENSE or query != self.last_query + 1:
            raise PairValidationError("contract must start at the next dense query")
        if horizon not in HORIZONS:
            raise PairValidationError("contract horizon is unsupported")
        return PairRuntime(
            episode_id=self.episode_id,
            mode=RuntimeMode.CONTRACT,
            last_query=query,
            profile_id=profile.profile_id,
            horizon=horizon,
            remaining=horizon,
            abort_history=self.abort_history,
        )

    def advance(self, *, query: int) -> "PairRuntime":
        if self.mode is not RuntimeMode.CONTRACT or query != self.last_query + 1:
            raise PairValidationError("contract queries must be active and contiguous")
        remaining = self.remaining - 1
        if remaining <= 0:
            return PairRuntime(
                episode_id=self.episode_id,
                mode=RuntimeMode.DENSE,
                last_query=query,
                abort_history=self.abort_history,
            )
        return PairRuntime(
            episode_id=self.episode_id,
            mode=self.mode,
            last_query=query,
            profile_id=self.profile_id,
            horizon=self.horizon,
            remaining=remaining,
            abort_history=self.abort_history,
        )

    def abort(self, *, query: int, reason: str) -> "PairRuntime":
        if query <= self.last_query or not reason:
            raise PairValidationError("contract abort identity is invalid")
        return PairRuntime(
            episode_id=self.episode_id,
            mode=RuntimeMode.DENSE,
            last_query=query,
            abort_history=(*self.abort_history, reason),
        )


def calibrated_route(
    router: PairRouter,
    calibrator: EmpiricalCalibrator,
    features: RouterFeatures,
    *,
    delta: float,
    gripper_veto: bool = False,
) -> RouteDecision:
    """Return dense fallback for every unsupported or invalid route state."""

    if gripper_veto:
        return RouteDecision(True, None, None, "gripper_transition_veto")
    if not np.isfinite(delta) or delta < 0:
        return RouteDecision(True, None, None, "invalid_risk_threshold")
    try:
        output = router.predict(features)
        bound = calibrator.upper_bound(output.positive_regret_q90, features)
    except (PairValidationError, ValueError, IndexError):
        return RouteDecision(True, None, None, "invalid_or_out_of_support")
    if bound > delta:
        return RouteDecision(True, None, bound, "calibrated_risk_exceeded")
    return RouteDecision(False, features.groups[0].profile_id, bound, None)
