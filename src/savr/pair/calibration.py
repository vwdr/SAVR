"""Empirical upper-tail calibration and fail-closed support checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from savr.pair.features import RouterFeatures
from savr.pair.types import PairValidationError


def conservative_quantile(values: Sequence[float], probability: float) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not array.size or not np.isfinite(array).all():
        raise PairValidationError("calibration values must be finite and nonempty")
    if not 0 <= probability <= 1:
        raise PairValidationError("calibration probability is invalid")
    ordered = np.sort(array)
    index = max(0, int(np.ceil(probability * len(ordered))) - 1)
    return float(ordered[index])


@dataclass(frozen=True)
class SupportEnvelope:
    group_minimum: tuple[float, ...]
    group_maximum: tuple[float, ...]
    global_minimum: tuple[float, ...]
    global_maximum: tuple[float, ...]
    profiles: tuple[str, ...]

    @classmethod
    def fit(cls, examples: Iterable[RouterFeatures]) -> "SupportEnvelope":
        rows = tuple(examples)
        if not rows:
            raise PairValidationError("support fit requires examples")
        group_values = []
        global_values = []
        profiles: set[str] = set()
        for example in rows:
            example.validate()
            group_values.extend(item.continuous for item in example.groups)
            global_values.append(example.global_continuous)
            profiles.update(item.profile_id for item in example.groups)
        group = np.asarray(group_values, dtype=np.float64)
        global_array = np.asarray(global_values, dtype=np.float64)
        return cls(
            tuple(group.min(axis=0)),
            tuple(group.max(axis=0)),
            tuple(global_array.min(axis=0)),
            tuple(global_array.max(axis=0)),
            tuple(sorted(profiles)),
        )

    def contains(self, features: RouterFeatures, *, tolerance: float = 1e-12) -> bool:
        try:
            features.validate(supported_profiles=self.profiles)
        except PairValidationError:
            return False
        group = np.asarray([item.continuous for item in features.groups])
        global_array = np.asarray(features.global_continuous)
        return bool(
            np.all(group >= np.asarray(self.group_minimum) - tolerance)
            and np.all(group <= np.asarray(self.group_maximum) + tolerance)
            and np.all(global_array >= np.asarray(self.global_minimum) - tolerance)
            and np.all(global_array <= np.asarray(self.global_maximum) + tolerance)
        )


@dataclass(frozen=True)
class EmpiricalCalibrator:
    probability: float
    residual_correction: float
    empirical_coverage: float
    support: SupportEnvelope

    @classmethod
    def fit(
        cls,
        predicted_q90: Sequence[float],
        observed_positive_regret: Sequence[float],
        support_examples: Iterable[RouterFeatures],
        *,
        probability: float = 0.9,
    ) -> "EmpiricalCalibrator":
        predicted = np.asarray(predicted_q90, dtype=np.float64)
        observed = np.asarray(observed_positive_regret, dtype=np.float64)
        if predicted.shape != observed.shape or predicted.ndim != 1 or not predicted.size:
            raise PairValidationError("calibration prediction/label arrays are misaligned")
        if (
            not np.isfinite(predicted).all()
            or not np.isfinite(observed).all()
            or np.any(observed < 0)
        ):
            raise PairValidationError("calibration arrays contain invalid risk")
        correction = conservative_quantile(observed - predicted, probability)
        correction = max(0.0, correction)
        coverage = float(np.mean(observed <= predicted + correction))
        return cls(probability, correction, coverage, SupportEnvelope.fit(support_examples))

    def upper_bound(self, predicted_q90: float, features: RouterFeatures) -> float:
        if not np.isfinite(predicted_q90) or predicted_q90 < 0:
            raise PairValidationError("predicted q90 risk is invalid")
        if not self.support.contains(features):
            raise PairValidationError("router input lies outside calibration support")
        return float(predicted_q90 + self.residual_correction)


@dataclass(frozen=True)
class CandidateRisk:
    profile_id: str
    saving_fraction: float
    predicted_q90: float
    features: RouterFeatures


def select_maximum_saving(
    candidates: Iterable[CandidateRisk], calibrator: EmpiricalCalibrator, *, delta: float
) -> CandidateRisk | None:
    if not np.isfinite(delta) or delta < 0:
        raise PairValidationError("risk threshold is invalid")
    accepted = []
    for candidate in candidates:
        if not 0 <= candidate.saving_fraction <= 1:
            raise PairValidationError("candidate saving is invalid")
        try:
            bound = calibrator.upper_bound(candidate.predicted_q90, candidate.features)
        except PairValidationError:
            continue
        if bound <= delta:
            accepted.append(candidate)
    return max(accepted, key=lambda item: (item.saving_fraction, item.profile_id), default=None)
