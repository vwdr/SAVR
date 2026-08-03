"""Dependency-free statistical procedures frozen by the ACR protocol."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from statistics import NormalDist


@dataclass(frozen=True)
class Interval:
    lower: float
    upper: float


@dataclass(frozen=True)
class PairedCounts:
    both_success: int
    candidate_only: int
    fr_only: int
    both_failure: int

    @property
    def total(self) -> int:
        return self.both_success + self.candidate_only + self.fr_only + self.both_failure

    @property
    def risk_difference(self) -> float:
        if self.total == 0:
            raise ValueError("Paired risk difference requires observations")
        return (self.candidate_only - self.fr_only) / self.total


@dataclass(frozen=True)
class PairedObservation:
    stratum: str
    candidate: float
    reference: float


def paired_counts(candidate: Sequence[bool], fr: Sequence[bool]) -> PairedCounts:
    if len(candidate) != len(fr) or not candidate:
        raise ValueError("Paired outcomes require equal non-empty sequences")
    cells = [0, 0, 0, 0]
    for left, right in zip(candidate, fr):
        if left and right:
            cells[0] += 1
        elif left:
            cells[1] += 1
        elif right:
            cells[2] += 1
        else:
            cells[3] += 1
    return PairedCounts(*cells)


def wilson_interval(successes: int, total: int, confidence: float = 0.95) -> Interval:
    if total <= 0 or not 0 <= successes <= total or not 0 < confidence < 1:
        raise ValueError("Invalid Wilson interval inputs")
    z = NormalDist().inv_cdf(1 - (1 - confidence) / 2)
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    radius = z / denominator * math.sqrt(
        proportion * (1 - proportion) / total + z * z / (4 * total * total)
    )
    lower = 0.0 if successes == 0 else max(0.0, center - radius)
    upper = 1.0 if successes == total else min(1.0, center + radius)
    return Interval(lower, upper)


def newcombe_paired_interval(
    counts: PairedCounts, confidence: float = 0.95
) -> Interval:
    """Newcombe (1998) method 10 score interval, candidate minus FR."""

    n = counts.total
    if n <= 0:
        raise ValueError("Newcombe interval requires paired observations")
    candidate_success = counts.both_success + counts.candidate_only
    fr_success = counts.both_success + counts.fr_only
    candidate_interval = wilson_interval(candidate_success, n, confidence)
    fr_interval = wilson_interval(fr_success, n, confidence)
    p_candidate = candidate_success / n
    p_fr = fr_success / n
    theta = p_candidate - p_fr
    denominator = math.sqrt(
        candidate_success
        * (n - candidate_success)
        * fr_success
        * (n - fr_success)
    )
    cross = counts.both_success * counts.both_failure - counts.candidate_only * counts.fr_only
    if denominator == 0:
        phi = 0.0
    else:
        corrected = max(cross - n / 2, 0.0) if cross > 0 else cross
        phi = corrected / denominator
    candidate_down = p_candidate - candidate_interval.lower
    candidate_up = candidate_interval.upper - p_candidate
    fr_down = p_fr - fr_interval.lower
    fr_up = fr_interval.upper - p_fr
    lower_distance = math.sqrt(
        max(0.0, candidate_down**2 - 2 * phi * candidate_down * fr_up + fr_up**2)
    )
    upper_distance = math.sqrt(
        max(0.0, candidate_up**2 - 2 * phi * candidate_up * fr_down + fr_down**2)
    )
    return Interval(max(-1.0, theta - lower_distance), min(1.0, theta + upper_distance))


def exact_mcnemar_pvalue(counts: PairedCounts) -> float:
    discordant = counts.candidate_only + counts.fr_only
    if discordant == 0:
        return 1.0
    smaller = min(counts.candidate_only, counts.fr_only)
    tail = sum(math.comb(discordant, value) for value in range(smaller + 1)) / 2**discordant
    return min(1.0, 2 * tail)


def stratified_paired_bootstrap(
    observations: Sequence[PairedObservation],
    *,
    resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Interval:
    if not observations or resamples < 1 or not 0 < confidence < 1:
        raise ValueError("Invalid paired bootstrap inputs")
    strata: dict[str, list[float]] = defaultdict(list)
    for observation in observations:
        strata[observation.stratum].append(observation.candidate - observation.reference)
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(resamples):
        total = 0.0
        count = 0
        for name in sorted(strata):
            values = strata[name]
            for _ in values:
                total += values[rng.randrange(len(values))]
                count += 1
        estimates.append(total / count)
    estimates.sort()
    alpha = (1 - confidence) / 2

    def percentile(probability: float) -> float:
        rank = probability * (len(estimates) - 1)
        lower = math.floor(rank)
        upper = math.ceil(rank)
        fraction = rank - lower
        return estimates[lower] * (1 - fraction) + estimates[upper] * fraction

    return Interval(percentile(alpha), percentile(1 - alpha))


def holm_adjust(pvalues: Iterable[float]) -> tuple[float, ...]:
    values = tuple(float(value) for value in pvalues)
    if any(not 0 <= value <= 1 for value in values):
        raise ValueError("P-values must lie in [0,1]")
    ordered = sorted(enumerate(values), key=lambda item: (item[1], item[0]))
    adjusted = [0.0] * len(values)
    running = 0.0
    count = len(values)
    for rank, (index, value) in enumerate(ordered):
        running = max(running, (count - rank) * value)
        adjusted[index] = min(1.0, running)
    return tuple(adjusted)


def planned_sample_size(
    discordances: int,
    total: int,
    *,
    power: float = 0.90,
    one_sided_alpha: float = 0.025,
    margin: float = 0.02,
    maximum: int = 1600,
    strata: int = 40,
) -> int | None:
    if total <= 0 or not 0 <= discordances <= total:
        raise ValueError("Invalid discordance inputs")
    if (
        not 0 < power < 1
        or not 0 < one_sided_alpha < 1
        or margin <= 0
        or maximum < 1
        or strata < 1
    ):
        raise ValueError("Invalid frozen sample-size design inputs")
    observed = discordances / total
    upper = wilson_interval(discordances, total, 0.95).upper
    planning_rate = max(observed, upper, 0.01)
    z_alpha = NormalDist().inv_cdf(1 - one_sided_alpha)
    z_power = NormalDist().inv_cdf(power)
    raw = math.ceil(planning_rate * (z_alpha + z_power) ** 2 / margin**2)
    rounded = math.ceil(raw / strata) * strata
    return rounded if rounded <= maximum else None
