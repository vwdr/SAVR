"""Paired binary-outcome and power calculations for SAVR."""

from __future__ import annotations

import math
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from statistics import NormalDist


DEFAULT_DISCORDANCE_GRID = tuple(value / 100 for value in range(1, 11))


@dataclass(frozen=True)
class PairedBinaryCounts:
    both_success: int
    fr_only_success: int
    candidate_only_success: int
    both_failure: int

    @property
    def total(self) -> int:
        return (
            self.both_success
            + self.fr_only_success
            + self.candidate_only_success
            + self.both_failure
        )

    @property
    def success_difference(self) -> float:
        """Candidate minus FR success-rate difference."""

        if not self.total:
            raise ValueError("Paired counts are empty")
        return (self.candidate_only_success - self.fr_only_success) / self.total

    @property
    def discordance_rate(self) -> float:
        if not self.total:
            raise ValueError("Paired counts are empty")
        return (
            self.fr_only_success + self.candidate_only_success
        ) / self.total


def paired_binary_counts(
    fr: Mapping[Hashable, bool],
    candidate: Mapping[Hashable, bool],
) -> PairedBinaryCounts:
    if not fr or set(fr) != set(candidate):
        raise ValueError("Paired outcome keys must be equal and non-empty")
    both_success = fr_only = candidate_only = both_failure = 0
    for key in fr:
        if fr[key] and candidate[key]:
            both_success += 1
        elif fr[key]:
            fr_only += 1
        elif candidate[key]:
            candidate_only += 1
        else:
            both_failure += 1
    return PairedBinaryCounts(
        both_success=both_success,
        fr_only_success=fr_only,
        candidate_only_success=candidate_only,
        both_failure=both_failure,
    )


def wilson_upper(successes: int, total: int, *, confidence: float = 0.95) -> float:
    """Two-sided Wilson-score upper limit for a binomial proportion."""

    if total < 1 or not 0 <= successes <= total:
        raise ValueError("Wilson inputs are invalid")
    if not 0.0 < confidence < 1.0:
        raise ValueError("Confidence must lie in (0,1)")
    z = NormalDist().inv_cdf(1.0 - (1.0 - confidence) / 2.0)
    proportion = successes / total
    z2 = z * z
    denominator = 1.0 + z2 / total
    center = proportion + z2 / (2.0 * total)
    radius = z * math.sqrt(
        proportion * (1.0 - proportion) / total + z2 / (4.0 * total * total)
    )
    return (center + radius) / denominator


def paired_noninferiority_sample_size(
    discordance_rate: float,
    *,
    margin: float = 0.02,
    alpha: float = 0.025,
    power: float = 0.90,
    planning_difference: float = 0.0,
) -> int:
    """Normal-approximation size for paired risk-difference non-inferiority."""

    if not 0.0 < discordance_rate <= 1.0:
        raise ValueError("Discordance rate must lie in (0,1]")
    if not 0.0 < margin < 1.0:
        raise ValueError("Margin must lie in (0,1)")
    if not 0.0 < alpha < 0.5 or not 0.5 < power < 1.0:
        raise ValueError("Alpha or power is invalid")
    distance = margin + planning_difference
    if distance <= 0:
        raise ValueError("Planning difference must remain above the NI boundary")
    z_alpha = NormalDist().inv_cdf(1.0 - alpha)
    z_power = NormalDist().inv_cdf(power)
    return math.ceil(
        discordance_rate * (z_alpha + z_power) ** 2 / (distance * distance)
    )


def planning_power_result(
    counts: PairedBinaryCounts,
    *,
    margin: float = 0.02,
    alpha: float = 0.025,
    power: float = 0.90,
    discordance_floor: float = 0.01,
    parent_sample_size: int = 1200,
    balanced_block: int = 400,
) -> dict[str, float | int]:
    if counts.total < 1:
        raise ValueError("Paired counts are empty")
    discordant = counts.fr_only_success + counts.candidate_only_success
    observed = discordant / counts.total
    upper = wilson_upper(discordant, counts.total)
    planning = max(observed, upper, discordance_floor)
    required = paired_noninferiority_sample_size(
        planning,
        margin=margin,
        alpha=alpha,
        power=power,
    )
    selected = max(parent_sample_size, required)
    rounded = math.ceil(selected / balanced_block) * balanced_block
    return {
        "paired_episodes": counts.total,
        "observed_discordance_rate": observed,
        "wilson_95_upper_discordance_rate": upper,
        "planning_discordance_rate": planning,
        "margin": margin,
        "alpha_one_sided": alpha,
        "target_power": power,
        "required_unrounded": required,
        "parent_planned_sample_size": parent_sample_size,
        "balanced_block": balanced_block,
        "recommended_sample_size": rounded,
    }


def power_sensitivity(
    discordance_rates: Sequence[float] = DEFAULT_DISCORDANCE_GRID,
    *,
    powers: Sequence[float] = (0.80, 0.90),
    margin: float = 0.02,
    alpha: float = 0.025,
) -> tuple[dict[str, float | int], ...]:
    rows = []
    for discordance in discordance_rates:
        for power in powers:
            rows.append(
                {
                    "discordance_rate": discordance,
                    "power": power,
                    "required_sample_size": paired_noninferiority_sample_size(
                        discordance,
                        margin=margin,
                        alpha=alpha,
                        power=power,
                    ),
                }
            )
    return tuple(rows)
