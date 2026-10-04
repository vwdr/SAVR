#!/usr/bin/env python3
"""Outcome-blind C0 power forecast for the frozen hierarchical C7 test."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports/cac_c0_power_simulation_v1.json"
SEED = 20260908
OUTER_REPLICATES = 100
BOOTSTRAP_REPLICATES = 20_000
TASKS = 40
STATES = 40
MARGIN = 0.02


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return center - half, center + half


def bootstrap_lower(matrix: np.ndarray, rng: np.random.Generator) -> float:
    # For each original task, draw the 20,000 within-task state-bootstrap means.
    state_means = np.empty((TASKS, BOOTSTRAP_REPLICATES), dtype=np.float32)
    for task in range(TASKS):
        counts = np.bincount(matrix[task].astype(np.int8) + 1, minlength=3)
        probabilities = counts / STATES
        resampled = rng.multinomial(STATES, probabilities, size=BOOTSTRAP_REPLICATES)
        state_means[task] = (resampled[:, 2] - resampled[:, 0]) / STATES
    task_indices = rng.integers(0, TASKS, size=(BOOTSTRAP_REPLICATES, TASKS))
    selected = state_means.T[np.arange(BOOTSTRAP_REPLICATES)[:, None], task_indices]
    estimates = selected.mean(axis=1)
    return float(np.quantile(estimates, 0.05, method="lower"))


def simulate_scenario(discordance: float, difference: float, rng: np.random.Generator) -> dict[str, object]:
    # Jeffreys-smoothed historical pattern: 9 tasks at 10/10 and one at 9/10,
    # repeated for four suites. Failure propensity allocates task heterogeneity;
    # the simulated variable is the paired CAC-minus-dense outcome itself.
    dense_rates = np.asarray(([10.5 / 11] * 9 + [9.5 / 11]) * 4, dtype=np.float64)
    multipliers = (1.0 - dense_rates) / float(np.mean(1.0 - dense_rates))
    task_discordance = discordance * multipliers
    task_difference = difference * multipliers
    if np.any(task_discordance > 1) or np.any(np.abs(task_difference) > task_discordance):
        raise RuntimeError("invalid hierarchical planning scenario")

    passes = 0
    observed_differences = []
    observed_lowers = []
    for _ in range(OUTER_REPLICATES):
        matrix = np.empty((TASKS, STATES), dtype=np.int8)
        for task in range(TASKS):
            positive = (task_discordance[task] + task_difference[task]) / 2
            negative = (task_discordance[task] - task_difference[task]) / 2
            matrix[task] = rng.choice(
                np.asarray([-1, 0, 1], dtype=np.int8),
                size=STATES,
                p=[negative, 1.0 - task_discordance[task], positive],
            )
        point = float(matrix.mean())
        lower = bootstrap_lower(matrix, rng)
        observed_differences.append(point)
        observed_lowers.append(lower)
        passes += int(point >= -MARGIN and lower > -MARGIN)
    low, high = wilson(passes, OUTER_REPLICATES)
    return {
        "discordance": discordance,
        "true_cac_minus_dense": difference,
        "passes": passes,
        "outer_replicates": OUTER_REPLICATES,
        "estimated_power": passes / OUTER_REPLICATES,
        "power_wilson_95": [low, high],
        "mean_observed_difference": float(np.mean(observed_differences)),
        "mean_bootstrap_lower": float(np.mean(observed_lowers)),
        "task_discordance_range": [float(task_discordance.min()), float(task_discordance.max())],
    }


def main() -> None:
    if OUTPUT.exists():
        raise RuntimeError("immutable CAC C0 power simulation already exists")
    rng = np.random.default_rng(SEED)
    rows = [
        simulate_scenario(discordance, difference, rng)
        for discordance in (0.05, 0.10, 0.15, 0.20)
        for difference in (0.0, -0.01, -0.02)
    ]
    payload = {
        "schema_version": "cac-c0-hierarchical-power-v1",
        "seed": SEED,
        "outer_replicates": OUTER_REPLICATES,
        "bootstrap_replicates_per_outer_dataset": BOOTSTRAP_REPLICATES,
        "tasks": TASKS,
        "states_per_task": STATES,
        "test": "C7 task-then-state paired bootstrap; point >= -0.02 and one-sided 95% lower > -0.02",
        "historical_pattern": "Jeffreys-smoothed 10/10 for tasks 0-8 and 9/10 for task 9, repeated over four suites",
        "simulation_scope": "paired-difference power forecast; not a simulator or policy-outcome forecast",
        "rows": rows,
        "limitation": "100 outer datasets give coarse power resolution; Wilson intervals are reported and the lower of analytic and simulated evidence governs planning",
    }
    payload["semantic_sha256"] = hashlib.sha256(canonical(payload)).hexdigest()
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
