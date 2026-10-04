"""Frozen scheduling and Gate-H analysis for CAC Phase C1H."""

from __future__ import annotations

import hashlib
import json
import random
import math
from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence


class C1HValidationError(RuntimeError):
    """Raised when C1H evidence violates its frozen contract."""


def worst_case_query_count(
    suite_max_steps: Mapping[str, int], *, conditions_per_suite_per_stage: int,
    stages: int, arms: int = 2, technical_controls: int = 2,
) -> int:
    """Exact upper bound for eight-action chunks plus frozen technical controls."""

    if conditions_per_suite_per_stage <= 0 or stages not in {1, 2} or arms != 2:
        raise C1HValidationError("invalid C1H query-bound inputs")
    episode_queries = sum(
        math.ceil(int(max_steps) / 8) * conditions_per_suite_per_stage * arms
        for max_steps in suite_max_steps.values()
    )
    return technical_controls + stages * episode_queries


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def semantic_sha256(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("semantic_sha256", None)
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def frozen_schedule(
    conditions: Sequence[Mapping[str, Any]], *, population: str, seed: int
) -> tuple[dict[str, Any], ...]:
    """Create an exactly balanced, paired two-arm schedule without outcomes."""

    selected = [dict(row) for row in conditions if row.get("population") == population]
    if len(selected) != 120:
        raise C1HValidationError(f"{population} must contain exactly 120 conditions")
    identities = {str(row.get("condition_id")) for row in selected}
    if len(identities) != 120:
        raise C1HValidationError(f"{population} condition identities are not unique")
    rng = random.Random(seed)
    rng.shuffle(selected)
    scheduled: list[dict[str, Any]] = []
    for pair_index, condition in enumerate(selected):
        arms = ("dense", "D62") if pair_index % 2 == 0 else ("D62", "dense")
        for pair_position, arm in enumerate(arms):
            payload = {
                "population": population,
                "pair_index": pair_index,
                "pair_position": pair_position,
                "condition_id": condition["condition_id"],
                "suite": condition["suite"],
                "task_index": int(condition["task_index"]),
                "task_id": condition["task_id"],
                "initial_state_id": int(condition["initial_state_id"]),
                "seed": int(condition["seed"]),
                "arm": arm,
            }
            payload["schedule_id"] = semantic_sha256(payload)
            scheduled.append(payload)
    return tuple(scheduled)


def _validate_terminal_records(
    records: Iterable[Mapping[str, Any]], expected_conditions: int
) -> list[Mapping[str, Any]]:
    rows = list(records)
    if len(rows) != expected_conditions * 2:
        raise C1HValidationError("terminal record count is incomplete")
    by_condition: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if row.get("status") != "completed" or not isinstance(row.get("success"), bool):
            raise C1HValidationError("C1H contains a nonterminal or invalid episode")
        arm = str(row.get("arm"))
        if arm not in {"dense", "D62"}:
            raise C1HValidationError("C1H contains an unknown arm")
        by_condition[str(row.get("condition_id"))].add(arm)
    if len(by_condition) != expected_conditions or any(
        arms != {"dense", "D62"} for arms in by_condition.values()
    ):
        raise C1HValidationError("C1H paired-condition coverage is incomplete")
    return rows


def summarize(records: Iterable[Mapping[str, Any]], *, expected_conditions: int) -> dict[str, Any]:
    rows = _validate_terminal_records(records, expected_conditions)
    arm_total = {"dense": 0, "D62": 0}
    arm_success = {"dense": 0, "D62": 0}
    suite_total: dict[str, dict[str, int]] = defaultdict(lambda: {"dense": 0, "D62": 0})
    suite_success: dict[str, dict[str, int]] = defaultdict(lambda: {"dense": 0, "D62": 0})
    for row in rows:
        arm, suite = str(row["arm"]), str(row["suite"])
        arm_total[arm] += 1
        suite_total[suite][arm] += 1
        if row["success"]:
            arm_success[arm] += 1
            suite_success[suite][arm] += 1
    rates = {arm: arm_success[arm] / arm_total[arm] for arm in arm_total}
    suites: dict[str, Any] = {}
    for suite in sorted(suite_total):
        suite_rates = {
            arm: suite_success[suite][arm] / suite_total[suite][arm] for arm in arm_total
        }
        suites[suite] = {
            "counts": {arm: [suite_success[suite][arm], suite_total[suite][arm]] for arm in arm_total},
            "rates": suite_rates,
            "dense_favors": suite_rates["dense"] > suite_rates["D62"],
        }
    gap_points = round(100.0 * (rates["dense"] - rates["D62"]), 12)
    return {
        "condition_count": expected_conditions,
        "episode_count": len(rows),
        "counts": {arm: [arm_success[arm], arm_total[arm]] for arm in arm_total},
        "rates": rates,
        "dense_minus_D62_points": gap_points,
        "suites_favoring_dense": sum(int(value["dense_favors"]) for value in suites.values()),
        "suites": suites,
    }


def stage1_decision(summary: Mapping[str, Any]) -> str:
    dense = float(summary["rates"]["dense"])
    cache = float(summary["rates"]["D62"])
    gap = float(summary["dense_minus_D62_points"])
    suites = int(summary["suites_favoring_dense"])
    if dense >= 0.75 and cache >= 0.50 and 8.0 <= gap <= 35.0 and suites >= 2:
        return "proceed"
    if dense < 0.75 or cache < 0.50 or gap <= 2.0 or gap > 35.0:
        return "stop"
    return "extend"


def cumulative_decision(summary: Mapping[str, Any]) -> str:
    dense = float(summary["rates"]["dense"])
    cache = float(summary["rates"]["D62"])
    gap = float(summary["dense_minus_D62_points"])
    suites = int(summary["suites_favoring_dense"])
    return (
        "proceed"
        if dense >= 0.75 and cache >= 0.50 and 5.0 <= gap <= 35.0 and suites >= 2
        else "stop"
    )
