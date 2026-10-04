#!/usr/bin/env python3
"""Frozen fixed-sequence CAC C7 analyzer.

The analyzer refuses partial populations and evaluates G3 through G6 in the
protocol order. It is published in C0 but must not be run on protected values
before C7 is separately authorized and complete.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


BOOTSTRAP_SEED = 20260907
BOOTSTRAP_REPLICATES = 20_000
ALPHA = 0.05
NONINFERIORITY_MARGIN = 0.02


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def semantic_hash(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def quantile_lower(values: np.ndarray) -> float:
    return float(np.quantile(values, ALPHA, method="lower"))


def validate_episode_population(rows: list[dict[str, Any]]) -> dict[tuple[str, str, int], dict[str, dict[str, Any]]]:
    grouped: dict[tuple[str, str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        suite = str(row["suite"])
        task = str(row["task_id"])
        state = int(row["initial_state_id"])
        policy = str(row["policy"])
        if policy not in {"dense", "d62", "cac"} or state not in range(10, 50):
            raise RuntimeError("unexpected C7 episode identity")
        if suite not in {"libero_10", "libero_goal", "libero_object", "libero_spatial"}:
            raise RuntimeError("unexpected C7 suite identity")
        key = (suite, task, state)
        if policy in grouped[key]:
            raise RuntimeError("duplicate C7 policy condition")
        if not isinstance(row.get("success"), bool):
            raise RuntimeError("terminal success must be boolean")
        grouped[key][policy] = row
    tasks = sorted({key[:2] for key in grouped})
    if len(tasks) != 40 or len(grouped) != 1600:
        raise RuntimeError("C7 requires exactly 40 tasks x 40 states")
    if any(set(arms) != {"dense", "d62", "cac"} for arms in grouped.values()):
        raise RuntimeError("C7 paired arms are incomplete")
    if any({state for suite, task, state in grouped if (suite, task) == name} != set(range(10, 50)) for name in tasks):
        raise RuntimeError("C7 state population is incomplete")
    return grouped


def task_equal_difference(
    grouped: Mapping[tuple[str, str, int], Mapping[str, Mapping[str, Any]]], left: str, right: str
) -> tuple[float, dict[str, float]]:
    by_task: dict[str, list[float]] = defaultdict(list)
    for (suite, task, _), arms in grouped.items():
        by_task[f"{suite}/{task}"].append(float(arms[left]["success"]) - float(arms[right]["success"]))
    estimates = {task: float(np.mean(values)) for task, values in by_task.items()}
    return float(np.mean(list(estimates.values()))), estimates


def hierarchical_bootstrap(
    grouped: Mapping[tuple[str, str, int], Mapping[str, Mapping[str, Any]]], left: str, right: str
) -> np.ndarray:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    tasks = sorted({key[:2] for key in grouped})
    arrays = {
        task: np.asarray([
            float(grouped[(task[0], task[1], state)][left]["success"])
            - float(grouped[(task[0], task[1], state)][right]["success"])
            for state in range(10, 50)
        ])
        for task in tasks
    }
    output = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for index in range(BOOTSTRAP_REPLICATES):
        task_indices = rng.integers(0, len(tasks), size=len(tasks))
        means = []
        for task_index in task_indices:
            values = arrays[tasks[int(task_index)]]
            means.append(float(np.mean(values[rng.integers(0, len(values), size=len(values))])))
        output[index] = float(np.mean(means))
    return output


def latency_gate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_episode: dict[tuple[str, str, int], dict[str, float]] = defaultdict(dict)
    for row in rows:
        key = (str(row["suite"]), str(row["task_id"]), int(row["initial_state_id"]))
        policy = str(row["policy"])
        if policy not in {"dense", "d62", "cac"} or policy in by_episode[key]:
            raise RuntimeError("invalid episode timing record")
        if row.get("d62_semantics_unchanged") is not True or int(row.get("hidden_dense_calls", -1)) != 0:
            raise RuntimeError("timing record does not authenticate unchanged D62 semantics/no fallback")
        value = float(row["complete_cycle_ms"])
        if not np.isfinite(value) or value <= 0:
            raise RuntimeError("invalid complete-cycle timing")
        by_episode[key][policy] = value
    if len(by_episode) != 1600 or any(set(arms) != {"dense", "d62", "cac"} for arms in by_episode.values()):
        raise RuntimeError("C7 requires 1,600 complete paired episode timing summaries")
    tasks = sorted({key[:2] for key in by_episode})
    if len(tasks) != 40:
        raise RuntimeError("C7 timing requires exactly 40 tasks")
    arrays = {
        task: np.asarray([
            1.0 - by_episode[(task[0], task[1], state)]["cac"] / by_episode[(task[0], task[1], state)]["dense"]
            for state in range(10, 50)
        ])
        for task in tasks
    }
    savings = np.concatenate(list(arrays.values()))
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    boot = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for index in range(BOOTSTRAP_REPLICATES):
        task_indices = rng.integers(0, len(tasks), size=len(tasks))
        means = []
        for task_index in task_indices:
            values = arrays[tasks[int(task_index)]]
            means.append(float(np.mean(values[rng.integers(0, len(values), size=len(values))])))
        boot[index] = float(np.mean(means))
    point = float(np.mean(savings))
    lower = quantile_lower(boot)
    return {"point_saving": point, "one_sided_95_lower": lower, "pass": point >= 0.05 and lower > 0.0}


def holm_pass(p_values: Iterable[float], alpha: float = ALPHA) -> list[bool]:
    values = list(float(value) for value in p_values)
    order = sorted(range(len(values)), key=values.__getitem__)
    passed = [False] * len(values)
    for rank_index, original in enumerate(order):
        threshold = alpha / (len(values) - rank_index)
        if values[original] > threshold:
            break
        passed[original] = True
    return passed


def mechanism_gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    controls = payload.get("controls", [])
    ablations = payload.get("ablations", [])
    suites = payload.get("suite_directions", {})
    if len(controls) != 2 or len(ablations) != 4 or set(suites) != {"libero_10", "libero_goal", "libero_object", "libero_spatial"}:
        raise RuntimeError("G6 mechanism summary is incomplete")
    control_holm = holm_pass([row["one_sided_p"] for row in controls])
    ablation_holm = holm_pass([row["one_sided_p"] for row in ablations])
    pass_controls = all(float(row["relative_full_advantage"]) >= 0.10 and control_holm[i] for i, row in enumerate(controls))
    pass_ablations = sum(float(row["relative_full_advantage"]) >= 0.05 and ablation_holm[i] for i, row in enumerate(ablations)) >= 3
    no_better_ablation = all(float(row["relative_full_advantage"]) >= -0.05 for row in ablations)
    suite_direction = all(float(value) > 0 for value in suites.values())
    return {
        "controls_holm_pass": control_holm,
        "ablations_holm_pass": ablation_holm,
        "pass": pass_controls and pass_ablations and no_better_ablation and suite_direction,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--timing", type=Path, required=True)
    parser.add_argument("--mechanism", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError("immutable analysis output already exists")

    grouped = validate_episode_population(read_jsonl(args.episodes))
    cac_dense, per_task_dense = task_equal_difference(grouped, "cac", "dense")
    cac_d62, per_task_d62 = task_equal_difference(grouped, "cac", "d62")
    dense_boot = hierarchical_bootstrap(grouped, "cac", "dense")
    d62_boot = hierarchical_bootstrap(grouped, "cac", "d62")
    per_suite_dense = {
        suite: float(np.mean([value for key, value in per_task_dense.items() if key.startswith(f"{suite}/")]))
        for suite in ("libero_10", "libero_goal", "libero_object", "libero_spatial")
    }
    zero_task_failures = []
    for (suite, task), states in sorted({(key[0], key[1]): range(10, 50) for key in grouped}.items()):
        cac_success = sum(grouped[(suite, task, state)]["cac"]["success"] for state in states)
        dense_success = sum(grouped[(suite, task, state)]["dense"]["success"] for state in states)
        if cac_success == 0 and dense_success >= 20:
            zero_task_failures.append(f"{suite}/{task}")
    g3 = {
        "point": cac_dense,
        "one_sided_95_lower": quantile_lower(dense_boot),
        "per_suite": per_suite_dense,
        "zero_cac_when_dense_at_least_half": zero_task_failures,
        "pass": (
            cac_dense >= -NONINFERIORITY_MARGIN
            and quantile_lower(dense_boot) > -NONINFERIORITY_MARGIN
            and all(value >= -0.05 for value in per_suite_dense.values())
            and not zero_task_failures
        ),
        "per_task": per_task_dense,
    }
    g4 = {
        "point": cac_d62,
        "one_sided_95_lower": quantile_lower(d62_boot),
        "pass": cac_d62 >= 0.05 and quantile_lower(d62_boot) > 0.0,
        "per_task": per_task_d62,
    }
    g5 = latency_gate(read_jsonl(args.timing))
    g6 = mechanism_gate(json.loads(args.mechanism.read_text(encoding="utf-8")))
    sequence = [g3["pass"], g4["pass"], g5["pass"], g6["pass"]]
    reached = []
    for name, passed in zip(("G3", "G4", "G5", "G6"), sequence):
        reached.append(name)
        if not passed:
            break
    classification = (
        "positive_cac" if all(sequence) else
        "reliability_only" if g3["pass"] and g4["pass"] and not g5["pass"] else
        "mechanism_ambiguous" if g3["pass"] and g4["pass"] and g5["pass"] and not g6["pass"] else
        "efficiency_only_negative" if g5["pass"] and (not g3["pass"] or not g4["pass"]) else
        "negative"
    )
    result = {
        "schema_version": "cac-c7-analysis-v1",
        "fixed_sequence_reached": reached,
        "gates": {"G3": g3, "G4": g4, "G5": g5, "G6": g6},
        "classification": classification,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }
    result["semantic_sha256"] = semantic_hash(result)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
