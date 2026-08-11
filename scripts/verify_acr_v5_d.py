#!/usr/bin/env python3
"""Independently recompute the frozen V5-D analysis and every gate."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import random
import statistics
import sys
from pathlib import Path
from typing import Any


EXPECTED_ROOT = Path("/home/ved/SAVR")


def encoded(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(record: dict[str, Any]) -> str:
    payload = dict(record)
    payload.pop("semantic_sha256", None)
    return hashlib.sha256(encoded(payload)).hexdigest()


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    coordinate = probability * (len(ordered) - 1)
    left = int(coordinate)
    right = min(left + 1, len(ordered) - 1)
    weight = coordinate - left
    return ordered[left] + (ordered[right] - ordered[left]) * weight


def verify(config: dict[str, Any], run: dict[str, Any], analysis: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if config.get("semantic_sha256") != digest(config):
        errors.append("configuration semantic hash mismatch")
    if run.get("semantic_sha256") != digest(run):
        errors.append("run semantic hash mismatch")
    if analysis.get("semantic_sha256") != digest(analysis):
        errors.append("analysis semantic hash mismatch")
    if analysis.get("schema_version") != "acr.v5d-analysis.v1":
        errors.append("analysis schema mismatch")
    if run.get("run_id") != config.get("run_id") or analysis.get("run_id") != config.get("run_id"):
        errors.append("run identity mismatch")

    paths = tuple(config["timing"]["paths"])
    expected_permutations = [list(order) for order in itertools.permutations(paths)]
    if config["timing"]["permutations"] != expected_permutations:
        errors.append("frozen permutations mismatch")
    expected_labels = list(config["correctness"]["labels"])
    for path in paths:
        for repetition in range(2):
            expected_labels.append(f"warmup-{path}-{repetition:02d}")
    for block, order in enumerate(expected_permutations):
        for position, path in enumerate(order):
            expected_labels.append(f"timed-{block:02d}-{position}-{path}")
    queries = run.get("queries", [])
    if [item.get("label") for item in queries] != expected_labels:
        errors.append("query schedule mismatch")
        return errors
    if [item.get("query_index") for item in queries] != list(range(111)):
        errors.append("query index mismatch")
    correctness = queries[:7]
    if any(
        item.get("kind") != "correctness"
        or item.get("passed") is not True
        or not item.get("checks")
        or not all(item["checks"].values())
        for item in correctness
    ):
        errors.append("correctness matrix mismatch")

    timed = queries[15:]
    if len(timed) != 96 or any(item.get("kind") != "timed" for item in timed):
        errors.append("timed population mismatch")
        return errors
    block_map: dict[int, dict[str, dict[str, Any]]] = {}
    for record in timed:
        block = int(record["block"])
        position = int(record["position"])
        path = record["path"]
        if expected_permutations[block][position] != path:
            errors.append("timed permutation mismatch")
        if record.get("input_label") != ("input-a" if block % 2 == 0 else "input-b"):
            errors.append("timed input alternation mismatch")
        block_map.setdefault(block, {})[path] = record
    if set(block_map) != set(range(24)) or any(
        set(value) != set(paths) for value in block_map.values()
    ):
        errors.append("timed block completeness mismatch")
        return errors

    def median(path: str, metric: str, selected: list[int] | None = None) -> float:
        if selected is None:
            values = [float(record["timing"][metric]) for record in timed if record["path"] == path]
        else:
            values = [float(block_map[index][path]["timing"][metric]) for index in selected]
        if any(not math.isfinite(value) or value <= 0 for value in values):
            errors.append(f"invalid timing: {path}:{metric}")
        return statistics.median(values)

    metrics = ("wall_ms", "total_cuda_ms", "visual_cuda_ms")
    point_paths = {path: {metric: median(path, metric) for metric in metrics} for path in paths}
    for path in ("eager-reuse", "optimized-reuse"):
        point_paths[path]["sequential_cuda_ms"] = median(path, "sequential_cuda_ms")
    if analysis.get("path_medians") != point_paths:
        errors.append("path medians mismatch")

    names = (
        "optimized_reuse_wall_over_batched_fr",
        "weighted_wall_over_batched_fr",
        "optimized_over_eager_sequential_cuda",
        "weighted_total_cuda_over_batched_fr",
        "weighted_visual_cuda_reduction",
        "v5_refresh_wall_over_batched_fr",
    )
    distributions = {name: [] for name in names}
    generator = random.Random(20260810)
    weight = float(config["analysis"]["reuse_weight"])
    for _ in range(10000):
        selected = [generator.randrange(24) for _ in range(24)]
        bfr_wall = median("batched-fr", "wall_ms", selected)
        refresh_wall = median("v5-refresh", "wall_ms", selected)
        optimized_wall = median("optimized-reuse", "wall_ms", selected)
        bfr_total = median("batched-fr", "total_cuda_ms", selected)
        refresh_total = median("v5-refresh", "total_cuda_ms", selected)
        optimized_total = median("optimized-reuse", "total_cuda_ms", selected)
        bfr_visual = median("batched-fr", "visual_cuda_ms", selected)
        refresh_visual = median("v5-refresh", "visual_cuda_ms", selected)
        optimized_visual = median("optimized-reuse", "visual_cuda_ms", selected)
        eager_sequential = median("eager-reuse", "sequential_cuda_ms", selected)
        optimized_sequential = median("optimized-reuse", "sequential_cuda_ms", selected)
        distributions[names[0]].append(optimized_wall / bfr_wall)
        distributions[names[1]].append(
            ((1 - weight) * refresh_wall + weight * optimized_wall) / bfr_wall
        )
        distributions[names[2]].append(optimized_sequential / eager_sequential)
        distributions[names[3]].append(
            ((1 - weight) * refresh_total + weight * optimized_total) / bfr_total
        )
        distributions[names[4]].append(
            1 - ((1 - weight) * refresh_visual + weight * optimized_visual) / bfr_visual
        )
        distributions[names[5]].append(refresh_wall / bfr_wall)
    comparisons = {
        name: {
            "point": statistics.median(values),
            "lower_95": quantile(values, 0.025),
            "upper_95": quantile(values, 0.975),
        }
        for name, values in distributions.items()
    }
    if analysis.get("comparisons") != comparisons:
        errors.append("bootstrap comparisons mismatch")

    order_rows: list[dict[str, Any]] = []
    maximum_deviation = 0.0
    for path in paths:
        available = list(metrics)
        if path in ("eager-reuse", "optimized-reuse"):
            available.append("sequential_cuda_ms")
        for metric in available:
            overall = point_paths[path][metric]
            for position in range(4):
                value = statistics.median(
                    float(record["timing"][metric])
                    for record in timed
                    if record["path"] == path and record["position"] == position
                )
                deviation = abs(value - overall) / overall
                maximum_deviation = max(maximum_deviation, deviation)
                order_rows.append(
                    {
                        "path": path,
                        "metric": metric,
                        "position": position,
                        "median": value,
                        "relative_deviation": deviation,
                    }
                )
    if (
        analysis.get("ordering") != order_rows
        or analysis.get("maximum_ordering_deviation") != maximum_deviation
    ):
        errors.append("ordering-bias calculation mismatch")

    limits = config["gates"]
    direct = point_paths["optimized-reuse"]["wall_ms"] / point_paths["batched-fr"]["wall_ms"]
    expected_gates = {
        "optimized_reuse_wall": direct <= limits["optimized_reuse_wall_over_batched_fr_median_max"],
        "weighted_wall": comparisons[names[1]]["upper_95"]
        <= limits["weighted_wall_over_batched_fr_upper_95_max"],
        "optimized_sequential_cuda": comparisons[names[2]]["upper_95"]
        <= limits["optimized_over_eager_sequential_cuda_upper_95_max"],
        "weighted_total_cuda": comparisons[names[3]]["upper_95"]
        <= limits["weighted_total_cuda_over_batched_fr_upper_95_max"],
        "weighted_visual_cuda": comparisons[names[4]]["lower_95"]
        >= limits["weighted_visual_cuda_reduction_lower_95_min"],
        "v5_refresh_wall": comparisons[names[5]]["upper_95"]
        <= limits["v5_refresh_wall_over_batched_fr_upper_95_max"],
        "ordering_bias": maximum_deviation <= limits["maximum_position_median_relative_deviation"],
    }
    if analysis.get("gates") != expected_gates:
        errors.append("analysis gate booleans mismatch")
    run_gate_names = (
        "correctness_pass",
        "memory_pass",
        "work_pass",
        "lifecycle_pass",
        "restoration_pass",
        "resource_pass",
    )
    expected_run_gates = {name: run.get(name) is True for name in run_gate_names}
    if analysis.get("run_gates") != expected_run_gates:
        errors.append("run gate booleans mismatch")
    passed = all(expected_gates.values()) and all(expected_run_gates.values())
    if analysis.get("passed") is not passed:
        errors.append("overall disposition mismatch")
    expected_disposition = "PASS_STOP_BEFORE_V5_E_PROTOCOL" if passed else "STOP_BEFORE_SIMULATOR"
    if analysis.get("disposition") != expected_disposition:
        errors.append("disposition label mismatch")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--run",
        type=Path,
        default=Path("results/acr-v5d-real-tensor-feasibility-v03/final/record.json"),
    )
    parser.add_argument(
        "--analysis", type=Path, default=Path("results/acr-v5d-analysis-v03/record.json")
    )
    args = parser.parse_args()
    root = args.root.resolve()
    if root != EXPECTED_ROOT:
        raise SystemExit(f"Refusing to verify outside {EXPECTED_ROOT}: {root}")
    sys.path.insert(0, str(root / "src"))
    from savr.acr.v5_d_runtime import load_v5_d_freeze

    config = load_v5_d_freeze(root)
    run_path = args.run if args.run.is_absolute() else root / args.run
    analysis_path = args.analysis if args.analysis.is_absolute() else root / args.analysis
    errors = verify(
        config,
        json.loads(run_path.read_text(encoding="utf-8")),
        json.loads(analysis_path.read_text(encoding="utf-8")),
    )
    print(json.dumps({"errors": errors, "verified": not errors}, sort_keys=True))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
