#!/usr/bin/env python3
"""Analyze the complete frozen V5-D real-tensor feasibility record."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import sys
from pathlib import Path
from typing import Any


EXPECTED_ROOT = Path("/home/ved/SAVR")


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def semantic_sha256(value: dict[str, Any], *, field: str = "semantic_sha256") -> str:
    payload = dict(value)
    payload.pop(field, None)
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def percentile(values: list[float], probability: float) -> float:
    if not values or not 0.0 <= probability <= 1.0:
        raise ValueError("Percentile requires values and probability in [0,1]")
    ordered = sorted(values)
    location = (len(ordered) - 1) * probability
    lower = math.floor(location)
    upper = math.ceil(location)
    if lower == upper:
        return ordered[lower]
    fraction = location - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def interval(values: list[float]) -> dict[str, float]:
    return {
        "lower_95": percentile(values, 0.025),
        "upper_95": percentile(values, 0.975),
    }


def _positive(record: dict[str, Any], key: str) -> float:
    value = float(record["timing"][key])
    if not math.isfinite(value) or value <= 0:
        raise RuntimeError(f"V5-D timing {key} must be finite and positive")
    return value


def _validate_records(
    config: dict[str, Any], run: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[int, dict[str, dict[str, Any]]]]:
    if run.get("schema_version") != "acr.v5d-run.v1":
        raise RuntimeError("V5-D run schema mismatch")
    if run.get("run_id") != config["run_id"] or run.get("status") != "completed":
        raise RuntimeError("V5-D run is not the complete frozen run")
    if run.get("configuration_semantic_sha256") != config["semantic_sha256"]:
        raise RuntimeError("V5-D run/configuration identity mismatch")
    if run.get("backend") not in config["backend_waterfall"]["order"]:
        raise RuntimeError("V5-D run backend is not frozen")
    queries = run.get("queries", [])
    if len(queries) != config["resource_caps"]["full_model_queries_if_complete"]:
        raise RuntimeError("V5-D full-query record count mismatch")
    if [int(item.get("query_index", -1)) for item in queries] != list(range(len(queries))):
        raise RuntimeError("V5-D query indices are not contiguous")

    expected_labels = list(config["correctness"]["labels"])
    for path in config["timing"]["paths"]:
        for repetition in range(config["timing"]["warmups_per_path"]):
            expected_labels.append(f"warmup-{path}-{repetition:02d}")
    for block, order in enumerate(config["timing"]["permutations"]):
        for position, path in enumerate(order):
            expected_labels.append(f"timed-{block:02d}-{position}-{path}")
    if [item.get("label") for item in queries] != expected_labels:
        raise RuntimeError("V5-D exact query schedule mismatch")

    correctness = queries[: config["correctness"]["query_count"]]
    for record in correctness:
        if record.get("kind") != "correctness" or record.get("passed") is not True:
            raise RuntimeError("V5-D correctness record failed")
        checks = record.get("checks")
        if not isinstance(checks, dict) or not checks or not all(checks.values()):
            raise RuntimeError("V5-D correctness assertions are incomplete")

    timed = [item for item in queries if item.get("kind") == "timed"]
    if len(timed) != config["timing"]["timed_query_count"]:
        raise RuntimeError("V5-D timed-query record count mismatch")
    blocks: dict[int, dict[str, dict[str, Any]]] = {}
    for record in timed:
        block = int(record.get("block", -1))
        position = int(record.get("position", -1))
        path = str(record.get("path"))
        if block not in range(config["timing"]["block_count"]):
            raise RuntimeError("V5-D timing block is invalid")
        expected_order = config["timing"]["permutations"][block]
        if position not in range(4) or expected_order[position] != path:
            raise RuntimeError("V5-D timing order differs from the frozen permutation")
        if record.get("input_label") != ("input-a" if block % 2 == 0 else "input-b"):
            raise RuntimeError("V5-D timing input alternation changed")
        for key in ("wall_ms", "total_cuda_ms", "visual_cuda_ms"):
            _positive(record, key)
        if path in ("eager-reuse", "optimized-reuse"):
            _positive(record, "sequential_cuda_ms")
        block_records = blocks.setdefault(block, {})
        if path in block_records:
            raise RuntimeError("V5-D timing block contains a duplicate path")
        block_records[path] = record
    paths = set(config["timing"]["paths"])
    if set(blocks) != set(range(24)) or any(set(records) != paths for records in blocks.values()):
        raise RuntimeError("V5-D timing blocks are incomplete")
    return timed, blocks


def _median(records: list[dict[str, Any]], path: str, metric: str) -> float:
    return statistics.median(
        float(item["timing"][metric]) for item in records if item["path"] == path
    )


def analyze(config: dict[str, Any], run: dict[str, Any]) -> dict[str, Any]:
    """Recompute every frozen statistical gate from raw complete query records."""

    if config.get("semantic_sha256") != semantic_sha256(config):
        raise RuntimeError("V5-D configuration semantic hash mismatch")
    timed, blocks = _validate_records(config, run)
    paths = tuple(config["timing"]["paths"])
    metrics = ("wall_ms", "total_cuda_ms", "visual_cuda_ms")
    points = {path: {metric: _median(timed, path, metric) for metric in metrics} for path in paths}
    for path in ("eager-reuse", "optimized-reuse"):
        points[path]["sequential_cuda_ms"] = _median(timed, path, "sequential_cuda_ms")

    reuse_weight = float(config["analysis"]["reuse_weight"])
    rng = random.Random(int(config["analysis"]["bootstrap_seed"]))
    samples: dict[str, list[float]] = {
        "optimized_reuse_wall_over_batched_fr": [],
        "weighted_wall_over_batched_fr": [],
        "optimized_over_eager_sequential_cuda": [],
        "weighted_total_cuda_over_batched_fr": [],
        "weighted_visual_cuda_reduction": [],
        "v5_refresh_wall_over_batched_fr": [],
    }
    block_count = int(config["timing"]["block_count"])
    for _ in range(int(config["analysis"]["bootstrap_resamples"])):
        selected = [rng.randrange(block_count) for _ in range(block_count)]

        def sampled(path: str, metric: str) -> float:
            return statistics.median(
                float(blocks[index][path]["timing"][metric]) for index in selected
            )

        bfr_wall = sampled("batched-fr", "wall_ms")
        refresh_wall = sampled("v5-refresh", "wall_ms")
        optimized_wall = sampled("optimized-reuse", "wall_ms")
        bfr_total = sampled("batched-fr", "total_cuda_ms")
        refresh_total = sampled("v5-refresh", "total_cuda_ms")
        optimized_total = sampled("optimized-reuse", "total_cuda_ms")
        bfr_visual = sampled("batched-fr", "visual_cuda_ms")
        refresh_visual = sampled("v5-refresh", "visual_cuda_ms")
        optimized_visual = sampled("optimized-reuse", "visual_cuda_ms")
        eager_sequential = sampled("eager-reuse", "sequential_cuda_ms")
        optimized_sequential = sampled("optimized-reuse", "sequential_cuda_ms")
        weighted_wall = (1.0 - reuse_weight) * refresh_wall + reuse_weight * optimized_wall
        weighted_total = (1.0 - reuse_weight) * refresh_total + reuse_weight * optimized_total
        weighted_visual = (1.0 - reuse_weight) * refresh_visual + reuse_weight * optimized_visual
        samples["optimized_reuse_wall_over_batched_fr"].append(optimized_wall / bfr_wall)
        samples["weighted_wall_over_batched_fr"].append(weighted_wall / bfr_wall)
        samples["optimized_over_eager_sequential_cuda"].append(
            optimized_sequential / eager_sequential
        )
        samples["weighted_total_cuda_over_batched_fr"].append(weighted_total / bfr_total)
        samples["weighted_visual_cuda_reduction"].append(1.0 - weighted_visual / bfr_visual)
        samples["v5_refresh_wall_over_batched_fr"].append(refresh_wall / bfr_wall)

    comparisons = {
        name: {
            "point": statistics.median(values),
            **interval(values),
        }
        for name, values in samples.items()
    }
    direct_reuse_ratio = points["optimized-reuse"]["wall_ms"] / points["batched-fr"]["wall_ms"]

    ordering: list[dict[str, Any]] = []
    maximum_ordering_deviation = 0.0
    for path in paths:
        available_metrics = list(metrics)
        if path in ("eager-reuse", "optimized-reuse"):
            available_metrics.append("sequential_cuda_ms")
        for metric in available_metrics:
            overall = points[path][metric]
            for position in range(4):
                values = [
                    float(record["timing"][metric])
                    for record in timed
                    if record["path"] == path and int(record["position"]) == position
                ]
                position_median = statistics.median(values)
                deviation = abs(position_median - overall) / overall
                maximum_ordering_deviation = max(maximum_ordering_deviation, deviation)
                ordering.append(
                    {
                        "path": path,
                        "metric": metric,
                        "position": position,
                        "median": position_median,
                        "relative_deviation": deviation,
                    }
                )

    frozen = config["gates"]
    gates = {
        "optimized_reuse_wall": direct_reuse_ratio
        <= frozen["optimized_reuse_wall_over_batched_fr_median_max"],
        "weighted_wall": comparisons["weighted_wall_over_batched_fr"]["upper_95"]
        <= frozen["weighted_wall_over_batched_fr_upper_95_max"],
        "optimized_sequential_cuda": comparisons["optimized_over_eager_sequential_cuda"]["upper_95"]
        <= frozen["optimized_over_eager_sequential_cuda_upper_95_max"],
        "weighted_total_cuda": comparisons["weighted_total_cuda_over_batched_fr"]["upper_95"]
        <= frozen["weighted_total_cuda_over_batched_fr_upper_95_max"],
        "weighted_visual_cuda": comparisons["weighted_visual_cuda_reduction"]["lower_95"]
        >= frozen["weighted_visual_cuda_reduction_lower_95_min"],
        "v5_refresh_wall": comparisons["v5_refresh_wall_over_batched_fr"]["upper_95"]
        <= frozen["v5_refresh_wall_over_batched_fr_upper_95_max"],
        "ordering_bias": maximum_ordering_deviation
        <= frozen["maximum_position_median_relative_deviation"],
    }
    required_run_gates = {
        name: run.get(name) is True
        for name in (
            "correctness_pass",
            "memory_pass",
            "work_pass",
            "lifecycle_pass",
            "restoration_pass",
            "resource_pass",
        )
    }
    all_pass = all(gates.values()) and all(required_run_gates.values())
    result = {
        "schema_version": "acr.v5d-analysis.v1",
        "run_id": config["run_id"],
        "configuration_semantic_sha256": config["semantic_sha256"],
        "run_semantic_sha256": run.get("semantic_sha256"),
        "backend": run["backend"],
        "timed_blocks": len(blocks),
        "timed_queries": len(timed),
        "bootstrap": {
            "resamples": config["analysis"]["bootstrap_resamples"],
            "seed": config["analysis"]["bootstrap_seed"],
            "unit": config["analysis"]["resampling_unit"],
        },
        "path_medians": points,
        "comparisons": comparisons,
        "direct_optimized_reuse_wall_ratio": direct_reuse_ratio,
        "ordering": ordering,
        "maximum_ordering_deviation": maximum_ordering_deviation,
        "gates": gates,
        "run_gates": required_run_gates,
        "passed": all_pass,
        "failed_gates": sorted(
            [name for name, passed in {**gates, **required_run_gates}.items() if not passed]
        ),
        "disposition": ("PASS_STOP_BEFORE_V5_E_PROTOCOL" if all_pass else "STOP_BEFORE_SIMULATOR"),
    }
    result["semantic_sha256"] = semantic_sha256(result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/acr-v5d-real-tensor-feasibility-v02/final/record.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/acr-v5d-analysis-v02/record.json"),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    if root != EXPECTED_ROOT:
        raise SystemExit(f"Refusing to analyze outside {EXPECTED_ROOT}: {root}")
    sys.path.insert(0, str(root / "src"))
    from savr.acr.v5_d_runtime import load_v5_d_freeze

    config = load_v5_d_freeze(root)
    input_path = args.input if args.input.is_absolute() else root / args.input
    output_path = args.output if args.output.is_absolute() else root / args.output
    if output_path.exists():
        raise SystemExit(f"Immutable V5-D analysis already exists: {output_path}")
    result = analyze(config, json.loads(input_path.read_text(encoding="utf-8")))
    output_path.parent.mkdir(parents=True, exist_ok=False)
    output_path.write_bytes(canonical_bytes(result))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
