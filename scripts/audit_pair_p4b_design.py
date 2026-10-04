#!/usr/bin/env python3
"""Reproduce P4B capacity and power planning without opening locked labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ROOT = Path("/home/ved/SAVR")


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_once(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(canonical_bytes(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def ranks(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    result = np.empty(len(array), dtype=np.float64)
    start = 0
    while start < len(array):
        end = start + 1
        while end < len(array) and array[order[end]] == array[order[start]]:
            end += 1
        result[order[start:end]] = (start + end - 1) / 2
        start = end
    return result


def spearman(left: Sequence[float], right: Sequence[float]) -> float:
    x, y = ranks(left), ranks(right)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return -1.0
    return float(np.corrcoef(x, y)[0, 1])


def cvar90(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    count = max(1, int(math.ceil(0.10 * len(array))))
    return float(np.sort(array)[-count:].mean())


def served_cvar(
    observed: np.ndarray,
    scores: np.ndarray,
    suites: np.ndarray,
    horizons: np.ndarray,
) -> float:
    selected = []
    for suite in sorted(set(suites.tolist())):
        for horizon in (1, 2, 4):
            candidates = np.flatnonzero((suites == suite) & (horizons == horizon))
            if not len(candidates):
                continue
            count = max(1, int(math.ceil(0.70 * len(candidates))))
            selected.extend(
                candidates[
                    np.argsort(scores[candidates], kind="mergesort")[:count]
                ].tolist()
            )
    return cvar90(observed[np.asarray(selected, dtype=int)])


def cvar_improvement(
    observed: np.ndarray,
    router_scores: np.ndarray,
    proxy_scores: np.ndarray,
    suites: np.ndarray,
    horizons: np.ndarray,
) -> float:
    router = served_cvar(observed, router_scores, suites, horizons)
    proxy = served_cvar(observed, proxy_scores, suites, horizons)
    return -1.0 if proxy <= 0 else float(1.0 - router / proxy)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if ROOT != EXPECTED_ROOT:
        raise SystemExit(f"P4B design audit refuses to run outside {EXPECTED_ROOT}")
    if os.environ.get("CUDA_VISIBLE_DEVICES", ""):
        raise SystemExit("P4B design audit requires CUDA to be hidden")
    sys.path.insert(0, str(ROOT / "src"))
    from savr.pair.features import GroupFeature, RouterFeatures
    from savr.pair.p4b import expected_counts, semantic_sha256, validate_config
    from savr.pair.router import PairRouter
    from savr.pair.types import AtomicGroup, Camera

    config = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    validate_config(config, ROOT)
    output = (ROOT / args.output).resolve()
    if not output.is_relative_to(ROOT / "reports/pair_p4b") or output.exists():
        raise SystemExit("P4B design audit output must be new and under reports/pair_p4b")

    trajectory_path = ROOT / config["authenticated_inputs"]["trajectory_index"]["path"]
    trajectories = read_jsonl(trajectory_path)
    p4_schedule_path = ROOT / config["authenticated_inputs"]["p4_schedule"]["path"]
    p4_schedule = read_jsonl(p4_schedule_path)
    p4_used = {row["trajectory_id"] for row in p4_schedule}
    if len(p4_used) != 400:
        raise SystemExit("P4B predecessor schedule identity changed")
    remaining = [row for row in trajectories if row["trajectory_id"] not in p4_used]
    by_task: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for row in remaining:
        by_task[(row["suite"], row["task_id"])][row["split"]] += 1
    available_calibration = [row for row in remaining if row["split"] == "calibration"]
    available_train = [row for row in remaining if row["split"] == "train"]
    locked = [row for row in remaining if row["split"] == "locked_test"]
    expected_pattern = Counter({"train": 27, "calibration": 6, "locked_test": 7})
    if (
        len(trajectories) != 2000
        or len(by_task) != 40
        or any(value != expected_pattern for value in by_task.values())
        or len(available_calibration) != 240
        or len(available_train) != 1080
        or len(locked) != 280
        or Counter(row["suite"] for row in available_calibration)
        != Counter(
            {
                "libero_10": 60,
                "libero_goal": 60,
                "libero_object": 60,
                "libero_spatial": 60,
            }
        )
        or Counter(row["suite"] for row in locked)
        != Counter(
            {
                "libero_10": 70,
                "libero_goal": 70,
                "libero_object": 70,
                "libero_spatial": 70,
            }
        )
        or min(row["step_count"] for row in available_calibration) < 40
    ):
        raise SystemExit("P4B independent confirmation capacity changed")

    p4_root = ROOT / "results/pair-p4-pilot-v02-schedule-recovery01"
    feature_rows = read_jsonl(p4_root / "calibration_features.sealed.jsonl")
    contract_rows = read_jsonl(p4_root / "calibration_contracts.sealed.jsonl")
    training = json.loads((p4_root / "router_training_summary.json").read_text())
    router = PairRouter.load(p4_root / "router_checkpoint.json")
    by_contract: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in feature_rows:
        by_contract[row["contract_id"]].append(row)
    for values in by_contract.values():
        values.sort(key=lambda row: row["query_ordinal"])
    contracts = sorted(
        (
            row
            for row in contract_rows
            if row["branch_kind"] == "base" and row["category"] == "structured"
        ),
        key=lambda row: row["contract_id"],
    )
    if len(contracts) != 40:
        raise SystemExit("P4B planning source population changed")

    def restore(row: Mapping[str, Any]) -> RouterFeatures:
        return RouterFeatures(
            tuple(
                GroupFeature(
                    AtomicGroup(
                        Camera(group["camera"]), group["tile"], group["onset_layer"]
                    ),
                    group["profile_id"],
                    group["horizon"],
                    tuple(group["continuous"]),
                )
                for group in row["features"]["groups"]
            ),
            tuple(row["features"]["global_continuous"]),
        )

    target = training["target_statistics"]

    def router_score(contract: Mapping[str, Any]) -> float:
        values = [
            router.predict(restore(row)).signed_contract_regret
            for row in by_contract[contract["contract_id"]]
        ]
        return float(np.mean(values) * target["signed_scale"] + target["signed_mean"])

    proxy = training["selected_proxy"]
    if proxy != {"name": "projected_change", "orientation": -1, "spearman": proxy["spearman"]}:
        raise SystemExit("P4B frozen strongest proxy changed")

    def proxy_score(contract: Mapping[str, Any]) -> float:
        return -float(
            np.mean(
                [
                    np.mean(
                        [group["continuous"][6] for group in row["features"]["groups"]]
                    )
                    for row in by_contract[contract["contract_id"]]
                ]
            )
        )

    signed = np.asarray([row["signed_contract_regret"] for row in contracts])
    positive = np.maximum(signed, 0)
    router_scores = np.asarray([router_score(row) for row in contracts])
    proxy_scores = np.asarray([proxy_score(row) for row in contracts])
    suites = np.asarray([row["suite"] for row in contracts])
    horizons = np.asarray([row["horizon"] for row in contracts], dtype=int)
    point = {
        "spearman": spearman(router_scores, signed),
        "p4b_cell_matched_cvar90_improvement": cvar_improvement(
            positive, router_scores, proxy_scores, suites, horizons
        ),
    }

    planning = config["planning"]
    rng = np.random.default_rng(int(planning["empirical_seed"]))
    curves = []
    for sample_size in planning["sample_sizes"]:
        if sample_size % 12:
            raise SystemExit("P4B planning sizes must balance all 12 cells")
        per_cell = sample_size // 12
        correlation = np.empty(int(planning["empirical_replicates"]))
        improvement = np.empty_like(correlation)
        for replicate in range(len(correlation)):
            indices = np.concatenate(
                [
                    rng.choice(
                        np.flatnonzero((suites == suite) & (horizons == horizon)),
                        per_cell,
                        replace=True,
                    )
                    for suite in sorted(set(suites.tolist()))
                    for horizon in (1, 2, 4)
                ]
            )
            correlation[replicate] = spearman(router_scores[indices], signed[indices])
            improvement[replicate] = cvar_improvement(
                positive[indices],
                router_scores[indices],
                proxy_scores[indices],
                suites[indices],
                horizons[indices],
            )
        curves.append(
            {
                "sample_size": sample_size,
                "spearman_lower": float(
                    np.quantile(correlation, planning["lower_quantile"])
                ),
                "cvar90_improvement_lower": float(
                    np.quantile(improvement, planning["lower_quantile"])
                ),
                "cvar90_nonpositive_fraction": float(np.mean(improvement <= 0)),
            }
        )

    normal_quantiles = {0.8: 0.841621, 0.9: 1.281552}
    fisher = []
    null = float(planning["fisher_null_correlation"])
    for assumed in planning["fisher_assumed_correlations"]:
        difference = np.arctanh(assumed) - np.arctanh(null)
        for power in planning["fisher_powers"]:
            required = 3 + ((1.644854 + normal_quantiles[power]) / difference) ** 2
            fisher.append(
                {
                    "assumed_correlation": assumed,
                    "power": power,
                    "required_contracts": int(math.ceil(required)),
                }
            )
    result = {
        "schema_version": "pair-p4b-design-audit-v1",
        "status": "passed",
        "config_semantic_sha256": config["semantic_sha256"],
        "independent_confirmation_capacity": {
            "structured_calibration_trajectories": len(available_calibration),
            "available_train_controls": len(available_train),
            "tasks": len(by_task),
            "structured_per_task": 6,
            "structured_per_suite": dict(
                sorted(Counter(row["suite"] for row in available_calibration).items())
            ),
            "minimum_step_count": min(row["step_count"] for row in available_calibration),
            "maximum_step_count": max(row["step_count"] for row in available_calibration),
            "p4_trajectory_overlap": 0,
        },
        "locked_test_preservation": {
            "trajectories": len(locked),
            "per_task": 7,
            "per_suite": dict(sorted(Counter(row["suite"] for row in locked).items())),
            "labels_accessed": False,
        },
        "expected_execution_counts": expected_counts(),
        "p4_design_aligned_point_estimates": point,
        "empirical_planning_curve": curves,
        "fisher_planning": fisher,
        "planning_caveats": [
            "P4 empirical distribution is treated as the planning distribution.",
            "The future locked-test distribution can differ from this development-confirmation population.",
            "Primary claim is conditional on the fixed 40 benchmark tasks.",
        ],
        "access": {
            "p4_calibration_records_accessed": True,
            "p1_split_metadata_accessed": True,
            "unused_calibration_action_values_accessed": False,
            "unused_calibration_observations_accessed": False,
            "unused_calibration_regret_accessed": False,
            "locked_action_values_accessed": False,
            "locked_observations_accessed": False,
            "locked_regret_accessed": False,
            "terminal_outcomes_accessed": False,
            "simulator_accessed": False,
            "cuda_visible": False,
            "model_accessed": False,
        },
        "input_sha256": {
            "trajectory_index": file_sha256(trajectory_path),
            "p4_schedule": file_sha256(p4_schedule_path),
            "p4_calibration_features": file_sha256(
                p4_root / "calibration_features.sealed.jsonl"
            ),
            "p4_calibration_contracts": file_sha256(
                p4_root / "calibration_contracts.sealed.jsonl"
            ),
            "router_checkpoint": file_sha256(p4_root / "router_checkpoint.json"),
        },
    }
    result["semantic_sha256"] = semantic_sha256(result)
    write_once(output, result)
    print(
        json.dumps(
            {
                "status": "passed",
                "unused_calibration_trajectories": len(available_calibration),
                "locked_trajectories_preserved": len(locked),
                "recommended_structured_contracts": 240,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
