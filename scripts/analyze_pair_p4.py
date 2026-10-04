#!/usr/bin/env python3
"""Unblind and reconcile the frozen PAIR P4 gates after router freeze."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def write_once(path: Path, value: Mapping[str, Any]) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(canonical_bytes(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


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
    observed: np.ndarray, scores: np.ndarray, horizons: np.ndarray, allowed: set[int]
) -> float:
    selected = []
    for horizon in sorted(allowed):
        candidates = np.flatnonzero(horizons == horizon)
        if not len(candidates):
            continue
        count = max(1, int(math.ceil(0.70 * len(candidates))))
        chosen = candidates[np.argsort(scores[candidates], kind="mergesort")[:count]]
        selected.extend(chosen.tolist())
    if not selected:
        return float("nan")
    return cvar90(observed[np.asarray(selected, dtype=int)])


def cvar_improvement(
    observed: np.ndarray,
    router_scores: np.ndarray,
    proxy_scores: np.ndarray,
    horizons: np.ndarray,
    allowed: set[int],
) -> float:
    router = served_cvar(observed, router_scores, horizons, allowed)
    proxy = served_cvar(observed, proxy_scores, horizons, allowed)
    if not np.isfinite(router) or not np.isfinite(proxy) or proxy <= 0:
        return -1.0
    return float(1.0 - router / proxy)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--recovery-config", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    if ROOT != EXPECTED_ROOT:
        raise SystemExit(f"P4 analyzer refuses to run outside {EXPECTED_ROOT}")
    if os.environ.get("CUDA_VISIBLE_DEVICES", ""):
        raise SystemExit("P4 analysis requires CUDA to be hidden")
    sys.path.insert(0, str(ROOT / "src"))
    from savr.pair.features import GroupFeature, RouterFeatures
    from savr.pair.router import PairRouter
    from savr.pair.types import AtomicGroup, Camera

    config = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    recovery = json.loads((ROOT / args.recovery_config).read_text(encoding="utf-8"))
    run_root = args.run_root.resolve()
    worker = json.loads((run_root / "worker_summary.json").read_text(encoding="utf-8"))
    training = json.loads((run_root / "router_training_summary.json").read_text(encoding="utf-8"))
    artifact = json.loads((run_root / "router_artifact.json").read_text(encoding="utf-8"))
    router = PairRouter.load(run_root / "router_checkpoint.json")
    if (
        worker.get("status") != "completed"
        or worker.get("run_id") != recovery["run_id"]
        or artifact["checkpoint_sha256"] != file_sha256(run_root / "router_checkpoint.json")
        or training["calibration_label_paths_opened"] != []
    ):
        raise SystemExit("P4 unblind prerequisites are invalid")
    paths = {
        "train_interventions": run_root / "train_interventions.jsonl",
        "calibration_interventions": run_root / "calibration_interventions.sealed.jsonl",
        "train_features": run_root / "train_features.jsonl",
        "calibration_features": run_root / "calibration_features.sealed.jsonl",
        "train_contracts": run_root / "train_contracts.jsonl",
        "calibration_contracts": run_root / "calibration_contracts.sealed.jsonl",
    }
    for kind in ("interventions", "features", "contracts"):
        for split in ("train", "calibration"):
            if file_sha256(paths[f"{split}_{kind}"]) != worker[f"{kind}_sha256"][split]:
                raise SystemExit("P4 protected artifact hash mismatch")
    interventions = read_jsonl(paths["train_interventions"]) + read_jsonl(
        paths["calibration_interventions"]
    )
    features = read_jsonl(paths["calibration_features"])
    contracts = read_jsonl(paths["train_contracts"]) + read_jsonl(
        paths["calibration_contracts"]
    )
    if len(interventions) != 2120 or len(features) != 160 or len(contracts) != 832:
        raise SystemExit("P4 unblinded record accounting changed")
    intervention_by_id = {row["record_id"]: row for row in interventions}
    features_by_contract: dict[str, list[dict[str, Any]]] = {}
    for row in features:
        features_by_contract.setdefault(row["contract_id"], []).append(row)
    for rows in features_by_contract.values():
        rows.sort(key=lambda value: value["query_ordinal"])

    def restore_features(row: Mapping[str, Any]) -> RouterFeatures:
        groups = tuple(
            GroupFeature(
                AtomicGroup(
                    Camera(item["camera"]), item["tile"], item["onset_layer"]
                ),
                item["profile_id"],
                item["horizon"],
                tuple(item["continuous"]),
            )
            for item in row["features"]["groups"]
        )
        return RouterFeatures(groups, tuple(row["features"]["global_continuous"]))

    target_stats = training["target_statistics"]
    calibration_structured = sorted(
        (
            row
            for row in contracts
            if row["split"] == "calibration"
            and row["branch_kind"] == "base"
            and row["category"] == "structured"
        ),
        key=lambda row: row["contract_id"],
    )
    if len(calibration_structured) != 40:
        raise SystemExit("P4 calibration structured population changed")

    def router_score(contract: Mapping[str, Any]) -> float:
        outputs = [
            router.predict(restore_features(row)).signed_contract_regret
            for row in features_by_contract[contract["contract_id"]]
        ]
        return float(np.mean(outputs) * target_stats["signed_scale"] + target_stats["signed_mean"])

    proxy_name = training["selected_proxy"]["name"]
    proxy_orientation = int(training["selected_proxy"]["orientation"])
    proxy_index = {
        "source_age": 0,
        "raw_change": 1,
        "projected_change": 6,
        "proprio_change": 12,
        "previous_action_change": 14,
        "retained_salience": 16,
    }

    def proxy_score(contract: Mapping[str, Any]) -> float:
        contract_id = contract["contract_id"]
        if proxy_name == "seeded_random":
            value = int(hashlib.sha256(f"proxy|{contract_id}".encode()).hexdigest()[:16], 16) / 16**16
        else:
            index = proxy_index[proxy_name]
            value = float(
                np.mean(
                    [
                        np.mean(
                            [
                                group["continuous"][index]
                                for group in row["features"]["groups"]
                            ]
                        )
                        for row in features_by_contract[contract_id]
                    ]
                )
            )
        return proxy_orientation * value

    observed = np.asarray(
        [row["signed_contract_regret"] for row in calibration_structured], dtype=np.float64
    )
    positive = np.maximum(observed, 0)
    router_scores = np.asarray([router_score(row) for row in calibration_structured])
    proxy_scores = np.asarray([proxy_score(row) for row in calibration_structured])
    horizons = np.asarray([row["horizon"] for row in calibration_structured], dtype=int)
    trajectories = np.asarray([row["trajectory_id"] for row in calibration_structured])
    spearman_point = spearman(router_scores, observed)
    improvement_point = cvar_improvement(
        positive, router_scores, proxy_scores, horizons, {1, 2, 4}
    )
    h24_improvement = cvar_improvement(
        positive, router_scores, proxy_scores, horizons, {2, 4}
    )
    rng = np.random.default_rng(int(config["gates"]["bootstrap_seed"]))
    spearman_bootstrap = np.empty(int(config["gates"]["bootstrap_replicates"]))
    improvement_bootstrap = np.empty_like(spearman_bootstrap)
    unique_trajectories = np.unique(trajectories)
    for replicate in range(len(spearman_bootstrap)):
        sampled = rng.choice(unique_trajectories, len(unique_trajectories), replace=True)
        indices = np.concatenate([np.flatnonzero(trajectories == item) for item in sampled])
        spearman_bootstrap[replicate] = spearman(
            router_scores[indices], observed[indices]
        )
        improvement_bootstrap[replicate] = cvar_improvement(
            positive[indices],
            router_scores[indices],
            proxy_scores[indices],
            horizons[indices],
            {1, 2, 4},
        )
    spearman_lower = float(np.quantile(spearman_bootstrap, 0.10))
    improvement_lower = float(np.quantile(improvement_bootstrap, 0.10))

    repeat_differences = np.asarray(
        [
            abs(row["signed_regret"] - intervention_by_id[row["repeat_of_record_id"]]["signed_regret"])
            for row in interventions
            if row.get("repeat_of_record_id")
        ],
        dtype=np.float64,
    )
    repeat_median = float(np.median(repeat_differences))
    repeat_p95 = float(np.quantile(repeat_differences, 0.95))
    base_contracts = [row for row in contracts if row["branch_kind"] == "base"]
    controls = [row for row in base_contracts if row["category"] == "all_fresh_control"]
    noncontrols = [row for row in base_contracts if row["category"] != "all_fresh_control"]
    base_interventions = [
        intervention_by_id[record_id]
        for contract in noncontrols
        for record_id in contract["intervention_record_ids"]
    ]
    signed_records = np.asarray([row["signed_regret"] for row in base_interventions])
    positive_records = np.maximum(signed_records, 0)
    signed_iqr = float(np.quantile(signed_records, 0.75) - np.quantile(signed_records, 0.25))
    positive_prevalence = float(np.mean(signed_records > 0))
    tail_count = max(1, int(math.ceil(0.10 * len(positive_records))))
    tail = np.sort(positive_records)[-tail_count:]
    maximum_cvar_mass = float(tail.max() / tail.sum()) if tail.sum() > 0 else 1.0
    interactions = []
    for base in (row for row in noncontrols if row["category"] == "structured"):
        related = [
            row
            for row in contracts
            if row["base_contract_id"] == base["base_contract_id"]
        ]
        parts = {row["branch_kind"]: row for row in related}
        interactions.append(
            abs(
                base["signed_contract_regret"]
                - parts["partition_a"]["signed_contract_regret"]
                - parts["partition_b"]["signed_contract_regret"]
            )
        )
    interaction_median = float(np.median(interactions))
    allfresh_action_max = max(row["maximum_action_abs_distortion"] for row in controls)
    allfresh_regret_max = max(abs(row["signed_contract_regret"]) for row in controls)
    gates_config = config["gates"]
    gate_values = {
        "repeat_noise": repeat_median
        <= gates_config["repeat_noise_median_absolute_maximum"]
        and repeat_p95 <= gates_config["repeat_noise_p95_absolute_maximum"],
        "all_fresh": allfresh_action_max
        <= gates_config["all_fresh_action_max_abs_maximum"]
        and allfresh_regret_max <= gates_config["all_fresh_regret_abs_maximum"],
        "identifiability": signed_iqr
        > max(
            gates_config["signed_regret_iqr_minimum"],
            gates_config["signed_regret_iqr_to_repeat_p95_minimum"] * repeat_p95,
        ),
        "positive_prevalence": gates_config["positive_regret_prevalence_interval"][0]
        <= positive_prevalence
        <= gates_config["positive_regret_prevalence_interval"][1],
        "cvar_mass": maximum_cvar_mass
        <= gates_config["single_record_positive_cvar_mass_maximum"],
        "spearman": spearman_point >= gates_config["spearman_point_minimum"]
        and spearman_lower > gates_config["spearman_one_sided_90pct_lower_minimum"],
        "cvar_improvement": improvement_point
        >= gates_config["cvar90_improvement_point_minimum"]
        and improvement_lower > gates_config["cvar90_improvement_one_sided_90pct_lower_minimum"],
        "horizon24_cvar": h24_improvement
        >= gates_config["horizon24_cvar_improvement_minimum"],
        "interaction": interaction_median
        >= gates_config["interaction_residual_to_repeat_noise_minimum"] * repeat_median,
        "resources_and_protection": worker["peak_gpu_memory_mib"]
        < gates_config["peak_gpu_memory_mib_strict_max"]
        and worker["model_calls"] == config["accounting"]["planned_model_calls"]
        and not worker["expert_values_persisted"]
        and not worker["locked_test_labels_accessed"]
        and not worker["simulator_accessed"],
    }
    failed = sorted(name for name, passed in gate_values.items() if not passed)
    if not failed:
        status = "passed"
    elif set(failed) <= {"horizon24_cvar", "interaction"}:
        status = "blocked_authority_single_query"
    else:
        status = "scientific_stop"
    analysis = {
        "schema_version": "pair-p4-analysis-v1",
        "run_id": recovery["run_id"],
        "status": status,
        "gates": gate_values,
        "failed_gates": failed,
        "metrics": {
            "repeat_noise_median": repeat_median,
            "repeat_noise_p95": repeat_p95,
            "allfresh_action_max_abs": allfresh_action_max,
            "allfresh_regret_abs_max": allfresh_regret_max,
            "signed_regret_iqr": signed_iqr,
            "positive_regret_prevalence": positive_prevalence,
            "maximum_single_record_cvar90_mass": maximum_cvar_mass,
            "calibration_structured_spearman": spearman_point,
            "calibration_structured_spearman_one_sided_90_lower": spearman_lower,
            "matched_service_cvar90_improvement": improvement_point,
            "matched_service_cvar90_improvement_one_sided_90_lower": improvement_lower,
            "horizon24_cvar90_improvement": h24_improvement,
            "interaction_residual_median_absolute": interaction_median,
            "calibration_structured_contracts": len(calibration_structured),
            "repeat_query_records": len(repeat_differences),
        },
        "router": {
            "selected_seed": training["selected_seed"],
            "selected_proxy": training["selected_proxy"],
            "artifact_id": artifact["artifact_id"],
        },
        "bootstrap": {
            "replicates": len(spearman_bootstrap),
            "seed": config["gates"]["bootstrap_seed"],
            "unit": "calibration_trajectory",
            "degenerate_resample_value": -1.0,
        },
        "protection": {
            "router_frozen_before_calibration_open": True,
            "terminal_outcomes_accessed": False,
            "locked_test_labels_accessed": False,
            "raw_action_arrays_persisted": False,
            "p5_authorized": False,
        },
        "input_sha256": {key: file_sha256(path) for key, path in paths.items()},
    }
    analysis["semantic_sha256"] = hashlib.sha256(canonical_bytes(analysis)).hexdigest()
    write_once(run_root / "analysis.json", analysis)
    report = "\n".join(
        (
            "# PAIR P4 Expert-Regret Pilot Report",
            "",
            f"- Status: **{status}**",
            f"- Gates passed: {sum(gate_values.values())}/{len(gate_values)}",
            f"- Failed gates: {', '.join(failed) if failed else 'none'}",
            f"- Calibration structured Spearman: {spearman_point:.4f} (one-sided 90% lower {spearman_lower:.4f})",
            f"- Matched-service CVaR90 improvement: {improvement_point:.2%} (one-sided 90% lower {improvement_lower:.2%})",
            f"- Horizon-2/4 CVaR90 improvement: {h24_improvement:.2%}",
            f"- Positive-regret prevalence: {positive_prevalence:.2%}",
            f"- Signed-regret IQR: {signed_iqr:.8f}",
            "",
            "P4 is an offline reliability-risk screening result. It does not use simulator success and does not authorize P5.",
            "",
        )
    )
    report_path = run_root / "PAIR_P4_REPORT.md"
    descriptor = os.open(report_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(report)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"status": status, "failed_gates": failed}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
