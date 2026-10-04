#!/usr/bin/env python3
"""Unblind PAIR P4B only after its immutable worker summary is complete."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ROOT = Path("/home/ved/SAVR")


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def file_sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def write_once(path: Path, value: Mapping[str, Any]) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(canonical_bytes(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    if ROOT != EXPECTED_ROOT:
        raise SystemExit(f"P4B analyzer refuses to run outside {EXPECTED_ROOT}")
    if os.environ.get("CUDA_VISIBLE_DEVICES", ""):
        raise SystemExit("P4B analyzer requires CUDA to be hidden")
    sys.path.insert(0, str(ROOT / "src"))
    from savr.pair.features import GroupFeature, RouterFeatures
    from savr.pair.p4b import (
        cvar_improvement,
        expected_counts,
        semantic_sha256,
        spearman,
        validate_config,
    )
    from savr.pair.router import PairRouter
    from savr.pair.types import AtomicGroup, Camera

    config = json.loads((ROOT / args.config).read_text())
    validate_config(config, ROOT)
    run_root = args.run_root.resolve()
    if not run_root.is_relative_to(ROOT / "results"):
        raise SystemExit("P4B analysis root must be inside project results")
    paths = {
        "worker": run_root / "worker_summary.json",
        "seal": run_root / "worker_seal.json",
        "interventions": run_root / "interventions.sealed.jsonl",
        "features": run_root / "features.sealed.jsonl",
        "contracts": run_root / "contracts.sealed.jsonl",
    }
    if (run_root / "analysis.json").exists() or any(not path.is_file() for path in paths.values()):
        raise SystemExit("P4B analysis inputs are missing or analysis already exists")
    worker = json.loads(paths["worker"].read_text())
    seal = json.loads(paths["seal"].read_text())
    counts = expected_counts(int(config["schedule"]["seed"]))
    if (
        worker.get("status") != "completed"
        or seal.get("status") != "sealed_complete"
        or seal.get("worker_summary_sha256") != file_sha256(paths["worker"])
        or seal.get("configuration_semantic_sha256") != config["semantic_sha256"]
        or worker.get("configuration_semantic_sha256") != config["semantic_sha256"]
        or worker.get("intervention_records") != counts["intervention_records"]
        or worker.get("feature_records") != counts["feature_records"]
        or worker.get("contract_records") != counts["contract_records"]
        or worker.get("model_calls") != counts["planned_model_calls"]
        or any(worker["artifact_sha256"][key] != file_sha256(paths[key]) for key in ("interventions", "features", "contracts"))
    ):
        raise SystemExit("P4B immutable worker summary is incomplete or inconsistent")

    interventions = read_jsonl(paths["interventions"])
    features = read_jsonl(paths["features"])
    contracts = read_jsonl(paths["contracts"])
    if (len(interventions), len(features), len(contracts)) != (
        counts["intervention_records"], counts["feature_records"], counts["contract_records"]
    ):
        raise SystemExit("P4B record counts changed")
    intervention_by_id = {row["record_id"]: row for row in interventions}
    base = [row for row in contracts if row["branch_kind"] == "base"]
    structured = [row for row in base if row["category"] == "structured"]
    controls = [row for row in base if row["category"] == "all_fresh_control"]
    repeats = [row for row in contracts if row["branch_kind"] == "repeat"]
    if (len(structured), len(controls), len(repeats)) != (240, 40, 24):
        raise SystemExit("P4B contract populations changed")

    feature_by_contract: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in features:
        feature_by_contract[row["contract_id"]].append(row)
    for rows in feature_by_contract.values():
        rows.sort(key=lambda row: row["query_ordinal"])
    p4_root = ROOT / "results/pair-p4-pilot-v02-schedule-recovery01"
    router = PairRouter.load(p4_root / "router_checkpoint.json")
    training = json.loads((p4_root / "router_training_summary.json").read_text())
    artifact = json.loads((p4_root / "router_artifact.json").read_text())
    if (
        artifact["artifact_id"] != config["router"]["artifact_id"]
        or file_sha256(p4_root / "router_checkpoint.json") != config["router"]["checkpoint_sha256"]
        or training["selected_seed"] != config["router"]["training_seed"]
        or training["selected_proxy"]["name"] != config["router"]["proxy_name"]
        or training["selected_proxy"]["orientation"] != config["router"]["proxy_orientation"]
    ):
        raise SystemExit("P4B frozen router or proxy changed")

    def restore(row: Mapping[str, Any]) -> RouterFeatures:
        return RouterFeatures(
            tuple(
                GroupFeature(
                    AtomicGroup(Camera(group["camera"]), group["tile"], group["onset_layer"]),
                    group["profile_id"], group["horizon"], tuple(group["continuous"])
                )
                for group in row["features"]["groups"]
            ),
            tuple(row["features"]["global_continuous"]),
        )

    target = training["target_statistics"]

    def router_score(contract: Mapping[str, Any]) -> float:
        values = [
            router.predict(restore(row)).signed_contract_regret
            for row in feature_by_contract[contract["contract_id"]]
        ]
        return float(np.mean(values) * target["signed_scale"] + target["signed_mean"])

    def proxy_score(contract: Mapping[str, Any]) -> float:
        return -float(np.mean([
            np.mean([group["continuous"][6] for group in row["features"]["groups"]])
            for row in feature_by_contract[contract["contract_id"]]
        ]))

    observed = np.asarray([row["signed_contract_regret"] for row in structured], dtype=float)
    positive = np.maximum(observed, 0)
    router_scores = np.asarray([router_score(row) for row in structured])
    proxy_scores = np.asarray([proxy_score(row) for row in structured])
    suites = np.asarray([row["suite"] for row in structured])
    horizons = np.asarray([row["horizon"] for row in structured], dtype=int)
    tasks = np.asarray([row["task_id"] for row in structured])
    cells = Counter(zip(suites, horizons, strict=True))
    point_rho = spearman(router_scores, observed)
    point_cvar, router_selected, proxy_selected = cvar_improvement(
        positive, router_scores, proxy_scores, suites, horizons
    )
    h24_cvar, _, _ = cvar_improvement(
        positive, router_scores, proxy_scores, suites, horizons,
        allowed_horizons=frozenset({2, 4}),
    )

    rng = np.random.default_rng(int(config["inference"]["bootstrap_seed"]))
    replicates = int(config["inference"]["bootstrap_replicates"])
    boot_rho = np.empty(replicates)
    boot_cvar = np.empty(replicates)
    cell_indices = {
        cell: np.flatnonzero((suites == cell[0]) & (horizons == cell[1])) for cell in cells
    }
    for replicate in range(replicates):
        indices = np.concatenate([
            rng.choice(values, len(values), replace=True)
            for _cell, values in sorted(cell_indices.items())
        ])
        boot_rho[replicate] = spearman(router_scores[indices], observed[indices])
        boot_cvar[replicate] = cvar_improvement(
            positive[indices], router_scores[indices], proxy_scores[indices],
            suites[indices], horizons[indices]
        )[0]
    rho_lower = float(np.quantile(boot_rho, 0.05))
    cvar_lower = float(np.quantile(boot_cvar, 0.05))

    task_rng = np.random.default_rng(int(config["inference"]["bootstrap_seed"]) + 1)
    task_rho = np.empty(replicates)
    task_cvar = np.empty(replicates)
    for replicate in range(replicates):
        indices = []
        for suite in sorted(set(suites.tolist())):
            suite_tasks = sorted(set(tasks[suites == suite].tolist()))
            for task in task_rng.choice(suite_tasks, len(suite_tasks), replace=True):
                indices.extend(np.flatnonzero((suites == suite) & (tasks == task)).tolist())
        chosen = np.asarray(indices, dtype=int)
        task_rho[replicate] = spearman(router_scores[chosen], observed[chosen])
        task_cvar[replicate] = cvar_improvement(
            positive[chosen], router_scores[chosen], proxy_scores[chosen],
            suites[chosen], horizons[chosen]
        )[0]

    repeat_differences = np.asarray([
        abs(row["signed_regret"] - intervention_by_id[row["repeat_of_record_id"]]["signed_regret"])
        for row in interventions if row.get("repeat_of_record_id")
    ])
    repeat_median = float(np.median(repeat_differences))
    repeat_p95 = float(np.quantile(repeat_differences, 0.95))
    allfresh_action = max(row["maximum_action_abs_distortion"] for row in controls)
    allfresh_regret = max(abs(row["signed_contract_regret"]) for row in controls)
    served_tail = np.sort(positive[router_selected])[-max(1, math.ceil(0.10 * len(router_selected))):]
    cvar_mass = float(served_tail.max() / served_tail.sum()) if served_tail.sum() > 0 else 1.0
    gates = config["gates"]
    gate_values = {
        "repeat_noise": repeat_median <= gates["repeat_noise_median_absolute_maximum"] and repeat_p95 <= gates["repeat_noise_p95_absolute_maximum"],
        "all_fresh": allfresh_action <= gates["all_fresh_action_max_abs_maximum"] and allfresh_regret <= gates["all_fresh_regret_abs_maximum"],
        "cell_balance": len(cells) == 12 and set(cells.values()) == {20} and len(router_selected) == len(proxy_selected) == 168,
        "cvar_mass": cvar_mass <= gates["single_record_positive_cvar_mass_maximum"],
        "spearman": point_rho >= gates["spearman_point_minimum"] and rho_lower > gates["spearman_one_sided_95pct_lower_minimum"],
        "cvar_improvement": point_cvar >= gates["cvar90_improvement_point_minimum"] and cvar_lower > gates["cvar90_improvement_one_sided_95pct_lower_minimum"],
        "horizon24_cvar": h24_cvar >= gates["horizon24_cvar_improvement_minimum"],
        "resources_and_protection": worker["peak_gpu_memory_mib"] < gates["peak_gpu_memory_mib_strict_max"] and not worker["expert_values_persisted"] and not worker["raw_model_actions_persisted"] and not worker["simulator_accessed"] and not worker["locked_test_labels_accessed"],
    }
    failed = sorted(name for name, passed in gate_values.items() if not passed)
    status = "passed" if not failed else "scientific_stop"
    result = {
        "schema_version": "pair-p4b-analysis-v1",
        "run_id": config["run_id"],
        "status": status,
        "gates": gate_values,
        "failed_gates": failed,
        "metrics": {
            "structured_contracts": len(structured),
            "spearman": point_rho,
            "spearman_one_sided_95_lower": rho_lower,
            "matched_cvar90_improvement": point_cvar,
            "matched_cvar90_improvement_one_sided_95_lower": cvar_lower,
            "horizon24_cvar90_improvement": h24_cvar,
            "router_served": len(router_selected),
            "proxy_served": len(proxy_selected),
            "maximum_single_contract_cvar90_mass": cvar_mass,
            "repeat_noise_median": repeat_median,
            "repeat_noise_p95": repeat_p95,
            "allfresh_action_max_abs": allfresh_action,
            "allfresh_regret_abs_max": allfresh_regret,
        },
        "sensitivity": {
            "task_cluster_spearman_lower": float(np.quantile(task_rho, 0.05)),
            "task_cluster_cvar90_improvement_lower": float(np.quantile(task_cvar, 0.05)),
        },
        "protection": {
            "p4b_only_primary": True,
            "p4_p4b_pooling_used_for_gate": False,
            "terminal_outcomes_accessed": False,
            "locked_test_labels_accessed": False,
            "p5_authorized": False,
        },
        "input_sha256": {key: file_sha256(path) for key, path in paths.items()},
    }
    result["semantic_sha256"] = semantic_sha256(result)
    write_once(run_root / "analysis.json", result)
    print(json.dumps({"status": status, "failed_gates": failed}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
