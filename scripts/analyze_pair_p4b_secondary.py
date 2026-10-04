#!/usr/bin/env python3
"""Prespecified non-gating P4B diagnostics after the primary analysis is frozen."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ROOT = Path("/home/ved/SAVR")


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    if ROOT != EXPECTED_ROOT or os.environ.get("CUDA_VISIBLE_DEVICES", ""):
        raise SystemExit("P4B secondary analysis requires project root and hidden CUDA")
    sys.path.insert(0, str(ROOT / "src"))
    from savr.pair.features import GroupFeature, RouterFeatures
    from savr.pair.p4b import cvar_improvement, semantic_sha256, spearman, validate_config
    from savr.pair.router import PairRouter
    from savr.pair.types import AtomicGroup, Camera

    config = json.loads((ROOT / args.config).read_text())
    validate_config(config, ROOT)
    run_root = args.run_root.resolve()
    primary_path = run_root / "analysis.json"
    output = run_root / "secondary_analysis.json"
    primary = json.loads(primary_path.read_text())
    if primary.get("status") not in {"passed", "scientific_stop"} or output.exists():
        raise SystemExit("P4B primary result is absent or secondary output already exists")
    contracts_path = run_root / "contracts.sealed.jsonl"
    features_path = run_root / "features.sealed.jsonl"
    interventions_path = run_root / "interventions.sealed.jsonl"
    contracts = [
        row for row in read_jsonl(contracts_path)
        if row["branch_kind"] == "base" and row["category"] == "structured"
    ]
    features_by_contract: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in read_jsonl(features_path):
        features_by_contract[row["contract_id"]].append(row)
    for rows in features_by_contract.values():
        rows.sort(key=lambda row: row["query_ordinal"])
    interventions = {row["record_id"]: row for row in read_jsonl(interventions_path)}
    p4_root = ROOT / "results/pair-p4-pilot-v02-schedule-recovery01"
    router = PairRouter.load(p4_root / "router_checkpoint.json")
    training = json.loads((p4_root / "router_training_summary.json").read_text())
    target = training["target_statistics"]

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

    def router_score(contract: Mapping[str, Any]) -> float:
        raw = [router.predict(restore(row)).signed_contract_regret for row in features_by_contract[contract["contract_id"]]]
        return float(np.mean(raw) * target["signed_scale"] + target["signed_mean"])

    def proxy_score(contract: Mapping[str, Any]) -> float:
        return -float(np.mean([
            np.mean([group["continuous"][6] for group in row["features"]["groups"]])
            for row in features_by_contract[contract["contract_id"]]
        ]))

    observed = np.asarray([row["signed_contract_regret"] for row in contracts])
    positive = np.maximum(observed, 0)
    router_scores = np.asarray([router_score(row) for row in contracts])
    proxy_scores = np.asarray([proxy_score(row) for row in contracts])
    suites = np.asarray([row["suite"] for row in contracts])
    horizons = np.asarray([row["horizon"] for row in contracts], dtype=int)
    transition = np.asarray([
        any(interventions[item]["gripper_transition"] for item in row["intervention_record_ids"])
        for row in contracts
    ])

    def subgroup(indices: np.ndarray) -> dict[str, Any]:
        if len(indices) < 3:
            return {"contracts": len(indices), "spearman": None, "cvar90_improvement": None}
        return {
            "contracts": len(indices),
            "spearman": spearman(router_scores[indices], observed[indices]),
            "cvar90_improvement": cvar_improvement(
                positive[indices], router_scores[indices], proxy_scores[indices],
                suites[indices], horizons[indices]
            )[0],
        }

    per_suite = {suite: subgroup(np.flatnonzero(suites == suite)) for suite in sorted(set(suites))}
    per_horizon = {str(h): subgroup(np.flatnonzero(horizons == h)) for h in (1, 2, 4)}
    gripper = {
        "transition": subgroup(np.flatnonzero(transition)),
        "no_transition": subgroup(np.flatnonzero(~transition)),
    }
    calibration = {}
    for quantile in (0.50, 0.90):
        threshold = float(np.quantile(router_scores, quantile))
        low = router_scores <= threshold
        high = ~low
        calibration[f"q{int(quantile * 100)}"] = {
            "prediction_threshold": threshold,
            "lower_risk_contracts": int(low.sum()),
            "lower_risk_mean_observed_regret": float(observed[low].mean()),
            "higher_risk_contracts": int(high.sum()),
            "higher_risk_mean_observed_regret": float(observed[high].mean()) if high.any() else None,
        }
    p4 = json.loads((p4_root / "analysis.json").read_text())
    rho4 = float(p4["metrics"]["calibration_structured_spearman"])
    rho4b = float(primary["metrics"]["spearman"])
    pooled_z = ((40 - 3) * np.arctanh(rho4) + (240 - 3) * np.arctanh(rho4b)) / (40 + 240 - 6)
    result = {
        "schema_version": "pair-p4b-secondary-analysis-v1",
        "run_id": config["run_id"],
        "status": "reported_non_gating",
        "primary_status_unchanged": primary["status"],
        "per_suite": per_suite,
        "per_horizon": per_horizon,
        "gripper_transition": gripper,
        "router_quantile_calibration": calibration,
        "p4_p4b_fixed_effect_fisher_spearman": float(np.tanh(pooled_z)),
        "gating_use": False,
        "protection": {"terminal_outcomes_accessed": False, "locked_test_labels_accessed": False, "p5_authorized": False},
        "input_sha256": {
            "primary": file_sha256(primary_path), "contracts": file_sha256(contracts_path),
            "features": file_sha256(features_path), "interventions": file_sha256(interventions_path),
        },
    }
    result["semantic_sha256"] = semantic_sha256(result)
    write_once(output, result)
    print(json.dumps({"status": "reported_non_gating", "primary_status": primary["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
