#!/usr/bin/env python3
"""Independently reconcile the frozen V5-B result and selection rule."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def semantic_sha256(record: dict[str, Any]) -> str:
    semantic = dict(record)
    semantic.pop("semantic_sha256", None)
    return hashlib.sha256(canonical_bytes(semantic)).hexdigest()


def verify(config: dict[str, Any], result: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if result.get("semantic_sha256") != semantic_sha256(result):
        errors.append("result semantic hash mismatch")
    if result.get("schema_version") != "acr.v5b-result.v1":
        errors.append("result schema mismatch")
    if result.get("candidate_count") != config["threshold_family"]["candidate_count"]:
        errors.append("candidate count mismatch")
    if len(result.get("candidates", [])) != config["threshold_family"]["candidate_count"]:
        errors.append("candidate result count mismatch")
    if result.get("replay_repetitions") != 2 or result.get("replay_byte_identical") is not True:
        errors.append("deterministic replay requirement failed")
    for key in ("trace_records", "trace_artifact_bytes", "ordered_path_content_sha256"):
        if result.get("input_manifest", {}).get(key) != config["input"][key]:
            errors.append(f"input manifest mismatch: {key}")

    gates = config["gates"]
    candidate_freeze = {item["candidate_id"]: item for item in config["candidates"]}
    eligible: list[dict[str, Any]] = []
    for candidate in result.get("candidates", []):
        identifier = candidate.get("candidate_id")
        if identifier not in candidate_freeze:
            errors.append(f"unknown candidate: {identifier}")
            continue
        frozen = candidate_freeze[identifier]
        for key in ("level", "scene_threshold", "translation_threshold", "hard_reuse_cap"):
            if candidate.get(key) != frozen[key]:
                errors.append(f"candidate freeze mismatch: {identifier}:{key}")
        recomputed = {
            "population": candidate["episodes"] == config["input"]["episodes"]
            and candidate["queries"] == config["input"]["trace_records"],
            "maximum_streak": candidate["maximum_reuse_streak"] == gates["maximum_reuse_streak"],
            "prefix_cap": candidate["prefix_cap_violations"]
            <= gates["maximum_prefix_cap_violations"],
            "gripper_transition": candidate["gripper_transition_reuses"]
            <= gates["gripper_transition_reuses_max"],
            "isolation_state": candidate["isolation_state_mismatches"]
            <= gates["isolation_state_mismatches_max"],
            "invariants": candidate["invariant_failures"] <= gates["invariant_failures_max"],
            "post_reuse_refresh": candidate["post_reuse_refreshes"]
            >= gates["post_reuse_refreshes_min"],
            "reuse_point": candidate["reuse_rate"] >= gates["reuse_point_min"],
            "reuse_lower_95": candidate["reuse_rate_interval"]["lower_95"]
            >= gates["reuse_lower_95_min"],
            "logical_visual_point": candidate["logical_visual_reduction_point"]
            >= gates["logical_visual_reduction_point_min"],
            "logical_visual_lower_95": candidate["logical_visual_reduction_interval"]["lower_95"]
            >= gates["logical_visual_reduction_lower_95_min"],
        }
        if candidate.get("gates") != recomputed:
            errors.append(f"gate mismatch: {identifier}")
        is_eligible = all(recomputed.values())
        if candidate.get("eligible") is not is_eligible:
            errors.append(f"eligibility mismatch: {identifier}")
        if is_eligible:
            eligible.append(candidate)

    eligible.sort(
        key=lambda item: (
            float(item["level"]),
            float(item["hard_reuse_cap"]),
            int(item["isolation_state_mismatches"]),
            float(item["reuse_rate"]),
        )
    )
    eligible_ids = [item["candidate_id"] for item in eligible]
    if result.get("eligible_candidate_ids") != eligible_ids:
        errors.append("eligible candidate ordering mismatch")
    selected = eligible_ids[0] if eligible_ids else None
    if result.get("selected_candidate_id") != selected:
        errors.append("selected candidate mismatch")
    disposition = "ADVANCE_TO_V5_C_PROTOCOL" if selected else "STOP_BEFORE_V5_C"
    if result.get("disposition") != disposition:
        errors.append("disposition mismatch")

    resources = result.get("resources", {})
    for key in (
        "gpu_count",
        "model_queries",
        "simulator_episodes",
        "simulator_resets",
        "downloads",
        "new_task_outcomes",
    ):
        if resources.get(key) != 0:
            errors.append(f"resource boundary mismatch: {key}")
    if result.get("protected") != config["protected"]:
        errors.append("protected-data boundary mismatch")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--result", type=Path, default=Path("reports/runtime/acr_v5_b.json"))
    args = parser.parse_args()
    config = json.loads(
        (args.root / "configs/acr/v5_b_output_blind_preflight.json").read_text(encoding="utf-8")
    )
    result_path = args.result if args.result.is_absolute() else args.root / args.result
    result = json.loads(result_path.read_text(encoding="utf-8"))
    errors = verify(config, result)
    print(json.dumps({"errors": errors, "verified": not errors}, sort_keys=True))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
