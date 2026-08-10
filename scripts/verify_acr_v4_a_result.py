#!/usr/bin/env python3
"""Independently verify the frozen V4-A result and mechanical stop rule."""

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


def verify(preflight: dict[str, Any], result: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    expected_hash = semantic_sha256(result)
    if result.get("semantic_sha256") != expected_hash:
        errors.append("result semantic hash mismatch")

    family = preflight["controller_family"]
    controller_gates = preflight["controller_gates"]
    complete_gates = preflight["complete_method_gates"]
    candidates = result.get("candidates", [])
    if len(candidates) != family["candidate_count"]:
        errors.append("candidate count mismatch")

    eligible_ids: list[str] = []
    for candidate in candidates:
        recomputed = {
            "reuse_point": candidate["reuse_rate"] >= controller_gates["replay_reuse_point_min"],
            "reuse_lower_bound": candidate["reuse_rate_interval"]["lower_95"]
            >= controller_gates["replay_reuse_lower_bound_min"],
            "maximum_streak": candidate["maximum_reuse_streak"]
            == controller_gates["maximum_reuse_streak"],
            "gripper_transition": candidate["gripper_transition_reuses"]
            <= controller_gates["gripper_transition_reuses_max"],
            "visual_point": candidate["predicted_visual_reduction_point"]
            >= controller_gates["predicted_visual_cuda_reduction_point_min"],
            "executor_feasibility": candidate["required_reuse_wall_ratio_vs_bfr"]
            >= complete_gates["required_reuse_wall_ratio_feasibility_floor"],
        }
        if recomputed != candidate["gates"]:
            errors.append(f"gate mismatch: {candidate['candidate_id']}")
        recomputed_eligible = all(recomputed.values())
        if recomputed_eligible != candidate["eligible"]:
            errors.append(f"eligibility mismatch: {candidate['candidate_id']}")
        if recomputed_eligible:
            eligible_ids.append(candidate["candidate_id"])

    expected_disposition = "ADVANCE_TO_V4_B_AUTHORIZATION" if eligible_ids else "STOP_BEFORE_V4_B"
    if result.get("disposition") != expected_disposition:
        errors.append("disposition mismatch")
    if not eligible_ids and (
        result.get("selected_candidate_id") is not None
        or result.get("selected_executor") is not None
    ):
        errors.append("negative result selected a method")

    caps = preflight["resource_caps"]
    resources = result.get("resources", {})
    for key in ("gpu_count", "model_queries", "simulator_episodes", "downloads"):
        if resources.get(key) != 0 or resources.get(key, 0) > caps[key]:
            errors.append(f"resource boundary mismatch: {key}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    preflight = json.loads(
        (args.root / "configs/acr/v4_a_diagnosis_preflight.json").read_text(encoding="utf-8")
    )
    result = json.loads((args.root / "reports/runtime/acr_v4_a.json").read_text(encoding="utf-8"))
    errors = verify(preflight, result)
    print(json.dumps({"errors": errors, "verified": not errors}, sort_keys=True))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
