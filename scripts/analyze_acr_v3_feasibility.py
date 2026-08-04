#!/usr/bin/env python3
"""Derive the ACR V3 feasibility bounds from immutable V2-C evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED_V2_RESULT_SHA256 = "9e4d8e0034c4410dbc35f4e4a2b987eda4a0645ab603c3479da05a97bd6f1ae6"
EXPECTED_V2_SEMANTIC_SHA256 = "e834b5dc04385ec6b5d2385cff4098016427f5ce0c6e2d77744f0c9f1b76afc6"
PINNED_OPENVLA_OFT_REVISION = "e4287e94541f459edc4feabc4e181f537cd569a8"
PINNED_MODEL_SOURCE_SHA256 = "b5431a074c0025a12e46dc954a5e18d1d73477babb5ae42e3a12ab4b907f33a6"
WEIGHTED_WALL_RATIO_GATE = 0.98


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def byte_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def semantic_record(payload: dict[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result["semantic_sha256"] = hashlib.sha256(canonical_bytes(payload)).hexdigest()
    return result


def derive(v2_result: dict[str, Any], *, source_sha256: str) -> dict[str, Any]:
    if v2_result.get("result_semantic_sha256") != EXPECTED_V2_SEMANTIC_SHA256:
        raise RuntimeError("V2-C semantic identity differs from the frozen source")
    timing = v2_result["timing_summary"]
    paths = timing["paths"]
    reuse_weight = float(timing["reuse_weight"])
    fr_wall_ms = float(paths["upstream-fr"]["median_wall_ms"])
    fr_visual_ms = float(paths["upstream-fr"]["median_visual_cuda_ms"])
    reuse_visual_ms = float(paths["dual-path-reuse"]["median_visual_cuda_ms"])
    skipped_scene_visual_ms = fr_visual_ms - reuse_visual_ms

    # Optimistic upper bound: remove the complete measured scene-camera visual
    # time from FR and assume zero controller, cache, concat, or audit overhead.
    ideal_reuse_wall_ms = fr_wall_ms - skipped_scene_visual_ms
    ideal_reuse_wall_ratio = ideal_reuse_wall_ms / fr_wall_ms
    scene_skip_only_weighted_ratio = (
        (1.0 - reuse_weight) + reuse_weight * ideal_reuse_wall_ratio
    )
    maximum_scene_skip_only_reduction = 1.0 - scene_skip_only_weighted_ratio
    required_refresh_ratio = (
        WEIGHTED_WALL_RATIO_GATE - reuse_weight * ideal_reuse_wall_ratio
    ) / (1.0 - reuse_weight)

    payload = {
        "schema_version": "acr.v3-feasibility.v1",
        "phase": "ACR-V3-A",
        "status": "FROZEN_BEFORE_V3_IMPLEMENTATION_OR_OUTCOMES",
        "source_evidence": {
            "v2_c_result_sha256": source_sha256,
            "v2_c_result_semantic_sha256": EXPECTED_V2_SEMANTIC_SHA256,
            "openvla_oft_revision": PINNED_OPENVLA_OFT_REVISION,
            "modeling_prismatic_sha256": PINNED_MODEL_SOURCE_SHA256,
        },
        "observed_v2_c": {
            "reuse_weight": reuse_weight,
            "upstream_fr_median_wall_ms": fr_wall_ms,
            "upstream_fr_median_visual_cuda_ms": fr_visual_ms,
            "dual_reuse_median_visual_cuda_ms": reuse_visual_ms,
            "measured_scene_camera_visual_ms": skipped_scene_visual_ms,
            "upstream_low_level_calls": {"siglip": 2, "dinov2": 2, "projector": 1},
            "reuse_low_level_calls": {"siglip": 1, "dinov2": 1, "projector": 1},
        },
        "scene_skip_only_ceiling": {
            "assumptions": [
                "refresh queries cost exactly upstream FR",
                "reuse removes all measured scene-camera visual time",
                "controller_cache_concat_and_evidence_overhead_is_zero",
            ],
            "ideal_reuse_wall_ms": ideal_reuse_wall_ms,
            "ideal_reuse_wall_ratio": ideal_reuse_wall_ratio,
            "weighted_wall_ratio": scene_skip_only_weighted_ratio,
            "maximum_weighted_wall_reduction": maximum_scene_skip_only_reduction,
            "target_weighted_wall_reduction": 1.0 - WEIGHTED_WALL_RATIO_GATE,
            "target_reachable": scene_skip_only_weighted_ratio <= WEIGHTED_WALL_RATIO_GATE,
        },
        "redesign_requirement": {
            "maximum_refresh_wall_ratio_if_reuse_is_ideal": required_refresh_ratio,
            "minimum_refresh_saving_ms_if_reuse_is_ideal": fr_wall_ms
            * (1.0 - required_refresh_ratio),
            "must_accelerate_refresh_queries": True,
            "selected_mechanism": "batch_scene_and_wrist_within_each_vision_tower",
            "required_ablation": "batched_full_refresh",
        },
        "interpretation_guards": [
            "The bound is optimistic and is not a measured V3 result.",
            "Removing V2 audit hashing is necessary for fair timing but cannot by "
            "itself establish the V3 mechanism.",
            "Two-camera batching is plausible because the pinned source loops over "
            "cameras, but acceleration and numerical equivalence remain unobserved.",
            "Batched Full Refresh is required to separate batching gains from "
            "asymmetric-refresh gains.",
            "V2-C remains negative and is neither rerun nor reinterpreted.",
        ],
    }
    return semantic_record(payload)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source_sha256 = byte_sha256(args.v2_result)
    if source_sha256 != EXPECTED_V2_RESULT_SHA256:
        raise RuntimeError("V2-C byte identity differs from the frozen source")
    result = derive(
        json.loads(args.v2_result.read_text(encoding="utf-8")),
        source_sha256=source_sha256,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, sort_keys=True, separators=(",", ":")), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
