#!/usr/bin/env python3
"""Analyze the immutable outcome-blind PAIR P3 timing population."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def semantic_sha256(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("semantic_sha256", None)
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_once(path: Path, value: Mapping[str, Any]) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(canonical_bytes(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def effective_reuse_fraction(profile: Any) -> float:
    totals = [
        primary + wrist
        for primary, wrist in zip(profile.primary_budgets, profile.wrist_budgets, strict=True)
    ]
    reused = 4 * totals[0] + 3 * totals[1] + 2 * totals[2] + 21 * totals[3]
    return reused / (32 * 512)


def validate_action_record(action: Mapping[str, Any]) -> None:
    if set(action) != {"shape", "finite", "sha256"}:
        raise RuntimeError("P3 action record exposes a protected field")
    if action["shape"] != [8, 7] or action["finite"] is not True:
        raise RuntimeError("P3 action structural record is invalid")
    if not isinstance(action["sha256"], str) or len(action["sha256"]) != 64:
        raise RuntimeError("P3 action hash is invalid")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--worker-summary", type=Path)
    parser.add_argument("--blocks", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT / "src"))
    from savr.pair.p3 import (
        P3Profile,
        block_schedule,
        net_saving_lower,
        one_sided_bootstrap_lower,
        profile_source_diversity,
        reject_protected_fields,
        validate_config,
    )

    config = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    inputs = json.loads((ROOT / config["input_manifest"]).read_text(encoding="utf-8"))
    validate_config(config, input_count=len(inputs["inputs"]))
    worker_path = args.worker_summary or (args.run_root / "worker_summary.json")
    blocks_path = args.blocks or (args.run_root / "blocks.jsonl")
    if not worker_path.is_file() or not blocks_path.is_file():
        raise RuntimeError("P3 terminal worker evidence is incomplete")
    worker = json.loads(worker_path.read_text(encoding="utf-8"))
    if worker.get("semantic_sha256") != semantic_sha256(worker):
        raise RuntimeError("P3 worker summary semantic hash mismatch")
    recovery = worker.get("schema_version") == "pair-p3-worker-recovery-v1"
    expected_status = "completed_recovery" if recovery else "completed"
    if worker.get("status") != expected_status:
        raise RuntimeError("P3 worker did not complete")
    blocks = [json.loads(line) for line in blocks_path.read_text(encoding="utf-8").splitlines()]
    expected_schedule = block_schedule(config, len(inputs["inputs"]))
    if len(blocks) != len(expected_schedule) or [row["block_id"] for row in blocks] != [
        block.block_id for block in expected_schedule
    ]:
        raise RuntimeError("P3 block population is incomplete or reordered")
    for row in blocks:
        reject_protected_fields(row)
        if row.get("semantic_sha256") != semantic_sha256(row):
            raise RuntimeError("P3 block semantic hash mismatch")
        for arm in ("dense", "cache"):
            for query in row["arms"][arm]["queries"]:
                action = query["action_record"]
                validate_action_record(action)

    if recovery:
        if worker.get("controls_inferred_from_authenticated_control_flow") is not True:
            raise RuntimeError("P3 recovery lacks authenticated control-flow evidence")
    else:
        for field in ("off_action_record", "on_action_record"):
            validate_action_record(worker["sidecar_control"][field])
        for field in ("dense_action_record", "all_fresh_action_record"):
            validate_action_record(worker["all_fresh_control"][field])

    profiles = {value["profile_id"]: P3Profile.from_mapping(value) for value in config["profiles"]}
    memory_values = [int(worker["memory"]["peak_aggregate_mib"])]
    if worker["memory"].get("peak_reserved_mib") is not None:
        memory_values.append(int(worker["memory"]["peak_reserved_mib"]))
    router_p99 = float(worker["overhead_mock"]["router_p99_ms"])
    reset_p99 = float(worker["overhead_mock"]["reset_p99_ms"])
    replicates = int(config["measurement"]["bootstrap_replicates"])
    bootstrap_seed = int(config["measurement"]["bootstrap_seed"])
    points = {}
    passing_points = []
    for profile_index, (profile_id, profile) in enumerate(profiles.items()):
        points[profile_id] = {}
        for horizon in (1, 2, 4):
            population = [
                row for row in blocks if row["profile_id"] == profile_id and row["horizon"] == horizon
            ]
            dense = [float(row["arms"]["dense"]["cycle_wall_ms"]) for row in population]
            cache = [float(row["arms"]["cache"]["cycle_wall_ms"]) for row in population]
            raw_point, raw_lower = one_sided_bootstrap_lower(
                dense,
                cache,
                replicates=replicates,
                seed=bootstrap_seed + profile_index * 10 + horizon,
            )
            service_count = sum(int(row["arms"]["cache"]["service_count"]) for row in population)
            fallback_count = sum(
                int(row["arms"]["cache"]["fallback_count"]) for row in population
            )
            opportunities = len(population) * horizon
            service_rate = service_count / opportunities
            overhead_fraction = max(
                (horizon * router_p99 + reset_p99)
                / float(row["arms"]["dense"]["cycle_wall_ms"])
                for row in population
            )
            total_overhead_fraction = max(
                (
                    sum(
                        float(query["gate_wall_ms"])
                        + float(query["provenance_wall_ms"])
                        for query in row["arms"]["cache"]["queries"]
                    )
                    + float(worker["sidecar_control"]["incremental_upper_ms"])
                    + horizon * router_p99
                    + reset_p99
                )
                / float(row["arms"]["dense"]["cycle_wall_ms"])
                for row in population
            )
            net_lower = net_saving_lower(raw_lower, overhead_fraction)
            source_invariants = all(
                len(row["arms"]["cache"]["final_source_digest"]) == 64
                and int(row["arms"]["cache"]["maximum_source_mixture_count"]) >= 2
                for row in population
            )
            expected_active_reduction = profile.primary_budgets[-1] + profile.wrist_budgets[-1]
            shape_invariants = all(
                all(
                    int(query["active_sequence_length"])
                    == int(query["full_sequence_length"]) - expected_active_reduction
                    for query in row["arms"]["cache"]["queries"][1:]
                )
                for row in population
            )
            memory_pass = max(memory_values) < int(worker["memory"]["strict_limit_mib"])
            diversity = profile_source_diversity(profile)
            passed = (
                raw_lower > 0
                and net_lower >= float(config["overhead"]["minimum_net_saving_fraction"])
                and service_rate >= float(config["overhead"]["minimum_service_rate"])
                and total_overhead_fraction
                <= float(config["overhead"]["maximum_total_overhead_fraction"])
                and source_invariants
                and shape_invariants
                and memory_pass
                and bool(diversity["mixed_source_possible"])
            )
            result = {
                "cycles": len(population),
                "raw_complete_cycle_saving": raw_point,
                "raw_complete_cycle_saving_one_sided_95_lower": raw_lower,
                "router_reset_overhead_fraction_upper": overhead_fraction,
                "total_feature_router_provenance_reset_overhead_fraction_upper": total_overhead_fraction,
                "net_saving_lower": net_lower,
                "service_rate": service_rate,
                "fallback_count": fallback_count,
                "effective_reuse_fraction": effective_reuse_fraction(profile),
                "source_invariants": source_invariants,
                "shape_invariants": shape_invariants,
                "source_diversity": diversity,
                "memory_pass": memory_pass,
                "passed": passed,
            }
            points[profile_id][str(horizon)] = result
            if passed:
                passing_points.append((profile_id, horizon, result))

    controls_pass = (
        worker["sidecar_control"]["cache_sample_match"]
        and int(worker["sidecar_control"]["underlying_sdpa_calls"]) == 32
        and worker["all_fresh_control"]["cache_sample_match"]
    )
    resource_pass = (
        int(worker["queries_used"]) == int(config["measurement"]["planned_model_queries"])
        and int(worker["queries_planned"])
        == int(config["measurement"]["planned_model_queries"])
        and int(worker["blocks_completed"]) == len(blocks)
        and int(worker["blocks_expected"]) == len(expected_schedule)
        and float(worker["wall_seconds"]) < int(config["resource_caps"]["wall_seconds"])
        and max(memory_values)
        < int(config["resource_caps"]["peak_gpu_memory_mib_strict_max"])
    )
    protection_pass = worker["protection"] == {
        "action_values_persisted": False,
        "action_comparisons_performed": False,
        "expert_actions_accessed": False,
        "terminal_outcomes_accessed": False,
        "simulator_used": False,
        "downloads": 0,
    }
    artifact_bytes = sum(path.stat().st_size for path in args.run_root.rglob("*") if path.is_file())
    if recovery and not blocks_path.is_relative_to(args.run_root):
        artifact_bytes += blocks_path.stat().st_size
    artifact_pass = artifact_bytes <= int(config["resource_caps"]["artifact_bytes"])
    gate_pass = bool(passing_points) and controls_pass and resource_pass and protection_pass and artifact_pass
    best = (
        max(passing_points, key=lambda item: (item[2]["net_saving_lower"], item[0], item[1]))
        if passing_points
        else None
    )
    analysis = {
        "schema_version": "pair-p3-analysis-v1",
        "run_id": config["run_id"],
        "status": "passed" if gate_pass else "scientific_stop",
        "configuration_semantic_sha256": config["semantic_sha256"],
        "input_manifest_semantic_sha256": inputs["semantic_sha256"],
        "worker_summary_sha256": file_sha256(worker_path),
        "blocks_sha256": file_sha256(blocks_path),
        "terminal_recovery": recovery,
        "queries": int(worker["queries_used"]),
        "blocks": len(blocks),
        "profiles": points,
        "passing_points": [
            {"profile_id": profile_id, "horizon": horizon, **result}
            for profile_id, horizon, result in passing_points
        ],
        "selected_timing_point": (
            {"profile_id": best[0], "horizon": best[1], **best[2]} if best else None
        ),
        "gates": {
            "at_least_one_headroom_point": bool(passing_points),
            "sidecar_sequence_all_fresh_controls": controls_pass,
            "resource_caps": resource_pass,
            "protected_population_boundary": protection_pass,
            "artifact_cap": artifact_pass,
        },
        "memory": worker["memory"],
        "overhead_mock": worker["overhead_mock"],
        "artifact_bytes": artifact_bytes,
        "interpretation_boundary": "Outcome-blind timing result only; no action quality, expert labels, simulator outcomes, or task success analyzed.",
        "advance": {"next_phase": "P4", "authorized": False, "stop_before_next_phase": True},
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    reject_protected_fields(analysis)
    analysis["semantic_sha256"] = semantic_sha256(analysis)
    write_once(args.run_root / "analysis.json", analysis)
    print(
        json.dumps(
            {
                "status": analysis["status"],
                "passing_points": len(passing_points),
                "selected_timing_point": analysis["selected_timing_point"],
            },
            sort_keys=True,
        )
    )
    return 0 if gate_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
