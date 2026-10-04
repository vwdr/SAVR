#!/usr/bin/env python3
"""Reconcile completed CAC C1H evidence and mechanically apply Gate H."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
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
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("semantic_sha256") != semantic_sha256(value):
        raise RuntimeError(f"semantic hash mismatch: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    for row in rows:
        if row.get("semantic_sha256") != semantic_sha256(row):
            raise RuntimeError(f"terminal semantic hash mismatch: {path}")
    return rows


def read_schedule(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    for row in rows:
        payload = dict(row)
        observed = payload.pop("schedule_id", None)
        if observed != semantic_sha256(payload):
            raise RuntimeError(f"schedule identity mismatch: {path}")
    return rows


def write_once(path: Path, value: Mapping[str, Any]) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(canonical_bytes(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    if not run_root.is_relative_to(ROOT / "results"):
        raise SystemExit("analysis input must remain below project results")
    sys.path.insert(0, str(ROOT / "src"))
    from savr.cac.c1h import cumulative_decision, frozen_schedule, stage1_decision, summarize

    if (run_root / "technical_stop.json").exists():
        raise RuntimeError("C1H run contains a technical stop")
    manifest = read_json(run_root / "worker_manifest.json")
    config_path = (ROOT / manifest["config"]).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("semantic_sha256") != semantic_sha256(config):
        raise RuntimeError("C1H config semantic hash mismatch")
    if file_sha256(config_path) != manifest["config_sha256"]:
        raise RuntimeError("C1H worker/config hash mismatch")
    if config["run_id"] != manifest["run_id"]:
        raise RuntimeError("C1H worker/config run identity mismatch")
    for relative, expected in config["code"].items():
        if file_sha256(ROOT / relative) != expected:
            raise RuntimeError(f"C1H frozen code changed: {relative}")
    population_path = ROOT / config["population_manifest"]
    if file_sha256(population_path) != config["population_manifest_sha256"]:
        raise RuntimeError("C1H population manifest changed")
    conditions = [
        json.loads(line)
        for line in population_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    expected_stage1 = list(
        frozen_schedule(
            conditions,
            population=config["populations"]["stage1"],
            seed=int(config["schedule"]["seed"]),
        )
    )
    expected_extension = list(
        frozen_schedule(
            conditions,
            population=config["populations"]["extension"],
            seed=int(config["schedule"]["seed"]) + 1,
        )
    )
    stage1_schedule = read_schedule(run_root / "stage1_schedule.jsonl")
    extension_schedule = read_schedule(run_root / "extension_schedule.jsonl")
    if stage1_schedule != expected_stage1 or extension_schedule != expected_extension:
        raise RuntimeError("C1H persisted schedule differs from the frozen C0 schedule")
    worker = read_json(run_root / "worker_summary.json")
    if worker.get("status") != "completed":
        raise RuntimeError("C1H worker is not complete")
    stage1 = read_jsonl(run_root / "stage1_terminal_records.jsonl")
    if len(stage1) != 240 or file_sha256(run_root / "stage1_terminal_records.jsonl") != worker[
        "stage1_terminal_records_sha256"
    ]:
        raise RuntimeError("C1H stage1 count/hash mismatch")
    terminal_schedule_fields = tuple(stage1_schedule[0])
    if any(
        {key: row[key] for key in terminal_schedule_fields} != scheduled
        for row, scheduled in zip(stage1, stage1_schedule, strict=True)
    ):
        raise RuntimeError("C1H stage1 terminals do not match the frozen episode schedule")
    stage1_summary = summarize(stage1, expected_conditions=120)
    initial = stage1_decision(stage1_summary)
    stage1_report = read_json(run_root / "stage1_summary.json")
    if (
        stage1_report["summary"] != stage1_summary
        or stage1_report["decision"] != initial
        or stage1_report["terminal_records_sha256"]
        != file_sha256(run_root / "stage1_terminal_records.jsonl")
    ):
        raise RuntimeError("C1H Stage-1 report does not reconcile")
    extension_opened = bool(worker["extension_opened"])
    if extension_opened != (initial == "extend"):
        raise RuntimeError("C1H extension did not follow the frozen ambiguity rule")
    rows = stage1
    if extension_opened:
        extension = read_jsonl(run_root / "extension_terminal_records.jsonl")
        if len(extension) != 240 or file_sha256(
            run_root / "extension_terminal_records.jsonl"
        ) != worker["extension_terminal_records_sha256"]:
            raise RuntimeError("C1H extension count/hash mismatch")
        if any(
            {key: row[key] for key in terminal_schedule_fields} != scheduled
            for row, scheduled in zip(extension, extension_schedule, strict=True)
        ):
            raise RuntimeError("C1H extension terminals do not match the frozen schedule")
        rows += extension
        final_summary = summarize(rows, expected_conditions=240)
        final_decision = cumulative_decision(final_summary)
        extension_report = read_json(run_root / "extension_summary.json")
        if extension_report["summary"] != final_summary or extension_report[
            "decision"
        ] != final_decision:
            raise RuntimeError("C1H extension report does not reconcile")
    else:
        if (run_root / "extension_terminal_records.jsonl").exists() or (
            run_root / "extension_summary.json"
        ).exists():
            raise RuntimeError("C1H extension artifacts exist without the ambiguity decision")
        final_summary = stage1_summary
        final_decision = initial
    progress_count = len(
        (run_root / "progress.jsonl").read_text(encoding="utf-8").splitlines()
    )
    restoration = worker.get("checkpoint_restoration", {})
    controls = worker.get("technical_controls", {})
    corrected_s6 = config.get("schema_version") in {
        "cac-c1h-corrected-s6-v1", "cac-c1h-corrected-s6-recovery01-v1"
    }
    gates = {
        "worker_complete": worker["status"] == "completed",
        "exact_episode_count": len(rows) == int(worker["terminal_episode_records"]),
        "extension_rule_obeyed": extension_opened == (initial == "extend"),
        "worker_decision_matches": worker["gate_h_decision"] == final_decision,
        "c2_not_started": worker["c2_started"] is False,
        "progress_count_matches": progress_count == len(rows),
        "queries_within_frozen_cap": int(config["query_accounting"]["technical_control_queries"])
        <= int(worker["model_queries"])
        <= int(config["resource_caps"]["model_queries"]),
        "memory_within_frozen_cap": int(worker["peak_aggregate_gpu_memory_mib"])
        < int(config["resource_caps"]["peak_gpu_memory_mib_strict_max"]),
        "official_dense_parity": controls.get("queries")
        == int(config["query_accounting"]["technical_control_queries"])
        and controls.get("official_helper_max_abs", 1.0) <= 1e-6
        and controls.get("official_observation_contract")
        == "current images copied into prev_images"
        and controls.get("official_return_contract") == "actions-cache-images-metrics"
        and controls.get("d62_anchor_and_reuse") is True
        and controls.get("terminal_outcome_access") is False,
        "corrected_official_boundary": (
            not corrected_s6
            or (
                controls.get("independent_official_oracle") is True
                and controls.get("observation_isolated") is True
                and controls.get("official_hidden_max_abs", 1.0) <= 1e-6
                and controls.get("official_normalized_action_max_abs", 1.0) <= 1e-6
                and controls.get("official_helper_max_abs", 1.0) <= 1e-6
                and config.get("cache_substrate", {}).get("profile")
                == "D62_BAL_PT1_S4C_V1"
            )
        ),
        "checkpoint_restored": all(
            restoration.get(key) is True
            for key in ("protected_bytes_restored", "backup_cleanup_complete", "inventory_equal")
        ),
        "outcome_sealing_attested": worker.get(
            "outcomes_opened_only_after_exact_stage_completion"
        )
        is True,
    }
    result = {
        "schema_version": "cac-c1h-analysis-v1",
        "run_id": worker["run_id"],
        "status": "pass" if all(gates.values()) and final_decision == "proceed" else "stop",
        "gate_h_decision": final_decision,
        "stage1_decision": initial,
        "extension_opened": extension_opened,
        "terminal_episode_records": len(rows),
        "model_queries": worker["model_queries"],
        "peak_aggregate_gpu_memory_mib": worker["peak_aggregate_gpu_memory_mib"],
        "summary": final_summary,
        "reconciliation_gates": gates,
        "next_phase": "C2" if final_decision == "proceed" else None,
        "next_phase_authorized": False,
        "config_sha256": file_sha256(config_path),
        "schedule_hashes": {
            "stage1": file_sha256(run_root / "stage1_schedule.jsonl"),
            "extension": file_sha256(run_root / "extension_schedule.jsonl"),
        },
    }
    result["semantic_sha256"] = semantic_sha256(result)
    write_once(run_root / "analysis.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if all(gates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
