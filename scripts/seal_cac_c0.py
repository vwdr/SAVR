#!/usr/bin/env python3
"""Seal the completed CAC C0 evidence after independent reconciliation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports/CAC_C0_SEMANTIC_MANIFEST.json"
RECOVERY_ROOT = ROOT / "reports/cac_c0_recovery01"

REQUIRED = [
    "PROJECT_STATUS.md",
    "docs/DECISIONS.md",
    "docs/CACHE_ACTION_CORRECTION_EXECUTION_PROTOCOL_V2.md",
    "docs/CACHE_ACTION_CORRECTION_FINAL_PREEXECUTION_AUDIT_2026-08-30.md",
    "docs/CAC_C0_LITERATURE_COLLISION_AUDIT_2026-08-30.md",
    "configs/cac/c0_freeze_v1.json",
    "configs/cac/c0_recovery01.json",
    "schemas/cac/feature_record.schema.json",
    "scripts/freeze_cac_c0.py",
    "scripts/analyze_cac_c7.py",
    "scripts/simulate_cac_c0_power.py",
    "scripts/seal_cac_c0.py",
    "tests/test_cac_c0.py",
    "reports/CAC_C0_TECHNICAL_STOP_01.md",
    "reports/CAC_C0_TECHNICAL_STOP_01.json",
    "reports/CAC_C0_REPORT.md",
    "reports/cac_c0_power_simulation_v1.json",
    "reports/cac_c0_recovery01/trajectory_roles_v1.jsonl",
    "reports/cac_c0_recovery01/c2_contracts_v1.jsonl",
    "reports/cac_c0_recovery01/c3_contracts_v1.jsonl",
    "reports/cac_c0_recovery01/c7_locked_contracts_v1.jsonl",
    "reports/cac_c0_recovery01/simulator_populations_v1.jsonl",
    "reports/cac_c0_recovery01/power_table_v1.json",
    "reports/cac_c0_recovery01/freeze_summary_v1.json",
]


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_semantic(payload: Mapping[str, Any]) -> None:
    body = dict(payload)
    declared = body.pop("semantic_sha256", None)
    if declared != hashlib.sha256(canonical(body)).hexdigest():
        raise RuntimeError("semantic hash mismatch")


def main() -> None:
    if OUTPUT.exists():
        raise RuntimeError("immutable C0 semantic manifest already exists")
    if not (ROOT / "reports/cac_c0").is_dir() or any((ROOT / "reports/cac_c0").iterdir()):
        raise RuntimeError("first technical-stop root is not preserved empty")
    for relative in REQUIRED:
        if not (ROOT / relative).is_file():
            raise RuntimeError(f"missing C0 evidence: {relative}")

    recovery = json.loads((ROOT / "configs/cac/c0_recovery01.json").read_text())
    stop = json.loads((ROOT / "reports/CAC_C0_TECHNICAL_STOP_01.json").read_text())
    power = json.loads((ROOT / "reports/cac_c0_power_simulation_v1.json").read_text())
    summary = json.loads((RECOVERY_ROOT / "freeze_summary_v1.json").read_text())
    for payload in (recovery, stop, power, summary):
        verify_semantic(payload)
    if summary["status"] != "completed" or summary["decision"] != "pass_stop_before_c1":
        raise RuntimeError("C0 recovery did not complete with the frozen decision")
    if summary["advance"] != {"next_phase": "C1", "authorized": False, "stop_before_next_phase": True}:
        raise RuntimeError("C1 authorization boundary changed")
    if summary["resources"] != {
        "gpu_used": False, "model_loaded": False, "simulator_used": False,
        "terminal_outcomes_accessed": False, "downloads": 0,
    }:
        raise RuntimeError("C0 resource/protection boundary changed")
    if summary["locked_boundaries"] != {
        "locked_test_action_values_opened": False,
        "state_ids_10_49_outcomes_opened": False,
        "locked_transition_labels_present": False,
    }:
        raise RuntimeError("C0 locked boundary changed")
    for name, metadata in summary["files"].items():
        if sha256_file(RECOVERY_ROOT / name) != metadata["sha256"]:
            raise RuntimeError(f"recovery artifact changed: {name}")

    files = {
        relative: {"bytes": (ROOT / relative).stat().st_size, "sha256": sha256_file(ROOT / relative)}
        for relative in REQUIRED
    }
    payload = {
        "schema_version": "cac-c0-semantic-manifest-v1",
        "date": "2026-08-31",
        "status": "complete_pass_stop_before_c1",
        "recovery": "c0_recovery01",
        "file_count": len(files),
        "files": files,
        "counts": summary["counts"],
        "model_calls": summary["model_calls"],
        "storage_bytes": summary["storage_bytes"],
        "transition_shortfalls": summary["transition_shortfalls"],
        "protection": summary["locked_boundaries"],
        "resources": summary["resources"],
        "c1_authorized": False,
    }
    payload["semantic_sha256"] = hashlib.sha256(canonical(payload)).hexdigest()
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
