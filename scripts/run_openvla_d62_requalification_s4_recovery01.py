#!/usr/bin/env python3
"""Versioned S4 Recovery 01 wrapper with compact-position qualification gate."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Mapping


ROOT = Path("/home/ved/SAVR")
BASE_WORKER = ROOT / "scripts/run_openvla_d62_requalification_s4.py"


def load_base() -> Any:
    spec = importlib.util.spec_from_file_location("s4_v01_for_recovery01", BASE_WORKER)
    if spec is None or spec.loader is None:
        raise RuntimeError("S4 v01 worker cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_recovery(config: Mapping[str, Any], v1: Any) -> None:
    base = load_base()
    if config.get("schema_version") != "openvla-d62-requalification-s4-recovery01-v1":
        raise base.S4TechnicalStop("S4 Recovery 01 schema changed")
    if config.get("semantic_sha256") != base.semantic_sha256(config):
        raise base.S4TechnicalStop("S4 Recovery 01 config hash mismatch")
    expected_authorization = {
        "s4_recovery01_authorized": True,
        "source": "User approved S4 Recovery 01 on 2026-09-01",
        "simulator": False,
        "terminal_outcomes": False,
        "expert_actions": False,
        "training": False,
        "automatic_retry": False,
    }
    if config.get("authorization") != expected_authorization:
        raise base.S4TechnicalStop("S4 Recovery 01 authorization changed")
    if config.get("output_root") != "results/openvla-d62-requalification-s4-v02-recovery01":
        raise base.S4TechnicalStop("S4 Recovery 01 output root changed")
    if config.get("population") != {
        "selection_audit_observations": 8,
        "allfresh_official_controls": 8,
        "recursive_windows": 1,
        "recursive_repetitions": 2,
        "recursive_ages": [1, 2, 3, 4],
        "cadence_steps": 8,
        "terminal_outcomes_accessed": False,
        "expert_actions_accessed": False,
    }:
        raise base.S4TechnicalStop("S4 Recovery 01 population changed")
    if config.get("call_schedule") != {
        "suite_balanced_anchor_calls": 8,
        "suite_balanced_official_calls": 8,
        "suite_balanced_allfresh_calls": 8,
        "recursive_anchor_calls": 2,
        "recursive_transition_calls": 8,
        "reset_official_calls": 1,
        "reset_custom_calls": 2,
        "planned_model_calls": 37,
    }:
        raise base.S4TechnicalStop("S4 Recovery 01 call schedule changed")
    if config.get("substrate") != {
        "historical_id": "D62_BAL_PT1",
        "corrected_candidate_id": "D62_BAL_PT1_S4C_V1",
        "canonical_action_positions": True,
        "instruction_only_positions": True,
        "compact_positions_explicitly_mapped": True,
        "selection_materiality_rule": (
            "material if any age-1 ordered position, onset assignment, or protected tile "
            "differs across the eight frozen observations"
        ),
        "freeze_new_identity": True,
    }:
        raise base.S4TechnicalStop("S4 Recovery 01 substrate changed")
    if config.get("resource_caps") != {
        "gpu_count": 1,
        "model_processes": 1,
        "model_call_hard_cap": 48,
        "planned_model_calls": 37,
        "wall_seconds": 1800,
        "artifact_bytes": 268435456,
        "peak_gpu_memory_mib_strict_max": 23552,
        "downloads": 0,
        "simulator_outcomes": 0,
        "raw_actions_persisted": False,
        "automatic_retry": False,
    }:
        raise base.S4TechnicalStop("S4 Recovery 01 resource boundary changed")
    if config.get("advance") != {
        "next_stage": "S5_CAC_C1_REQUALIFICATION",
        "authorized": False,
        "stop_before_next_stage": True,
    }:
        raise base.S4TechnicalStop("S4 Recovery 01 advance boundary changed")
    parent = config.get("technical_parent", {})
    for key in ("stop", "traceback"):
        path = ROOT / parent[key]
        if not path.is_file() or base.file_sha256(path) != parent[f"{key}_sha256"]:
            raise base.S4TechnicalStop(f"S4 Recovery 01 parent {key} changed")
    for relative, expected in config["authenticated_files"].items():
        path = ROOT / relative
        if not path.is_file() or base.file_sha256(path) != expected:
            raise base.S4TechnicalStop(f"authenticated recovery input changed: {relative}")
    s3_path = ROOT / config["s3_parent"]["summary"]
    s3 = json.loads(s3_path.read_text(encoding="utf-8"))
    if (
        base.file_sha256(s3_path) != config["s3_parent"]["summary_sha256"]
        or s3.get("semantic_sha256") != base.semantic_sha256(s3)
        or s3.get("complete") is not True
        or s3.get("passed") is not True
        or s3.get("observations_completed") != 8
        or s3.get("model_calls") != 32
        or s3.get("comparison_count") != 344
        or float(s3.get("maximum_abs_error", 1.0)) > 0.000001
    ):
        raise base.S4TechnicalStop("S3 parent did not pass")
    manifest = json.loads((ROOT / config["input_manifest"]).read_text(encoding="utf-8"))
    if (
        manifest.get("semantic_sha256") != base.semantic_sha256(manifest)
        or len(manifest.get("inputs", [])) != 8
        or {row["suite"] for row in manifest["inputs"]}
        != {"libero_spatial", "libero_object", "libero_goal", "libero_10"}
        or manifest.get("terminal_outcome_fields_accessed") is not False
        or manifest.get("expert_action_fields_accessed") is not False
    ):
        raise base.S4TechnicalStop("S4 Recovery 01 input manifest changed")
    data_root = ROOT / manifest["data_root_relative"]
    for row in manifest["inputs"]:
        source = data_root / row["source_path"]
        if not source.is_file() or base.file_sha256(source) != row["source_sha256"]:
            raise base.S4TechnicalStop("S4 Recovery 01 observation source changed")
    checkpoint = ROOT / config["model"]["checkpoint_relative"]
    for name, expected in config["model"]["checkpoint_metadata_sha256"].items():
        if not (checkpoint / name).is_file() or base.file_sha256(checkpoint / name) != expected:
            raise base.S4TechnicalStop(f"S4 Recovery 01 checkpoint changed: {name}")
    qualification_path = ROOT / config["qualification"]["summary"]
    if not qualification_path.is_file():
        raise base.S4TechnicalStop("compact-position qualification is absent")
    qualification = json.loads(qualification_path.read_text(encoding="utf-8"))
    if (
        qualification.get("semantic_sha256") != base.semantic_sha256(qualification)
        or qualification.get("complete") is not True
        or qualification.get("passed") is not True
        or qualification.get("config_sha256") != config["semantic_sha256"]
        or qualification.get("model_calls") != 2
        or qualification.get("openvla_checkpoint_loads") != 0
        or qualification.get("active_sequence_length") >= qualification.get("full_sequence_length")
        or qualification.get("action_positions_survived") is not True
        or qualification.get("sentinel_mapping_exact") is not True
        or qualification.get("malformed_contracts_rejected") != 3
    ):
        raise base.S4TechnicalStop("compact-position qualification did not pass")
    if (ROOT / config["output_root"]).exists():
        raise base.S4TechnicalStop("immutable S4 Recovery 01 output already exists")


def main() -> int:
    base = load_base()
    base.validate_config = validate_recovery
    return int(base.main())


if __name__ == "__main__":
    raise SystemExit(main())
