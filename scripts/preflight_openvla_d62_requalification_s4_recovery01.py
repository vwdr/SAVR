#!/usr/bin/env python3
"""CUDA-hidden static preflight for S4 Recovery 01."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path("/home/ved/SAVR")
CONFIG = ROOT / "configs/openvla/d62_requalification_s4_recovery01_v01.json"


def file_sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    if Path.cwd().resolve() != ROOT:
        raise SystemExit(f"preflight must start in {ROOT}")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    worker_path = ROOT / "scripts/run_openvla_d62_requalification_s4.py"
    spec = importlib.util.spec_from_file_location("s4_preflight_base", worker_path)
    if spec is None or spec.loader is None:
        raise SystemExit("base S4 worker cannot be loaded")
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    if config.get("semantic_sha256") != base.semantic_sha256(config):
        raise SystemExit("recovery config hash mismatch")
    if config.get("authorization", {}).get("s4_recovery01_authorized") is not True:
        raise SystemExit("S4 Recovery 01 is not authorized")
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
        raise SystemExit("recovery call schedule changed")
    for relative, expected in config["authenticated_files"].items():
        path = ROOT / relative
        if not path.is_file() or file_sha256(path) != expected:
            raise SystemExit(f"authenticated recovery input changed: {relative}")
    for relative in (
        "src/savr/brace/b3_openvla.py",
        "src/savr/pair/p3_openvla.py",
        "src/savr/pair/p4_openvla.py",
    ):
        source = (ROOT / relative).read_text(encoding="utf-8")
        if "-57:-1" in source or "cache_fork_active_positions" not in source:
            raise SystemExit(f"active compact-position contract changed: {relative}")
    if (ROOT / config["qualification"]["output_root"]).exists():
        raise SystemExit("immutable qualification output already exists")
    if (ROOT / config["output_root"]).exists():
        raise SystemExit("immutable recovery output already exists")
    print(json.dumps({
        "ready": True,
        "authorized": True,
        "compact_position_mapping": True,
        "tiny_model_calls_before_openvla": 2,
        "planned_openvla_calls": 37,
        "simulator_outcomes": 0,
        "automatic_retry": False,
        "next_stage_authorized": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
