#!/usr/bin/env python3
"""CUDA-hidden full static audit for S4 D62 requalification."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path("/home/ved/SAVR")
WORKER = ROOT / "scripts/run_openvla_d62_requalification_s4.py"
CONFIG = ROOT / "configs/openvla/d62_requalification_s4_v01.json"


def main() -> int:
    if Path.cwd().resolve() != ROOT:
        raise SystemExit(f"S4 preflight must start in {ROOT}")
    spec = importlib.util.spec_from_file_location("openvla_d62_s4", WORKER)
    if spec is None or spec.loader is None:
        raise SystemExit("S4 worker cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    recovery05 = module.load_recovery05()
    recovery04 = recovery05.load_recovery04()
    recovery03 = recovery04.load_recovery03()
    recovery02 = recovery03.load_recovery02()
    recovery01 = recovery02.load_recovery01()
    v1 = recovery01.load_v1()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    module.validate_config(config, v1)
    active = (
        ROOT / "src/savr/brace/b3_openvla.py",
        ROOT / "src/savr/pair/p3_openvla.py",
        ROOT / "src/savr/pair/p4_openvla.py",
    )
    for path in active:
        source = path.read_text(encoding="utf-8")
        if "-57:-1" in source:
            raise SystemExit(f"active shifted action slice remains: {path.name}")
        if "select_official_action_hidden" not in source:
            raise SystemExit(f"active path does not use canonical action selection: {path.name}")
    p3_source = active[1].read_text(encoding="utf-8")
    if "derive_semantic_runtime_positions" not in p3_source:
        raise SystemExit("D62 does not use canonical semantic positions")
    if (ROOT / config["output_root"]).exists():
        raise SystemExit("immutable S4 output already exists")
    print(
        json.dumps(
            {
                "ready": True,
                "authorized": True,
                "s3_parent_verified": True,
                "active_shifted_slices": 0,
                "canonical_action_and_instruction_positions": True,
                "planned_model_calls": 37,
                "model_call_hard_cap": 48,
                "simulator_outcomes": 0,
                "next_stage_authorized": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
