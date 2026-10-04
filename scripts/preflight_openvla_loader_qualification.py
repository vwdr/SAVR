#!/usr/bin/env python3
"""CUDA-hidden static audit for the zero-call OpenVLA loader qualification."""

from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path


ROOT = Path("/home/ved/SAVR")
WORKER = ROOT / "scripts/run_openvla_loader_qualification.py"
CONFIG = ROOT / "configs/openvla/loader_qualification_s3l_v1.json"


def prohibited_calls(source: str) -> list[str]:
    prohibited: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if isinstance(function, ast.Name) and function.id in {
            "model",
            "action_head",
            "get_libero_env",
        }:
            prohibited.append(function.id)
        elif isinstance(function, ast.Attribute) and function.attr in {
            "get_action",
            "forward",
            "predict_action",
            "step",
        }:
            prohibited.append(function.attr)
    return prohibited


def main() -> int:
    if Path.cwd().resolve() != ROOT:
        raise SystemExit(f"loader preflight must start in {ROOT}")
    spec = importlib.util.spec_from_file_location("openvla_loader_qualification", WORKER)
    if spec is None or spec.loader is None:
        raise SystemExit("loader qualification worker cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    module.validate_config(config, require_authorized=True)
    worker_text = WORKER.read_text(encoding="utf-8")
    if prohibited_calls(worker_text):
        raise SystemExit("loader qualification contains a prohibited policy/outcome call")
    print(
        json.dumps(
            {
                "ready": True,
                "authorized": True,
                "policy_calls": 0,
                "actions": 0,
                "simulator_outcomes": 0,
                "next_stage_authorized": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
