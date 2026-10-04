#!/usr/bin/env python3
"""CUDA-hidden static and CPU matrix audit for S3 Recovery 03."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path("/home/ved/SAVR")
WORKER = ROOT / "scripts/run_openvla_semantic_parity_recovery03.py"
CONFIG = ROOT / "configs/openvla/semantic_parity_s3_v04_recovery03.json"


def main() -> int:
    if Path.cwd().resolve() != ROOT:
        raise SystemExit(f"S3 Recovery 03 preflight must start in {ROOT}")
    spec = importlib.util.spec_from_file_location("s3_recovery03", WORKER)
    if spec is None or spec.loader is None:
        raise SystemExit("S3 Recovery 03 worker cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    recovery02 = module.load_recovery02()
    recovery01 = recovery02.load_recovery01()
    v1 = recovery01.load_v1()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    module.validate_recovery03(config, recovery02, recovery01, v1)
    qualification_root = ROOT / config["comparator_qualification"]["output_root"]
    if qualification_root.exists():
        raise SystemExit("comparator qualification output already exists")

    import numpy as np
    import torch

    for boundary in sorted(module.ARRAY_BOUNDARIES):
        record = module.compare_with_contract(
            v1.compare_values,
            boundary=boundary,
            reference=np.zeros(2, dtype=np.float32),
            candidate=np.zeros(2, dtype=np.float32),
            tolerance=0.0,
            torch_module=torch,
            np_module=np,
        )
        if not record["passed"]:
            raise SystemExit(f"CPU array contract failed: {boundary}")
    print(
        json.dumps(
            {
                "ready": True,
                "authorized": True,
                "boundary_contract_count": len(module.ALL_BOUNDARIES),
                "gpu_comparator_qualification_required": True,
                "model_calls": 32,
                "simulator_outcomes": 0,
                "next_stage_authorized": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
