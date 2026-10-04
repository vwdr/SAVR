#!/usr/bin/env python3
"""CUDA-hidden readiness check for S3 Recovery 02."""

from __future__ import annotations

import dataclasses
import importlib.util
import json
from pathlib import Path


ROOT = Path("/home/ved/SAVR")
WORKER = ROOT / "scripts/run_openvla_semantic_parity_recovery02.py"
CONFIG = ROOT / "configs/openvla/semantic_parity_s3_v03_recovery02.json"


@dataclasses.dataclass(frozen=True)
class PreparedProbe:
    normalized_proprio: object


def main() -> int:
    if Path.cwd().resolve() != ROOT:
        raise SystemExit(f"S3 Recovery 02 preflight must start in {ROOT}")
    spec = importlib.util.spec_from_file_location("s3_recovery02", WORKER)
    if spec is None or spec.loader is None:
        raise SystemExit("S3 Recovery 02 worker cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    recovery01 = module.load_recovery01()
    v1 = recovery01.load_v1()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    module.validate_recovery02(config, recovery01, v1, require_authorized=True)

    import torch

    probe = PreparedProbe(torch.zeros((1, 8), dtype=torch.bfloat16))
    corrected = module.canonicalize_prepared_proprio(probe)
    if tuple(corrected.normalized_proprio.shape) != (8,):
        raise SystemExit("Recovery 02 proprio canonicalization failed")
    safe = module.json_safe_comparison(
        {"passed": False, "finite": True, "max_abs": float("inf")}
    )
    if safe != {
        "passed": False,
        "finite": True,
        "max_abs": None,
        "max_abs_nonfinite": "positive_infinity",
    }:
        raise SystemExit("Recovery 02 JSON failure encoding changed")

    print(
        json.dumps(
            {
                "ready": True,
                "authorized": True,
                "model_calls": 32,
                "simulator_outcomes": 0,
                "canonical_proprio_shape": [8],
                "strict_json_failure_records": True,
                "next_stage_authorized": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
