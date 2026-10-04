#!/usr/bin/env python3
"""CUDA-hidden full static audit for S3 Recovery 05."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path("/home/ved/SAVR")
WORKER = ROOT / "scripts/run_openvla_semantic_parity_recovery05.py"
CONFIG = ROOT / "configs/openvla/semantic_parity_s3_v06_recovery05.json"


def main() -> int:
    if Path.cwd().resolve() != ROOT:
        raise SystemExit(f"S3 Recovery 05 preflight must start in {ROOT}")
    spec = importlib.util.spec_from_file_location("s3_recovery05", WORKER)
    if spec is None or spec.loader is None:
        raise SystemExit("S3 Recovery 05 worker cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    recovery04 = module.load_recovery04()
    recovery03 = recovery04.load_recovery03()
    recovery02 = recovery03.load_recovery02()
    recovery01 = recovery02.load_recovery01()
    v1 = recovery01.load_v1()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    module.validate_recovery05(
        config, recovery04, recovery03, recovery02, recovery01, v1
    )
    qualification_root = ROOT / config["full_helper_qualification"]["output_root"]
    if qualification_root.exists():
        raise SystemExit("dense-helper qualification output already exists")
    source = (ROOT / "src/savr/openvla/official_semantics.py").read_text(encoding="utf-8")
    if "if use_cache not in (None, False, True):" not in source:
        raise SystemExit("dense-helper accepted cache modes changed")
    if '"use_cache": use_cache,' not in source:
        raise SystemExit("dense-helper cache forwarding changed")
    received = []

    def probe(*values, **kwargs):
        received.append(kwargs["use_cache"])
        return {"cache": kwargs["use_cache"]}

    if recovery04.explicit_no_cache_forward(probe, use_cache=None)["cache"] is not False:
        raise SystemExit("Recovery 05 no-cache translation failed")
    if recovery04.explicit_no_cache_forward(probe, use_cache=True)["cache"] is not True:
        raise SystemExit("Recovery 05 True cache control changed")
    if received != [False, True]:
        raise SystemExit("Recovery 05 wrapper sequence changed")
    print(
        json.dumps(
            {
                "ready": True,
                "authorized": True,
                "full_helper_gpu_qualification_required": True,
                "prior_runtime_cache_qualification_verified": True,
                "prior_comparator_qualification_verified": True,
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
