#!/usr/bin/env python3
"""CUDA-hidden full-path audit for S3 Recovery 04."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path("/home/ved/SAVR")
WORKER = ROOT / "scripts/run_openvla_semantic_parity_recovery04.py"
CONFIG = ROOT / "configs/openvla/semantic_parity_s3_v05_recovery04.json"


def main() -> int:
    if Path.cwd().resolve() != ROOT:
        raise SystemExit(f"S3 Recovery 04 preflight must start in {ROOT}")
    spec = importlib.util.spec_from_file_location("s3_recovery04", WORKER)
    if spec is None or spec.loader is None:
        raise SystemExit("S3 Recovery 04 worker cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    recovery03 = module.load_recovery03()
    recovery02 = recovery03.load_recovery02()
    recovery01 = recovery02.load_recovery01()
    v1 = recovery01.load_v1()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    module.validate_recovery04(config, recovery03, recovery02, recovery01, v1)
    qualification_root = ROOT / config["cache_mode_qualification"]["output_root"]
    if qualification_root.exists():
        raise SystemExit("cache-mode qualification output already exists")
    source = (
        ROOT
        / "envs/vla-cache-compat/lib/python3.10/site-packages/transformers/models/llama/modeling_llama.py"
    ).read_text(encoding="utf-8")
    if "use_cache = use_cache if use_cache is not None else self.config.use_cache" not in source:
        raise SystemExit("pinned Transformers cache resolution changed")
    received = []

    def probe(*values, **kwargs):
        received.append(kwargs["use_cache"])
        return {"cache": kwargs["use_cache"]}

    if module.explicit_no_cache_forward(probe, use_cache=None)["cache"] is not False:
        raise SystemExit("Recovery 04 no-cache translation failed")
    if module.explicit_no_cache_forward(probe, use_cache=True)["cache"] is not True:
        raise SystemExit("Recovery 04 True cache control changed")
    if received != [False, True]:
        raise SystemExit("Recovery 04 wrapper sequence changed")
    print(
        json.dumps(
            {
                "ready": True,
                "authorized": True,
                "post_comparison_path_audited": True,
                "gpu_cache_mode_qualification_required": True,
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
