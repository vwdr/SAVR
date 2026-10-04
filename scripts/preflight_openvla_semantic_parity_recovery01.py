#!/usr/bin/env python3
"""Static readiness check for the proposed S3 Recovery 01 package."""

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path


ROOT = Path("/home/ved/SAVR")
CONFIG = ROOT / "configs/openvla/semantic_parity_s3_v02_recovery01.json"
WORKER = ROOT / "scripts/run_openvla_semantic_parity_recovery01.py"


def main() -> int:
    if Path.cwd().resolve() != ROOT:
        raise SystemExit(f"S3 Recovery 01 preflight must start in {ROOT}")
    spec = importlib.util.spec_from_file_location("s3_recovery01", WORKER)
    if spec is None or spec.loader is None:
        raise SystemExit("S3 Recovery 01 worker cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    v1 = module.load_v1()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    module.validate_recovery(config, v1, require_authorized=False)
    official_root = ROOT / config["loader_recovery"]["official_source_tree"]
    revision = subprocess.check_output(
        ["git", "-C", str(official_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != config["loader_recovery"]["official_source_revision"]:
        raise SystemExit("official OpenVLA-OFT source revision changed")
    checkpoint_source = (
        ROOT / config["model"]["checkpoint_relative"] / "modeling_prismatic.py"
    ).read_text(encoding="utf-8")
    official_evaluator = (
        official_root / "experiments/robot/openvla_utils.py"
    ).read_text(encoding="utf-8")
    official_config = (
        official_root / "experiments/robot/libero/run_libero_eval.py"
    ).read_text(encoding="utf-8")
    cache_model = (
        ROOT
        / "third_party/vla-cache/src/openvla-oft/prismatic/extern/hf/modeling_prismatic.py"
    ).read_text(encoding="utf-8")
    if (
        "NUM_PATCHES + NUM_PROMPT_TOKENS" not in checkpoint_source
        or "- ACTION_DIM * NUM_ACTIONS_CHUNK - 1" in checkpoint_source
        or "return [action[i] for i in range(len(action))]" not in official_evaluator
        or "use_vla_cache" in official_config
        or "- ACTION_DIM * NUM_ACTIONS_CHUNK - 1" not in cache_model
    ):
        raise SystemExit("official/cache source-surface assumptions changed")
    evaluation, originals = module.install_official_loader_guard()
    try:
        cfg = v1.base_config(
            evaluation,
            ROOT / config["model"]["checkpoint_relative"],
            ROOT / config["output_root"],
        )
        evaluation.validate_config(cfg)
        if hasattr(cfg, "use_vla_cache"):
            raise SystemExit("official evaluator config unexpectedly retained VLA-Cache")
    finally:
        for name, value in originals.items():
            setattr(evaluation, name, value)
    print(
        json.dumps(
            {
                "ready": True,
                "authorized": False,
                "model_calls": 32,
                "simulator_outcomes": 0,
                "official_checkpoint_logic_guard": True,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
