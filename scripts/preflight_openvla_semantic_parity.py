#!/usr/bin/env python3
"""Outcome-free static preflight for OpenVLA semantic parity S3."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path("/home/ved/SAVR")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/openvla/semantic_parity_s3_v1.json"),
    )
    args = parser.parse_args()
    if Path.cwd().resolve() != ROOT:
        raise SystemExit(f"semantic-parity preflight must start in {ROOT}")
    sys.path.insert(0, str(ROOT / "scripts"))
    from run_openvla_semantic_parity import validate_static

    config = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    validate_static(config)
    print(
        json.dumps(
            {
                "ready": True,
                "run_id": config["run_id"],
                "observations": config["resource_caps"]["observations"],
                "model_calls": config["resource_caps"]["model_call_hard_cap"],
                "simulator_outcomes": 0,
                "next_stage_authorized": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
