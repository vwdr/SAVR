#!/usr/bin/env python3
"""Freeze schedule, full preflight, and executable identities for one P4B attempt."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ROOT = Path("/home/ved/SAVR")


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_once(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(canonical_bytes(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    if ROOT != EXPECTED_ROOT or os.environ.get("CUDA_VISIBLE_DEVICES", ""):
        raise SystemExit("P4B launch freeze requires project root and hidden CUDA")
    sys.path.insert(0, str(ROOT / "src"))
    from savr.pair.p4b import semantic_sha256, validate_config

    config_path = ROOT / args.config
    config = json.loads(config_path.read_text())
    validate_config(config, ROOT)
    schedule_path = ROOT / config["artifacts"]["schedule"]
    summary_path = ROOT / config["artifacts"]["schedule_summary"]
    preflight_path = ROOT / config["artifacts"]["preflight"]
    launch_path = ROOT / config["artifacts"]["launch_manifest"]
    preflight = json.loads(preflight_path.read_text())
    summary = json.loads(summary_path.read_text())
    if (
        launch_path.exists()
        or preflight.get("status") != "passed"
        or preflight.get("mode") != "full"
        or preflight.get("configuration_semantic_sha256") != config["semantic_sha256"]
        or preflight["schedule_validation"]["schedule_sha256"] != summary["schedule_sha256"]
        or (ROOT / config["artifacts"]["run_root"]).exists()
    ):
        raise SystemExit("P4B launch inputs are incomplete, changed, or already consumed")
    code = (
        "scripts/run_pair_p4b_worker.py",
        "scripts/seal_pair_p4b_worker.py",
        "scripts/analyze_pair_p4b.py",
        "src/savr/pair/p4b.py",
        "src/savr/pair/p4_openvla.py",
    )
    manifest = {
        "schema_version": "pair-p4b-launch-manifest-v1",
        "run_id": config["run_id"],
        "status": "frozen_not_started",
        "configuration_semantic_sha256": config["semantic_sha256"],
        "config_sha256": file_sha256(config_path),
        "schedule_sha256": file_sha256(schedule_path),
        "schedule_summary_sha256": file_sha256(summary_path),
        "preflight_sha256": file_sha256(preflight_path),
        "code_sha256": {path: file_sha256(ROOT / path) for path in code},
        "model_call_hard_cap": config["resource_caps"]["model_call_hard_cap"],
        "automatic_retry": False,
        "locked_test_use": False,
        "p5_authorized": False,
    }
    manifest["semantic_sha256"] = semantic_sha256(manifest)
    write_once(launch_path, manifest)
    print(json.dumps({"status": "frozen_not_started"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
