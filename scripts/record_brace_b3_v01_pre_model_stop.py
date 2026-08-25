#!/usr/bin/env python3
"""Seal the immutable zero-query B3-v01 launch-guard technical stop."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ROOT = Path("/home/ved/SAVR")
RUN = ROOT / "results/brace-b3-physical-v01"
RECORD = ROOT / "reports/runtime/brace_b3_v01_technical_stop.json"


def canonical_bytes(value: dict) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def main() -> int:
    if ROOT != EXPECTED_ROOT or Path.cwd().resolve() != EXPECTED_ROOT:
        raise SystemExit(f"B3 stop sealer is restricted to {EXPECTED_ROOT}")
    launch_path = RUN / "launch.json"
    target = RUN / "technical_stop.json"
    if target.exists() or (RUN / "workers").exists() or (RUN / "run_summary.json").exists():
        raise SystemExit("B3-v01 terminal or worker evidence already exists")
    record = json.loads(RECORD.read_text())
    launch = json.loads(launch_path.read_text())
    semantic = dict(record)
    recorded_semantic = semantic.pop("semantic_sha256")
    if hashlib.sha256(canonical_bytes(semantic)).hexdigest() != recorded_semantic:
        raise SystemExit("B3-v01 technical-stop semantic identity changed")
    if hashlib.sha256(launch_path.read_bytes()).hexdigest() != record["launch_file_sha256"]:
        raise SystemExit("B3-v01 launch file changed")
    if launch["semantic_sha256"] != record["launch_semantic_sha256"]:
        raise SystemExit("B3-v01 launch semantic identity changed")
    payload = canonical_bytes(record) + b"\n"
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"status": record["status"], "model_queries": 0}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
