#!/usr/bin/env python3
"""Select one idle TITAN GPU from aggregate telemetry for frozen BRACE-B3."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ROOT = Path("/home/ved/SAVR")
CONFIG = Path("configs/brace/b3_physical_v1.json")
RUN = Path("results/brace-b3-physical-v01")


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def semantic_sha256(value: dict[str, Any]) -> str:
    payload = dict(value)
    payload.pop("semantic_sha256", None)
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def sample() -> list[dict[str, Any]]:
    fields = "index,uuid,name,driver_version,memory.total,memory.used,utilization.gpu"
    output = subprocess.check_output(
        ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"], text=True
    )
    rows = []
    for row in csv.reader(io.StringIO(output), skipinitialspace=True):
        if len(row) != 7:
            raise RuntimeError("Unexpected aggregate nvidia-smi response")
        rows.append(
            {
                "index": int(row[0]),
                "uuid": row[1],
                "name": row[2],
                "driver_version": row[3],
                "memory_total_mib": int(row[4]),
                "memory_used_mib": int(row[5]),
                "utilization_percent": int(row[6]),
            }
        )
    return rows


def main() -> int:
    if ROOT != EXPECTED_ROOT or Path.cwd().resolve() != EXPECTED_ROOT:
        raise SystemExit(f"B3 GPU selection is restricted to {EXPECTED_ROOT}")
    config = json.loads((ROOT / CONFIG).read_text())
    if config["semantic_sha256"] != "30825fd30eb2c0566b30564740a212c51e09dc60f7e9c4bc7315b24b369dea11":
        raise SystemExit("B3 frozen configuration changed")
    launch_path = ROOT / RUN / "launch.json"
    if launch_path.exists():
        raise SystemExit("Immutable B3 launch record already exists")
    snapshots = []
    for ordinal in range(3):
        snapshots.append(
            {"sample": ordinal, "captured_at_utc": datetime.now(timezone.utc).isoformat(), "gpus": sample()}
        )
        if ordinal < 2:
            time.sleep(2)
    by_index: dict[int, list[dict[str, Any]]] = {}
    for snapshot in snapshots:
        for gpu in snapshot["gpus"]:
            by_index.setdefault(gpu["index"], []).append(gpu)
    eligible = [
        values[-1]
        for _, values in sorted(by_index.items())
        if len(values) == 3
        and all(item["memory_used_mib"] <= 512 for item in values)
        and all(item["utilization_percent"] <= 5 for item in values)
    ]
    if not eligible:
        raise SystemExit("No GPU met the frozen aggregate-idle rule; B3 was not launched")
    selected = eligible[0]
    record = {
        "schema_version": "brace.b3-launch.v1",
        "run_id": config["run_id"],
        "configuration_semantic_sha256": config["semantic_sha256"],
        "source_revision": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        "selected_gpu": selected,
        "selection_rule": {"samples": 3, "maximum_memory_used_mib": 512, "maximum_utilization_percent": 5},
        "aggregate_snapshots": snapshots,
        "process_identity_inspection": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    record["semantic_sha256"] = semantic_sha256(record)
    launch_path.parent.mkdir(parents=True, exist_ok=False)
    descriptor = os.open(launch_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(canonical_bytes(record) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"status": "selected", "index": selected["index"], "uuid": selected["uuid"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
