#!/usr/bin/env python3
"""User-coordinated aggregate-only GPU selection for the frozen V5-D run.

Do not execute this script until the user explicitly approves entry into the
one-GPU phase.  It never inspects process identities or commands.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_ROOT = Path("/home/ved/SAVR")
COORDINATION_VARIABLE = "SAVR_V5D_GPU_COORDINATION_APPROVED"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def semantic_sha256(value: dict[str, Any]) -> str:
    payload = dict(value)
    payload.pop("semantic_sha256", None)
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def git_output(root: Path, *arguments: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *arguments], text=True).strip()


def sample() -> list[dict[str, Any]]:
    fields = "index,uuid,name,driver_version,memory.total,memory.used,utilization.gpu"
    output = subprocess.check_output(
        ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
        text=True,
    )
    recorded = utc_now()
    rows = []
    for raw in output.splitlines():
        if not raw.strip():
            continue
        values = [item.strip() for item in raw.split(",")]
        if len(values) != 7:
            raise RuntimeError("Aggregate GPU telemetry schema changed")
        rows.append(
            {
                "index": int(values[0]),
                "uuid": values[1],
                "name": values[2],
                "driver_version": values[3],
                "memory_total_mib": int(values[4]),
                "memory_used_mib": int(values[5]),
                "utilization_percent": int(values[6]),
                "recorded_at_utc": recorded,
            }
        )
    if not rows or len({row["index"] for row in rows}) != len(rows):
        raise RuntimeError("Aggregate GPU selection did not return unique devices")
    return sorted(rows, key=lambda row: row["index"])


def select_index(
    samples: list[list[dict[str, Any]]], config: dict[str, Any]
) -> tuple[int, list[int]]:
    if len(samples) != config["gpu_selection"]["aggregate_samples"]:
        raise RuntimeError("V5-D aggregate sample count changed")
    indices = {row["index"] for group in samples for row in group}
    eligible = []
    for index in sorted(indices):
        observations = [
            next((row for row in group if row["index"] == index), None) for group in samples
        ]
        if any(row is None for row in observations):
            continue
        if all(
            row["utilization_percent"]
            <= config["gpu_selection"]["maximum_utilization_percent_each_sample"]
            and row["memory_used_mib"]
            <= config["gpu_selection"]["maximum_memory_used_mib_each_sample"]
            for row in observations
        ):
            eligible.append(index)
    if not eligible:
        raise RuntimeError("No GPU satisfied every frozen aggregate eligibility sample")
    return min(eligible), eligible


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    if root != EXPECTED_ROOT:
        raise SystemExit(f"Refusing GPU selection outside {EXPECTED_ROOT}: {root}")
    if os.environ.get(COORDINATION_VARIABLE) != "1":
        raise SystemExit(f"Set {COORDINATION_VARIABLE}=1 only after explicit user GPU coordination")
    if git_output(root, "status", "--porcelain"):
        raise SystemExit("Refusing V5-D GPU selection from a dirty repository")
    if git_output(root, "branch", "--show-current") != "main":
        raise SystemExit("V5-D GPU selection requires the synchronized main branch")
    sys.path.insert(0, str(root / "src"))
    from savr.acr.records import ImmutableRecordStore
    from savr.acr.v5_d_runtime import load_v5_d_freeze

    config = load_v5_d_freeze(root)
    run_root = root / "results" / config["run_id"]
    if run_root.exists():
        raise SystemExit(f"Immutable V5-D run already exists: {run_root}")
    started_at = utc_now()
    samples = []
    for index in range(config["gpu_selection"]["aggregate_samples"]):
        samples.append(sample())
        if index + 1 < config["gpu_selection"]["aggregate_samples"]:
            time.sleep(config["gpu_selection"]["seconds_between_samples"])
    try:
        selected_index, eligible = select_index(samples, config)
    except RuntimeError as error:
        raise SystemExit(str(error)) from error
    selected_samples = [
        next(row for row in group if row["index"] == selected_index) for group in samples
    ]
    identities = {(row["uuid"], row["name"], row["driver_version"]) for row in selected_samples}
    if len(identities) != 1:
        raise RuntimeError("Selected GPU identity changed across samples")
    manifest = {
        "schema_version": "acr.v5d-launch.v1",
        "run_id": config["run_id"],
        "status": "selected-not-launched",
        "started_at_utc": started_at,
        "finished_at_utc": utc_now(),
        "execution_revision": git_output(root, "rev-parse", "HEAD"),
        "configuration_semantic_sha256": config["semantic_sha256"],
        "selection_rule": config["gpu_selection"]["selection_rule"],
        "all_aggregate_samples": samples,
        "eligible_indices": eligible,
        "selected_gpu": selected_samples[-1],
        "selected_gpu_samples": selected_samples,
        "process_identities_inspected": False,
        "model_loaded": False,
        "model_queries": 0,
        "simulator_episodes": 0,
        "simulator_resets": 0,
        "downloads": 0,
        "new_task_outcomes": 0,
    }
    manifest["semantic_sha256"] = semantic_sha256(manifest)
    ImmutableRecordStore(root / "results").write_once(f"{config['run_id']}/launch", manifest)
    print(
        json.dumps(
            {
                "run_id": config["run_id"],
                "selected_gpu": manifest["selected_gpu"],
                "manifest_semantic_sha256": manifest["semantic_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
