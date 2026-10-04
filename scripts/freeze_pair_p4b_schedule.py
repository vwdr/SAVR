#!/usr/bin/env python3
"""Create the outcome-blind PAIR P4B schedule without opening array values."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ROOT = Path("/home/ved/SAVR")


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def file_sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_once(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if ROOT != EXPECTED_ROOT:
        raise SystemExit(f"P4B scheduler refuses to run outside {EXPECTED_ROOT}")
    if os.environ.get("CUDA_VISIBLE_DEVICES", ""):
        raise SystemExit("P4B scheduler requires CUDA to be hidden")
    sys.path.insert(0, str(ROOT / "src"))
    from savr.pair.p4b import expected_counts, semantic_sha256, slot_spec, validate_config

    config = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    validate_config(config, ROOT)
    trajectory_path = ROOT / config["authenticated_inputs"]["trajectory_index"]["path"]
    p4_path = ROOT / config["authenticated_inputs"]["p4_schedule"]["path"]
    trajectories = [json.loads(line) for line in trajectory_path.read_text().splitlines()]
    p4_rows = [json.loads(line) for line in p4_path.read_text().splitlines()]
    p4_ids = {row["trajectory_id"] for row in p4_rows}
    if len(trajectories) != 2000 or len(p4_ids) != 400:
        raise RuntimeError("P4B predecessor population changed")

    remaining: dict[tuple[str, str], dict[str, list[Mapping[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    locked_ids = set()
    for row in trajectories:
        if row["split"] == "locked_test":
            locked_ids.add(row["trajectory_id"])
        if row["trajectory_id"] not in p4_ids:
            remaining[(row["suite"], row["task_id"])][row["split"]].append(row)
    tasks = sorted(remaining)
    if len(tasks) != 40 or len(locked_ids) != 280:
        raise RuntimeError("P4B task or locked-test population changed")

    seed = int(config["schedule"]["seed"])
    rows = []
    for task_index, (suite, task_id) in enumerate(tasks):
        splits = remaining[(suite, task_id)]
        if {key: len(value) for key, value in splits.items()} != {
            "train": 27,
            "calibration": 6,
            "locked_test": 7,
        }:
            raise RuntimeError("P4B per-task independent capacity changed")
        calibration = sorted(
            splits["calibration"],
            key=lambda row: digest(
                f"{seed}|structured|{suite}|{task_id}|{row['original_trajectory_id']}"
            ),
        )
        control = min(
            splits["train"],
            key=lambda row: digest(
                f"{seed}|control|{suite}|{task_id}|{row['original_trajectory_id']}"
            ),
        )
        assigned = [control, *calibration]
        for slot, trajectory in enumerate(assigned):
            spec = slot_spec(task_index, slot, seed=seed)
            if (slot == 0) != (trajectory["split"] == "train"):
                raise RuntimeError("P4B split assignment changed")
            maximum_start = int(trajectory["step_count"]) - (8 * spec.horizon + 8)
            eligible = range(0, maximum_start + 1, int(config["schedule"]["aligned_start_stride"]))
            starts = sorted(
                eligible,
                key=lambda start: digest(
                    f"{seed}|start|{suite}|{task_id}|{slot}|{trajectory['trajectory_id']}|{start}"
                ),
            )
            if not starts:
                raise RuntimeError("P4B trajectory has no complete aligned window")
            start = starts[0]
            future = [start + 8 * ordinal for ordinal in range(1, spec.horizon + 1)]
            payload = {
                "schema_version": "pair-p4b-schedule-row-v1",
                "run_id": config["run_id"],
                "task_index": task_index,
                "slot": slot,
                "category": spec.category,
                "horizon": spec.horizon,
                "split": trajectory["split"],
                "suite": suite,
                "task_id": task_id,
                "trajectory_id": trajectory["trajectory_id"],
                "original_trajectory_id": trajectory["original_trajectory_id"],
                "source_path": trajectory["source_path"],
                "source_sha256": trajectory["source_sha256"],
                "action_dataset": trajectory["action_dataset"],
                "primary_camera_dataset": trajectory["primary_camera_dataset"],
                "wrist_camera_dataset": trajectory["wrist_camera_dataset"],
                "proprio_datasets": trajectory["proprio_datasets"],
                "normalization_statistics_key": trajectory["normalization_statistics_key"],
                "language_instruction": trajectory["language_instruction"],
                "step_count": trajectory["step_count"],
                "anchor_original_step": start,
                "anchor_query_id": digest(f"{trajectory['trajectory_id']}|query|{start}"),
                "future_original_steps": future,
                "future_query_ids": [
                    digest(f"{trajectory['trajectory_id']}|query|{step}") for step in future
                ],
                "profile_id": config["profile"]["profile_id"],
                "exact_repeat": spec.repeat,
            }
            payload["schedule_id"] = semantic_sha256(payload)
            rows.append(payload)

    counts = expected_counts(seed)
    identities = {row["trajectory_id"] for row in rows}
    cells = Counter(
        (row["suite"], row["horizon"])
        for row in rows
        if row["category"] == "structured"
    )
    if (
        len(rows) != counts["anchors"]
        or len(identities) != counts["anchors"]
        or identities & p4_ids
        or identities & locked_ids
        or len(cells) != 12
        or set(cells.values()) != {20}
        or sum(row["exact_repeat"] for row in rows) != counts["exact_repeats"]
    ):
        raise RuntimeError("P4B schedule population or balance changed")
    schedule_bytes = b"".join(canonical_bytes(row) + b"\n" for row in rows)
    summary = {
        "schema_version": "pair-p4b-schedule-summary-v1",
        "run_id": config["run_id"],
        "status": "dry_run_passed" if args.dry_run else "completed",
        "configuration_semantic_sha256": config["semantic_sha256"],
        "schedule_sha256": hashlib.sha256(schedule_bytes).hexdigest(),
        "counts": counts,
        "unique_trajectories": len(identities),
        "p4_overlap": 0,
        "locked_test_trajectories_used": 0,
        "suite_horizon_counts": {
            f"{suite}/h{horizon}": value
            for (suite, horizon), value in sorted(cells.items())
        },
        "expert_values_accessed": False,
        "observation_values_accessed": False,
        "regret_accessed": False,
        "model_accessed": False,
        "cuda_visible": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    summary["semantic_sha256"] = semantic_sha256(summary)
    if args.dry_run:
        print(json.dumps({"status": "dry_run_passed", "summary": summary}, sort_keys=True))
        return 0
    schedule_path = ROOT / config["artifacts"]["schedule"]
    summary_path = ROOT / config["artifacts"]["schedule_summary"]
    if schedule_path.exists() or summary_path.exists():
        raise RuntimeError("P4B immutable schedule output already exists")
    write_once(schedule_path, schedule_bytes)
    write_once(summary_path, canonical_bytes(summary) + b"\n")
    print(json.dumps({"status": "completed", "anchors": len(rows)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
