#!/usr/bin/env python3
"""Freeze outcome-blind P3 timing inputs from the authenticated P1 index."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REVISION = "f13aa24a3da8c43c7225569f28c562979fa0e35a"
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def main() -> int:
    index = ROOT / f"data/pair/index/{REVISION}/trajectory_index.jsonl"
    data_root = ROOT / f"data/pair/libero_hdf5/{REVISION}"
    rows = [json.loads(line) for line in index.read_text(encoding="utf-8").splitlines()]
    selected = []
    for suite in SUITES:
        candidates = sorted(
            (
                row
                for row in rows
                if row["suite"] == suite
                and row["split"] == "train"
                and int(row["eligible_query_count"]) >= 8
            ),
            key=lambda row: (row["split_hash"], row["trajectory_id"]),
        )
        if len(candidates) < 2:
            raise RuntimeError(f"P3 lacks two eligible training trajectories for {suite}")
        for row in candidates[:2]:
            source = data_root / row["source_path"]
            if not source.is_file() or source.stat().st_size <= 0:
                raise RuntimeError(f"P3 source is unavailable: {row['source_path']}")
            selected.append(
                {
                    "suite": row["suite"],
                    "task_id": row["task_id"],
                    "trajectory_id": row["trajectory_id"],
                    "original_trajectory_id": row["original_trajectory_id"],
                    "source_path": row["source_path"],
                    "source_sha256": row["source_sha256"],
                    "language_instruction": row["language_instruction"],
                    "normalization_statistics_key": row["normalization_statistics_key"],
                    "step_count": int(row["step_count"]),
                    "eligible_query_count": int(row["eligible_query_count"]),
                    "split": "train",
                    "split_hash": row["split_hash"],
                }
            )
    payload = {
        "schema_version": "pair-p3-input-manifest-v1",
        "selection_rule": "two lowest split_hash training trajectories per suite with at least eight complete deployment queries",
        "dataset_revision": REVISION,
        "trajectory_index_relative": f"data/pair/index/{REVISION}/trajectory_index.jsonl",
        "data_root_relative": f"data/pair/libero_hdf5/{REVISION}",
        "actions_per_query": 8,
        "expert_action_fields_accessed": False,
        "terminal_outcome_fields_accessed": False,
        "inputs": selected,
    }
    payload["semantic_sha256"] = hashlib.sha256(canonical_bytes(payload)).hexdigest()
    destination = ROOT / "configs/pair/p3_inputs_v1.json"
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(payload["semantic_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
