#!/usr/bin/env python3
"""Freeze the P4 anchor schedule while exposing only transition strata."""

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


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def semantic_sha256(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("semantic_sha256", None)
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_once(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--recovery-config", type=Path)
    args = parser.parse_args()
    if ROOT != EXPECTED_ROOT:
        raise SystemExit(f"P4 scheduler refuses to run outside {EXPECTED_ROOT}")
    if os.environ.get("CUDA_VISIBLE_DEVICES", ""):
        raise RuntimeError("P4 scheduler requires CUDA to be hidden")
    sys.path.insert(0, str(ROOT / "src"))
    from savr.pair.p4 import atomic_group, expected_counts, slot_spec, validate_config

    import h5py
    import numpy as np

    config = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    validate_config(config, ROOT)
    recovery = None
    execution_gripper_mapping = False
    effective_run_id = config["run_id"]
    schedule_artifacts = config["schedule_artifacts"]
    configuration_semantic_sha256 = config["semantic_sha256"]
    if args.recovery_config is not None:
        recovery_path = ROOT / args.recovery_config
        recovery = json.loads(recovery_path.read_text(encoding="utf-8"))
        if recovery.get("schema_version") != "pair-p4-schedule-recovery-config-v1":
            raise RuntimeError("P4 schedule recovery schema changed")
        if recovery.get("semantic_sha256") != semantic_sha256(recovery):
            raise RuntimeError("P4 schedule recovery semantic hash mismatch")
        authenticated = (
            ("base_config", "base_config_sha256"),
            ("protocol", "protocol_sha256"),
            ("technical_stop_report", "technical_stop_report_sha256"),
            ("invalid_schedule", "invalid_schedule_sha256"),
            ("invalid_summary", "invalid_summary_sha256"),
        )
        for path_key, hash_key in authenticated:
            path = ROOT / recovery[path_key]
            if not path.is_file() or file_sha256(path) != recovery[hash_key]:
                raise RuntimeError("P4 schedule recovery evidence changed")
        if recovery["base_config_semantic_sha256"] != config["semantic_sha256"]:
            raise RuntimeError("P4 recovery/base semantic identity mismatch")
        if recovery["correction"] != {
            "gripper_coordinate_mapping": "2*x-1 before sign classification",
            "other_scientific_settings_changed": False,
        }:
            raise RuntimeError("P4 schedule recovery correction changed")
        effective_run_id = recovery["run_id"]
        schedule_artifacts = recovery["schedule_artifacts"]
        configuration_semantic_sha256 = recovery["semantic_sha256"]
        execution_gripper_mapping = True
    schedule_path = ROOT / schedule_artifacts["schedule"]
    summary_path = ROOT / schedule_artifacts["summary"]
    if schedule_path.exists() or summary_path.exists():
        raise RuntimeError("P4 immutable schedule output already exists")

    trajectory_path = ROOT / config["authenticated_inputs"]["trajectory_index"]["path"]
    trajectories = [
        json.loads(line) for line in trajectory_path.read_text(encoding="utf-8").splitlines()
    ]
    if any(
        row["split"] not in {"train", "calibration", "locked_test"}
        for row in trajectories
    ):
        raise RuntimeError("P4 trajectory split is invalid")
    by_task: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in trajectories:
        if row["split"] in {"train", "calibration"}:
            by_task[(row["suite"], row["task_id"])].append(row)
    tasks = sorted(by_task)
    if len(tasks) != 40 or any(len(by_task[task]) != 43 for task in tasks):
        raise RuntimeError("P4 training/calibration trajectory population changed")

    statistics = json.loads(
        (ROOT / "checkpoints/openvla-7b-oft-libero-four-suite/dataset_statistics.json").read_text(
            encoding="utf-8"
        )
    )
    data_root = ROOT / "data/pair/libero_hdf5/f13aa24a3da8c43c7225569f28c562979fa0e35a"
    action_cache: dict[str, Any] = {}

    def normalized_actions(row: Mapping[str, Any]) -> Any:
        trajectory_id = str(row["trajectory_id"])
        if trajectory_id in action_cache:
            return action_cache[trajectory_id]
        with h5py.File(data_root / row["source_path"], "r") as handle:
            raw = np.asarray(handle[row["action_dataset"]], dtype=np.float32)
        values = raw.copy()
        values[:, -1] = np.float32(1.0) - np.clip(
            values[:, -1], np.float32(0.0), np.float32(1.0)
        )
        metadata = statistics[row["normalization_statistics_key"]]["action"]
        low = np.asarray(metadata["q01"], dtype=np.float32)
        high = np.asarray(metadata["q99"], dtype=np.float32)
        mask = np.asarray(metadata.get("mask", [True] * 7), dtype=bool)
        scaled = np.clip(
            np.float32(2.0) * (values - low) / (high - low + np.float32(1e-8))
            - np.float32(1.0),
            np.float32(-1.0),
            np.float32(1.0),
        )
        result = np.where(mask, scaled, values)
        result = np.where(
            np.asarray(metadata["min"], dtype=np.float32)
            == np.asarray(metadata["max"], dtype=np.float32),
            0.0,
            result,
        ).astype(np.float32)
        if result.shape != (int(row["step_count"]), 7) or not np.isfinite(result).all():
            raise RuntimeError("P4 normalized action chronology changed")
        action_cache[trajectory_id] = result
        return result

    def transition_status(actions: Any, start: int, horizon: int) -> bool:
        first = start + 8
        final = start + 8 * horizon + 8
        sequence = actions[first - 1 : final, -1]
        if execution_gripper_mapping:
            sequence = np.float32(2.0) * sequence - np.float32(1.0)
        signs = sequence >= 0
        return bool(np.any(signs[1:] != signs[:-1]))

    rows = []
    selected_trajectories: dict[tuple[str, str], set[str]] = defaultdict(set)
    transition_fallbacks = 0
    for task_index, (suite, task_id) in enumerate(tasks):
        for slot in range(10):
            spec = slot_spec(task_index, slot)
            desired_transition = bool((task_index + slot) % 2)
            candidates = []
            for trajectory in by_task[(suite, task_id)]:
                if trajectory["split"] != spec.split or trajectory["trajectory_id"] in selected_trajectories[(suite, task_id)]:
                    continue
                actions = normalized_actions(trajectory)
                maximum_start = int(trajectory["step_count"]) - (8 * spec.horizon + 8)
                for start in range(0, maximum_start + 1, 8):
                    observed = transition_status(actions, start, spec.horizon)
                    rank = sha256_text(
                        f"{config['schedule']['seed']}|{suite}|{task_id}|{spec.split}|{slot}|{trajectory['trajectory_id']}|{start}"
                    )
                    candidates.append((observed != desired_transition, rank, trajectory, start, observed))
            if not candidates:
                raise RuntimeError("P4 schedule lacks an eligible trajectory/query window")
            fallback, _rank, trajectory, start, observed_transition = min(candidates)
            transition_fallbacks += int(fallback)
            selected_trajectories[(suite, task_id)].add(trajectory["trajectory_id"])
            atomic = atomic_group(suite, task_id, slot, int(config["schedule"]["seed"]))
            future_steps = [start + 8 * ordinal for ordinal in range(1, spec.horizon + 1)]
            payload = {
                "schema_version": "pair-p4-schedule-row-v1",
                "run_id": effective_run_id,
                "task_index": task_index,
                "slot": slot,
                "category": spec.category,
                "horizon": spec.horizon,
                "split": spec.split,
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
                "normalization_statistics_key": trajectory[
                    "normalization_statistics_key"
                ],
                "language_instruction": trajectory["language_instruction"],
                "step_count": trajectory["step_count"],
                "anchor_original_step": start,
                "anchor_query_id": sha256_text(
                    f"{trajectory['trajectory_id']}|query|{start}"
                ),
                "future_original_steps": future_steps,
                "future_query_ids": [
                    sha256_text(f"{trajectory['trajectory_id']}|query|{step}")
                    for step in future_steps
                ],
                "desired_gripper_transition": desired_transition,
                "observed_gripper_transition": observed_transition,
                "transition_stratum_fallback": bool(fallback),
                "atomic_group": {
                    "camera": atomic.camera.value,
                    "tile": atomic.tile,
                    "onset_layer": atomic.onset_layer,
                },
                "profile_id": config["profile"]["profile_id"],
                "exact_repeat": spec.repeat,
                "partition_branches": 2 if spec.category == "structured" else 0,
            }
            payload["schedule_id"] = semantic_sha256(payload)
            rows.append(payload)

    counts = expected_counts()
    if len(rows) != counts["anchors"] or len({row["trajectory_id"] for row in rows}) != 400:
        raise RuntimeError("P4 schedule identity or anchor count changed")
    observed_counts = Counter((row["split"], row["category"]) for row in rows)
    transition_counts = Counter(
        (
            row["split"],
            row["category"],
            bool(row["observed_gripper_transition"]),
        )
        for row in rows
    )
    camera_counts = Counter(row["atomic_group"]["camera"] for row in rows if row["category"] == "atomic")
    layer_counts = Counter(row["atomic_group"]["onset_layer"] for row in rows if row["category"] == "atomic")
    schedule_bytes = b"".join(canonical_bytes(row) + b"\n" for row in rows)
    summary = {
        "schema_version": "pair-p4-schedule-summary-v1",
        "run_id": effective_run_id,
        "status": "completed",
        "configuration_semantic_sha256": configuration_semantic_sha256,
        "base_configuration_semantic_sha256": config["semantic_sha256"],
        "schedule_recovery": recovery is not None,
        "execution_gripper_mapping": execution_gripper_mapping,
        "schedule_sha256": hashlib.sha256(schedule_bytes).hexdigest(),
        "counts": counts,
        "split_category_counts": {
            f"{split}/{category}": count
            for (split, category), count in sorted(observed_counts.items())
        },
        "transition": {
            "desired_true": sum(row["desired_gripper_transition"] for row in rows),
            "observed_true": sum(row["observed_gripper_transition"] for row in rows),
            "fallbacks": transition_fallbacks,
            "by_split_category": {
                f"{split}/{category}": {
                    "false": transition_counts[(split, category, False)],
                    "true": transition_counts[(split, category, True)],
                }
                for split, category in sorted(observed_counts)
            },
        },
        "atomic_camera_counts": dict(sorted(camera_counts.items())),
        "atomic_onset_layer_counts": {
            str(key): value for key, value in sorted(layer_counts.items())
        },
        "expert_values_persisted": False,
        "regret_accessed": False,
        "dense_or_cache_actions_accessed": False,
        "locked_test_labels_accessed": False,
        "simulator_used": False,
        "cuda_visible": False,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    summary["semantic_sha256"] = semantic_sha256(summary)
    write_once(schedule_path, schedule_bytes)
    write_once(summary_path, canonical_bytes(summary) + b"\n")
    print(
        json.dumps(
            {
                "status": "completed",
                "anchors": len(rows),
                "transition_fallbacks": transition_fallbacks,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
