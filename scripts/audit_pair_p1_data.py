#!/usr/bin/env python3
"""Build and verify the outcome-free PAIR-VLA P1 data index."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import h5py
import numpy as np
import tensorflow as tf
from huggingface_hub import HfApi


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def numeric_demo_key(name: str) -> int:
    prefix = "demo_"
    if not name.startswith(prefix):
        raise ValueError(f"Unexpected trajectory key: {name}")
    return int(name[len(prefix) :])


def lfs_sha256(sibling: Any) -> str | None:
    lfs = getattr(sibling, "lfs", None)
    if isinstance(lfs, dict):
        return lfs.get("sha256")
    return getattr(lfs, "sha256", None) if lfs else None


def allocate_splits(count: int, fractions: dict[str, float], tie_order: list[str]) -> dict[str, int]:
    raw = {name: count * fraction for name, fraction in fractions.items()}
    allocated = {name: math.floor(value) for name, value in raw.items()}
    remaining = count - sum(allocated.values())
    tie_rank = {name: index for index, name in enumerate(tie_order)}
    order = sorted(raw, key=lambda name: (-(raw[name] - allocated[name]), tie_rank[name], name))
    for name in order[:remaining]:
        allocated[name] += 1
    if sum(allocated.values()) != count:
        raise AssertionError("Split allocation does not reconcile")
    return allocated


def manual_standardize(
    raw_actions: np.ndarray,
    raw_proprio: np.ndarray,
    statistics: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    actions = np.asarray(raw_actions, dtype=np.float32).copy()
    actions[:, -1] = np.float32(1.0) - np.clip(actions[:, -1], np.float32(0.0), np.float32(1.0))

    def normalize(values: np.ndarray, metadata: dict[str, Any]) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        low = np.asarray(metadata["q01"], dtype=np.float32)
        high = np.asarray(metadata["q99"], dtype=np.float32)
        mask = np.asarray(metadata.get("mask", [True] * values.shape[-1]), dtype=bool)
        normalized = np.clip(
            np.float32(2.0) * (values - low) / (high - low + np.float32(1e-8)) - np.float32(1.0),
            np.float32(-1.0),
            np.float32(1.0),
        )
        result = np.where(mask, normalized, values)
        result = np.where(np.asarray(metadata["min"], dtype=np.float32) == np.asarray(metadata["max"], dtype=np.float32), 0.0, result)
        return np.asarray(result, dtype=np.float32)

    return normalize(actions, statistics["action"]), normalize(raw_proprio, statistics["proprio"])


def load_official_functions(source_root: Path) -> tuple[Any, Any, Any]:
    sys.path.insert(0, str(source_root.resolve()))
    from prismatic.vla.constants import NormalizationType
    from prismatic.vla.datasets.rlds.oxe.transforms import libero_dataset_transform
    from prismatic.vla.datasets.rlds.traj_transforms import chunk_act_obs
    from prismatic.vla.datasets.rlds.utils.data_utils import normalize_action_and_proprio

    return libero_dataset_transform, normalize_action_and_proprio, (chunk_act_obs, NormalizationType)


def official_standardize_and_chunk(
    raw_actions: np.ndarray,
    raw_proprio: np.ndarray,
    statistics: dict[str, Any],
    official: tuple[Any, Any, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    libero_transform, normalize, packed = official
    chunk_act_obs, normalization_type = packed
    action_tensor = tf.convert_to_tensor(np.asarray(raw_actions, dtype=np.float32))
    state_tensor = tf.convert_to_tensor(np.asarray(raw_proprio, dtype=np.float32))
    trajectory = {"action": action_tensor, "observation": {"state": state_tensor}}
    trajectory = libero_transform(trajectory)
    trajectory["observation"]["proprio"] = tf.concat(
        [trajectory["observation"]["EEF_state"], trajectory["observation"]["gripper_state"]], axis=-1
    )
    metadata = {
        key: {subkey: tf.convert_to_tensor(value) for subkey, value in section.items()}
        for key, section in statistics.items()
        if key in {"action", "proprio"}
    }
    trajectory = normalize(trajectory, metadata, normalization_type.BOUNDS_Q99)
    normalized_actions = np.asarray(trajectory["action"].numpy(), dtype=np.float32)
    normalized_proprio = np.asarray(trajectory["observation"]["proprio"].numpy(), dtype=np.float32)

    length = normalized_actions.shape[0]
    chunk_input = {
        "action": tf.convert_to_tensor(normalized_actions),
        "observation": {"proprio": tf.convert_to_tensor(normalized_proprio)},
        "task": {"language_instruction": tf.fill([length], "")},
        "dataset_name": tf.fill([length], "libero"),
        "absolute_action_mask": tf.tile(tf.constant([[False] * 6 + [True]]), [length, 1]),
    }
    chunked = chunk_act_obs(chunk_input, window_size=1, future_action_window_size=7)
    return normalized_actions, normalized_proprio, np.asarray(chunked["action"].numpy(), dtype=np.float32)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()

    project_root = args.config.resolve().parents[2]
    config = json.loads(args.config.read_text(encoding="utf-8"))
    dataset_root = project_root / config["dataset"]["root"]
    output_root = project_root / config["outputs"]["index_root"]
    summary_path = project_root / config["outputs"]["run_summary"]
    statistics_path = project_root / config["checkpoint_statistics"]
    source_root = project_root / config["official_source"]["root"]
    started_at = datetime.now(timezone.utc).isoformat()

    if tf.config.list_physical_devices("GPU"):
        raise RuntimeError("P1 requires CUDA/GPU to remain hidden")
    source_revision = subprocess.check_output(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if source_revision != config["official_source"]["revision"]:
        raise RuntimeError(f"OpenVLA-OFT revision mismatch: {source_revision}")

    api = HfApi()
    info = api.dataset_info(
        config["dataset"]["repo_id"],
        revision=config["dataset"]["revision"],
        files_metadata=True,
    )
    if info.sha != config["dataset"]["revision"]:
        raise RuntimeError(f"Dataset revision mismatch: {info.sha}")

    allowed_suites = {"libero_10", "libero_goal", "libero_object", "libero_spatial"}
    remote_files: dict[str, dict[str, Any]] = {}
    for sibling in info.siblings or []:
        parts = sibling.rfilename.split("/", 1)
        if len(parts) == 2 and parts[0] in allowed_suites and parts[1].endswith(".hdf5"):
            remote_files[sibling.rfilename] = {
                "bytes": int(sibling.size or 0),
                "sha256": lfs_sha256(sibling),
            }
    if len(remote_files) != config["dataset"]["expected_files"]:
        raise RuntimeError(f"Expected 40 remote HDF5 files, found {len(remote_files)}")
    if sum(item["bytes"] for item in remote_files.values()) != config["dataset"]["expected_bytes"]:
        raise RuntimeError("Remote byte count does not match the freeze")

    local_paths = sorted(dataset_root.glob("libero_*/*.hdf5"))
    local_relatives = [path.relative_to(dataset_root).as_posix() for path in local_paths]
    if set(local_relatives) != set(remote_files):
        raise RuntimeError("Local and frozen remote file sets differ")

    with ThreadPoolExecutor(max_workers=4) as executor:
        local_hashes = dict(zip(local_relatives, executor.map(sha256_file, local_paths), strict=True))
    source_rows = []
    for relative in sorted(remote_files):
        path = dataset_root / relative
        remote = remote_files[relative]
        if path.stat().st_size != remote["bytes"]:
            raise RuntimeError(f"Byte mismatch: {relative}")
        if not remote["sha256"] or local_hashes[relative] != remote["sha256"]:
            raise RuntimeError(f"SHA-256 mismatch: {relative}")
        source_rows.append(
            {
                "path": relative,
                "bytes": remote["bytes"],
                "hf_lfs_sha256": remote["sha256"],
                "local_sha256": local_hashes[relative],
            }
        )
    source_manifest_payload = {
        "schema_version": "pair-p1-source-manifest-v1",
        "repo_id": config["dataset"]["repo_id"],
        "repo_type": "dataset",
        "revision": info.sha,
        "license": config["dataset"]["license"],
        "license_source": config["dataset"]["license_source"],
        "file_count": len(source_rows),
        "total_bytes": sum(row["bytes"] for row in source_rows),
        "files": source_rows,
    }
    source_manifest = dict(source_manifest_payload)
    source_manifest["semantic_sha256"] = sha256_bytes(canonical_bytes(source_manifest_payload))

    statistics_all = json.loads(statistics_path.read_text(encoding="utf-8"))
    suite_to_statistics = {
        "libero_spatial": "libero_spatial_no_noops",
        "libero_object": "libero_object_no_noops",
        "libero_goal": "libero_goal_no_noops",
        "libero_10": "libero_10_no_noops",
    }
    official = load_official_functions(source_root)
    tolerance = float(config["numerical_audit"]["absolute_tolerance"])
    split_seed = int(config["split"]["seed"])
    fractions = {key: float(value) for key, value in config["split"]["fractions"].items()}
    tie_order = list(config["split"]["largest_remainder_tie_order"])

    trajectories: list[dict[str, Any]] = []
    trajectory_arrays: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    task_samples: list[dict[str, Any]] = []
    boundary_samples: dict[int, dict[str, Any]] = {}
    boundary_counts: Counter[int] = Counter()
    max_action_difference = 0.0
    max_proprio_difference = 0.0
    max_chunk_difference = 0.0
    total_steps = 0

    for path in local_paths:
        relative = path.relative_to(dataset_root).as_posix()
        suite = relative.split("/", 1)[0]
        task_id = path.stem.removesuffix("_demo")
        statistics_key = suite_to_statistics[suite]
        statistics = statistics_all[statistics_key]
        with h5py.File(path, "r") as handle:
            data = handle["data"]
            demos = sorted(data.keys(), key=numeric_demo_key)
            if len(demos) != config["dataset"]["expected_trajectories_per_task"]:
                raise RuntimeError(f"Expected 50 trajectories in {relative}")
            if int(data.attrs["num_demos"]) != len(demos):
                raise RuntimeError(f"num_demos mismatch in {relative}")
            problem_info = json.loads(data.attrs["problem_info"])
            instruction = problem_info["language_instruction"]
            file_steps = 0
            for demo_index, demo_name in enumerate(demos):
                demo = data[demo_name]
                obs = demo["obs"]
                required = ["actions", "dones", "rewards", "states", "robot_states"]
                required_obs = ["agentview_rgb", "eye_in_hand_rgb", "ee_states", "gripper_states", "joint_states"]
                if any(key not in demo for key in required) or any(key not in obs for key in required_obs):
                    raise RuntimeError(f"Missing required field in {relative}:{demo_name}")
                length = int(demo["actions"].shape[0])
                if length < 8 or demo["actions"].shape[1:] != (7,):
                    raise RuntimeError(f"Invalid action shape in {relative}:{demo_name}")
                for key in required:
                    if int(demo[key].shape[0]) != length:
                        raise RuntimeError(f"Length mismatch for {key} in {relative}:{demo_name}")
                for key in required_obs:
                    if int(obs[key].shape[0]) != length:
                        raise RuntimeError(f"Length mismatch for obs/{key} in {relative}:{demo_name}")
                if obs["agentview_rgb"].shape[1:] != (128, 128, 3) or obs["eye_in_hand_rgb"].shape[1:] != (128, 128, 3):
                    raise RuntimeError(f"Camera shape mismatch in {relative}:{demo_name}")
                if obs["agentview_rgb"].dtype != np.uint8 or obs["eye_in_hand_rgb"].dtype != np.uint8:
                    raise RuntimeError(f"Camera dtype mismatch in {relative}:{demo_name}")
                if obs["ee_states"].shape[1:] != (6,) or obs["gripper_states"].shape[1:] != (2,):
                    raise RuntimeError(f"Proprio shape mismatch in {relative}:{demo_name}")
                if int(demo.attrs["num_samples"]) != length:
                    raise RuntimeError(f"num_samples mismatch in {relative}:{demo_name}")
                dones = np.asarray(demo["dones"])
                if np.count_nonzero(dones) != 1 or not bool(dones[-1]):
                    raise RuntimeError(f"Terminal marker mismatch in {relative}:{demo_name}")

                raw_actions = np.asarray(demo["actions"])
                raw_proprio = np.concatenate(
                    [np.asarray(obs["ee_states"]), np.asarray(obs["gripper_states"])], axis=-1
                )
                official_actions, official_proprio, official_chunks = official_standardize_and_chunk(
                    raw_actions, raw_proprio, statistics, official
                )
                manual_actions, manual_proprio = manual_standardize(raw_actions, raw_proprio, statistics)
                action_difference = float(np.max(np.abs(official_actions - manual_actions)))
                proprio_difference = float(np.max(np.abs(official_proprio - manual_proprio)))
                expected_chunks = np.stack(
                    [official_actions[start : start + 8] for start in range(length - 7)], axis=0
                )
                chunk_difference = float(np.max(np.abs(official_chunks - expected_chunks)))
                max_action_difference = max(max_action_difference, action_difference)
                max_proprio_difference = max(max_proprio_difference, proprio_difference)
                max_chunk_difference = max(max_chunk_difference, chunk_difference)
                if max(action_difference, proprio_difference, chunk_difference) > tolerance:
                    raise RuntimeError(f"Official preprocessing mismatch in {relative}:{demo_name}")

                original_trajectory_id = demo_name
                trajectory_id = sha256_bytes(
                    f"{info.sha}|{relative}|{local_hashes[relative]}|{original_trajectory_id}".encode()
                )
                split_hash = sha256_bytes(
                    f"{info.sha}|{suite}|{task_id}|{original_trajectory_id}|{split_seed}".encode()
                )
                remainder = length % 8
                boundary_counts[remainder] += 1
                starts = list(range(0, length - 7, 8))
                record = {
                    "schema_version": "pair-p1-trajectory-v1",
                    "trajectory_id": trajectory_id,
                    "original_trajectory_id": original_trajectory_id,
                    "suite": suite,
                    "task_id": task_id,
                    "language_instruction": instruction,
                    "source_path": relative,
                    "source_sha256": local_hashes[relative],
                    "step_count": length,
                    "original_step_start": 0,
                    "original_step_end_inclusive": length - 1,
                    "terminal_remainder_actions": remainder,
                    "eligible_query_count": len(starts),
                    "split_hash": split_hash,
                    "split": None,
                    "primary_camera_dataset": f"data/{demo_name}/obs/agentview_rgb",
                    "wrist_camera_dataset": f"data/{demo_name}/obs/eye_in_hand_rgb",
                    "proprio_datasets": [
                        f"data/{demo_name}/obs/ee_states",
                        f"data/{demo_name}/obs/gripper_states",
                    ],
                    "action_dataset": f"data/{demo_name}/actions",
                    "terminal_dataset": f"data/{demo_name}/dones",
                    "normalization_statistics_key": statistics_key,
                }
                trajectories.append(record)
                trajectory_arrays[trajectory_id] = (raw_actions, official_actions, official_proprio)
                file_steps += length
                total_steps += length

                if demo_index == 0:
                    task_samples.append(
                        {
                            "suite": suite,
                            "task_id": task_id,
                            "trajectory_id": trajectory_id,
                            "primary_first_sha256": sha256_bytes(np.asarray(obs["agentview_rgb"][0]).tobytes()),
                            "primary_last_sha256": sha256_bytes(np.asarray(obs["agentview_rgb"][-1]).tobytes()),
                            "wrist_first_sha256": sha256_bytes(np.asarray(obs["eye_in_hand_rgb"][0]).tobytes()),
                            "wrist_last_sha256": sha256_bytes(np.asarray(obs["eye_in_hand_rgb"][-1]).tobytes()),
                        }
                    )
                if remainder not in boundary_samples:
                    boundary_samples[remainder] = {
                        "suite": suite,
                        "task_id": task_id,
                        "trajectory_id": trajectory_id,
                        "step_count": length,
                        "terminal_remainder_actions": remainder,
                        "primary_terminal_sha256": sha256_bytes(np.asarray(obs["agentview_rgb"][-1]).tobytes()),
                        "wrist_terminal_sha256": sha256_bytes(np.asarray(obs["eye_in_hand_rgb"][-1]).tobytes()),
                    }
            if int(data.attrs["total"]) != file_steps:
                raise RuntimeError(f"File total mismatch in {relative}")

    by_task: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in trajectories:
        by_task[(record["suite"], record["task_id"])].append(record)
    split_counts: Counter[str] = Counter()
    split_task_counts: dict[str, dict[str, int]] = {}
    for (suite, task_id), records in sorted(by_task.items()):
        allocation = allocate_splits(len(records), fractions, tie_order)
        ordered = sorted(records, key=lambda row: (row["split_hash"], row["trajectory_id"]))
        cursor = 0
        task_counts: dict[str, int] = {}
        for split_name in ["train", "calibration", "locked_test"]:
            count = allocation[split_name]
            for record in ordered[cursor : cursor + count]:
                record["split"] = split_name
            cursor += count
            split_counts[split_name] += count
            task_counts[split_name] = count
        if cursor != len(ordered) or any(record["split"] is None for record in ordered):
            raise AssertionError("Split assignment failed")
        split_task_counts[f"{suite}/{task_id}"] = task_counts

    queries: list[dict[str, Any]] = []
    valid_mask = np.ones((8, 7), dtype=np.bool_)
    valid_mask_sha256 = sha256_bytes(valid_mask.tobytes())
    chronology_pair_count = 0
    for trajectory in sorted(trajectories, key=lambda row: row["trajectory_id"]):
        raw_actions, official_actions, official_proprio = trajectory_arrays.pop(trajectory["trajectory_id"])
        starts = list(range(0, trajectory["step_count"] - 7, 8))
        query_ids = [
            sha256_bytes(f"{trajectory['trajectory_id']}|query|{start}".encode()) for start in starts
        ]
        for index, start in enumerate(starts):
            if index + 1 < len(starts):
                if starts[index + 1] - start != 8:
                    raise AssertionError("Query chronology is not exactly eight original actions")
                chronology_pair_count += 1
            max_horizon = min(4, len(starts) - index)
            queries.append(
                {
                    "schema_version": "pair-p1-query-v1",
                    "query_id": query_ids[index],
                    "trajectory_id": trajectory["trajectory_id"],
                    "split": trajectory["split"],
                    "suite": trajectory["suite"],
                    "task_id": trajectory["task_id"],
                    "source_path": trajectory["source_path"],
                    "original_trajectory_id": trajectory["original_trajectory_id"],
                    "original_step": start,
                    "action_start_step": start,
                    "action_end_step_inclusive": start + 7,
                    "next_query_id": query_ids[index + 1] if index + 1 < len(query_ids) else None,
                    "next_query_original_step": starts[index + 1] if index + 1 < len(starts) else None,
                    "next_query_action_delta": 8 if index + 1 < len(starts) else None,
                    "maximum_complete_contract_horizon": max_horizon,
                    "valid_action_coordinates": 56,
                    "valid_action_mask_sha256": valid_mask_sha256,
                    "raw_action_chunk_sha256": sha256_bytes(np.ascontiguousarray(raw_actions[start : start + 8]).tobytes()),
                    "normalized_action_chunk_sha256": sha256_bytes(
                        np.ascontiguousarray(official_actions[start : start + 8]).tobytes()
                    ),
                    "normalized_proprio_sha256": sha256_bytes(
                        np.ascontiguousarray(official_proprio[start]).tobytes()
                    ),
                    "primary_image_reference": f"{trajectory['primary_camera_dataset']}[{start}]",
                    "wrist_image_reference": f"{trajectory['wrist_camera_dataset']}[{start}]",
                }
            )

    trajectory_ids = [row["trajectory_id"] for row in trajectories]
    if len(set(trajectory_ids)) != len(trajectory_ids):
        raise AssertionError("Duplicate trajectory identity")
    split_by_trajectory = {row["trajectory_id"]: row["split"] for row in trajectories}
    if any(query["split"] != split_by_trajectory[query["trajectory_id"]] for query in queries):
        raise AssertionError("Query split inheritance failure")
    if set(boundary_samples) != set(range(8)):
        raise RuntimeError(f"Not all terminal boundary classes are represented: {sorted(boundary_samples)}")

    output_root.mkdir(parents=True, exist_ok=True)
    source_manifest_path = output_root / config["outputs"]["source_manifest"]
    trajectory_path = output_root / config["outputs"]["trajectory_index"]
    query_path = output_root / config["outputs"]["query_index"]
    split_path = output_root / config["outputs"]["split_manifest"]
    audit_path = output_root / config["outputs"]["audit_report"]
    artifact_manifest_path = output_root / config["outputs"]["artifact_manifest"]
    write_json(source_manifest_path, source_manifest)
    write_jsonl(trajectory_path, sorted(trajectories, key=lambda row: row["trajectory_id"]))
    write_jsonl(query_path, sorted(queries, key=lambda row: row["query_id"]))

    split_payload = {
        "schema_version": "pair-p1-split-manifest-v1",
        "seed": split_seed,
        "unit": "complete trajectory within task",
        "fractions": fractions,
        "largest_remainder_tie_order": tie_order,
        "counts": dict(sorted(split_counts.items())),
        "per_task_counts": split_task_counts,
        "trajectory_assignments_sha256": sha256_bytes(
            canonical_bytes(sorted((row["trajectory_id"], row["split"]) for row in trajectories))
        ),
    }
    split_manifest = dict(split_payload)
    split_manifest["semantic_sha256"] = sha256_bytes(canonical_bytes(split_payload))
    write_json(split_path, split_manifest)

    audit_payload = {
        "schema_version": "pair-p1-audit-v1",
        "status": "PASS",
        "dataset_revision": info.sha,
        "openvla_oft_revision": source_revision,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "tensorflow_visible_gpu_count": len(tf.config.list_physical_devices("GPU")),
        "file_count": len(source_rows),
        "source_bytes": source_manifest["total_bytes"],
        "suite_count": len({row["suite"] for row in trajectories}),
        "task_count": len(by_task),
        "trajectory_count": len(trajectories),
        "original_environment_action_count": total_steps,
        "eligible_query_count": len(queries),
        "exact_eight_action_query_pair_count": chronology_pair_count,
        "terminal_boundary_class_counts": {str(key): boundary_counts[key] for key in range(8)},
        "terminal_boundary_samples": [boundary_samples[key] for key in range(8)],
        "every_suite_task_camera_samples": task_samples,
        "maximum_official_vs_numpy_action_abs_difference": max_action_difference,
        "maximum_official_vs_numpy_proprio_abs_difference": max_proprio_difference,
        "maximum_official_chunk_abs_difference": max_chunk_difference,
        "absolute_tolerance": tolerance,
        "filtered_frame_shortcut_used": False,
        "terminal_incomplete_query_starts_in_primary_index": 0,
        "split_counts": dict(sorted(split_counts.items())),
        "trajectory_leakage_count": 0,
        "query_split_inheritance_failures": 0,
        "source_manifest_semantic_sha256": source_manifest["semantic_sha256"],
    }
    audit = dict(audit_payload)
    audit["semantic_sha256"] = sha256_bytes(canonical_bytes(audit_payload))
    write_json(audit_path, audit)

    artifact_paths = [source_manifest_path, trajectory_path, query_path, split_path, audit_path]
    artifact_rows = [
        {
            "path": path.relative_to(project_root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in artifact_paths
    ]
    artifact_payload = {
        "schema_version": "pair-p1-artifact-manifest-v1",
        "dataset_revision": info.sha,
        "files": artifact_rows,
    }
    artifact_manifest = dict(artifact_payload)
    artifact_manifest["semantic_sha256"] = sha256_bytes(canonical_bytes(artifact_payload))
    write_json(artifact_manifest_path, artifact_manifest)
    artifact_bytes = sum(row["bytes"] for row in artifact_rows) + artifact_manifest_path.stat().st_size
    config_sha256 = sha256_file(args.config)
    completed_at = datetime.now(timezone.utc).isoformat()
    run_summary = {
        "schema_version": "pair-run-summary-v1",
        "run_id": "pair-p1-data-v1",
        "phase": "P1",
        "status": "completed",
        "expected_terminal_records": config["dataset"]["expected_files"],
        "observed_terminal_records": len(source_rows),
        "records_sha256": artifact_manifest["semantic_sha256"],
        "config_sha256": config_sha256,
        "project_revision": subprocess.check_output(["git", "-C", str(project_root), "rev-parse", "HEAD"], text=True).strip(),
        "gpu_ids": [],
        "peak_gpu_memory_mib": None,
        "artifact_bytes": artifact_bytes,
        "stop_reason": None,
        "started_at_utc": started_at,
        "completed_at_utc": completed_at,
    }
    write_json(summary_path, run_summary)
    print(json.dumps({"audit": audit, "artifacts": artifact_manifest, "run_summary": run_summary}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
