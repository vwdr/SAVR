#!/usr/bin/env python3
"""Freeze CAC C0 identities and outcome-blind schedules.

This program reads only authenticated P1 metadata and development expert-action
gripper coordinates. It never reads locked-test action values, simulator
outcomes, model weights, or GPU state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from statistics import NormalDist
from typing import Any, Iterable, Mapping

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
INDEX_REL = Path("data/pair/index/f13aa24a3da8c43c7225569f28c562979fa0e35a")
DATA_REL = Path("data/pair/libero_hdf5/f13aa24a3da8c43c7225569f28c562979fa0e35a")
CONFIG_REL = Path("configs/cac/c0_freeze_v1.json")
DEFAULT_OUT_ROOT = ROOT / "reports/cac_c0"


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_semantic_hash(payload: Mapping[str, Any]) -> bool:
    body = dict(payload)
    declared = body.pop("semantic_sha256", None)
    return declared == sha256_bytes(canonical(body))


def rank(*parts: Any) -> str:
    return sha256_bytes("|".join(str(part) for part in parts).encode())


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    body = dict(payload)
    body["semantic_sha256"] = sha256_bytes(canonical(body))
    path.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> tuple[int, str]:
    count = 0
    digest = hashlib.sha256()
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            body = dict(row)
            body["semantic_sha256"] = sha256_bytes(canonical(body))
            line = json.dumps(body, sort_keys=True, separators=(",", ":")) + "\n"
            handle.write(line)
            digest.update(line.encode())
            count += 1
    return count, digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def verify_inputs(config: Mapping[str, Any]) -> dict[str, str]:
    paths = {
        "trajectory_index.jsonl": ROOT / INDEX_REL / "trajectory_index.jsonl",
        "query_index.jsonl": ROOT / INDEX_REL / "query_index.jsonl",
        "split_manifest.json": ROOT / INDEX_REL / "split_manifest.json",
    }
    observed_p1 = {name: sha256_file(path) for name, path in paths.items()}
    if observed_p1 != config["identities"]["p1_index_files"]:
        raise RuntimeError("authenticated P1 index identity mismatch")
    identities = config["identities"]

    def revision(path: Path) -> str:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
        ).strip()

    observed_revisions = {
        "project_git_base": revision(ROOT),
        "openvla_oft_git": revision(ROOT / "third_party/openvla-oft"),
        "vla_cache_git": revision(ROOT / "third_party/vla-cache"),
        "libero_git": revision(ROOT / "third_party/LIBERO"),
    }
    for key, value in observed_revisions.items():
        if value != identities[key]:
            raise RuntimeError(f"authenticated revision mismatch: {key}")

    checkpoint_root = ROOT / "checkpoints/openvla-7b-oft-libero-four-suite"
    observed_checkpoint = {
        name: sha256_file(checkpoint_root / name)
        for name in identities["checkpoint_files"]
    }
    if observed_checkpoint != identities["checkpoint_files"]:
        raise RuntimeError("authenticated checkpoint metadata mismatch")

    substrate_paths = {
        "profile_config": ROOT / "configs/pair/p3r_vectorized_v02_recovery01.json",
        "vectorized_source_helper": ROOT / "src/savr/pair/p3_openvla.py",
    }
    observed_substrate = {name: sha256_file(path) for name, path in substrate_paths.items()}
    if observed_substrate != {
        "profile_config": config["cache_substrate"]["profile_config_sha256"],
        "vectorized_source_helper": config["cache_substrate"]["vectorized_source_helper_sha256"],
    }:
        raise RuntimeError("authenticated D62 substrate mismatch")

    source_manifest = json.loads((ROOT / INDEX_REL / "source_manifest.json").read_text(encoding="utf-8"))
    manifest_text = json.dumps(source_manifest, sort_keys=True)
    if identities["dataset_revision"] not in manifest_text:
        raise RuntimeError("dataset revision is absent from the authenticated source manifest")
    return {
        **{f"p1_{name}": value for name, value in observed_p1.items()},
        **observed_revisions,
        **{f"checkpoint_{name}": value for name, value in observed_checkpoint.items()},
        **{f"substrate_{name}": value for name, value in observed_substrate.items()},
        "dataset_revision": identities["dataset_revision"],
        "checkpoint_revision": identities["checkpoint_revision"],
    }


def assign_roles(
    trajectories: list[dict[str, Any]], seed: int
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    by_task: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in trajectories:
        by_task[(row["suite"], row["task_id"])].append(row)
    if len(by_task) != 40:
        raise RuntimeError("expected exactly 40 tasks")

    roles: dict[str, str] = {}
    output: list[dict[str, Any]] = []
    for task in sorted(by_task):
        rows = by_task[task]
        train = sorted(
            (row for row in rows if row["split"] == "train"),
            key=lambda row: rank(seed, *task, row["trajectory_id"]),
        )
        calibration = sorted(
            (row for row in rows if row["split"] == "calibration"),
            key=lambda row: rank(seed, *task, "development", row["trajectory_id"]),
        )
        locked = sorted(
            (row for row in rows if row["split"] == "locked_test"),
            key=lambda row: rank(seed, *task, "locked", row["trajectory_id"]),
        )
        if (len(train), len(calibration), len(locked)) != (35, 8, 7):
            raise RuntimeError(f"trajectory split mismatch for {task}")
        for index, row in enumerate(train):
            role = "adapter_fit" if index < 24 else "architecture_selection" if index < 29 else "checkpoint_validation"
            roles[row["trajectory_id"]] = role
            output.append({
                "schema_version": "cac-trajectory-role-v1",
                "suite": task[0],
                "task_id": task[1],
                "trajectory_id": row["trajectory_id"],
                "original_trajectory_id": row["original_trajectory_id"],
                "p1_split": "train",
                "cac_role": role,
                "role_rank": rank(seed, *task, row["trajectory_id"]),
            })
        for row in calibration:
            roles[row["trajectory_id"]] = "development_calibration"
            output.append({
                "schema_version": "cac-trajectory-role-v1",
                "suite": task[0], "task_id": task[1],
                "trajectory_id": row["trajectory_id"],
                "original_trajectory_id": row["original_trajectory_id"],
                "p1_split": "calibration", "cac_role": "development_calibration",
                "role_rank": rank(seed, *task, "development", row["trajectory_id"]),
            })
        for row in locked:
            roles[row["trajectory_id"]] = "locked_test"
            output.append({
                "schema_version": "cac-trajectory-role-v1",
                "suite": task[0], "task_id": task[1],
                "trajectory_id": row["trajectory_id"],
                "original_trajectory_id": row["original_trajectory_id"],
                "p1_split": "locked_test", "cac_role": "locked_test",
                "role_rank": rank(seed, *task, "locked", row["trajectory_id"]),
            })
    return output, roles


class TransitionClassifier:
    def __init__(self, trajectories: Mapping[str, Mapping[str, Any]], statistics: Mapping[str, Any]):
        self.trajectories = trajectories
        self.statistics = statistics
        self.cache: dict[str, np.ndarray] = {}

    def actions(self, trajectory_id: str) -> np.ndarray:
        row = self.trajectories[trajectory_id]
        if row["split"] == "locked_test":
            raise RuntimeError("locked-test action access is forbidden in C0")
        if trajectory_id in self.cache:
            return self.cache[trajectory_id]
        with h5py.File(ROOT / DATA_REL / row["source_path"], "r") as handle:
            raw = np.asarray(handle[row["action_dataset"]], dtype=np.float32)
        values = raw.copy()
        values[:, -1] = np.float32(1.0) - np.clip(values[:, -1], 0.0, 1.0)
        metadata = self.statistics[row["normalization_statistics_key"]]["action"]
        low = np.asarray(metadata["q01"], dtype=np.float32)
        high = np.asarray(metadata["q99"], dtype=np.float32)
        mask = np.asarray(metadata.get("mask", [True] * 7), dtype=bool)
        scaled = np.clip(np.float32(2.0) * (values - low) / (high - low + np.float32(1e-8)) - 1.0, -1.0, 1.0)
        normalized = np.where(mask, scaled, values).astype(np.float32)
        if normalized.shape != (int(row["step_count"]), 7) or not np.isfinite(normalized).all():
            raise RuntimeError("development action chronology mismatch")
        self.cache[trajectory_id] = normalized
        return normalized

    def is_transition(self, query: Mapping[str, Any], horizon: int) -> bool:
        actions = self.actions(str(query["trajectory_id"]))
        start = int(query["original_step"])
        first = start + 8
        final = start + 8 * horizon + 8
        execution_coordinate = np.float32(2.0) * actions[first - 1 : final, -1] - np.float32(1.0)
        signs = execution_coordinate >= 0
        return bool(np.any(signs[1:] != signs[:-1]))


def future_chain(query: Mapping[str, Any], horizon: int, by_id: Mapping[str, Mapping[str, Any]]) -> list[str]:
    ids: list[str] = []
    current = query
    for _ in range(horizon):
        next_id = current.get("next_query_id")
        if not next_id or next_id not in by_id:
            return []
        ids.append(str(next_id))
        current = by_id[str(next_id)]
    return ids


def select_cell(
    *, task_index: int, task: tuple[str, str], role: str, horizon: int, count: int,
    queries: list[dict[str, Any]], by_id: Mapping[str, Mapping[str, Any]],
    classifier: TransitionClassifier, seed: int, expose_transition: bool,
) -> list[dict[str, Any]]:
    candidates: list[tuple[str, dict[str, Any], list[str], bool | None]] = []
    for query in queries:
        if query["cac_role"] != role or int(query["maximum_complete_contract_horizon"]) < horizon:
            continue
        chain = future_chain(query, horizon, by_id)
        if len(chain) != horizon:
            continue
        transition = classifier.is_transition(query, horizon) if expose_transition else None
        candidates.append((rank(seed, role, *task, horizon, query["query_id"]), query, chain, transition))
    if len(candidates) < count:
        raise RuntimeError(f"insufficient {role} candidates for {task} h={horizon}: {len(candidates)} < {count}")
    candidates.sort(key=lambda item: item[0])
    selected: list[dict[str, Any]] = []
    used: set[str] = set()
    for slot in range(count):
        desired = bool((task_index + horizon + slot) % 2) if expose_transition else None
        choices = [item for item in candidates if item[1]["query_id"] not in used and (desired is None or item[3] == desired)]
        fallback = False
        if not choices:
            choices = [item for item in candidates if item[1]["query_id"] not in used]
            fallback = True
        if not choices:
            raise RuntimeError("candidate reuse exhausted")
        _, query, chain, observed = choices[0]
        used.add(str(query["query_id"]))
        payload = {
            "schema_version": "cac-contract-v1",
            "task_index": task_index,
            "suite": task[0],
            "task_id": task[1],
            "role": role,
            "horizon": horizon,
            "slot": slot,
            "trajectory_id": query["trajectory_id"],
            "anchor_query_id": query["query_id"],
            "anchor_original_step": query["original_step"],
            "future_query_ids": chain,
            "endpoint_query_id": chain[-1],
            "desired_gripper_transition": desired,
            "observed_gripper_transition": observed,
            "transition_fallback": fallback,
            "contract_id": rank("cac-contract-v1", role, horizon, query["query_id"]),
        }
        selected.append(payload)
    return selected


def transition_availability(
    *, task: tuple[str, str], queries: list[dict[str, Any]],
    by_id: Mapping[str, Mapping[str, Any]], classifier: TransitionClassifier,
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for query in queries:
        role = str(query["cac_role"])
        if role not in {"adapter_fit", "architecture_selection"}:
            continue
        for horizon in (1, 2, 4):
            if int(query["maximum_complete_contract_horizon"]) < horizon:
                continue
            if len(future_chain(query, horizon, by_id)) != horizon:
                continue
            if classifier.is_transition(query, horizon):
                counts[f"{role}_h{horizon}"] += 1
    counts["total"] = sum(value for key, value in counts.items() if key != "total")
    return dict(counts)


def validate_transition_minimum(
    tasks: Iterable[tuple[str, str]],
    selected_counts: Mapping[tuple[str, str], int],
    availability_by_task: Mapping[tuple[str, str], Mapping[str, int]],
) -> list[dict[str, Any]]:
    shortfalls = []
    for task in tasks:
        selected = int(selected_counts.get(task, 0))
        availability = dict(availability_by_task[task])
        available = int(availability.get("total", 0))
        if selected < 8 and available >= 8:
            raise RuntimeError(f"C2 transition selector failed despite availability: {task}")
        if selected < 8:
            shortfalls.append({
                "suite": task[0], "task_id": task[1],
                "selected_transition_contracts": selected,
                "eligible_transition_contracts": available,
                "availability_by_role_horizon": availability,
                "reason": "fewer than eight eligible demonstrated transition contracts",
            })
    return shortfalls


def normal_power(n: int, discordance: float, true_difference: float, margin: float, alpha: float) -> float:
    variance = max(discordance - true_difference * true_difference, 1e-12)
    noncentral = math.sqrt(n) * (true_difference + margin) / math.sqrt(variance)
    return NormalDist().cdf(noncentral - NormalDist().inv_cdf(1 - alpha))


def power_table(config: Mapping[str, Any]) -> dict[str, Any]:
    rows = []
    for discordance in (0.05, 0.10, 0.15, 0.20):
        for difference in (0.0, -0.01, -0.02):
            power = normal_power(1600, discordance, difference, 0.02, 0.05)
            required = None
            for n in range(40, 20001):
                if normal_power(n, discordance, difference, 0.02, 0.05) >= 0.80:
                    required = n
                    break
            rows.append({
                "discordance": discordance,
                "true_cac_minus_dense": difference,
                "analytic_power_n1600": power,
                "minimum_n_for_80_percent": required,
                "within_final_population": required is not None and required <= 1600,
            })
    return {
        "schema_version": "cac-c0-power-v1",
        "test": "one-sided paired noninferiority normal approximation",
        "alpha": 0.05,
        "margin": 0.02,
        "n": 1600,
        "rows": rows,
        "hierarchical_sensitivity": {
            "historical_task_pattern": "LIBERO-Spatial FR task successes 10/10 for tasks 0-8 and 9/10 for task 9, repeated only as a planning sensitivity across four suites",
            "result": "task-level heterogeneity cannot improve analytic power; the final analyzer uses a 20,000-replicate task-then-state bootstrap and the lower analytic/hierarchical forecast governs",
            "limitation": "historical development outcomes do not identify final paired discordance; C5/C6 must update the forecast without changing the margin or population"
        }
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--recovery-config", type=Path)
    args = parser.parse_args()
    if args.root.resolve() != ROOT.resolve():
        raise RuntimeError("C0 freezer may operate only in the authenticated SAVR root")
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in {"", "-1"}:
        raise RuntimeError("C0 requires CUDA_VISIBLE_DEVICES to be empty or -1")
    recovery: dict[str, Any] | None = None
    out_root = DEFAULT_OUT_ROOT
    if args.recovery_config is not None:
        recovery_path = ROOT / args.recovery_config
        recovery = json.loads(recovery_path.read_text(encoding="utf-8"))
        if recovery.get("schema_version") != "cac-c0-recovery-config-v1" or not verify_semantic_hash(recovery):
            raise RuntimeError("invalid C0 recovery configuration")
        authenticated = {
            "base_config": ROOT / recovery["authenticated_inputs"]["base_config"]["path"],
            "technical_stop_report": ROOT / recovery["authenticated_inputs"]["technical_stop_report"]["path"],
            "technical_stop_machine": ROOT / recovery["authenticated_inputs"]["technical_stop_machine"]["path"],
            "protocol": ROOT / recovery["authenticated_inputs"]["protocol"]["path"],
            "expanded_test": ROOT / recovery["authenticated_inputs"]["expanded_test"]["path"],
        }
        for name, path in authenticated.items():
            if sha256_file(path) != recovery["authenticated_inputs"][name]["sha256"]:
                raise RuntimeError(f"C0 recovery input changed: {name}")
        if sha256_file(Path(__file__)) != recovery["recovery_freezer_sha256"]:
            raise RuntimeError("C0 recovery freezer identity mismatch")
        expected_correction = {
            "preserve_first_attempt_root": True,
            "enforce_eight_when_at_least_eight_are_available": True,
            "report_unavailable_shortfalls": True,
            "scientific_settings_changed": False,
        }
        if recovery["correction"] != expected_correction:
            raise RuntimeError("C0 recovery correction changed")
        out_root = ROOT / recovery["output_root"]
        if DEFAULT_OUT_ROOT.is_dir() and any(DEFAULT_OUT_ROOT.iterdir()):
            raise RuntimeError("first C0 attempt root is not the preserved empty root")
    if out_root.exists():
        raise RuntimeError("immutable C0 output root already exists")
    out_root.mkdir(parents=True)

    config = json.loads((ROOT / CONFIG_REL).read_text(encoding="utf-8"))
    observed_inputs = verify_inputs(config)
    trajectories = load_jsonl(ROOT / INDEX_REL / "trajectory_index.jsonl")
    queries = load_jsonl(ROOT / INDEX_REL / "query_index.jsonl")
    trajectory_by_id = {row["trajectory_id"]: row for row in trajectories}
    query_by_id = {row["query_id"]: row for row in queries}
    role_rows, roles = assign_roles(trajectories, int(config["data_roles"]["trajectory_role_seed"]))
    for query in queries:
        query["cac_role"] = roles[query["trajectory_id"]]
    statistics = json.loads((ROOT / "checkpoints/openvla-7b-oft-libero-four-suite/dataset_statistics.json").read_text())
    classifier = TransitionClassifier(trajectory_by_id, statistics)
    tasks = sorted({(row["suite"], row["task_id"]) for row in trajectories})

    all_cells: dict[tuple[str, tuple[str, str], int], list[dict[str, Any]]] = {}
    required = {
        "adapter_fit": 50,
        "architecture_selection": 6,
        "checkpoint_validation": 20,
        "development_calibration": 40,
        "locked_test": 24,
    }
    for task_index, task in enumerate(tasks):
        task_queries = [row for row in queries if (row["suite"], row["task_id"]) == task]
        for role, count in required.items():
            for horizon in (1, 2, 4):
                all_cells[(role, task, horizon)] = select_cell(
                    task_index=task_index, task=task, role=role, horizon=horizon,
                    count=count, queries=task_queries, by_id=query_by_id,
                    classifier=classifier, seed=int(config["contract_schedules"]["seed"]),
                    expose_transition=role != "locked_test",
                )

    c2_primary: list[dict[str, Any]] = []
    c3_rows: list[dict[str, Any]] = []
    locked_rows: list[dict[str, Any]] = []
    for task in tasks:
        for horizon in (1, 2, 4):
            fit = all_cells[("adapter_fit", task, horizon)]
            architecture = all_cells[("architecture_selection", task, horizon)]
            c2_primary.extend([{**row, "schedule": "c2_primary"} for row in fit[:24]])
            c2_primary.extend([{**row, "schedule": "c2_primary"} for row in architecture])
            c3_rows.extend([{**row, "schedule": "c3_adapter_fit"} for row in fit])
            c3_rows.extend([{**row, "schedule": "c3_checkpoint_validation"} for row in all_cells[("checkpoint_validation", task, horizon)]])
            c3_rows.extend([{**row, "schedule": "c3_development_calibration"} for row in all_cells[("development_calibration", task, horizon)]])
            locked_rows.extend([{**row, "schedule": "c7_locked"} for row in all_cells[("locked_test", task, horizon)]])

    c2_controls: list[dict[str, Any]] = []
    c2_repeats: list[dict[str, Any]] = []
    by_cell: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in c2_primary:
        by_cell[(row["suite"], row["task_id"], row["horizon"])].append(row)
    for cell in sorted(by_cell):
        rows = by_cell[cell]
        c2_controls.extend([{**row, "schedule": "c2_all_fresh", "control_id": rank("all-fresh", row["contract_id"])} for row in rows[:2]])
        c2_repeats.extend([{**row, "schedule": "c2_exact_repeat", "repeat_of": row["contract_id"], "repeat_id": rank("repeat", row["contract_id"])} for row in rows[:3]])
    c2_rows = c2_primary + c2_controls + c2_repeats

    if (len(c2_primary), len(c2_controls), len(c2_repeats), len(c3_rows), len(locked_rows)) != (3600, 240, 360, 13200, 2880):
        raise RuntimeError("frozen contract totals do not match C0")
    if any(row["observed_gripper_transition"] is not None for row in locked_rows):
        raise RuntimeError("locked transition leakage")
    transition_counts = Counter(
        ((row["suite"], row["task_id"]), bool(row["observed_gripper_transition"]))
        for row in c2_primary
    )
    availability_by_task = {}
    for task in tasks:
        task_queries = [row for row in queries if (row["suite"], row["task_id"]) == task]
        availability_by_task[task] = transition_availability(
            task=task, queries=task_queries, by_id=query_by_id, classifier=classifier
        )
    transition_shortfalls = validate_transition_minimum(
        tasks,
        {task: transition_counts[(task, True)] for task in tasks},
        availability_by_task,
    )

    simulator_rows = []
    populations = {
        "headroom_stage1": range(0, 3), "headroom_extension": range(3, 6),
        "r0_development": range(0, 6), "r1_holdout": range(6, 10),
        "final_confirmation": range(10, 50),
    }
    for population, states in populations.items():
        for task_index, task in enumerate(tasks):
            for state_id in states:
                simulator_rows.append({
                    "schema_version": "cac-simulator-condition-v1", "population": population,
                    "task_index": task_index, "suite": task[0], "task_id": task[1],
                    "initial_state_id": state_id, "seed": 7,
                    "condition_id": rank("cac-condition-v1", population, *task, state_id, 7),
                })

    files: dict[str, dict[str, Any]] = {}
    for name, rows in (
        ("trajectory_roles_v1.jsonl", role_rows),
        ("c2_contracts_v1.jsonl", c2_rows),
        ("c3_contracts_v1.jsonl", c3_rows),
        ("c7_locked_contracts_v1.jsonl", locked_rows),
        ("simulator_populations_v1.jsonl", simulator_rows),
    ):
        count, semantic = write_jsonl(out_root / name, rows)
        files[name] = {"count": count, "sha256": sha256_file(out_root / name), "semantic_stream_sha256": semantic}

    write_json(out_root / "power_table_v1.json", power_table(config))
    files["power_table_v1.json"] = {"sha256": sha256_file(out_root / "power_table_v1.json")}
    summary = {
        "schema_version": "cac-c0-freeze-summary-v1",
        "status": "completed",
        "decision": "pass_stop_before_c1",
        "authenticated_inputs": observed_inputs,
        "counts": {
            "tasks": 40, "trajectory_roles": len(role_rows), "c2_primary": len(c2_primary),
            "c2_all_fresh": len(c2_controls), "c2_exact_repeats": len(c2_repeats),
            "c3_contracts": len(c3_rows), "c7_locked_contracts": len(locked_rows),
            "simulator_condition_rows": len(simulator_rows),
        },
        "model_calls": {"c2": 17640, "c3": 57200, "c7_locked": 12480},
        "storage_bytes": {
            "c2": config["resource_caps"]["c2_bytes"],
            "c3_plus_c7": config["resource_caps"]["c3_and_c7_feature_bytes"],
        },
        "locked_boundaries": {
            "locked_test_action_values_opened": False,
            "state_ids_10_49_outcomes_opened": False,
            "locked_transition_labels_present": False,
        },
        "resources": {
            "gpu_used": False, "model_loaded": False, "simulator_used": False,
            "terminal_outcomes_accessed": False, "downloads": 0,
        },
        "transition_shortfalls": transition_shortfalls,
        "recovery": {
            "used": recovery is not None,
            "configuration_semantic_sha256": recovery.get("semantic_sha256") if recovery else None,
            "first_attempt_root_preserved": recovery is not None,
        },
        "files": files,
        "advance": {"next_phase": "C1", "authorized": False, "stop_before_next_phase": True},
    }
    write_json(out_root / "freeze_summary_v1.json", summary)


if __name__ == "__main__":
    main()
