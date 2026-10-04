"""Frozen design primitives for the proposed PAIR P4B confirmation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from savr.pair.types import PairValidationError


SLOT_DESIGN = (
    ("all_fresh_control", 4),
    ("structured", 1),
    ("structured", 1),
    ("structured", 2),
    ("structured", 2),
    ("structured", 4),
    ("structured", 4),
)
SUITE_COUNT = 4
TASKS_PER_SUITE = 10


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


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


@dataclass(frozen=True)
class P4BSlot:
    task_index: int
    slot: int
    category: str
    horizon: int
    repeat: bool


def repeat_slots(seed: int = 20260901) -> frozenset[tuple[int, int]]:
    """Select exactly two repeats per suite/horizon before labels exist."""

    selected = set()
    for suite_index in range(SUITE_COUNT):
        tasks = range(suite_index * TASKS_PER_SUITE, (suite_index + 1) * TASKS_PER_SUITE)
        for horizon in (1, 2, 4):
            candidates = [
                (task, slot)
                for task in tasks
                for slot, (category, value) in enumerate(SLOT_DESIGN)
                if category == "structured" and value == horizon
            ]
            ranked = sorted(
                candidates,
                key=lambda item: hashlib.sha256(
                    f"{seed}|repeat|{suite_index}|{horizon}|{item[0]}|{item[1]}".encode()
                ).hexdigest(),
            )
            selected.update(ranked[:2])
    if len(selected) != 24:
        raise PairValidationError("P4B repeat schedule is not exactly balanced")
    return frozenset(selected)


def slot_spec(task_index: int, slot: int, *, seed: int = 20260901) -> P4BSlot:
    if not 0 <= task_index < 40 or not 0 <= slot < 7:
        raise PairValidationError("P4B task or slot is outside the frozen population")
    category, horizon = SLOT_DESIGN[slot]
    return P4BSlot(
        task_index,
        slot,
        category,
        horizon,
        (task_index, slot) in repeat_slots(seed),
    )


def expected_counts(seed: int = 20260901) -> dict[str, int]:
    specs = [slot_spec(task, slot, seed=seed) for task in range(40) for slot in range(7)]
    base_horizon_sum = sum(spec.horizon for spec in specs)
    structured_horizon_sum = sum(
        spec.horizon for spec in specs if spec.category == "structured"
    )
    repeat_horizon_sum = sum(spec.horizon for spec in specs if spec.repeat)
    return {
        "anchors": len(specs),
        "all_fresh_controls": sum(
            spec.category == "all_fresh_control" for spec in specs
        ),
        "structured_contracts": sum(spec.category == "structured" for spec in specs),
        "structured_horizon1": sum(
            spec.category == "structured" and spec.horizon == 1 for spec in specs
        ),
        "structured_horizon2": sum(
            spec.category == "structured" and spec.horizon == 2 for spec in specs
        ),
        "structured_horizon4": sum(
            spec.category == "structured" and spec.horizon == 4 for spec in specs
        ),
        "exact_repeats": sum(spec.repeat for spec in specs),
        "intervention_records": base_horizon_sum + repeat_horizon_sum,
        "feature_records": structured_horizon_sum,
        "contract_records": len(specs) + sum(spec.repeat for spec in specs),
        "scheduled_model_calls": len(specs) + 2 * base_horizon_sum + repeat_horizon_sum,
        "planned_model_calls": len(specs) + 2 * base_horizon_sum + repeat_horizon_sum + 8,
    }


def ranks(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    result = np.empty(len(array), dtype=np.float64)
    start = 0
    while start < len(array):
        end = start + 1
        while end < len(array) and array[order[end]] == array[order[start]]:
            end += 1
        result[order[start:end]] = (start + end - 1) / 2
        start = end
    return result


def spearman(left: Sequence[float], right: Sequence[float]) -> float:
    x, y = ranks(left), ranks(right)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return -1.0
    return float(np.corrcoef(x, y)[0, 1])


def cvar90(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise PairValidationError("P4B CVaR population is invalid")
    count = max(1, int(np.ceil(0.10 * len(array))))
    return float(np.sort(array)[-count:].mean())


def served_indices(
    scores: Sequence[float],
    suites: Sequence[str],
    horizons: Sequence[int],
    *,
    allowed_horizons: frozenset[int] = frozenset({1, 2, 4}),
) -> np.ndarray:
    scores_array = np.asarray(scores, dtype=np.float64)
    suites_array = np.asarray(suites)
    horizons_array = np.asarray(horizons, dtype=int)
    if not (len(scores_array) == len(suites_array) == len(horizons_array)):
        raise PairValidationError("P4B service arrays have inconsistent lengths")
    selected: list[int] = []
    for suite in sorted(set(suites_array.tolist())):
        for horizon in sorted(allowed_horizons):
            candidates = np.flatnonzero(
                (suites_array == suite) & (horizons_array == horizon)
            )
            if not len(candidates):
                continue
            count = int(np.ceil(0.70 * len(candidates)))
            selected.extend(
                candidates[
                    np.argsort(scores_array[candidates], kind="mergesort")[:count]
                ].tolist()
            )
    return np.asarray(selected, dtype=int)


def cvar_improvement(
    observed_positive: Sequence[float],
    router_scores: Sequence[float],
    proxy_scores: Sequence[float],
    suites: Sequence[str],
    horizons: Sequence[int],
    *,
    allowed_horizons: frozenset[int] = frozenset({1, 2, 4}),
) -> tuple[float, np.ndarray, np.ndarray]:
    observed = np.asarray(observed_positive, dtype=np.float64)
    router_selected = served_indices(
        router_scores, suites, horizons, allowed_horizons=allowed_horizons
    )
    proxy_selected = served_indices(
        proxy_scores, suites, horizons, allowed_horizons=allowed_horizons
    )
    router_tail = cvar90(observed[router_selected])
    proxy_tail = cvar90(observed[proxy_selected])
    value = -1.0 if proxy_tail <= 0 else float(1.0 - router_tail / proxy_tail)
    return value, router_selected, proxy_selected


def validate_config(
    config: Mapping[str, Any], root: Path, *, require_large_inputs: bool = True
) -> None:
    if config.get("schema_version") != "pair-p4b-confirmatory-config-v1":
        raise PairValidationError("P4B configuration schema changed")
    if config.get("semantic_sha256") != semantic_sha256(config):
        raise PairValidationError("P4B configuration semantic hash mismatch")
    if file_sha256(root / config["protocol"]) != config["protocol_sha256"]:
        raise PairValidationError("P4B protocol hash mismatch")
    for item in config["authenticated_inputs"].values():
        path = root / item["path"]
        if not path.is_file():
            if not require_large_inputs and item.get("large_remote_input", False):
                continue
            raise PairValidationError("P4B authenticated input is missing")
        if file_sha256(path) != item["sha256"]:
            raise PairValidationError("P4B authenticated input changed")
    if tuple(tuple(value) for value in config["schedule"]["slot_design"]) != SLOT_DESIGN:
        raise PairValidationError("P4B slot design changed")
    if config["model_config"] != "configs/pair/p3_physical_v03.json":
        raise PairValidationError("P4B model configuration changed")
    if config["schedule"]["instruction_projection_seed"] != 20260828:
        raise PairValidationError("P4B instruction projection changed")
    if config["artifacts"] != {
        "schedule": "configs/pair/p4b_schedule_v01.jsonl",
        "schedule_summary": "configs/pair/p4b_schedule_v01.summary.json",
        "preflight": "reports/pair_p4b/preflight_v01.json",
        "launch_manifest": "configs/pair/p4b_launch_v01.json",
        "run_root": "results/pair-p4b-confirmatory-v01",
    }:
        raise PairValidationError("P4B artifact boundary changed")
    counts = expected_counts(int(config["schedule"]["seed"]))
    if counts != config["accounting"] or counts["planned_model_calls"] != 1784:
        raise PairValidationError("P4B call or record accounting changed")
    if config["population"] != {
        "task_count": 40,
        "suite_count": 4,
        "structured_split": "calibration",
        "structured_trajectories_per_task": 6,
        "structured_trajectory_count": 240,
        "structured_trajectories_per_suite": 60,
        "control_split": "train",
        "control_trajectories_per_task": 1,
        "control_trajectory_count": 40,
        "locked_test_trajectory_access": 0,
        "locked_test_trajectories_preserved": 280,
        "base_anchor_overlap": 0,
        "consume_once": True,
    }:
        raise PairValidationError("P4B independent population changed")
    if config["router"] != {
        "artifact_id": "76a28416f57e94a86511f003772c5887417a3914218fd416e7bfb580b6f14519",
        "checkpoint_sha256": "ec1f83ea76c6977e2401db212e8e5373966ee491eea660d5017ec7ff030b435f",
        "training_seed": 43,
        "retraining": False,
        "proxy_name": "projected_change",
        "proxy_orientation": -1,
        "matched_service_rate": 0.70,
        "selection_cell": "suite_by_horizon",
        "contracts_per_cell": 20,
        "served_per_cell": 14,
    }:
        raise PairValidationError("P4B frozen router or comparator changed")
    if config["resource_caps"] != {
        "gpu_count": 1,
        "model_processes": 1,
        "model_call_hard_cap": 1900,
        "wall_seconds": 28800,
        "artifact_bytes": 4294967296,
        "downloads": 0,
        "simulator_outcomes": 0,
        "automatic_retry": False,
    }:
        raise PairValidationError("P4B resource boundary changed")
    if config["protection"] != {
        "scientific_analysis_before_completed_worker_summary": False,
        "worker_label_access_only_after_frozen_schedule_and_controls": True,
        "raw_action_arrays_persisted": False,
        "terminal_outcomes_accessed": False,
        "simulator_accessed": False,
        "locked_test_labels_accessed": False,
        "network_accessed": False,
        "p4_router_mutation": False,
        "partial_scientific_monitoring": False,
    }:
        raise PairValidationError("P4B protection or unblinding boundary changed")
    if config["planning"] != {
        "sample_sizes": [48, 84, 120, 156, 204, 240],
        "empirical_replicates": 20000,
        "empirical_seed": 20260830,
        "lower_quantile": 0.05,
        "fisher_null_correlation": 0.15,
        "fisher_assumed_correlations": [0.3452157598499062, 0.30, 0.275],
        "fisher_powers": [0.80, 0.90],
    }:
        raise PairValidationError("P4B planning audit settings changed")
    if config["advance"] != {
        "next_phase": "P5",
        "authorized": False,
        "stop_before_next_phase": True,
        "p4c_permitted": False,
    }:
        raise PairValidationError("P4B terminal decision boundary changed")
