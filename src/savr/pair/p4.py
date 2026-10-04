"""Frozen scheduling, accounting, and statistics for the PAIR P4 pilot."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from savr.pair.types import AtomicGroup, Camera, ONSET_LAYERS, PairValidationError


SLOT_DESIGN = (
    ("all_fresh_control", 1),
    ("all_fresh_control", 4),
    ("atomic", 1),
    ("atomic", 2),
    ("atomic", 4),
    ("structured", 1),
    ("structured", 2),
    ("structured", 4),
    ("structured", 2),
    ("structured", 4),
)


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
class SlotSpec:
    task_index: int
    slot: int
    category: str
    horizon: int
    split: str
    repeat: bool


def slot_spec(task_index: int, slot: int) -> SlotSpec:
    if not 0 <= task_index < 40 or not 0 <= slot < 10:
        raise PairValidationError("P4 task or slot is outside the frozen schedule")
    category, horizon = SLOT_DESIGN[slot]
    calibration = {task_index % 10, (task_index + 5) % 10}
    split = "calibration" if slot in calibration else "train"
    atomic_repeat = task_index < 12 and slot == 2 + task_index % 3
    structured_repeat = task_index < 20 and slot == 5 + task_index % 5
    return SlotSpec(
        task_index,
        slot,
        category,
        horizon,
        split,
        atomic_repeat or structured_repeat,
    )


def atomic_group(suite: str, task_id: str, slot: int, seed: int) -> AtomicGroup:
    payload = f"{suite}|{task_id}|{slot}|{seed}|atomic".encode()
    ordinal = int(hashlib.sha256(payload).hexdigest(), 16) % 128
    camera = Camera.PRIMARY if ordinal < 64 else Camera.WRIST
    within = ordinal % 64
    onset_layer = ONSET_LAYERS[within // 16]
    tile = within % 16
    result = AtomicGroup(camera, tile, onset_layer)
    result.validate()
    return result


def expected_counts() -> dict[str, int]:
    specs = [slot_spec(task, slot) for task in range(40) for slot in range(10)]
    horizon_sum = sum(spec.horizon for spec in specs)
    structured_horizon_sum = sum(
        spec.horizon for spec in specs if spec.category == "structured"
    )
    repeat_horizon_sum = sum(spec.horizon for spec in specs if spec.repeat)
    return {
        "anchors": len(specs),
        "train_anchors": sum(spec.split == "train" for spec in specs),
        "calibration_anchors": sum(spec.split == "calibration" for spec in specs),
        "controls": sum(spec.category == "all_fresh_control" for spec in specs),
        "atomic": sum(spec.category == "atomic" for spec in specs),
        "structured": sum(spec.category == "structured" for spec in specs),
        "repeats": sum(spec.repeat for spec in specs),
        "intervention_records": horizon_sum + 2 * structured_horizon_sum + repeat_horizon_sum,
        "contract_summaries": len(specs)
        + 2 * sum(spec.category == "structured" for spec in specs)
        + sum(spec.repeat for spec in specs),
        "base_noncontrol_feature_rows": sum(
            spec.horizon for spec in specs if spec.category != "all_fresh_control"
        ),
        "scheduled_model_calls": len(specs)
        + 2 * horizon_sum
        + 2 * structured_horizon_sum
        + repeat_horizon_sum,
        "planned_model_calls": len(specs)
        + 2 * horizon_sum
        + 2 * structured_horizon_sum
        + repeat_horizon_sum
        + 8,
    }


def validate_config(
    config: Mapping[str, Any], root: Path, *, require_large_inputs: bool = True
) -> None:
    if config.get("schema_version") != "pair-p4-pilot-config-v1":
        raise PairValidationError("P4 configuration schema changed")
    if config.get("semantic_sha256") != semantic_sha256(config):
        raise PairValidationError("P4 configuration semantic hash mismatch")
    if file_sha256(root / config["protocol"]) != config["protocol_sha256"]:
        raise PairValidationError("P4 protocol hash mismatch")
    for item in config["authenticated_inputs"].values():
        path = root / item["path"]
        if not path.is_file():
            if not require_large_inputs and item["path"].startswith("data/pair/index/"):
                continue
            raise PairValidationError("P4 authenticated input is missing")
        if file_sha256(path) != item["sha256"]:
            raise PairValidationError("P4 authenticated input changed")
    schedule = config["schedule"]
    if tuple(tuple(value) for value in schedule["slot_design"]) != SLOT_DESIGN:
        raise PairValidationError("P4 slot design changed")
    counts = expected_counts()
    expected = {
        "planned_model_calls": counts["planned_model_calls"],
        "model_call_hard_cap": 4000,
        "planned_intervention_records": counts["intervention_records"],
        "planned_contract_summaries": counts["contract_summaries"],
        "planned_base_noncontrol_feature_rows": counts["base_noncontrol_feature_rows"],
        "warmup_calls": 4,
        "technical_control_calls": 4,
    }
    if config["accounting"] != expected or counts["planned_model_calls"] != 3528:
        raise PairValidationError("P4 model-call or record accounting changed")
    if config["resource_caps"] != {
        "gpu_count": 1,
        "model_processes": 1,
        "wall_seconds": 28800,
        "artifact_bytes": 4294967296,
        "downloads": 0,
        "simulator_outcomes": 0,
        "locked_test_labels": 0,
        "automatic_retry": False,
    }:
        raise PairValidationError("P4 resource boundary changed")
    if config["advance"] != {
        "next_phase": "P5",
        "authorized": False,
        "stop_before_next_phase": True,
    }:
        raise PairValidationError("P4 advance boundary changed")


def instruction_projection(values: Any, *, seed: int = 20260828) -> tuple[float, ...]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.shape != (4096,) or not np.isfinite(array).all():
        raise PairValidationError("P4 instruction embedding is invalid")
    rng = np.random.default_rng(seed)
    matrix = rng.choice((-1.0, 1.0), size=(4096, 16)) / np.sqrt(4096.0)
    return tuple(float(value) for value in array @ matrix)


def cvar90(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all() or np.any(array < 0):
        raise PairValidationError("P4 CVaR population is invalid")
    count = max(1, int(np.ceil(0.10 * len(array))))
    return float(np.sort(array)[-count:].mean())
