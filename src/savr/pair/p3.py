"""Frozen scheduling and outcome-blind analysis for PAIR Phase P3."""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from savr.pair.types import HORIZONS, ONSET_LAYERS, PairValidationError, ProfileSpec


FORBIDDEN_P3_FIELDS = {
    "success",
    "terminal_success",
    "reward",
    "task_success",
    "episode_outcome",
    "expert_action",
    "expert_actions",
    "action_values",
    "action_parity",
    "action_difference",
    "gripper_decisions",
}


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def semantic_sha256(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("semantic_sha256", None)
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def reject_protected_fields(value: Any, *, path: str = "p3") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).lower() in FORBIDDEN_P3_FIELDS:
                raise PairValidationError(f"protected P3 field at {path}.{key}")
            reject_protected_fields(item, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            reject_protected_fields(item, path=f"{path}[{index}]")


@dataclass(frozen=True)
class P3Profile:
    profile_id: str
    base_profile_id: str
    primary_budgets: tuple[int, ...]
    wrist_budgets: tuple[int, ...]
    protected_primary_tiles: int
    protected_wrist_tiles: int

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "P3Profile":
        result = cls(
            str(value["profile_id"]),
            str(value["base_profile_id"]),
            tuple(int(item) for item in value["primary_budgets"]),
            tuple(int(item) for item in value["wrist_budgets"]),
            int(value["protected_primary_tiles"]),
            int(value["protected_wrist_tiles"]),
        )
        result.validate()
        return result

    def validate(self) -> None:
        ProfileSpec(self.profile_id, self.primary_budgets, self.wrist_budgets).validate()
        if not self.base_profile_id:
            raise PairValidationError("P3 base profile identity is empty")
        for protected, budgets in (
            (self.protected_primary_tiles, self.primary_budgets),
            (self.protected_wrist_tiles, self.wrist_budgets),
        ):
            if protected < 0 or protected > 4:
                raise PairValidationError(
                    "protected-tile allocation is outside the frozen frontier"
                )
            if protected + budgets[-1] // 16 > 16:
                raise PairValidationError("protected tiles leave too few tiles for the profile")


@dataclass(frozen=True)
class P3Block:
    block_id: str
    profile_id: str
    horizon: int
    repetition: int
    input_index: int
    arm_order: tuple[str, str]


def block_schedule(config: Mapping[str, Any], input_count: int) -> tuple[P3Block, ...]:
    profiles = tuple(P3Profile.from_mapping(value) for value in config["profiles"])
    repetitions = int(config["measurement"]["timed_repetitions_per_profile_horizon"])
    seed = int(config["measurement"]["seed"])
    if input_count < 4 or repetitions != 4:
        raise PairValidationError("P3 input or repetition count changed")
    rows = [
        (profile.profile_id, horizon, repetition)
        for profile in profiles
        for horizon in HORIZONS
        for repetition in range(repetitions)
    ]
    random.Random(seed).shuffle(rows)
    blocks = []
    for index, (profile_id, horizon, repetition) in enumerate(rows):
        order = (
            ("dense", "cache") if random.Random(seed + index).random() < 0.5 else ("cache", "dense")
        )
        payload = {
            "profile_id": profile_id,
            "horizon": horizon,
            "repetition": repetition,
            "input_index": index % input_count,
            "arm_order": order,
        }
        blocks.append(P3Block(semantic_sha256(payload), **payload))
    return tuple(blocks)


def planned_query_count(config: Mapping[str, Any], input_count: int) -> int:
    measurement = config["measurement"]
    blocks = block_schedule(config, input_count)
    timed = sum(2 * (1 + block.horizon) for block in blocks)
    warmup = int(measurement["model_warmup_queries"])
    controls = int(measurement["control_queries"])
    profile_warmup = len(config["profiles"]) * (1 + int(measurement["profile_warmup_horizon"]))
    return warmup + controls + profile_warmup + timed


@dataclass
class QueryLedger:
    hard_cap: int
    planned: int

    def __post_init__(self) -> None:
        self.used = 0
        if self.planned > self.hard_cap:
            raise PairValidationError("P3 planned queries exceed the hard cap")

    def consume(self) -> None:
        if self.used >= self.planned or self.used >= self.hard_cap:
            raise PairValidationError("P3 model-query allocation exceeded")
        self.used += 1

    def require_complete(self) -> None:
        if self.used != self.planned:
            raise PairValidationError("P3 model-query ledger is incomplete")


def validate_config(config: Mapping[str, Any], *, input_count: int) -> None:
    if config.get("schema_version") != "pair-p3-physical-config-v1":
        raise PairValidationError("P3 configuration schema changed")
    if config.get("semantic_sha256") != semantic_sha256(config):
        raise PairValidationError("P3 configuration semantic hash mismatch")
    if tuple(config["measurement"]["horizons"]) != HORIZONS:
        raise PairValidationError("P3 horizons changed")
    profiles = tuple(P3Profile.from_mapping(value) for value in config["profiles"])
    if len(profiles) != 8 or len({item.profile_id for item in profiles}) != 8:
        raise PairValidationError("P3 requires exactly eight unique frontier profiles")
    if tuple(config["model"]["onset_layers"]) != ONSET_LAYERS:
        raise PairValidationError("P3 onset layers changed")
    caps = config["resource_caps"]
    expected_caps = {
        "gpu_count": 1,
        "model_processes": 1,
        "model_query_hard_cap": 700,
        "wall_seconds": 21600,
        "artifact_bytes": 2147483648,
        "peak_gpu_memory_mib_strict_max": 23552,
        "downloads": 0,
        "simulator_outcomes": 0,
        "protected_outcome_access": False,
        "automatic_retry": False,
    }
    if caps != expected_caps:
        raise PairValidationError("P3 resource boundary changed")
    if planned_query_count(config, input_count) != int(
        config["measurement"]["planned_model_queries"]
    ):
        raise PairValidationError("P3 planned query accounting changed")
    if config["advance"] != {
        "next_phase": "P4",
        "authorized": False,
        "stop_before_next_phase": True,
    }:
        raise PairValidationError("P3 advance boundary changed")


def one_sided_bootstrap_lower(
    dense_ms: Sequence[float],
    cache_ms: Sequence[float],
    *,
    replicates: int,
    seed: int,
) -> tuple[float, float]:
    dense = np.asarray(dense_ms, dtype=np.float64)
    cache = np.asarray(cache_ms, dtype=np.float64)
    if dense.shape != cache.shape or dense.ndim != 1 or len(dense) < 4:
        raise PairValidationError("paired P3 timing arrays are insufficient or misaligned")
    if (
        not np.isfinite(dense).all()
        or not np.isfinite(cache).all()
        or np.any(dense <= 0)
        or np.any(cache <= 0)
    ):
        raise PairValidationError("P3 timing arrays are nonfinite or nonpositive")
    point = 1.0 - float(cache.sum() / dense.sum())
    rng = np.random.default_rng(seed)
    values = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        sample = rng.integers(0, len(dense), len(dense))
        values[index] = 1.0 - cache[sample].sum() / dense[sample].sum()
    return point, float(np.quantile(values, 0.05))


def net_saving_lower(raw_lower: float, overhead_fraction_upper: float) -> float:
    if not math.isfinite(raw_lower) or not 0 <= overhead_fraction_upper <= 1:
        raise PairValidationError("P3 net-speed inputs are invalid")
    return 0.70 * raw_lower - overhead_fraction_upper


def profile_source_diversity(profile: P3Profile) -> dict[str, int | bool]:
    fresh_primary_tiles = 16 - profile.primary_budgets[-1] // 16
    fresh_wrist_tiles = 16 - profile.wrist_budgets[-1] // 16
    return {
        "fresh_primary_tiles_deepest": fresh_primary_tiles,
        "fresh_wrist_tiles_deepest": fresh_wrist_tiles,
        "mixed_source_possible": fresh_primary_tiles > 0
        and fresh_wrist_tiles > 0
        and (profile.primary_budgets[-1] + profile.wrist_budgets[-1]) > 0,
    }
