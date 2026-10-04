"""Chronology-safe, deployment-available PAIR router features."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from savr.pair.types import AtomicGroup, HORIZONS, PairValidationError


GROUP_CONTINUOUS_NAMES = (
    "source_age",
    "raw_l1",
    "raw_cosine_change",
    "raw_mean_change",
    "raw_max_change",
    "raw_q90_change",
    "projected_l1",
    "projected_l2",
    "projected_cosine_change",
    "projected_norm_change",
    "projected_max_change",
    "projected_q90_change",
    "proprio_l1",
    "proprio_linf",
    "previous_action_l1",
    "previous_action_linf",
    "previous_salience",
    "gripper_transition",
    "remaining_fraction",
    "source_mixture_fraction",
)
GLOBAL_CONTINUOUS_NAMES = tuple(f"instruction_projection_{index:02d}" for index in range(16))
FORBIDDEN_FEATURE_FRAGMENTS = (
    "expert",
    "future",
    "success",
    "reward",
    "outcome",
    "current_dense",
    "dense_action",
    "current_logits",
    "current_attention",
    "future_observation",
)


def feature_schema_sha256() -> str:
    payload = {
        "schema_version": "pair-feature-v1",
        "group_continuous": GROUP_CONTINUOUS_NAMES,
        "global_continuous": GLOBAL_CONTINUOUS_NAMES,
        "categorical": ["camera", "tile", "onset_layer", "profile", "horizon"],
    }
    data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


def reject_forbidden_feature_fields(value: Any, *, path: str = "feature") -> None:
    """Recursively reject outcome, future, expert, or hidden-dense inputs."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).lower()
            if any(fragment in normalized for fragment in FORBIDDEN_FEATURE_FRAGMENTS):
                raise PairValidationError(f"forbidden router feature at {path}.{key}")
            reject_forbidden_feature_fields(item, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            reject_forbidden_feature_fields(item, path=f"{path}[{index}]")


@dataclass(frozen=True)
class GroupFeature:
    group: AtomicGroup
    profile_id: str
    horizon: int
    continuous: tuple[float, ...]

    def validate(self, *, supported_profiles: Sequence[str] | None = None) -> None:
        self.group.validate()
        if self.horizon not in HORIZONS:
            raise PairValidationError("unsupported horizon category")
        if not self.profile_id or (
            supported_profiles is not None and self.profile_id not in set(supported_profiles)
        ):
            raise PairValidationError("unsupported profile category")
        if len(self.continuous) != len(GROUP_CONTINUOUS_NAMES):
            raise PairValidationError("group feature width differs from the frozen schema")
        if not np.isfinite(np.asarray(self.continuous, dtype=np.float64)).all():
            raise PairValidationError("group feature contains nonfinite values")


@dataclass(frozen=True)
class RouterFeatures:
    groups: tuple[GroupFeature, ...]
    global_continuous: tuple[float, ...]

    def validate(self, *, supported_profiles: Sequence[str] | None = None) -> None:
        if not self.groups:
            raise PairValidationError("router input has no selected group")
        if len(self.groups) > 128:
            raise PairValidationError("router input exceeds the frozen group cap")
        for group in self.groups:
            group.validate(supported_profiles=supported_profiles)
        if len(self.global_continuous) != len(GLOBAL_CONTINUOUS_NAMES):
            raise PairValidationError("global feature width differs from the frozen schema")
        if not np.isfinite(np.asarray(self.global_continuous, dtype=np.float64)).all():
            raise PairValidationError("global feature contains nonfinite values")


def _change_statistics(current: np.ndarray, source: np.ndarray) -> tuple[float, ...]:
    left = np.asarray(current, dtype=np.float64).reshape(-1)
    right = np.asarray(source, dtype=np.float64).reshape(-1)
    if left.shape != right.shape or not left.size:
        raise PairValidationError("current/source feature tensors are misaligned")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise PairValidationError("current/source feature tensors contain nonfinite values")
    difference = np.abs(left - right)
    denominator = max(left.size, 1)
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm == 0 and right_norm == 0:
        cosine_change = 0.0
    elif left_norm == 0 or right_norm == 0:
        cosine_change = 1.0
    else:
        cosine_change = (1.0 - float(np.clip(left @ right / (left_norm * right_norm), -1, 1))) / 2
    return (
        float(difference.sum() / denominator),
        cosine_change,
        float(difference.mean()),
        float(difference.max()),
        float(np.quantile(difference, 0.9)),
    )


def build_group_feature(
    *,
    group: AtomicGroup,
    profile_id: str,
    horizon: int,
    source_age: int,
    current_raw_tile: np.ndarray,
    source_raw_tile: np.ndarray,
    current_projected_tile: np.ndarray,
    source_projected_tile: np.ndarray,
    current_proprio: Sequence[float],
    source_proprio: Sequence[float],
    current_previous_action: Sequence[float],
    source_previous_action: Sequence[float],
    previous_salience: float,
    gripper_transition: bool,
    remaining: int,
    source_mixture_count: int,
) -> GroupFeature:
    """Construct one feature row using only values available before routing."""

    group.validate()
    if not 0 <= source_age <= 4 or horizon not in HORIZONS or not 0 <= remaining <= horizon:
        raise PairValidationError("feature chronology or contract state is invalid")
    if not 1 <= source_mixture_count <= 128:
        raise PairValidationError("source mixture count is unsupported")
    raw = _change_statistics(current_raw_tile, source_raw_tile)
    projected_base = _change_statistics(current_projected_tile, source_projected_tile)
    projected_difference = np.asarray(current_projected_tile, dtype=np.float64) - np.asarray(
        source_projected_tile, dtype=np.float64
    )
    projected = (
        projected_base[0],
        float(np.linalg.norm(projected_difference.reshape(-1))),
        projected_base[1],
        abs(
            float(np.linalg.norm(np.asarray(current_projected_tile)))
            - float(np.linalg.norm(np.asarray(source_projected_tile)))
        ),
        projected_base[3],
        projected_base[4],
    )
    proprio = np.asarray(current_proprio, dtype=np.float64) - np.asarray(
        source_proprio, dtype=np.float64
    )
    previous_action = np.asarray(current_previous_action, dtype=np.float64) - np.asarray(
        source_previous_action, dtype=np.float64
    )
    if (
        proprio.ndim != 1
        or previous_action.ndim != 1
        or not proprio.size
        or not previous_action.size
    ):
        raise PairValidationError("state/action feature dimensions are invalid")
    continuous = (
        float(source_age),
        *raw,
        *projected,
        float(np.mean(np.abs(proprio))),
        float(np.max(np.abs(proprio))),
        float(np.mean(np.abs(previous_action))),
        float(np.max(np.abs(previous_action))),
        float(previous_salience),
        float(bool(gripper_transition)),
        float(remaining / horizon),
        float(source_mixture_count / 128),
    )
    result = GroupFeature(group, profile_id, horizon, continuous)
    result.validate()
    return result


def fixed_instruction_projection(values: Sequence[float]) -> tuple[float, ...]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.shape != (16,) or not np.isfinite(array).all():
        raise PairValidationError("instruction projection must be a finite frozen 16-vector")
    return tuple(float(item) for item in array)
