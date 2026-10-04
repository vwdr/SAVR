"""Exact PAIR expert-regret and action diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from savr.pair.types import PairValidationError


@dataclass(frozen=True)
class RegretDiagnostics:
    valid_action_count: int
    dense_expert_l1: float
    cached_expert_l1: float
    signed_regret: float
    positive_regret: float
    action_l1_distortion: float
    action_max_abs_distortion: float
    gripper_coordinate_error: float


def normalized_l1_regret(
    expert: np.ndarray,
    dense: np.ndarray,
    cached: np.ndarray,
    valid_mask: np.ndarray,
) -> RegretDiagnostics:
    target = np.asarray(expert, dtype=np.float64)
    reference = np.asarray(dense, dtype=np.float64)
    candidate = np.asarray(cached, dtype=np.float64)
    mask = np.asarray(valid_mask, dtype=bool)
    if target.shape != (8, 7) or reference.shape != target.shape or candidate.shape != target.shape:
        raise PairValidationError("PAIR actions must have exact 8x7 shape")
    if mask.shape == (8,):
        mask = np.broadcast_to(mask[:, None], target.shape)
    if mask.shape != target.shape or not mask.any():
        raise PairValidationError("action validity mask is empty or misaligned")
    if not all(np.isfinite(value).all() for value in (target, reference, candidate)):
        raise PairValidationError("action diagnostic contains nonfinite values")
    dense_error = np.abs(target - reference)[mask]
    cached_error = np.abs(target - candidate)[mask]
    distortion = np.abs(reference - candidate)[mask]
    signed = float(cached_error.mean() - dense_error.mean())
    gripper_mask = mask[:, -1]
    gripper_error = (
        float(np.mean(np.abs(reference[:, -1][gripper_mask] - candidate[:, -1][gripper_mask])))
        if gripper_mask.any()
        else 0.0
    )
    return RegretDiagnostics(
        valid_action_count=int(mask.sum()),
        dense_expert_l1=float(dense_error.mean()),
        cached_expert_l1=float(cached_error.mean()),
        signed_regret=signed,
        positive_regret=max(0.0, signed),
        action_l1_distortion=float(distortion.mean()),
        action_max_abs_distortion=float(distortion.max()),
        gripper_coordinate_error=gripper_error,
    )
