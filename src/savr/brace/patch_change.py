"""Exact preprocessed-patch source-change operator for BRACE."""

from __future__ import annotations

import numpy as np

from savr.brace.types import B2ValidationError


def patch_change_scores(
    current: np.ndarray,
    source: np.ndarray,
    *,
    lower: np.ndarray | float,
    upper: np.ndarray | float,
    l1_weight: float,
    cosine_weight: float,
    epsilon: float = 1e-12,
) -> np.ndarray:
    """Compute bounded L1/cosine change against each entry's actual source."""

    left = np.asarray(current, dtype=np.float64)
    right = np.asarray(source, dtype=np.float64)
    if left.shape != right.shape or left.ndim < 2:
        raise B2ValidationError("current/source patch tensors must have equal [patch,...] shape")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise B2ValidationError("nonfinite patch input invalidates the contract")
    if l1_weight < 0 or cosine_weight < 0 or not np.isclose(l1_weight + cosine_weight, 1):
        raise B2ValidationError("patch-change weights must be nonnegative and sum to one")
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise B2ValidationError("cosine epsilon must be finite and positive")

    low = np.broadcast_to(np.asarray(lower, dtype=np.float64), left.shape[1:])
    high = np.broadcast_to(np.asarray(upper, dtype=np.float64), left.shape[1:])
    if not np.isfinite(low).all() or not np.isfinite(high).all() or np.any(high <= low):
        raise B2ValidationError("preprocessed coordinate bounds are invalid")
    denominator = float(np.sum(high - low))
    flat_left = left.reshape(left.shape[0], -1)
    flat_right = right.reshape(right.shape[0], -1)
    l1 = np.sum(np.abs(flat_left - flat_right), axis=1) / denominator

    left_norm = np.linalg.norm(flat_left, axis=1)
    right_norm = np.linalg.norm(flat_right, axis=1)
    both_zero = (left_norm <= epsilon) & (right_norm <= epsilon)
    one_zero = (left_norm <= epsilon) ^ (right_norm <= epsilon)
    cosine = np.sum(flat_left * flat_right, axis=1) / np.maximum(
        left_norm * right_norm, epsilon
    )
    cosine = np.clip(cosine, -1, 1)
    cosine_change = (1 - cosine) / 2
    cosine_change[both_zero] = 0
    cosine_change[one_zero] = 1

    result = np.clip(l1_weight * np.clip(l1, 0, 1) + cosine_weight * cosine_change, 0, 1)
    if not np.isfinite(result).all():
        raise B2ValidationError("nonfinite patch score invalidates the contract")
    return result
