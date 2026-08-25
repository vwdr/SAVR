"""Detached synthetic sidecar attention and dense-gate identities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from savr.brace.b1 import semantic_sha256
from savr.brace.types import B2ValidationError


@dataclass(frozen=True)
class DenseGateIdentity:
    anchor_query: int
    model_sha256: str
    sequence_map_sha256: str
    backend: str
    selected_layers: tuple[int, ...]
    semantic_sha256: str

    def age(self, current_query: int) -> int:
        if current_query < self.anchor_query:
            raise B2ValidationError("dense gate cannot predate its anchor")
        return current_query - self.anchor_query


def sidecar_attention(
    post_rope_query: np.ndarray,
    post_rope_key: np.ndarray,
    *,
    additive_mask: np.ndarray | None = None,
    scale: float | None = None,
) -> np.ndarray:
    """Reproduce attention probabilities from detached post-RoPE Q/K tensors."""

    query = np.asarray(post_rope_query, dtype=np.float64)
    key = np.asarray(post_rope_key, dtype=np.float64)
    if query.ndim != 4 or key.ndim != 4 or query.shape[:2] != key.shape[:2]:
        raise B2ValidationError("sidecar Q/K must be [batch,head,sequence,dimension]")
    if query.shape[-1] != key.shape[-1] or not np.isfinite(query).all() or not np.isfinite(key).all():
        raise B2ValidationError("sidecar Q/K dimensions or values are invalid")
    factor = float(scale) if scale is not None else query.shape[-1] ** -0.5
    if not np.isfinite(factor) or factor <= 0:
        raise B2ValidationError("sidecar attention scale is invalid")
    logits = np.matmul(query, np.swapaxes(key, -1, -2)) * factor
    if additive_mask is not None:
        mask = np.asarray(additive_mask, dtype=np.float64)
        try:
            logits = logits + mask
        except ValueError as error:
            raise B2ValidationError("sidecar mask does not broadcast to attention logits") from error
    if np.isnan(logits).any() or np.isposinf(logits).any():
        raise B2ValidationError("sidecar logits contain invalid values")
    maximum = np.max(logits, axis=-1, keepdims=True)
    exponent = np.exp(logits - maximum)
    denominator = np.sum(exponent, axis=-1, keepdims=True)
    if np.any(denominator <= 0) or not np.isfinite(denominator).all():
        raise B2ValidationError("sidecar attention row is fully masked or invalid")
    return exponent / denominator


def _rank_normalize(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(values.size, dtype=np.float64)
    return ranks / max(values.size - 1, 1)


def semantic_salience(
    probabilities: Sequence[np.ndarray],
    *,
    instruction_positions: Sequence[int],
    action_positions: Sequence[int],
    visual_positions: Sequence[int],
) -> np.ndarray:
    """Max-combine separately rank-normalized instruction/action salience."""

    if not probabilities:
        raise B2ValidationError("semantic sidecar requires at least one dense layer")
    instruction = tuple(int(value) for value in instruction_positions)
    action = tuple(int(value) for value in action_positions)
    visual = tuple(int(value) for value in visual_positions)
    if not instruction or not action or not visual:
        raise B2ValidationError("semantic sidecar runtime spans are empty")
    family_scores: list[np.ndarray] = []
    for queries in (instruction, action):
        layer_scores = []
        for raw in probabilities:
            attention = np.asarray(raw, dtype=np.float64)
            if attention.ndim != 4:
                raise B2ValidationError("attention probability tensor must be rank four")
            if max((*queries, *visual)) >= attention.shape[-1]:
                raise B2ValidationError("runtime sequence map exceeds the captured attention")
            layer_scores.append(attention[:, :, queries, :][:, :, :, visual].mean(axis=(0, 1, 2)))
        family_scores.append(_rank_normalize(np.stack(layer_scores).mean(axis=0)))
    return np.maximum(family_scores[0], family_scores[1])


def gate_identity(
    *,
    anchor_query: int,
    model_sha256: str,
    sequence_map_sha256: str,
    backend: str,
    selected_layers: Sequence[int],
) -> DenseGateIdentity:
    if anchor_query < 0 or not backend or not selected_layers:
        raise B2ValidationError("dense gate identity is incomplete")
    payload = {
        "anchor_query": anchor_query,
        "model_sha256": model_sha256,
        "sequence_map_sha256": sequence_map_sha256,
        "backend": backend,
        "selected_layers": [int(value) for value in selected_layers],
    }
    return DenseGateIdentity(**payload, semantic_sha256=semantic_sha256(payload))
