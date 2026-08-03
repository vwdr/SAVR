"""Deterministic Version 1 scene, state, transition, and audit signals."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Sequence

from savr.signals import (
    IMAGE_SIZE,
    SignalValidationError,
    action_transition,
    downsample_image,
    freeze_numeric,
    normalize_bounds,
)


GRID_SIZE = 8
TOP_K = 4


@dataclass(frozen=True)
class SceneChange:
    representation: tuple[float, ...]
    patch_scores: tuple[float, ...]
    top_four_mean: float
    global_mean: float


@dataclass(frozen=True)
class TransitionSignal:
    gripper_veto: bool
    mixed_latest_gripper: bool
    final_gripper_changed: bool
    translation_direction_reversals: tuple[bool, bool, bool]


def prepare_scene_representation(scene_image: Any) -> tuple[float, ...]:
    return downsample_image(scene_image, size=IMAGE_SIZE)


def _channels(representation: Sequence[float]) -> int:
    pixels = IMAGE_SIZE * IMAGE_SIZE
    if len(representation) < pixels or len(representation) % pixels:
        raise SignalValidationError("Scene representation has incompatible dimensions")
    return len(representation) // pixels


def scene_change_from_representations(
    current: Sequence[float], reference: Sequence[float]
) -> SceneChange:
    current_values = freeze_numeric(current)
    reference_values = freeze_numeric(reference)
    if len(current_values) != len(reference_values):
        raise SignalValidationError("Scene representation shape changed")
    channels = _channels(current_values)
    patch_width = IMAGE_SIZE // GRID_SIZE
    patch_scores: list[float] = []
    total = 0.0
    for patch_y in range(GRID_SIZE):
        for patch_x in range(GRID_SIZE):
            differences: list[float] = []
            count = patch_width * patch_width * channels
            for offset_y in range(patch_width):
                y = patch_y * patch_width + offset_y
                for offset_x in range(patch_width):
                    x = patch_x * patch_width + offset_x
                    start = (y * IMAGE_SIZE + x) * channels
                    for channel in range(channels):
                        difference = abs(
                            current_values[start + channel]
                            - reference_values[start + channel]
                        )
                        differences.append(difference)
            patch_total = math.fsum(differences)
            total += patch_total
            patch_scores.append(patch_total / count)
    largest = sorted(patch_scores, reverse=True)[:TOP_K]
    return SceneChange(
        representation=current_values,
        patch_scores=tuple(patch_scores),
        top_four_mean=math.fsum(largest) / TOP_K,
        global_mean=total / len(current_values),
    )


def scene_change(scene_image: Any, reference: Sequence[float]) -> SceneChange:
    return scene_change_from_representations(
        prepare_scene_representation(scene_image), reference
    )


def normalized_eef_position(
    state: Any,
    q01: Sequence[float],
    q99: Sequence[float],
) -> tuple[float, float, float]:
    if len(q01) != 8 or len(q99) != 8:
        raise SignalValidationError("ACR proprioception statistics require eight dimensions")
    normalized = normalize_bounds(state, q01, q99)
    if len(normalized) != 8:
        raise SignalValidationError("ACR proprioception requires exactly eight dimensions")
    return normalized[0], normalized[1], normalized[2]


def scene_relative_translation(
    current: Sequence[float], reference: Sequence[float]
) -> float:
    current_values = freeze_numeric(current)
    reference_values = freeze_numeric(reference)
    if len(current_values) != 3 or len(reference_values) != 3:
        raise SignalValidationError("EEF translation requires two three-dimensional positions")
    return math.sqrt(
        sum((left - right) ** 2 for left, right in zip(current_values, reference_values))
    )


def transition_signal(newer_chunk: Any, older_chunk: Any) -> TransitionSignal:
    transition = action_transition(newer_chunk, older_chunk)
    return TransitionSignal(
        gripper_veto=transition.gripper_veto,
        mixed_latest_gripper=transition.mixed_latest_gripper,
        final_gripper_changed=transition.final_gripper_changed,
        translation_direction_reversals=transition.translation_direction_reversals,
    )


def audit_sha256(value: Any) -> str:
    """Hash array-like values with stable shape/value semantics."""

    if hasattr(value, "detach") and callable(value.detach):
        value = value.detach()
    if hasattr(value, "cpu") and callable(value.cpu):
        value = value.cpu()
    if hasattr(value, "numpy") and callable(value.numpy):
        value = value.numpy()
    if hasattr(value, "tolist") and callable(value.tolist):
        value = value.tolist()
    frozen = freeze_numeric(value)
    payload = json.dumps(frozen, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
