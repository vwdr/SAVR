"""Deterministic low-cost signal calculations for refresh controllers."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any

_np: Any
try:
    import numpy as _np
except ImportError:  # pragma: no cover - exercised by dependency-free CI
    _np = None


IMAGE_SIZE = 32


class SignalValidationError(ValueError):
    """Raised when signal inputs are missing, non-finite, or incompatible."""


@dataclass(frozen=True)
class ImageChange:
    mean: float
    per_camera: dict[str, float]


def _finite_float(value: Any) -> float:
    if not isinstance(value, Real):
        raise SignalValidationError(f"Expected a real number, found {type(value).__name__}")
    result = float(value)
    if not math.isfinite(result):
        raise SignalValidationError("Signal data must be finite")
    return result


def _to_nested_list(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _flatten(value: Any) -> list[float]:
    value = _to_nested_list(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        result: list[float] = []
        for item in value:
            result.extend(_flatten(item))
        return result
    return [_finite_float(value)]


def _sample_index(position: int, source_size: int, target_size: int) -> int:
    if source_size <= 0:
        raise SignalValidationError("Image dimensions must be non-empty")
    if target_size == 1:
        return 0
    return round(position * (source_size - 1) / (target_size - 1))


def _normalize_pixels(values: list[float]) -> tuple[float, ...]:
    if not values:
        raise SignalValidationError("Image must contain pixels")
    low, high = min(values), max(values)
    if low < 0 or high > 255:
        raise SignalValidationError("Pixels must lie in [0,1] or [0,255]")
    scale = 1.0 if high <= 1.0 else 255.0
    return tuple(value / scale for value in values)


def downsample_image(image: Any, size: int = IMAGE_SIZE) -> tuple[float, ...]:
    """Return a deterministic nearest-sampled HxWxC representation in [0,1]."""

    if size <= 0:
        raise SignalValidationError("Downsample size must be positive")

    if _np is not None:
        try:
            array = _np.asarray(image)
            if array.ndim not in (2, 3):
                raise SignalValidationError("Image must have shape HxW or HxWxC")
            if array.shape[0] == 0 or array.shape[1] == 0:
                raise SignalValidationError("Image dimensions must be non-empty")
            if not _np.issubdtype(array.dtype, _np.number):
                raise SignalValidationError("Image pixels must be numeric")
            array = array.astype(_np.float64, copy=False)
            if not _np.isfinite(array).all():
                raise SignalValidationError("Image pixels must be finite")
            low, high = float(array.min()), float(array.max())
            if low < 0 or high > 255:
                raise SignalValidationError("Pixels must lie in [0,1] or [0,255]")
            if high > 1:
                array = array / 255.0
            y_indices = [_sample_index(i, array.shape[0], size) for i in range(size)]
            x_indices = [_sample_index(i, array.shape[1], size) for i in range(size)]
            sampled = array[_np.ix_(y_indices, x_indices)]
            return tuple(float(value) for value in sampled.reshape(-1))
        except SignalValidationError:
            raise
        except Exception as error:
            raise SignalValidationError(f"Invalid image: {error}") from error

    rows = _to_nested_list(image)
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)):
        raise SignalValidationError("Image must be a nested sequence")
    if not rows:
        raise SignalValidationError("Image height must be non-empty")
    first_row = _to_nested_list(rows[0])
    if not isinstance(first_row, Sequence) or isinstance(
        first_row, (str, bytes, bytearray)
    ):
        raise SignalValidationError("Image must have shape HxW or HxWxC")
    width = len(first_row)
    if width == 0:
        raise SignalValidationError("Image width must be non-empty")
    for row in rows:
        if len(_to_nested_list(row)) != width:
            raise SignalValidationError("Image rows must have equal width")

    sampled_values: list[float] = []
    for output_y in range(size):
        source_y = _sample_index(output_y, len(rows), size)
        row = _to_nested_list(rows[source_y])
        for output_x in range(size):
            source_x = _sample_index(output_x, width, size)
            sampled_values.extend(_flatten(row[source_x]))
    return _normalize_pixels(sampled_values)


def prepare_image_representations(
    images: Mapping[str, Any], size: int = IMAGE_SIZE
) -> dict[str, tuple[float, ...]]:
    if not images:
        raise SignalValidationError("At least one camera image is required")
    return {name: downsample_image(image, size=size) for name, image in images.items()}


def image_change(
    current_images: Mapping[str, Any],
    reference_representations: Mapping[str, Sequence[float]],
    size: int = IMAGE_SIZE,
) -> ImageChange:
    current = prepare_image_representations(current_images, size=size)
    if set(current) != set(reference_representations):
        raise SignalValidationError("Current and reference camera sets differ")

    per_camera: dict[str, float] = {}
    for name in sorted(current):
        reference = tuple(_finite_float(value) for value in reference_representations[name])
        if len(current[name]) != len(reference):
            raise SignalValidationError(f"Camera representation shape changed: {name}")
        per_camera[name] = sum(
            abs(current_value - reference_value)
            for current_value, reference_value in zip(current[name], reference)
        ) / len(current[name])
    return ImageChange(
        mean=sum(per_camera.values()) / len(per_camera),
        per_camera=per_camera,
    )


def normalize_bounds(
    values: Any,
    q01: Sequence[float],
    q99: Sequence[float],
) -> tuple[float, ...]:
    flattened = _flatten(values)
    lows = tuple(_finite_float(value) for value in q01)
    highs = tuple(_finite_float(value) for value in q99)
    if not lows or len(lows) != len(highs):
        raise SignalValidationError("q01/q99 statistics are missing or incompatible")
    if len(flattened) % len(lows) != 0:
        raise SignalValidationError(
            f"Value length {len(flattened)} is not divisible by statistic width {len(lows)}"
        )

    normalized = []
    for index, value in enumerate(flattened):
        low, high = lows[index % len(lows)], highs[index % len(highs)]
        if high <= low:
            raise SignalValidationError("Every q99 value must exceed q01")
        scaled = 2 * (value - low) / (high - low + 1e-8) - 1
        normalized.append(min(1.0, max(-1.0, scaled)))
    return tuple(normalized)


def rms_change(first: Sequence[float], second: Sequence[float]) -> float:
    if not first or len(first) != len(second):
        raise SignalValidationError("RMS inputs must be non-empty and shape-compatible")
    return math.sqrt(
        sum((left - right) ** 2 for left, right in zip(first, second)) / len(first)
    )


def state_change(
    current_state: Any,
    previous_state: Any,
    q01: Sequence[float],
    q99: Sequence[float],
) -> float:
    current = normalize_bounds(current_state, q01, q99)
    previous = normalize_bounds(previous_state, q01, q99)
    if len(current) != len(q01) or len(previous) != len(q01):
        raise SignalValidationError("State must contain exactly one statistic-width vector")
    return rms_change(current, previous)


def action_change(
    newer_chunk: Any,
    older_chunk: Any,
    q01: Sequence[float],
    q99: Sequence[float],
) -> float:
    newer = normalize_bounds(newer_chunk, q01, q99)
    older = normalize_bounds(older_chunk, q01, q99)
    return rms_change(newer, older)


def freeze_numeric(value: Any) -> tuple[float, ...]:
    """Validate and copy arbitrary numeric array-like input."""

    return tuple(_flatten(value))
