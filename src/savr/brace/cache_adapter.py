"""Transactional, exact-clone adapters for mutable DynamicCache-like objects."""

from __future__ import annotations

import copy
from contextlib import contextmanager
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

from savr.brace.types import B2ValidationError


_MISSING = object()


def _clone_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, list):
        return [_clone_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_value(item) for item in value)
    if isinstance(value, dict):
        return {key: _clone_value(item) for key, item in value.items()}
    if hasattr(value, "detach") and hasattr(value, "clone"):
        return value.detach().clone()
    try:
        return copy.deepcopy(value)
    except Exception:
        return value


def clone_dynamic_cache(cache: Any) -> Any:
    """Clone every mutable cache field without sharing tensor storage."""

    if not hasattr(cache, "__dict__"):
        raise B2ValidationError("DynamicCache-like object lacks inspectable state")
    try:
        result = cache.__class__()
    except Exception:
        result = copy.copy(cache)
    for key, value in cache.__dict__.items():
        setattr(result, key, _clone_value(value))
    return result


def restore_dynamic_cache(target: Any, snapshot: Any) -> None:
    """Restore exact cache contents and delete fields added inside a transaction."""

    if not hasattr(target, "__dict__") or not hasattr(snapshot, "__dict__"):
        raise B2ValidationError("DynamicCache-like object lacks restorable state")
    for key in tuple(target.__dict__):
        if key not in snapshot.__dict__:
            delattr(target, key)
    for key, value in snapshot.__dict__.items():
        setattr(target, key, _clone_value(value))


@contextmanager
def transactional_cache_configuration(
    cache: Any,
    configuration: Any,
    updates: Mapping[str, Any],
) -> Iterator[Any]:
    """Isolate one arm and restore cache/configuration on success or failure."""

    cache_snapshot = clone_dynamic_cache(cache)
    config_snapshot = {key: getattr(configuration, key, _MISSING) for key in updates}
    arm_cache = clone_dynamic_cache(cache)
    try:
        for key, value in updates.items():
            setattr(configuration, key, _clone_value(value))
        yield arm_cache
    finally:
        restore_dynamic_cache(cache, cache_snapshot)
        for key, value in config_snapshot.items():
            if value is _MISSING:
                if hasattr(configuration, key):
                    delattr(configuration, key)
            else:
                setattr(configuration, key, _clone_value(value))


def position_preserving_index_update(
    cached: Any,
    current: Any,
    positions: Sequence[int],
) -> Any:
    """Return a copy with current values written only at absolute positions."""

    result = _clone_value(cached)
    indices = [int(value) for value in positions]
    if len(indices) != len(set(indices)) or any(value < 0 for value in indices):
        raise B2ValidationError("cache update positions must be unique and nonnegative")
    if hasattr(result, "index_copy_"):
        import torch

        index = torch.as_tensor(indices, dtype=torch.long, device=result.device)
        if current.shape[-2] != len(indices):
            raise B2ValidationError("current cache rows do not match update positions")
        result.index_copy_(-2, index, current)
        return result
    array = np.asarray(result)
    source = np.asarray(current)
    if source.shape[-2] != len(indices) or (indices and max(indices) >= array.shape[-2]):
        raise B2ValidationError("cache update shape or positions are invalid")
    array[..., indices, :] = source
    return result
