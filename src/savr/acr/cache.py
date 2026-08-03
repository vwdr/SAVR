"""Strict one-entry projected scene-token cache for ACR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from savr.acr.types import ACRContext, SceneTensorMetadata


class SceneCacheError(RuntimeError):
    """Base scene-cache failure."""


class SceneCacheMiss(SceneCacheError):
    """No compatible scene block is available."""


class SceneCacheCompatibilityError(SceneCacheError):
    """Cached scene metadata is incompatible with the current query."""


@dataclass(frozen=True)
class SceneCacheEntry:
    context: ACRContext
    tokens: Any
    metadata: SceneTensorMetadata
    refresh_query_index: int


def _owned_detached_copy(value: Any) -> Any:
    detached = value.detach() if callable(getattr(value, "detach", None)) else value
    copied = detached.clone() if callable(getattr(detached, "clone", None)) else detached
    if callable(getattr(copied, "requires_grad_", None)):
        copied.requires_grad_(False)
    return copied


class SceneTokenCache:
    """Cache only one detached projected scene-camera token block."""

    def __init__(self) -> None:
        self._entry: SceneCacheEntry | None = None
        self._age = 0

    @property
    def entry(self) -> SceneCacheEntry | None:
        return self._entry

    @property
    def age(self) -> int:
        return self._age

    def available(self, context: ACRContext) -> bool:
        return self._entry is not None and self._entry.context == context

    def compatible(
        self,
        context: ACRContext,
        expected: SceneTensorMetadata | None = None,
    ) -> bool:
        if not self.available(context):
            return False
        return expected is None or self._entry is not None and self._entry.metadata == expected

    def store(
        self,
        *,
        context: ACRContext,
        tokens: Any,
        refresh_query_index: int,
    ) -> SceneTensorMetadata:
        if refresh_query_index < 0:
            raise ValueError("Refresh query index cannot be negative")
        if tokens is None:
            raise SceneCacheCompatibilityError("Cannot cache missing scene tokens")
        owned = _owned_detached_copy(tokens)
        try:
            metadata = SceneTensorMetadata.from_value(owned, patch_count=context.patch_count)
        except ValueError as error:
            raise SceneCacheCompatibilityError(str(error)) from error
        if metadata.dtype != context.dtype or metadata.device != context.device:
            raise SceneCacheCompatibilityError("Scene tensor dtype/device differs from context")
        self._entry = SceneCacheEntry(
            context=context,
            tokens=owned,
            metadata=metadata,
            refresh_query_index=refresh_query_index,
        )
        self._age = 0
        return metadata

    def load(self, context: ACRContext, expected: SceneTensorMetadata) -> Any:
        if not self.available(context):
            raise SceneCacheMiss("No scene token block exists for the current context")
        assert self._entry is not None
        if self._entry.metadata != expected:
            raise SceneCacheCompatibilityError(
                f"Cached scene metadata {self._entry.metadata} differs from {expected}"
            )
        return self._entry.tokens

    def mark_reused(self) -> None:
        if self._entry is None:
            raise SceneCacheMiss("Cannot age an empty scene cache")
        self._age += 1

    def invalidate(self) -> None:
        self._entry = None
        self._age = 0
