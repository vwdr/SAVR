"""Context-safe cache for projected visual feature tensors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class CacheError(RuntimeError):
    """Base projected-feature cache error."""


class CacheMissError(CacheError):
    """Raised when no entry exists for the requested context."""


class CacheCompatibilityError(CacheError):
    """Raised when cached tensor metadata is incompatible."""


@dataclass(frozen=True)
class CacheContext:
    episode_id: str
    task_id: str
    checkpoint_id: str
    configuration_id: str

    def __post_init__(self) -> None:
        if not all(
            isinstance(value, str) and value
            for value in (
                self.episode_id,
                self.task_id,
                self.checkpoint_id,
                self.configuration_id,
            )
        ):
            raise ValueError("Every cache-context field must be a non-empty string")


@dataclass(frozen=True)
class TensorMetadata:
    shape: tuple[int, ...]
    dtype: str
    device: str

    @classmethod
    def from_value(cls, value: Any) -> "TensorMetadata":
        if not hasattr(value, "shape"):
            raise CacheCompatibilityError("Cached feature lacks tensor shape metadata")
        try:
            shape = tuple(int(dimension) for dimension in value.shape)
        except Exception as error:
            raise CacheCompatibilityError("Invalid tensor shape metadata") from error
        if not shape or any(dimension <= 0 for dimension in shape):
            raise CacheCompatibilityError("Tensor shape must have positive dimensions")
        return cls(
            shape=shape,
            dtype=str(getattr(value, "dtype", "unknown")),
            device=str(getattr(value, "device", "unknown")),
        )


@dataclass(frozen=True)
class CacheEntry:
    context: CacheContext
    feature: Any
    metadata: TensorMetadata


class ProjectedFeatureCache:
    """One-entry projected-feature cache with explicit context and age."""

    def __init__(self) -> None:
        self._entry: CacheEntry | None = None
        self._age = 0

    @property
    def age(self) -> int:
        return self._age

    @property
    def entry(self) -> CacheEntry | None:
        return self._entry

    def available(self, context: CacheContext) -> bool:
        return self._entry is not None and self._entry.context == context

    def store(self, context: CacheContext, feature: Any) -> TensorMetadata:
        if feature is None:
            raise CacheCompatibilityError("Cannot cache a missing feature")
        detached = feature.detach() if callable(getattr(feature, "detach", None)) else feature
        metadata = TensorMetadata.from_value(detached)
        self._entry = CacheEntry(context=context, feature=detached, metadata=metadata)
        self._age = 0
        return metadata

    def load(self, context: CacheContext, expected: TensorMetadata) -> Any:
        if not self.available(context):
            raise CacheMissError("No projected feature exists for the current context")
        assert self._entry is not None
        if self._entry.metadata != expected:
            raise CacheCompatibilityError(
                f"Cached metadata {self._entry.metadata} does not match {expected}"
            )
        return self._entry.feature

    def mark_reused(self) -> None:
        if self._entry is None:
            raise CacheMissError("Cannot age an empty cache")
        self._age += 1

    def invalidate(self) -> None:
        self._entry = None
        self._age = 0
