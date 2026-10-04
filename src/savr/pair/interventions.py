"""Tile masks, recursive provenance, and synthetic K/V interventions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from savr.brace.cache_adapter import clone_dynamic_cache
from savr.pair.types import (
    ONSET_LAYERS,
    PATCHES_PER_CAMERA,
    AtomicGroup,
    Camera,
    PairValidationError,
    ProfileSpec,
    SourcePointer,
)


def tile_patch_indices(tile: int) -> tuple[int, ...]:
    if not 0 <= int(tile) < 16:
        raise PairValidationError("tile lies outside the frozen 4x4 grid")
    tile_row, tile_column = divmod(int(tile), 4)
    return tuple(
        (tile_row * 4 + row) * 16 + tile_column * 4 + column
        for row in range(4)
        for column in range(4)
    )


@dataclass(frozen=True)
class InterventionMask:
    groups: tuple[AtomicGroup, ...]

    @classmethod
    def fresh(cls) -> "InterventionMask":
        return cls(())

    def validate(self) -> None:
        identities: set[str] = set()
        onset_by_tile: dict[tuple[Camera, int], int] = {}
        for group in self.groups:
            group.validate()
            if group.identity in identities:
                raise PairValidationError("intervention mask contains a duplicate group")
            identities.add(group.identity)
            key = (group.camera, group.tile)
            if key in onset_by_tile:
                raise PairValidationError("one tile cannot have multiple onset layers")
            onset_by_tile[key] = group.onset_layer

    def reuses(self, layer: int, camera: Camera, tile: int) -> bool:
        self.validate()
        return any(
            group.camera is camera and group.tile == tile and layer >= group.onset_layer
            for group in self.groups
        )


def profile_mask(
    profile: ProfileSpec, rankings: Mapping[Camera, Sequence[int]]
) -> InterventionMask:
    """Build the unique tile-onset mask matching every nested token budget."""

    profile.validate()
    groups: list[AtomicGroup] = []
    for camera, budgets in (
        (Camera.PRIMARY, profile.primary_budgets),
        (Camera.WRIST, profile.wrist_budgets),
    ):
        ranking = tuple(int(tile) for tile in rankings.get(camera, ()))
        if len(ranking) != 16 or set(ranking) != set(range(16)):
            raise PairValidationError("camera ranking must be a permutation of all 16 tiles")
        previous = 0
        for layer, budget in zip(ONSET_LAYERS, budgets, strict=True):
            count = budget // 16
            for tile in ranking[previous:count]:
                groups.append(AtomicGroup(camera, tile, layer))
            previous = count
    result = InterventionMask(tuple(sorted(groups)))
    result.validate()
    for layer_index, layer in enumerate(ONSET_LAYERS):
        for camera, expected in (
            (Camera.PRIMARY, profile.primary_budgets[layer_index]),
            (Camera.WRIST, profile.wrist_budgets[layer_index]),
        ):
            observed = sum(result.reuses(layer, camera, tile) * 16 for tile in range(16))
            if observed != expected:
                raise PairValidationError("mask does not exactly realize its profile budget")
    return result


class TensorSourceRing:
    """Bounded, exact-clone source tensor storage with live-source protection."""

    def __init__(self, capacity: int = 6) -> None:
        if capacity < 2:
            raise PairValidationError("source ring capacity must be at least two")
        self.capacity = int(capacity)
        self._records: dict[int, dict[int, np.ndarray]] = {}
        self._order: list[int] = []

    def clone(self) -> "TensorSourceRing":
        result = TensorSourceRing(self.capacity)
        result._records = {
            query: {layer: tensor.copy() for layer, tensor in layers.items()}
            for query, layers in self._records.items()
        }
        result._order = list(self._order)
        return result

    def add(
        self,
        query: int,
        tensors: Mapping[int, np.ndarray],
        *,
        live_sources: Sequence[int] = (),
    ) -> None:
        frozen = _validate_tensor_bank(tensors)
        if query in self._records:
            if any(
                not np.array_equal(self._records[query][layer], value)
                for layer, value in frozen.items()
            ):
                raise PairValidationError("source tensor identity was mutated")
            return
        live = set(int(item) for item in live_sources)
        while len(self._order) >= self.capacity:
            evictable = next((item for item in self._order if item not in live), None)
            if evictable is None:
                raise PairValidationError("ring eviction would remove a live source")
            self._order.remove(evictable)
            del self._records[evictable]
        self._records[int(query)] = frozen
        self._order.append(int(query))

    def get(self, query: int, layer: int) -> np.ndarray:
        try:
            return self._records[int(query)][int(layer)]
        except KeyError as error:
            raise PairValidationError("requested K/V source record is unavailable") from error

    def queries(self) -> tuple[int, ...]:
        return tuple(self._order)


def _validate_tensor_bank(tensors: Mapping[int, np.ndarray]) -> dict[int, np.ndarray]:
    if set(int(layer) for layer in tensors) != set(ONSET_LAYERS):
        raise PairValidationError("tensor bank must cover every frozen layer")
    result: dict[int, np.ndarray] = {}
    feature_width: int | None = None
    for layer in ONSET_LAYERS:
        value = np.asarray(tensors[layer])
        if value.ndim != 3 or value.shape[:2] != (2, PATCHES_PER_CAMERA):
            raise PairValidationError("tensor must have [camera,256,feature] shape")
        if feature_width is None:
            feature_width = value.shape[2]
        if value.shape[2] != feature_width or feature_width <= 0:
            raise PairValidationError("tensor feature width is inconsistent")
        if not np.isfinite(value).all():
            raise PairValidationError("source tensor contains nonfinite values")
        result[layer] = value.copy()
    return result


class GroupSourceLedger:
    """Per-layer/camera/tile source ownership across recursive contracts."""

    def __init__(
        self,
        *,
        current_query: int,
        entries: Mapping[tuple[int, Camera, int], SourcePointer],
        sources: TensorSourceRing,
    ) -> None:
        self.current_query = int(current_query)
        self.entries = dict(entries)
        self.sources = sources
        self.validate()

    @classmethod
    def dense(
        cls, query: int, tensors: Mapping[int, np.ndarray], *, ring_capacity: int = 6
    ) -> "GroupSourceLedger":
        ring = TensorSourceRing(ring_capacity)
        ring.add(query, tensors)
        entries = {
            (layer, camera, tile): SourcePointer(query, 0)
            for layer in ONSET_LAYERS
            for camera in Camera
            for tile in range(16)
        }
        return cls(current_query=query, entries=entries, sources=ring)

    def clone(self) -> "GroupSourceLedger":
        return GroupSourceLedger(
            current_query=self.current_query,
            entries=self.entries,
            sources=self.sources.clone(),
        )

    def live_sources(self) -> tuple[int, ...]:
        return tuple(sorted({pointer.query for pointer in self.entries.values()}))

    def validate(self) -> None:
        expected = {
            (layer, camera, tile)
            for layer in ONSET_LAYERS
            for camera in Camera
            for tile in range(16)
        }
        if set(self.entries) != expected:
            raise PairValidationError("ledger has a missing or off-by-one group")
        for (layer, camera, tile), pointer in self.entries.items():
            if layer not in ONSET_LAYERS or not isinstance(camera, Camera) or not 0 <= tile < 16:
                raise PairValidationError("ledger group identity is invalid")
            pointer.validate(self.current_query)
            self.sources.get(pointer.query, layer)

    def advance(
        self,
        current_query: int,
        current_tensors: Mapping[int, np.ndarray],
        mask: InterventionMask,
    ) -> "GroupSourceLedger":
        if current_query != self.current_query + 1:
            raise PairValidationError("recursive ledger queries must be contiguous")
        mask.validate()
        result = self.clone()
        result.sources.add(current_query, current_tensors, live_sources=self.live_sources())
        result.current_query = current_query
        for key, old in self.entries.items():
            layer, camera, tile = key
            source = old.query if mask.reuses(layer, camera, tile) else current_query
            result.entries[key] = SourcePointer(source, current_query - source)
        result.validate()
        return result

    def abort(
        self, current_query: int, current_tensors: Mapping[int, np.ndarray]
    ) -> "GroupSourceLedger":
        return GroupSourceLedger.dense(
            current_query, current_tensors, ring_capacity=self.sources.capacity
        )


def apply_intervention(
    current_tensors: Mapping[int, np.ndarray],
    ledger: GroupSourceLedger,
    mask: InterventionMask,
) -> dict[int, np.ndarray]:
    """Apply a mask using each group's actual pre-update source."""

    current = _validate_tensor_bank(current_tensors)
    ledger.validate()
    mask.validate()
    output = {layer: value.copy() for layer, value in current.items()}
    for layer in ONSET_LAYERS:
        for camera_index, camera in enumerate(Camera):
            for tile in range(16):
                if not mask.reuses(layer, camera, tile):
                    continue
                pointer = ledger.entries[(layer, camera, tile)]
                source = ledger.sources.get(pointer.query, layer)
                indices = tile_patch_indices(tile)
                output[layer][camera_index, indices, :] = source[camera_index, indices, :]
    return output


def clone_runtime_cache(cache: object) -> object:
    """Expose BRACE's exact-clone cache contract to PAIR runtimes."""

    return clone_dynamic_cache(cache)
