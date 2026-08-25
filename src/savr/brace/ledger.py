"""Per-layer/token source ownership and bounded source-record retention."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Mapping, Sequence

from savr.brace.b1 import semantic_sha256
from savr.brace.sequence_map import SequenceMap
from savr.brace.types import B2ValidationError, CacheIdentity, Camera


@dataclass(frozen=True)
class SourceRecord:
    query: int
    scene_image_sha256: str
    wrist_image_sha256: str
    normalized_proprio: tuple[float, ...]
    prompt_sha256: str
    action_mask_sha256: str
    sequence_map_sha256: str
    previous_action_summary: tuple[float, ...]
    rng_sha256: str
    configuration_sha256: str
    counters: tuple[tuple[str, int], ...]
    semantic_sha256: str

    @classmethod
    def create(
        cls,
        *,
        query: int,
        scene_image_sha256: str,
        wrist_image_sha256: str,
        normalized_proprio: Sequence[float],
        prompt_sha256: str,
        action_mask_sha256: str,
        sequence_map_sha256: str,
        previous_action_summary: Sequence[float],
        rng_sha256: str,
        configuration_sha256: str,
        counters: Mapping[str, int],
    ) -> "SourceRecord":
        if query < 0:
            raise B2ValidationError("source query must be nonnegative")
        digests = (
            scene_image_sha256,
            wrist_image_sha256,
            prompt_sha256,
            action_mask_sha256,
            sequence_map_sha256,
            rng_sha256,
            configuration_sha256,
        )
        if any(len(value) != 64 for value in digests):
            raise B2ValidationError("source record contains an invalid digest")
        proprio = tuple(float(value) for value in normalized_proprio)
        action = tuple(float(value) for value in previous_action_summary)
        if not proprio or not action:
            raise B2ValidationError("source record lacks multimodal context")
        payload = {
            "query": query,
            "scene_image_sha256": scene_image_sha256,
            "wrist_image_sha256": wrist_image_sha256,
            "normalized_proprio": proprio,
            "prompt_sha256": prompt_sha256,
            "action_mask_sha256": action_mask_sha256,
            "sequence_map_sha256": sequence_map_sha256,
            "previous_action_summary": action,
            "rng_sha256": rng_sha256,
            "configuration_sha256": configuration_sha256,
            "counters": tuple(sorted((str(key), int(value)) for key, value in counters.items())),
        }
        return cls(**payload, semantic_sha256=semantic_sha256(payload))


class SourceRingBuffer:
    """Bounded records that refuse to evict any source still referenced by K/V."""

    def __init__(self, capacity: int) -> None:
        if capacity < 2:
            raise B2ValidationError("source ring capacity must be at least two")
        self.capacity = int(capacity)
        self._records: dict[int, SourceRecord] = {}
        self._order: list[int] = []

    def clone(self) -> "SourceRingBuffer":
        result = SourceRingBuffer(self.capacity)
        result._records = dict(self._records)
        result._order = list(self._order)
        return result

    def add(self, record: SourceRecord, *, live_sources: Iterable[int] = ()) -> None:
        if record.query in self._records:
            if self._records[record.query] != record:
                raise B2ValidationError("source query identity was mutated")
            return
        live = set(int(value) for value in live_sources)
        while len(self._order) >= self.capacity:
            evictable = next((query for query in self._order if query not in live), None)
            if evictable is None:
                raise B2ValidationError("ring eviction would remove a live K/V source")
            self._order.remove(evictable)
            del self._records[evictable]
        self._records[record.query] = record
        self._order.append(record.query)

    def get(self, query: int) -> SourceRecord:
        try:
            return self._records[int(query)]
        except KeyError as error:
            raise B2ValidationError("K/V source record is unavailable") from error

    def digest(self, query: int) -> str:
        return self.get(query).semantic_sha256

    def queries(self) -> tuple[int, ...]:
        return tuple(self._order)


@dataclass(frozen=True)
class LedgerEntry:
    layer: int
    token_position: int
    camera: Camera
    patch_id: int
    source_query: int
    source_record_sha256: str
    age: int
    reused: bool
    kv_position: int
    dtype: str
    shape: tuple[int, ...]
    kv_sha256: str
    gate_sha256: str
    anchor_query: int
    profile_id: str
    horizon: int
    remaining: int
    abort_history: tuple[str, ...]


class SourceLedger:
    """Position-aligned visual ownership ledger across every decoder layer."""

    def __init__(
        self,
        *,
        identity: CacheIdentity,
        sequence_map: SequenceMap,
        entries: Mapping[tuple[int, int], LedgerEntry],
        source_records: SourceRingBuffer,
    ) -> None:
        identity.validate()
        if identity.sequence_map_sha256 != sequence_map.semantic_sha256:
            raise B2ValidationError("cache and runtime sequence-map identities differ")
        self.identity = identity
        self.sequence_map = sequence_map
        self.entries = dict(entries)
        self.source_records = source_records
        self.validate()

    @classmethod
    def dense(
        cls,
        *,
        identity: CacheIdentity,
        sequence_map: SequenceMap,
        layers: Sequence[int],
        source_record: SourceRecord,
        gate_sha256: str,
        dtype: str,
        shape: Sequence[int],
        kv_digests: Mapping[tuple[int, int], str],
        ring_capacity: int = 6,
    ) -> "SourceLedger":
        ring = SourceRingBuffer(ring_capacity)
        ring.add(source_record)
        entries: dict[tuple[int, int], LedgerEntry] = {}
        for layer in layers:
            for camera in Camera:
                for position in sequence_map.positions(camera):
                    key = (int(layer), position)
                    if key not in kv_digests:
                        raise B2ValidationError("dense ledger lacks a visual K/V digest")
                    entries[key] = LedgerEntry(
                        layer=int(layer),
                        token_position=position,
                        camera=camera,
                        patch_id=sequence_map.patch_for_position(position),
                        source_query=source_record.query,
                        source_record_sha256=source_record.semantic_sha256,
                        age=0,
                        reused=False,
                        kv_position=position,
                        dtype=dtype,
                        shape=tuple(int(value) for value in shape),
                        kv_sha256=kv_digests[key],
                        gate_sha256=gate_sha256,
                        anchor_query=identity.anchor_query,
                        profile_id=identity.profile_id,
                        horizon=0,
                        remaining=0,
                        abort_history=(),
                    )
        return cls(
            identity=identity,
            sequence_map=sequence_map,
            entries=entries,
            source_records=ring,
        )

    def clone(self) -> "SourceLedger":
        return SourceLedger(
            identity=self.identity,
            sequence_map=self.sequence_map,
            entries=self.entries,
            source_records=self.source_records.clone(),
        )

    def live_sources(self) -> set[int]:
        return {entry.source_query for entry in self.entries.values()}

    def validate(self) -> None:
        expected_positions = set(self.sequence_map.scene_positions + self.sequence_map.wrist_positions)
        layers = {layer for layer, _position in self.entries}
        if not layers:
            raise B2ValidationError("source ledger is empty")
        for layer in layers:
            positions = {position for entry_layer, position in self.entries if entry_layer == layer}
            if positions != expected_positions:
                raise B2ValidationError("source ledger does not cover every visual position")
        for (layer, position), entry in self.entries.items():
            if entry.layer != layer or entry.token_position != position or entry.kv_position != position:
                raise B2ValidationError("ledger K/V positions are not preserved")
            expected_camera = (
                Camera.SCENE if position in self.sequence_map.scene_positions else Camera.WRIST
            )
            if entry.camera is not expected_camera:
                raise B2ValidationError("scene/wrist ledger positions were swapped")
            record = self.source_records.get(entry.source_query)
            if record.semantic_sha256 != entry.source_record_sha256:
                raise B2ValidationError("ledger source digest does not resolve")
            if entry.age < 0 or entry.age != max(0, entry.anchor_query + entry.age - entry.anchor_query):
                raise B2ValidationError("ledger age is invalid")

    def update(
        self,
        *,
        current_query: int,
        source_record: SourceRecord,
        reuse_sets: Mapping[int, Iterable[int]],
        kv_digests: Mapping[tuple[int, int], str],
        profile_id: str,
        horizon: int,
        remaining: int,
        abort_history: Sequence[str] = (),
    ) -> "SourceLedger":
        if source_record.query != current_query:
            raise B2ValidationError("current ledger update lacks its exact source record")
        result = self.clone()
        result.source_records.add(source_record, live_sources=self.live_sources())
        for key, entry in self.entries.items():
            reused = entry.token_position in set(reuse_sets.get(entry.layer, ()))
            if reused:
                updated = replace(
                    entry,
                    age=current_query - entry.source_query,
                    reused=True,
                    profile_id=profile_id,
                    horizon=horizon,
                    remaining=remaining,
                    abort_history=tuple(abort_history),
                )
            else:
                if key not in kv_digests:
                    raise B2ValidationError("recomputed ledger entry lacks current K/V digest")
                updated = replace(
                    entry,
                    source_query=current_query,
                    source_record_sha256=source_record.semantic_sha256,
                    age=0,
                    reused=False,
                    kv_sha256=kv_digests[key],
                    profile_id=profile_id,
                    horizon=horizon,
                    remaining=remaining,
                    abort_history=tuple(abort_history),
                )
            result.entries[key] = updated
        result.validate()
        return result

    def context_envelope(
        self,
        *,
        current_proprio: Sequence[float],
        previous_proprio: Sequence[float],
        current_action_summary: Sequence[float],
        source_proprio_limit: float,
        step_proprio_limit: float,
        source_action_limit: float,
        gripper_transition: bool,
    ) -> tuple[bool, str | None]:
        if gripper_transition:
            return False, "unexpected_gripper_transition"
        current_z = tuple(float(value) for value in current_proprio)
        previous_z = tuple(float(value) for value in previous_proprio)
        current_a = tuple(float(value) for value in current_action_summary)
        if len(current_z) != len(previous_z) or not current_z or not current_a:
            return False, "context_dimension_mismatch"
        if max(abs(a - b) for a, b in zip(current_z, previous_z, strict=True)) > step_proprio_limit:
            return False, "step_proprio_drift"
        for query in self.live_sources():
            record = self.source_records.get(query)
            if len(record.normalized_proprio) != len(current_z):
                return False, "source_proprio_dimension_mismatch"
            if max(
                abs(a - b)
                for a, b in zip(current_z, record.normalized_proprio, strict=True)
            ) > source_proprio_limit:
                return False, "source_proprio_drift"
            if len(record.previous_action_summary) != len(current_a):
                return False, "source_action_dimension_mismatch"
            if max(
                abs(a - b)
                for a, b in zip(current_a, record.previous_action_summary, strict=True)
            ) > source_action_limit:
                return False, "source_action_drift"
        return True, None
