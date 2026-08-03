"""Immutable ACR identities, compact FR traces, and record reconciliation."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import struct
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_SAFE_ID = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def semantic_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


@dataclass(frozen=True)
class AttemptIdentity:
    run_id: str
    policy: str
    suite: str
    task_id: int
    initial_state_id: int
    seed: int
    attempt_index: int

    def __post_init__(self) -> None:
        for name, value in (
            ("run_id", self.run_id),
            ("policy", self.policy),
            ("suite", self.suite),
        ):
            if not _SAFE_ID.fullmatch(value):
                raise ValueError(f"{name} must be a lowercase path-safe identifier")
        if not 0 <= self.task_id <= 99 or not 0 <= self.initial_state_id <= 99:
            raise ValueError("Task/state identities must fit their two-digit record fields")
        if self.seed < 0 or not 0 <= self.attempt_index <= 9999:
            raise ValueError("Seed/attempt identities must be non-negative and bounded")

    @property
    def value(self) -> str:
        return (
            f"{self.run_id}/{self.policy}/{self.suite}/task-{self.task_id:02d}/"
            f"state-{self.initial_state_id:02d}/seed-{self.seed}/"
            f"attempt-{self.attempt_index:04d}"
        )

    def query_id(self, query_index: int) -> str:
        if not 0 <= query_index <= 999999:
            raise ValueError("Query index must fit the immutable six-digit field")
        return f"{self.value}/query-{query_index:06d}"

    @property
    def episode_id(self) -> str:
        return f"{self.value}/episode"


@dataclass(frozen=True)
class CompactFloatSequence:
    encoding: str
    count: int
    data: str
    raw_sha256: str

    def as_record(self) -> dict[str, Any]:
        return {
            "encoding": self.encoding,
            "count": self.count,
            "data": self.data,
            "raw_sha256": self.raw_sha256,
        }


def encode_float_sequence(values: Sequence[float]) -> CompactFloatSequence:
    frozen = tuple(float(value) for value in values)
    raw = struct.pack(f"<{len(frozen)}d", *frozen)
    compressed = zlib.compress(raw, level=9)
    return CompactFloatSequence(
        encoding="f64le+zlib+base64-v1",
        count=len(frozen),
        data=base64.b64encode(compressed).decode("ascii"),
        raw_sha256=hashlib.sha256(raw).hexdigest(),
    )


def decode_float_sequence(record: Mapping[str, Any]) -> tuple[float, ...]:
    if record.get("encoding") != "f64le+zlib+base64-v1":
        raise ValueError("Unsupported compact float encoding")
    count = record.get("count")
    if not isinstance(count, int) or count < 0:
        raise ValueError("Compact float count is invalid")
    try:
        raw = zlib.decompress(base64.b64decode(str(record["data"]), validate=True))
    except Exception as error:
        raise ValueError("Compact float payload is corrupt") from error
    if len(raw) != count * 8:
        raise ValueError("Compact float payload length differs from its count")
    if hashlib.sha256(raw).hexdigest() != record.get("raw_sha256"):
        raise ValueError("Compact float payload hash mismatch")
    return tuple(struct.unpack(f"<{count}d", raw))


class ImmutableRecordStore:
    """Write-once JSON records under one explicitly bounded artifact root."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, identity: str) -> Path:
        if identity.startswith("/") or ".." in Path(identity).parts:
            raise ValueError("Record identity escapes its artifact root")
        target = (self.root / identity / "record.json").resolve()
        if target != self.root and self.root not in target.parents:
            raise ValueError("Record identity escapes its artifact root")
        return target

    def write_once(self, identity: str, record: Mapping[str, Any]) -> Path:
        target = self._path(identity)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = canonical_json_bytes(dict(record)) + b"\n"
        try:
            descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError as error:
            raise FileExistsError(f"Immutable record already exists: {identity}") from error
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        except Exception:
            target.unlink(missing_ok=True)
            raise
        return target

    def next_attempt_index(self, pairing_prefix: str) -> int:
        directory = (self.root / pairing_prefix).resolve()
        if directory != self.root and self.root not in directory.parents:
            raise ValueError("Pairing identity escapes its artifact root")
        if not directory.exists():
            return 0
        observed: list[int] = []
        for child in directory.iterdir():
            match = re.fullmatch(r"attempt-([0-9]{4})", child.name)
            if match:
                observed.append(int(match.group(1)))
        return max(observed, default=-1) + 1


def validate_record(record: Mapping[str, Any], schema: Mapping[str, Any]) -> None:
    """Validate against frozen JSON Schema, requiring the pinned dependency."""

    try:
        import jsonschema  # type: ignore[import-untyped]
    except ImportError as error:  # pragma: no cover - packaging guard
        raise RuntimeError("jsonschema is required for immutable ACR records") from error
    validator = jsonschema.Draft202012Validator(schema)
    errors = sorted(
        validator.iter_errors(dict(record)),
        key=lambda item: tuple(str(part) for part in item.path),
    )
    if errors:
        location = ".".join(str(part) for part in errors[0].path) or "<root>"
        raise ValueError(f"ACR record violates schema at {location}: {errors[0].message}")


def reconcile_episode_counts(counts: Mapping[str, int]) -> None:
    required = {
        "queries",
        "scene_refreshes",
        "scene_reuses",
        "wrist_refreshes",
        "scene_siglip_calls",
        "scene_dinov2_calls",
        "scene_projector_calls",
        "wrist_siglip_calls",
        "wrist_dinov2_calls",
        "wrist_projector_calls",
        "downstream_calls",
    }
    if not required.issubset(counts):
        raise ValueError(f"Missing reconciliation counts: {sorted(required - set(counts))}")
    if any(not isinstance(counts[key], int) or counts[key] < 0 for key in required):
        raise ValueError("Reconciliation counts must be non-negative integers")
    queries = counts["queries"]
    refreshes = counts["scene_refreshes"]
    if refreshes + counts["scene_reuses"] != queries:
        raise ValueError("Scene refresh/reuse counts do not reconcile to queries")
    if counts["wrist_refreshes"] != queries or counts["downstream_calls"] != queries:
        raise ValueError("Wrist/downstream counts do not reconcile to queries")
    for key in ("scene_siglip_calls", "scene_dinov2_calls", "scene_projector_calls"):
        if counts[key] != refreshes:
            raise ValueError("Scene component calls do not reconcile to refreshes")
    for key in ("wrist_siglip_calls", "wrist_dinov2_calls", "wrist_projector_calls"):
        if counts[key] != queries:
            raise ValueError("Wrist component calls do not reconcile to queries")


def reconcile_run(*, scheduled_attempts: int, terminal_episodes: int, failures: int) -> None:
    if any(value < 0 for value in (scheduled_attempts, terminal_episodes, failures)):
        raise ValueError("Run reconciliation counts cannot be negative")
    if scheduled_attempts != terminal_episodes + failures:
        raise ValueError("Scheduled attempts do not reconcile to terminal records")
