"""Immutable, outcome-protected PAIR records and schema validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from jsonschema import Draft202012Validator, FormatChecker

from savr.pair.types import PairValidationError


FORBIDDEN_OUTCOME_FIELDS = {
    "success",
    "terminal_success",
    "reward",
    "task_success",
    "episode_outcome",
}


def canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    except (TypeError, ValueError) as error:
        raise PairValidationError("record is not canonical finite JSON") from error


def semantic_sha256(value: Mapping[str, Any], *, identity_field: str = "record_id") -> str:
    payload = {key: item for key, item in value.items() if key != identity_field}
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def reject_outcome_fields(value: Any, *, path: str = "record") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).lower() in FORBIDDEN_OUTCOME_FIELDS:
                raise PairValidationError(f"protected outcome field at {path}.{key}")
            reject_outcome_fields(item, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            reject_outcome_fields(item, path=f"{path}[{index}]")


def freeze_technical_record(record: Mapping[str, Any]) -> dict[str, Any]:
    reject_outcome_fields(record)
    payload = dict(record)
    if payload.get("schema_version") != "pair-technical-v1" or payload.get("phase") != "P2":
        raise PairValidationError("technical record schema or phase is invalid")
    payload["record_id"] = semantic_sha256(payload)
    return payload


def validate_schema(record: Mapping[str, Any], schema_path: Path) -> None:
    reject_outcome_fields(record) if record.get("schema_version") != "pair-episode-v1" else None
    try:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PairValidationError("record schema is unavailable or malformed") from error
    errors = sorted(
        Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(record),
        key=lambda error: tuple(str(item) for item in error.path),
    )
    if errors:
        raise PairValidationError(f"record schema validation failed: {errors[0].message}")
    if "record_id" in record and semantic_sha256(record) != record["record_id"]:
        raise PairValidationError("record semantic hash mismatch")


def records_sha256(records: list[Mapping[str, Any]]) -> str:
    for record in records:
        if "record_id" in record and semantic_sha256(record) != record["record_id"]:
            raise PairValidationError("record semantic hash mismatch")
    return hashlib.sha256(
        b"".join(canonical_bytes(record) + b"\n" for record in records)
    ).hexdigest()
