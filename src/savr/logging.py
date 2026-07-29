"""Immutable JSON record storage for SAVR queries and episodes."""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


class RecordExistsError(FileExistsError):
    """Raised when an immutable record path already exists."""


def _safe_identifier(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        raise ValueError(f"Unsafe record identifier: {value!r}")
    return value


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except Exception:
            pass
    return value


class ImmutableRecordStore:
    """Write each record once using an atomic hard-link publication step."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.query_dir = self.root / "queries"
        self.episode_dir = self.root / "episodes"
        self.query_dir.mkdir(parents=True, exist_ok=True)
        self.episode_dir.mkdir(parents=True, exist_ok=True)

    def _write_once(self, path: Path, record: Any) -> Path:
        payload = json.dumps(
            _jsonable(record),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ) + "\n"
        if path.exists():
            raise RecordExistsError(f"Immutable record already exists: {path}")

        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary_path = Path(handle.name)
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary_path, path)
            except FileExistsError as error:
                raise RecordExistsError(
                    f"Immutable record already exists: {path}"
                ) from error
            return path
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def write_query(self, query_index: int, record: Any) -> Path:
        if query_index < 0:
            raise ValueError("Query index cannot be negative")
        return self._write_once(
            self.query_dir / f"query_{query_index:08d}.json",
            record,
        )

    def write_episode(self, episode_id: str, record: Any) -> Path:
        identifier = _safe_identifier(episode_id)
        return self._write_once(self.episode_dir / f"{identifier}.json", record)
