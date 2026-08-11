"""Fail-closed V5-D recovery helpers with no GPU or simulator dependency."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


EXPECTED_PROJECT_ROOT = Path("/home/ved/SAVR")
V02_RUN_ID = "acr-v5d-real-tensor-feasibility-v02"
V03_RUN_ID = "acr-v5d-real-tensor-feasibility-v03"
CONFIG_KEYS = ("assets", "bddl_files", "benchmark_root", "datasets", "init_states")
PROTECTED_CHECKPOINT_NAMES = (
    "config.json",
    "configuration_prismatic.py",
    "modeling_prismatic.py",
)
_LOADER_BACKUP_PATTERN = re.compile(
    r"^(?P<protected>config\.json|configuration_prismatic\.py|modeling_prismatic\.py)"
    r"(?:\.bak|\.backup(?:\.\d{8}_\d{6})?|\.back\.\d{8}_\d{6})$"
)


class V5DRecoveryViolation(RuntimeError):
    """The proposed recovery operation does not match the frozen scope."""


@dataclass(frozen=True)
class CheckpointBaseline:
    """In-memory pre-load checkpoint state required for exact restoration."""

    checkpoint: Path
    names: tuple[str, ...]
    protected_bytes: Mapping[str, bytes]
    protected_hashes: Mapping[str, str]
    nonprotected_signatures: Mapping[str, tuple[str, int, int]]


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def semantic_sha256(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("semantic_sha256", None)
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_checkpoint_entry_signature(path: Path) -> tuple[str, int, int]:
    if path.is_symlink():
        raise V5DRecoveryViolation(f"Checkpoint symlink is prohibited: {path.name}")
    stat = path.stat()
    if path.is_file():
        return ("file", stat.st_size, stat.st_mtime_ns)
    if path.is_dir():
        return ("directory", 0, 0)
    raise V5DRecoveryViolation(f"Unsupported checkpoint entry: {path.name}")


def capture_checkpoint_baseline(
    checkpoint: Path,
    protected_names: tuple[str, ...] = PROTECTED_CHECKPOINT_NAMES,
) -> CheckpointBaseline:
    """Capture the exact pre-load inventory without reading model-weight contents."""

    if checkpoint.is_symlink() or not checkpoint.is_dir():
        raise V5DRecoveryViolation("Checkpoint root is absent or unsafe")
    checkpoint = checkpoint.resolve()
    entries = {item.name: item for item in checkpoint.iterdir()}
    protected = tuple(protected_names)
    if len(set(protected)) != len(protected) or any(name not in entries for name in protected):
        raise V5DRecoveryViolation("Protected checkpoint inventory is incomplete")
    protected_bytes: dict[str, bytes] = {}
    protected_hashes: dict[str, str] = {}
    nonprotected: dict[str, tuple[str, int, int]] = {}
    for name, path in entries.items():
        signature = _safe_checkpoint_entry_signature(path)
        if name in protected:
            if signature[0] != "file":
                raise V5DRecoveryViolation(f"Protected checkpoint entry is not a file: {name}")
            payload = path.read_bytes()
            protected_bytes[name] = payload
            protected_hashes[name] = hashlib.sha256(payload).hexdigest()
        else:
            nonprotected[name] = signature
    return CheckpointBaseline(
        checkpoint=checkpoint,
        names=tuple(sorted(entries)),
        protected_bytes=protected_bytes,
        protected_hashes=protected_hashes,
        nonprotected_signatures=nonprotected,
    )


def restore_checkpoint_exact(
    checkpoint: Path,
    baseline: CheckpointBaseline,
) -> dict[str, Any]:
    """Restore protected bytes and remove only verified loader-created backups."""

    if checkpoint.is_symlink() or not checkpoint.is_dir():
        raise V5DRecoveryViolation("Checkpoint root is absent or unsafe during restoration")
    checkpoint = checkpoint.resolve()
    if checkpoint != baseline.checkpoint:
        raise V5DRecoveryViolation("Checkpoint restoration target changed")
    entries = {item.name: item for item in checkpoint.iterdir()}
    baseline_names = set(baseline.names)
    missing = sorted(baseline_names - set(entries))
    if missing:
        raise V5DRecoveryViolation(f"Checkpoint baseline entries disappeared: {missing}")

    for name, expected in baseline.nonprotected_signatures.items():
        if _safe_checkpoint_entry_signature(entries[name]) != expected:
            raise V5DRecoveryViolation(f"Pre-existing checkpoint entry changed: {name}")
    for name in baseline.protected_bytes:
        if _safe_checkpoint_entry_signature(entries[name])[0] != "file":
            raise V5DRecoveryViolation(f"Protected checkpoint entry became unsafe: {name}")

    new_names = sorted(set(entries) - baseline_names)
    verified_backups: list[tuple[Path, str]] = []
    for name in new_names:
        path = entries[name]
        if _safe_checkpoint_entry_signature(path)[0] != "file":
            raise V5DRecoveryViolation(f"Unexpected non-file checkpoint artifact: {name}")
        match = _LOADER_BACKUP_PATTERN.fullmatch(name)
        if match is None:
            raise V5DRecoveryViolation(f"Unexpected checkpoint loader artifact: {name}")
        protected_name = match.group("protected")
        if file_sha256(path) != baseline.protected_hashes[protected_name]:
            raise V5DRecoveryViolation(f"Checkpoint loader backup content changed: {name}")
        verified_backups.append((path, name))

    for name, payload in baseline.protected_bytes.items():
        (checkpoint / name).write_bytes(payload)

    removed: list[str] = []
    for path, name in verified_backups:
        try:
            path.unlink()
        except OSError as error:
            raise V5DRecoveryViolation(f"Checkpoint loader backup cleanup failed: {name}") from error
        removed.append(name)

    final_entries = {item.name: item for item in checkpoint.iterdir()}
    if set(final_entries) != baseline_names:
        raise V5DRecoveryViolation("Checkpoint inventory was not restored exactly")
    for name, expected in baseline.nonprotected_signatures.items():
        if _safe_checkpoint_entry_signature(final_entries[name]) != expected:
            raise V5DRecoveryViolation(f"Checkpoint baseline drift remained: {name}")
    protected_hashes = {
        name: file_sha256(checkpoint / name) for name in baseline.protected_bytes
    }
    if protected_hashes != dict(baseline.protected_hashes):
        raise V5DRecoveryViolation("Protected checkpoint hashes were not restored")
    return {
        "schema_version": "acr.v5d-checkpoint-restoration.v3",
        "protected_bytes_restored": True,
        "protected_hashes": protected_hashes,
        "removed_loader_backups": removed,
        "backup_cleanup_complete": True,
        "inventory_equal": True,
        "idempotent_ready": True,
    }


def expected_libero_mapping(project_root: Path, recovery: Mapping[str, Any]) -> dict[str, str]:
    relative = recovery["libero_config"]["paths_relative_to_project"]
    if tuple(sorted(relative)) != CONFIG_KEYS:
        raise V5DRecoveryViolation("V5-D v02 LIBERO keys changed")
    project = project_root.resolve()
    mapping = {}
    for key in CONFIG_KEYS:
        candidate = project / relative[key]
        _assert_no_symlink_between(project, candidate)
        mapping[key] = str(candidate.resolve())
    for path in mapping.values():
        if not Path(path).is_relative_to(project):
            raise V5DRecoveryViolation("V5-D v02 LIBERO path escaped the project")
    return mapping


def canonical_libero_bytes(mapping: Mapping[str, str]) -> bytes:
    if tuple(sorted(mapping)) != CONFIG_KEYS:
        raise V5DRecoveryViolation("V5-D v02 LIBERO mapping is incomplete")
    # JSON is valid YAML and prevents dumper/version-dependent byte drift.
    return json.dumps(dict(mapping), indent=2, sort_keys=True).encode() + b"\n"


def _assert_no_symlink_between(project_root: Path, target: Path) -> None:
    project = project_root.resolve()
    lexical_target = target.absolute()
    if not lexical_target.is_relative_to(project):
        raise V5DRecoveryViolation("V5-D v02 target escaped the project")
    relative = lexical_target.relative_to(project)
    current = project
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise V5DRecoveryViolation(f"V5-D v02 symlink is prohibited: {current}")


def libero_config_path(project_root: Path, run_root: Path, recovery: Mapping[str, Any]) -> Path:
    project = project_root.resolve()
    run = run_root.absolute()
    if not run.is_relative_to(project / "results"):
        raise V5DRecoveryViolation("V5-D v02 run root escaped project results")
    _assert_no_symlink_between(project, run)
    relative = Path(recovery["libero_config"]["config_relative_to_run"])
    if relative.is_absolute() or ".." in relative.parts:
        raise V5DRecoveryViolation("V5-D v02 LIBERO config relative path is unsafe")
    config_path = run / relative
    _assert_no_symlink_between(project, config_path)
    if config_path.resolve(strict=False) != config_path.absolute():
        raise V5DRecoveryViolation("V5-D v02 LIBERO config path is not canonical")
    return config_path


def validate_libero_config(
    project_root: Path, run_root: Path, recovery: Mapping[str, Any]
) -> dict[str, Any]:
    path = libero_config_path(project_root, run_root, recovery)
    if not path.is_file() or path.is_symlink():
        raise V5DRecoveryViolation("V5-D v02 LIBERO config is absent or unsafe")
    mapping = expected_libero_mapping(project_root, recovery)
    expected = canonical_libero_bytes(mapping)
    observed = path.read_bytes()
    if observed != expected:
        raise V5DRecoveryViolation("V5-D v02 LIBERO config bytes changed")
    parsed = json.loads(observed)
    if parsed != mapping:
        raise V5DRecoveryViolation("V5-D v02 LIBERO config mapping changed")
    return {
        "path": str(path),
        "mapping": mapping,
        "bytes": len(observed),
        "sha256": hashlib.sha256(observed).hexdigest(),
    }


def create_libero_config_once(
    project_root: Path, run_root: Path, recovery: Mapping[str, Any]
) -> dict[str, Any]:
    path = libero_config_path(project_root, run_root, recovery)
    if path.exists() or path.is_symlink():
        raise V5DRecoveryViolation("V5-D v02 LIBERO config already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    _assert_no_symlink_between(project_root, path)
    payload = canonical_libero_bytes(expected_libero_mapping(project_root, recovery))
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        if path.exists() and not path.is_symlink():
            path.unlink()
        raise
    return validate_libero_config(project_root, run_root, recovery)


def write_json_once(path: Path, value: Mapping[str, Any]) -> None:
    payload = canonical_json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def build_pre_model_stop_record(
    *,
    run_id: str,
    backend: str,
    execution_revision: str | None,
    configuration_semantic_sha256: str | None,
    launch_manifest_semantic_sha256: str | None,
    error: BaseException,
    recorded_at_utc: str,
    selected_gpu_after: Mapping[str, Any] | None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "schema_version": "acr.v5d-pre-model-technical-stop.v2",
        "run_id": run_id,
        "status": "technical-stop-before-model-load",
        "backend_requested": backend,
        "recorded_at_utc": recorded_at_utc,
        "execution_revision": execution_revision,
        "configuration_semantic_sha256": configuration_semantic_sha256,
        "launch_manifest_semantic_sha256": launch_manifest_semantic_sha256,
        "failure_stage": "pre-model-envelope",
        "failure_type": type(error).__name__,
        "failure_message": str(error),
        "raw_fallback_attempted": False,
        "raw_fallback_permitted": False,
        "model_loaded": False,
        "model_queries": 0,
        "backend_preparation_core_launches": 0,
        "correctness_records": 0,
        "warmup_records": 0,
        "timed_records": 0,
        "simulator_episodes": 0,
        "simulator_resets": 0,
        "downloads": 0,
        "new_task_outcomes": 0,
        "success_reward_or_outcome_fields_accessed": False,
        "selected_gpu_after": None if selected_gpu_after is None else dict(selected_gpu_after),
        "checkpoint_write_or_loader_backup_created": False,
        "restoration_required": False,
        "disposition": "STOP_NO_RAW_FALLBACK_NO_AUTOMATIC_RETRY",
    }
    record["semantic_sha256"] = semantic_sha256(record)
    return record
