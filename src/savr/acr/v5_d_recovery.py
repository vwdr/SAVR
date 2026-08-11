"""Fail-closed V5-D v02 recovery helpers with no GPU or simulator dependency."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping


EXPECTED_PROJECT_ROOT = Path("/home/ved/SAVR")
V02_RUN_ID = "acr-v5d-real-tensor-feasibility-v02"
CONFIG_KEYS = ("assets", "bddl_files", "benchmark_root", "datasets", "init_states")


class V5DRecoveryViolation(RuntimeError):
    """The proposed recovery operation does not match the frozen v02 scope."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def semantic_sha256(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("semantic_sha256", None)
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


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
