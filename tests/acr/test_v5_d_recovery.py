from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from savr.acr.v5_d_recovery import (
    V5DRecoveryViolation,
    build_pre_model_stop_record,
    create_libero_config_once,
    libero_config_path,
    semantic_sha256,
    validate_libero_config,
    write_json_once,
)
from savr.acr.v5_d_runtime import load_v5_d_freeze, validate_v5_d_freeze


ROOT = Path(__file__).resolve().parents[2]


def recovery_config() -> dict:
    return json.loads((ROOT / "configs/acr/v5_d_gpu_feasibility_recovery_v02.json").read_text())


def project_tree(tmp_path: Path) -> tuple[Path, Path, dict]:
    project = tmp_path / "SAVR"
    run = project / "results" / "acr-v5d-real-tensor-feasibility-v02"
    for relative in recovery_config()["libero_config"]["paths_relative_to_project"].values():
        (project / relative).mkdir(parents=True, exist_ok=True)
    return project, run, {"libero_config": recovery_config()["libero_config"]}


def test_v02_overlay_preserves_every_scientific_section() -> None:
    base = json.loads((ROOT / "configs/acr/v5_d_gpu_feasibility_freeze.json").read_text())
    resolved = load_v5_d_freeze(ROOT)
    validate_v5_d_freeze(resolved)
    assert resolved["run_id"] == "acr-v5d-real-tensor-feasibility-v02"
    assert resolved["recovery_v02"]["v01_technical_stop_semantic_sha256"] == (
        "edf5872fa818f5806601f52143cb17cec7dd4974e03cc4e2ed43c3d042fb4412"
    )
    scientific = (
        "selected_method",
        "pinned_stack",
        "environment_hashes",
        "checkpoint_hashes",
        "upstream_source_hashes",
        "identities",
        "backend_waterfall",
        "inputs",
        "tensor_contract",
        "correctness",
        "timing",
        "analysis",
        "gates",
        "gpu_selection",
        "memory",
        "resource_caps",
        "recovery",
    )
    assert all(resolved[key] == base[key] for key in scientific)


def test_canonical_libero_config_is_created_once_and_verified(tmp_path: Path) -> None:
    project, run, config = project_tree(tmp_path)
    attestation = create_libero_config_once(project, run, config)
    path = Path(attestation["path"])
    original = path.read_bytes()
    assert validate_libero_config(project, run, config) == attestation
    with pytest.raises(V5DRecoveryViolation, match="already exists"):
        create_libero_config_once(project, run, config)
    assert path.read_bytes() == original


def test_libero_config_rejects_mismatch_escaping_path_and_symlink(tmp_path: Path) -> None:
    project, run, config = project_tree(tmp_path)
    path = libero_config_path(project, run, config)
    path.parent.mkdir(parents=True)
    path.write_text('{"wrong":"mapping"}\n')
    with pytest.raises(V5DRecoveryViolation, match="bytes changed"):
        validate_libero_config(project, run, config)

    escaping = copy.deepcopy(config)
    escaping["libero_config"]["config_relative_to_run"] = "../../outside.yaml"
    with pytest.raises(V5DRecoveryViolation, match="unsafe"):
        libero_config_path(project, run, escaping)

    symlink_run = project / "results" / "symlink-run"
    (symlink_run / "cache").mkdir(parents=True)
    outside = project / "elsewhere"
    outside.mkdir()
    (symlink_run / "cache" / "libero").symlink_to(outside, target_is_directory=True)
    with pytest.raises(V5DRecoveryViolation, match="symlink"):
        libero_config_path(project, symlink_run, config)


def test_libero_config_rejects_changed_keys(tmp_path: Path) -> None:
    project, run, config = project_tree(tmp_path)
    changed = copy.deepcopy(config)
    changed["libero_config"]["paths_relative_to_project"].pop("assets")
    with pytest.raises(V5DRecoveryViolation, match="keys changed"):
        create_libero_config_once(project, run, changed)


def test_libero_config_rejects_symlinked_source_mapping(tmp_path: Path) -> None:
    project, run, config = project_tree(tmp_path)
    assets = project / config["libero_config"]["paths_relative_to_project"]["assets"]
    assets.rmdir()
    replacement = project / "replacement-assets"
    replacement.mkdir()
    assets.symlink_to(replacement, target_is_directory=True)
    with pytest.raises(V5DRecoveryViolation, match="symlink"):
        create_libero_config_once(project, run, config)


def test_pre_model_stop_is_zero_query_semantic_and_write_once(tmp_path: Path) -> None:
    error = EOFError("closed stdin")
    record = build_pre_model_stop_record(
        run_id="acr-v5d-real-tensor-feasibility-v02",
        backend="torch-compile",
        execution_revision="revision",
        configuration_semantic_sha256="configuration",
        launch_manifest_semantic_sha256="launch",
        error=error,
        recorded_at_utc="2026-08-11T00:00:00+00:00",
        selected_gpu_after={"index": 0, "memory_used_mib": 6, "utilization_percent": 0},
    )
    assert record["semantic_sha256"] == semantic_sha256(record)
    assert record["model_queries"] == record["backend_preparation_core_launches"] == 0
    assert record["correctness_records"] == record["warmup_records"] == 0
    assert record["timed_records"] == 0
    assert record["raw_fallback_permitted"] is False
    path = tmp_path / "record.json"
    write_json_once(path, record)
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        write_json_once(path, record)
    assert path.read_bytes() == original
