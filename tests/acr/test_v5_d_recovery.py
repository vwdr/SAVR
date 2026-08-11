from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from savr.acr.v5_d_recovery import (
    PROTECTED_CHECKPOINT_NAMES,
    V5DRecoveryViolation,
    build_pre_model_stop_record,
    capture_checkpoint_baseline,
    create_libero_config_once,
    libero_config_path,
    restore_checkpoint_exact,
    semantic_sha256,
    validate_libero_config,
    write_json_once,
)
from savr.acr.v5_d_runtime import (
    load_v5_d_freeze,
    resolve_v5_d_recovery,
    validate_v5_d_freeze,
)


ROOT = Path(__file__).resolve().parents[2]


def recovery_config() -> dict:
    return json.loads((ROOT / "configs/acr/v5_d_gpu_feasibility_recovery_v02.json").read_text())


def project_tree(tmp_path: Path) -> tuple[Path, Path, dict]:
    project = tmp_path / "SAVR"
    run = project / "results" / "acr-v5d-real-tensor-feasibility-v03"
    for relative in recovery_config()["libero_config"]["paths_relative_to_project"].values():
        (project / relative).mkdir(parents=True, exist_ok=True)
    return project, run, {"libero_config": recovery_config()["libero_config"]}


def checkpoint_tree(tmp_path: Path) -> tuple[Path, dict[str, bytes]]:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    protected = {
        "config.json": b"frozen-config\n",
        "configuration_prismatic.py": b"frozen-configuration\n",
        "modeling_prismatic.py": b"frozen-modeling\n",
    }
    for name, payload in protected.items():
        (checkpoint / name).write_bytes(payload)
    (checkpoint / "dataset_statistics.json").write_bytes(b"frozen-statistics\n")
    (checkpoint / "lora_adapter").mkdir()
    return checkpoint, protected


def test_v03_overlay_preserves_every_scientific_section() -> None:
    base = json.loads((ROOT / "configs/acr/v5_d_gpu_feasibility_freeze.json").read_text())
    resolved_v02 = resolve_v5_d_recovery(base, recovery_config())
    resolved = load_v5_d_freeze(ROOT)
    validate_v5_d_freeze(resolved)
    assert resolved["run_id"] == "acr-v5d-real-tensor-feasibility-v03"
    assert resolved["recovery_v02"]["v01_technical_stop_semantic_sha256"] == (
        "edf5872fa818f5806601f52143cb17cec7dd4974e03cc4e2ed43c3d042fb4412"
    )
    assert resolved["recovery_v03"]["v02_technical_stop_semantic_sha256"] == (
        "0a30bd847bf2e1549c376200e559a23c670b33c0b01215926c90a15704487661"
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
    assert all(resolved[key] == resolved_v02[key] for key in scientific)


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
        run_id="acr-v5d-real-tensor-feasibility-v03",
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


def test_v03_exact_observed_backups_restore_and_repeat_idempotently(tmp_path: Path) -> None:
    checkpoint, protected = checkpoint_tree(tmp_path)
    baseline = capture_checkpoint_baseline(checkpoint)
    (checkpoint / "config.json").write_bytes(b"loader-config\n")
    (checkpoint / "modeling_prismatic.py").write_bytes(b"loader-modeling\n")
    (checkpoint / "config.json.back.20260810_212317").write_bytes(protected["config.json"])
    (checkpoint / "modeling_prismatic.py.back.20260810_212317").write_bytes(
        protected["modeling_prismatic.py"]
    )

    restored = restore_checkpoint_exact(checkpoint, baseline)
    assert restored["protected_bytes_restored"] is True
    assert restored["backup_cleanup_complete"] is True
    assert restored["removed_loader_backups"] == [
        "config.json.back.20260810_212317",
        "modeling_prismatic.py.back.20260810_212317",
    ]
    assert {name: (checkpoint / name).read_bytes() for name in protected} == protected
    assert restore_checkpoint_exact(checkpoint, baseline)["removed_loader_backups"] == []


@pytest.mark.parametrize(
    "suffix",
    (".back.20260811_010203", ".bak", ".backup", ".backup.20260811_010203"),
)
def test_v03_permitted_backup_forms_are_exact_and_content_verified(
    tmp_path: Path, suffix: str
) -> None:
    checkpoint, protected = checkpoint_tree(tmp_path)
    baseline = capture_checkpoint_baseline(checkpoint)
    backup = checkpoint / f"configuration_prismatic.py{suffix}"
    backup.write_bytes(protected["configuration_prismatic.py"])
    restored = restore_checkpoint_exact(checkpoint, baseline)
    assert restored["removed_loader_backups"] == [backup.name]
    assert not backup.exists()


def test_v03_rejects_backup_content_mismatch_without_deleting_any_artifact(
    tmp_path: Path,
) -> None:
    checkpoint, protected = checkpoint_tree(tmp_path)
    baseline = capture_checkpoint_baseline(checkpoint)
    valid = checkpoint / "config.json.back.20260810_212317"
    invalid = checkpoint / "modeling_prismatic.py.back.20260810_212317"
    valid.write_bytes(protected["config.json"])
    invalid.write_bytes(b"not-the-frozen-modeling-file\n")
    with pytest.raises(V5DRecoveryViolation, match="backup content changed"):
        restore_checkpoint_exact(checkpoint, baseline)
    assert valid.exists() and invalid.exists()


@pytest.mark.parametrize("kind", ("file", "directory", "symlink"))
def test_v03_rejects_unexpected_artifact_types_without_cleanup(
    tmp_path: Path, kind: str
) -> None:
    checkpoint, protected = checkpoint_tree(tmp_path)
    baseline = capture_checkpoint_baseline(checkpoint)
    valid = checkpoint / "config.json.back.20260810_212317"
    valid.write_bytes(protected["config.json"])
    unexpected = checkpoint / "unexpected-loader-artifact"
    if kind == "file":
        unexpected.write_bytes(b"unexpected\n")
    elif kind == "directory":
        unexpected.mkdir()
    else:
        unexpected.symlink_to(checkpoint / "config.json")
    with pytest.raises(V5DRecoveryViolation, match="Unexpected|symlink"):
        restore_checkpoint_exact(checkpoint, baseline)
    assert valid.exists()


def test_v03_restores_changed_protected_file_but_rejects_baseline_drift(
    tmp_path: Path,
) -> None:
    checkpoint, protected = checkpoint_tree(tmp_path)
    baseline = capture_checkpoint_baseline(checkpoint)
    (checkpoint / "config.json").write_bytes(b"loader-mutated-config\n")
    restore_checkpoint_exact(checkpoint, baseline)
    assert (checkpoint / "config.json").read_bytes() == protected["config.json"]

    statistics = checkpoint / "dataset_statistics.json"
    statistics.write_bytes(b"changed-statistics\n")
    with pytest.raises(V5DRecoveryViolation, match="Pre-existing checkpoint entry changed"):
        restore_checkpoint_exact(checkpoint, baseline)


def test_v03_partial_cleanup_failure_stays_closed_and_is_recoverable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint, protected = checkpoint_tree(tmp_path)
    baseline = capture_checkpoint_baseline(checkpoint)
    first = checkpoint / "config.json.back.20260810_212317"
    second = checkpoint / "modeling_prismatic.py.back.20260810_212317"
    first.write_bytes(protected["config.json"])
    second.write_bytes(protected["modeling_prismatic.py"])
    original_unlink = Path.unlink

    def fail_second(path: Path, *args: object, **kwargs: object) -> None:
        if path.name == second.name:
            raise OSError("injected cleanup failure")
        original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_second)
    with pytest.raises(V5DRecoveryViolation, match="cleanup failed"):
        restore_checkpoint_exact(checkpoint, baseline)
    assert not first.exists()
    assert second.exists()
    assert {name: (checkpoint / name).read_bytes() for name in protected} == protected

    monkeypatch.undo()
    restored = restore_checkpoint_exact(checkpoint, baseline)
    assert restored["removed_loader_backups"] == [second.name]


def test_v03_capture_rejects_unsafe_or_incomplete_protected_inventory(tmp_path: Path) -> None:
    checkpoint, _ = checkpoint_tree(tmp_path)
    (checkpoint / PROTECTED_CHECKPOINT_NAMES[0]).unlink()
    with pytest.raises(V5DRecoveryViolation, match="inventory is incomplete"):
        capture_checkpoint_baseline(checkpoint)
