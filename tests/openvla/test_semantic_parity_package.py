from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "openvla_semantic_parity_worker", ROOT / "scripts/run_openvla_semantic_parity.py"
)
assert SPEC is not None and SPEC.loader is not None
WORKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(WORKER)
ParityTechnicalStop = WORKER.ParityTechnicalStop
compare_values = WORKER.compare_values
file_sha256 = WORKER.file_sha256
semantic_sha256 = WORKER.semantic_sha256
validate_static = WORKER.validate_static


torch = pytest.importorskip("torch")


def _write(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _fixture(tmp_path: Path) -> tuple[dict, Path]:
    source = tmp_path / "data/source.hdf5"
    source_hash = _write(source, b"frozen-observation-source")
    suites = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
    inputs = []
    for suite in suites:
        for ordinal in range(2):
            inputs.append(
                {
                    "suite": suite,
                    "source_path": "source.hdf5",
                    "source_sha256": source_hash,
                }
            )
    manifest = {
        "inputs": inputs,
        "data_root_relative": "data",
        "terminal_outcome_fields_accessed": False,
        "expert_action_fields_accessed": False,
    }
    manifest["semantic_sha256"] = semantic_sha256(manifest)
    manifest_path = tmp_path / "inputs.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    checkpoint_hashes = {}
    for name in ("config.json", "configuration_prismatic.py", "modeling_prismatic.py"):
        checkpoint_hashes[name] = _write(tmp_path / "checkpoint" / name, name.encode())
    authenticated = tmp_path / "contract.py"
    authenticated_hash = _write(authenticated, b"canonical-contract")
    config = {
        "schema_version": "openvla-semantic-parity-s3-v1",
        "run_id": "test",
        "authorization": {
            "s3_authorized": True,
            "source": "User approved continuation on 2026-08-31",
            "simulator": False,
            "terminal_outcomes": False,
            "training": False,
            "automatic_retry": False,
        },
        "resource_caps": {
            "gpu_count": 1,
            "model_processes": 1,
            "observations": 8,
            "calls_per_observation": 4,
            "model_call_hard_cap": 32,
            "wall_seconds": 1800,
            "artifact_bytes": 268435456,
            "peak_gpu_memory_mib_strict_max": 23552,
            "downloads": 0,
            "simulator_outcomes": 0,
            "raw_actions_persisted": False,
            "automatic_retry": False,
        },
        "advance": {
            "next_stage": "S4_D62_REQUALIFICATION",
            "authorized": False,
            "stop_before_next_stage": True,
        },
        "output_root": "results/openvla-semantic-parity-s3-v01",
        "input_manifest": "inputs.json",
        "authenticated_files": {"contract.py": authenticated_hash},
        "model": {
            "checkpoint_relative": "checkpoint",
            "checkpoint_metadata_sha256": checkpoint_hashes,
        },
    }
    config["semantic_sha256"] = semantic_sha256(config)
    return config, tmp_path


def test_static_validation_authenticates_population_and_caps(tmp_path: Path) -> None:
    config, root = _fixture(tmp_path)
    validate_static(config, root)
    config["resource_caps"]["model_call_hard_cap"] = 33
    config["semantic_sha256"] = semantic_sha256(config)
    with pytest.raises(ParityTechnicalStop, match="resource boundary"):
        validate_static(config, root)


def test_static_validation_rejects_source_drift_and_output_reuse(tmp_path: Path) -> None:
    config, root = _fixture(tmp_path)
    (root / "data/source.hdf5").write_bytes(b"changed")
    with pytest.raises(ParityTechnicalStop, match="observation source"):
        validate_static(config, root)
    config, root = _fixture(tmp_path / "reuse")
    (root / config["output_root"]).mkdir(parents=True)
    with pytest.raises(ParityTechnicalStop, match="already exists"):
        validate_static(config, root)


def test_comparison_records_hashes_and_detects_first_difference() -> None:
    reference = torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16)
    same = reference.clone()
    exact = compare_values(
        boundary="hidden",
        reference=reference,
        candidate=same,
        tolerance=0.0,
        torch_module=torch,
        np_module=np,
    )
    assert exact["passed"] and exact["max_abs"] == 0.0
    changed = compare_values(
        boundary="hidden",
        reference=reference,
        candidate=reference + torch.tensor([[0.0, 0.01]], dtype=torch.bfloat16),
        tolerance=1e-6,
        torch_module=torch,
        np_module=np,
    )
    assert not changed["passed"] and changed["max_abs"] > 1e-6
    assert changed["reference_sha256"] != changed["candidate_sha256"]


def test_launch_package_has_no_outcome_path_or_legacy_tail_slice() -> None:
    root = ROOT
    worker = (root / "scripts/run_openvla_semantic_parity.py").read_text(encoding="utf-8")
    contract = (root / "src/savr/openvla/official_semantics.py").read_text(encoding="utf-8")
    launcher = (root / "scripts/launch_openvla_semantic_parity.sh").read_text(encoding="utf-8")
    assert "-57:-1" not in contract
    assert "environment.step" not in worker
    assert "get_libero_env" not in worker
    assert "CUDA_VISIBLE_DEVICES" in launcher
    assert "OPENVLA_PHYSICAL_GPU_ID" in launcher
    assert "rm " not in launcher and "sudo" not in launcher


def test_file_hash_is_stable(tmp_path: Path) -> None:
    path = tmp_path / "value"
    path.write_bytes(b"semantic")
    assert file_sha256(path) == hashlib.sha256(b"semantic").hexdigest()
