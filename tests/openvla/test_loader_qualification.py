from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "openvla_loader_qualification",
    ROOT / "scripts/run_openvla_loader_qualification.py",
)
assert SPEC is not None and SPEC.loader is not None
WORKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(WORKER)
PREFLIGHT_SPEC = importlib.util.spec_from_file_location(
    "openvla_loader_qualification_preflight",
    ROOT / "scripts/preflight_openvla_loader_qualification.py",
)
assert PREFLIGHT_SPEC is not None and PREFLIGHT_SPEC.loader is not None
PREFLIGHT = importlib.util.module_from_spec(PREFLIGHT_SPEC)
PREFLIGHT_SPEC.loader.exec_module(PREFLIGHT)


def test_method_source_extraction_is_exact_and_unambiguous() -> None:
    source = """
class Model:
    def predict_action(self, value):
        return value + 1
"""
    observed = WORKER.extract_method_source(source, "predict_action")
    assert observed == "def predict_action(self, value):\n    return value + 1"
    with pytest.raises(WORKER.LoaderQualificationStop, match="0 definitions"):
        WORKER.extract_method_source(source, "missing")
    with pytest.raises(WORKER.LoaderQualificationStop, match="2 definitions"):
        WORKER.extract_method_source(source + source, "predict_action")


def test_loader_package_has_zero_policy_or_outcome_calls() -> None:
    worker = (ROOT / "scripts/run_openvla_loader_qualification.py").read_text(
        encoding="utf-8"
    )
    launcher = (ROOT / "scripts/launch_openvla_loader_qualification.sh").read_text(
        encoding="utf-8"
    )
    assert PREFLIGHT.prohibited_calls(worker) == []
    assert PREFLIGHT.prohibited_calls("result = model(batch)") == ["model"]
    assert PREFLIGHT.prohibited_calls("environment.step(action)") == ["step"]
    assert "CUDA_VISIBLE_DEVICES" in launcher
    assert "OPENVLA_PHYSICAL_GPU_ID" in launcher
    assert "rm " not in launcher and "sudo" not in launcher


def test_semantic_hash_ignores_only_its_own_field() -> None:
    value = {"schema_version": "test", "policy_calls": 0}
    expected = WORKER.semantic_sha256(value)
    value["semantic_sha256"] = expected
    assert WORKER.semantic_sha256(value) == expected
    value["policy_calls"] = 1
    assert WORKER.semantic_sha256(value) != expected
