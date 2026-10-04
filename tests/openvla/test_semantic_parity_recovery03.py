from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "openvla_semantic_parity_recovery03",
    ROOT / "scripts/run_openvla_semantic_parity_recovery03.py",
)
assert SPEC is not None and SPEC.loader is not None
RECOVERY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RECOVERY)


def test_boundary_contract_is_complete_and_disjoint() -> None:
    assert len(RECOVERY.ALL_BOUNDARIES) == 22
    assert not (RECOVERY.TENSOR_BOUNDARIES & RECOVERY.MIXED_BOUNDARIES)
    assert not (RECOVERY.TENSOR_BOUNDARIES & RECOVERY.ARRAY_BOUNDARIES)
    assert not (RECOVERY.MIXED_BOUNDARIES & RECOVERY.ARRAY_BOUNDARIES)
    assert RECOVERY.expected_representation_pair("normalized_actions") == (
        "tensor_cuda",
        "ndarray_cpu",
    )
    with pytest.raises(RuntimeError, match="uncontracted"):
        RECOVERY.expected_representation_pair("unknown")


def test_array_contract_and_strict_failure_records() -> None:
    torch = pytest.importorskip("torch")
    if not Path("/home/ved/SAVR/scripts/run_openvla_semantic_parity.py").is_file():
        pytest.skip("remote frozen worker is not available")
    recovery02 = RECOVERY.load_recovery02()
    v1 = recovery02.load_recovery01().load_v1()
    exact = RECOVERY.compare_with_contract(
        v1.compare_values,
        boundary="unnormalized_actions",
        reference=np.zeros(8, dtype=np.float32),
        candidate=np.zeros(8, dtype=np.float32),
        tolerance=0.0,
        torch_module=torch,
        np_module=np,
    )
    mismatch = RECOVERY.compare_with_contract(
        v1.compare_values,
        boundary="unnormalized_actions",
        reference=np.zeros(8, dtype=np.float32),
        candidate=np.zeros((1, 8), dtype=np.float32),
        tolerance=0.0,
        torch_module=torch,
        np_module=np,
    )
    assert exact["passed"] is True
    assert mismatch["passed"] is False
    assert mismatch["max_abs"] is None
    assert mismatch["max_abs_nonfinite"] == "positive_infinity"
    json.dumps({"records": [exact, mismatch]}, allow_nan=False)


def test_representation_drift_is_rejected() -> None:
    torch = pytest.importorskip("torch")
    if not Path("/home/ved/SAVR/scripts/run_openvla_semantic_parity.py").is_file():
        pytest.skip("remote frozen worker is not available")
    recovery02 = RECOVERY.load_recovery02()
    v1 = recovery02.load_recovery01().load_v1()
    with pytest.raises(RuntimeError, match="representation changed"):
        RECOVERY.compare_with_contract(
            v1.compare_values,
            boundary="normalized_actions",
            reference=np.zeros((8, 7), dtype=np.float32),
            candidate=np.zeros((8, 7), dtype=np.float32),
            tolerance=0.0,
            torch_module=torch,
            np_module=np,
        )


def test_launcher_requires_qualification_before_one_model_attempt() -> None:
    launcher = (
        ROOT / "scripts/launch_openvla_semantic_parity_recovery03.sh"
    ).read_text(encoding="utf-8")
    qualification = launcher.index("run_openvla_comparator_qualification.py")
    model = launcher.index("run_openvla_semantic_parity_recovery03.py")
    assert qualification < model
    assert "CUDA_VISIBLE_DEVICES" in launcher
    assert "while " not in launcher and "retry" not in launcher.lower()
    assert "rm " not in launcher and "sudo" not in launcher
