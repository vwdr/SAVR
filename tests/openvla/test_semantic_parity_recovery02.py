from __future__ import annotations

import dataclasses
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "openvla_semantic_parity_recovery02",
    ROOT / "scripts/run_openvla_semantic_parity_recovery02.py",
)
assert SPEC is not None and SPEC.loader is not None
RECOVERY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RECOVERY)


@dataclasses.dataclass(frozen=True)
class Prepared:
    normalized_proprio: object


def test_canonicalizes_only_normalized_proprio_shape() -> None:
    torch = pytest.importorskip("torch")
    tensor = torch.arange(8, dtype=torch.bfloat16).reshape(1, 8)
    original = Prepared(tensor)
    corrected = RECOVERY.canonicalize_prepared_proprio(original)
    assert tuple(corrected.normalized_proprio.shape) == (8,)
    assert torch.equal(corrected.normalized_proprio, tensor.reshape(-1))
    assert tuple(original.normalized_proprio.shape) == (1, 8)


def test_rejects_wrong_proprio_cardinality() -> None:
    torch = pytest.importorskip("torch")
    with pytest.raises(RuntimeError, match="exactly eight"):
        RECOVERY.canonicalize_prepared_proprio(
            Prepared(torch.zeros((1, 7), dtype=torch.bfloat16))
        )


@pytest.mark.parametrize(
    ("value", "label"),
    [
        (float("inf"), "positive_infinity"),
        (float("-inf"), "negative_infinity"),
        (float("nan"), "nan"),
    ],
)
def test_nonfinite_mismatch_is_strict_json_safe(value: float, label: str) -> None:
    record = RECOVERY.json_safe_comparison(
        {"boundary": "test", "max_abs": value, "passed": False}
    )
    assert record["max_abs"] is None
    assert record["max_abs_nonfinite"] == label
    json.dumps(record, allow_nan=False)


def test_finite_comparison_is_unchanged() -> None:
    record = {"boundary": "test", "max_abs": 0.25, "passed": False}
    assert RECOVERY.json_safe_comparison(record) == record


def test_real_shape_mismatch_record_can_be_sealed() -> None:
    torch = pytest.importorskip("torch")
    if not Path("/home/ved/SAVR/scripts/run_openvla_semantic_parity.py").is_file():
        pytest.skip("remote frozen worker is not available")
    recovery01 = RECOVERY.load_recovery01()
    v1 = recovery01.load_v1()
    record = v1.compare_values(
        boundary="normalized_proprio",
        reference=torch.zeros(8, dtype=torch.bfloat16),
        candidate=torch.zeros((1, 8), dtype=torch.bfloat16),
        tolerance=0.0,
        torch_module=torch,
        np_module=__import__("numpy"),
    )
    safe = RECOVERY.json_safe_comparison(record)
    assert safe["passed"] is False
    assert safe["max_abs"] is None
    assert safe["max_abs_nonfinite"] == "positive_infinity"
    json.dumps(
        {"records": [safe], "raw_values_persisted": False},
        allow_nan=False,
    )


def test_launcher_is_one_gpu_and_has_no_retry_or_cleanup() -> None:
    launcher = (
        ROOT / "scripts/launch_openvla_semantic_parity_recovery02.sh"
    ).read_text(encoding="utf-8")
    assert "CUDA_VISIBLE_DEVICES" in launcher
    assert "OPENVLA_PHYSICAL_GPU_ID" in launcher
    assert "while " not in launcher and "retry" not in launcher.lower()
    assert "rm " not in launcher and "sudo" not in launcher
