from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def load_verifier():
    path = ROOT / "scripts/verify_acr_v5_d_v10_preflight.py"
    spec = importlib.util.spec_from_file_location("verify_acr_v5_d_v10_preflight", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v10_preflight_is_deterministic_complete_and_stops_before_gpu() -> None:
    verifier = load_verifier()
    first = verifier.verify()
    second = verifier.verify()
    assert first == second
    assert first["status"] == "pass"
    assert all(first["checks"].values())
    assert len(first["checks"]) == 15
    assert all(
        value == 0
        for key, value in first["resources_used"].items()
        if key != "cuda_visible_devices" and not isinstance(value, bool)
    )
    assert first["resources_used"]["cuda_visible_devices"] == ""
    assert first["resources_used"]["cuda_initialized"] is False
    assert first["advance_only_to"] == "STOP_BEFORE_V10_GPU_INSPECTION_OR_SELECTION"
    assert len(first["source_hashes"]) == 15
    assert len(first["semantic_sha256"]) == 64
