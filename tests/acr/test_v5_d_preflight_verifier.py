from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def load_verifier():
    path = ROOT / "scripts/verify_acr_v5_d_preflight.py"
    spec = importlib.util.spec_from_file_location("verify_acr_v5_d_preflight", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_preflight_verifier_is_deterministic_complete_and_published() -> None:
    verifier = load_verifier()
    first = verifier.verify()
    second = verifier.verify()
    assert first == second
    assert first["status"] == "pass"
    assert all(first["checks"].values())
    assert all(value == 0 for value in first["resources_used"].values())
    assert first["advance_only_to"] == "EXPLICIT_USER_COORDINATION_BEFORE_GPU_SELECTION"
    assert len(first["source_hashes"]) == 8
    assert len(first["semantic_sha256"]) == 64
    published = json.loads(
        (ROOT / "reports/runtime/acr_v5_d_preflight.json").read_text(encoding="utf-8")
    )
    assert published == first
