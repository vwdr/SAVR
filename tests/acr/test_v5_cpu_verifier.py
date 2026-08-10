from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType


ROOT = Path(__file__).resolve().parents[2]


def load_verifier() -> ModuleType:
    path = ROOT / "scripts/verify_acr_v5_isolation.py"
    spec = importlib.util.spec_from_file_location("verify_acr_v5_isolation", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_dependency_free_v5_verifier_passes() -> None:
    record = load_verifier().verify()
    assert record["verified"] is True
    assert record["isolated_trace"]["maximum_reuse_streak"] == 1
    assert record["legacy_trace"]["maximum_reuse_streak"] == 2
    assert "post-reuse-refresh" in record["mismatch_reasons"]
    assert "isolation-state-mismatch" in record["mismatch_reasons"]
    assert set(record["resources"].values()) == {0}
    assert len(record["semantic_sha256"]) == 64
    published = json.loads(
        (ROOT / "reports/runtime/acr_v5_cpu_verification.json").read_text(encoding="utf-8")
    )
    assert published == record
