from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType


ROOT = Path(__file__).resolve().parents[2]


def load_verifier() -> ModuleType:
    path = ROOT / "scripts/verify_acr_v4_a_result.py"
    spec = importlib.util.spec_from_file_location("verify_acr_v4_a_result", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_published_v4_a_result_mechanically_stops() -> None:
    verifier = load_verifier()
    preflight = json.loads(
        (ROOT / "configs/acr/v4_a_diagnosis_preflight.json").read_text(encoding="utf-8")
    )
    result = json.loads((ROOT / "reports/runtime/acr_v4_a.json").read_text(encoding="utf-8"))
    assert verifier.verify(preflight, result) == []
    assert result["disposition"] == "STOP_BEFORE_V4_B"
    assert result["selected_candidate_id"] is None
    assert result["selected_executor"] is None
    assert len(result["candidates"]) == 6
    assert all(not candidate["eligible"] for candidate in result["candidates"])
    assert all(candidate["maximum_reuse_streak"] == 2 for candidate in result["candidates"])
