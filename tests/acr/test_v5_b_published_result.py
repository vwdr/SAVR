from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/acr/v5_b_output_blind_preflight.json"
RESULT = ROOT / "reports/runtime/acr_v5_b.json"


def load_verifier():
    path = ROOT / "scripts/verify_acr_v5_b_result.py"
    spec = importlib.util.spec_from_file_location("verify_acr_v5_b_published", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_published_v5_b_result_reconciles_and_selects_frozen_candidate() -> None:
    verifier = load_verifier()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    assert verifier.verify(config, result) == []
    assert result["semantic_sha256"] == (
        "8a9f15b818b58ed2868d4b1123a222a4c062507161ab7de911d8d233f3b1efec"
    )
    assert result["selected_candidate_id"] == "v5-a100-b40"
    assert result["disposition"] == "ADVANCE_TO_V5_C_PROTOCOL"
    assert result["eligible_candidate_ids"] == [
        "v5-a100-b40",
        "v5-a150-b40",
        "v5-a200-b40",
    ]


def test_selected_candidate_passes_every_gate_without_outcomes_or_gpu() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    selected = next(
        item
        for item in result["candidates"]
        if item["candidate_id"] == result["selected_candidate_id"]
    )
    assert selected["eligible"] is True
    assert all(selected["gates"].values())
    assert selected["maximum_reuse_streak"] == 1
    assert selected["gripper_transition_reuses"] == 0
    assert selected["isolation_state_mismatches"] == 0
    assert selected["invariant_failures"] == 0
    assert result["resources"]["gpu_count"] == 0
    assert result["resources"]["new_task_outcomes"] == 0
    assert result["protected"]["success_fields"] == "SEALED"
