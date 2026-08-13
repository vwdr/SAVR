from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from savr.acr.v5_d_runtime import semantic_sha256
from savr.acr.v5_d_v09_runtime import load_v09
from savr.acr.v5_d_v10_runtime import (
    HYBRID_ARCHITECTURE,
    LIVE_QUERY,
    PRE_CAPTURE_WARMUP,
    PRIOR_TIMING_RATIONALE,
    load_v10,
    validate_v10_resolved,
)
from savr.acr.v5_d_v10_verification import verify_v10_architecture


ROOT = Path(__file__).resolve().parents[2]


def test_v10_changes_only_identity_architecture_and_authorization() -> None:
    v09 = load_v09(ROOT)
    v10 = load_v10(ROOT)
    changed = {
        "schema_version",
        "status",
        "authorized_at",
        "authorized_scope",
        "protocol",
        "run_id",
        "raw_cuda_graph",
        "pre_capture_warmup",
        "live_query",
        "recovery_v10",
        "current_authorization",
        "advance_only_to",
        "semantic_sha256",
    }
    assert {key: value for key, value in v10.items() if key not in changed} == {
        key: value for key, value in v09.items() if key not in changed
    }
    assert v10["allocator"] == v09["allocator"]
    assert v10["inference_semantics"] == v09["inference_semantics"]
    assert v10["correctness"] == v09["correctness"]
    assert v10["timing"] == v09["timing"]
    assert v10["gates"] == v09["gates"]
    assert v10["memory"] == v09["memory"]


def test_v10_hybrid_contract_and_rationale_are_exact() -> None:
    config = load_v10(ROOT)
    assert config["raw_cuda_graph"] == HYBRID_ARCHITECTURE
    assert config["pre_capture_warmup"] == PRE_CAPTURE_WARMUP
    assert config["live_query"] == LIVE_QUERY
    assert config["recovery_v10"]["prior_timing_rationale"] == PRIOR_TIMING_RATIONALE
    assert HYBRID_ARCHITECTURE["graph_object_count"] == 1
    assert HYBRID_ARCHITECTURE["capture_label"] == "downstream"
    assert HYBRID_ARCHITECTURE["wrist_capture_count"] == 0
    assert HYBRID_ARCHITECTURE["shared_private_pool"] is False


def test_v10_configuration_and_v09_evidence_are_authentic() -> None:
    config = load_v10(ROOT)
    stop = json.loads(
        (ROOT / "reports/runtime/acr_v5_d_v09_technical_stop.json").read_text(encoding="utf-8")
    )
    assert config["semantic_sha256"] == semantic_sha256(config)
    assert stop["semantic_sha256"] == semantic_sha256(stop)
    assert config["recovery_v10"]["v09_technical_stop_semantic_sha256"] == stop["semantic_sha256"]
    assert (
        config["recovery_v10"]["v09_raw_attempt_semantic_sha256"]
        == stop["raw_attempt_semantic_sha256"]
    )


def test_v10_implementation_authorization_stops_before_gpu() -> None:
    authorization = load_v10(ROOT)["current_authorization"]
    assert authorization["pre_gpu_implementation"] is True
    assert authorization["gpu_inspection_or_selection"] is False
    assert authorization["model_queries"] == 0
    assert authorization["simulator_use"] is False
    assert authorization["protected_outcome_access"] is False
    assert authorization["manuscript_changes"] is False


def test_v10_rejects_architecture_mutation() -> None:
    config = deepcopy(load_v10(ROOT))
    config["raw_cuda_graph"]["wrist_capture_count"] = 1
    config["semantic_sha256"] = semantic_sha256(config)
    with pytest.raises(ValueError, match="architecture changed"):
        validate_v10_resolved(config)


def complete_architecture_evidence() -> dict:
    return {
        "wrist_backend": "eager-static-buffer",
        "downstream_backend": "raw-cudagraph",
        "pre_capture_warmup_order": ["wrist", "downstream"],
        "pre_capture_warmup_calls": {"wrist": 3, "downstream": 3},
        "capture_attempt_order": ["downstream"],
        "capture_order": ["downstream"],
        "graph_objects_created": 1,
        "graph_objects_retained": 1,
        "wrist_capture_count": 0,
        "shared_pool_api_calls": 0,
        "supplied_pool_token": False,
        "empty_cache_calls": 0,
        "preparation_labels": [
            "raw-wrist-warmup-0",
            "raw-wrist-warmup-1",
            "raw-wrist-warmup-2",
            "raw-downstream-warmup-0",
            "raw-downstream-warmup-1",
            "raw-downstream-warmup-2",
            "raw-downstream-capture-0",
        ],
        "memory_trace": [
            {"stage": "wrist-after-pre-capture-warmup"},
            {"stage": "downstream-after-pre-capture-warmup"},
            {"stage": "downstream-after-capture"},
        ],
    }


def test_v10_independent_architecture_verifier_is_fail_closed() -> None:
    evidence = complete_architecture_evidence()
    record = {
        "hybrid_architecture": evidence,
        "preparation_labels": evidence["preparation_labels"],
    }
    assert verify_v10_architecture(record) == []
    changed = deepcopy(record)
    changed["hybrid_architecture"]["capture_order"] = ["wrist", "downstream"]
    assert "V10 completed capture order changed" in verify_v10_architecture(changed)
    assert verify_v10_architecture({}) == ["missing V10 hybrid architecture provenance"]


def test_v10_execution_wrappers_are_gated_before_gpu() -> None:
    runner = (ROOT / "scripts/run_acr_v5_d_v10.py").read_text(encoding="utf-8")
    selector = (ROOT / "scripts/select_acr_v5_d_v10_gpu.py").read_text(encoding="utf-8")
    launcher = (ROOT / "scripts/launch_acr_v5_d_v10.sh").read_text(encoding="utf-8")
    assert "V10 GPU execution is not authorized" in runner
    assert "V10 GPU inspection or selection is not authorized" in selector
    assert launcher.index("--backend torch-compile") < launcher.index("--backend raw-cudagraph")
    assert "expandable_segments:True" not in launcher
    assert "PYTORCH_CUDA_ALLOC_CONF=" not in launcher
