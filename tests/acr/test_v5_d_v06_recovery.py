from __future__ import annotations

from pathlib import Path

from savr.acr.v5_d_v05_runtime import load_v05
from savr.acr.v5_d_v06_runtime import PRE_CAPTURE_WARMUP, V05_MEASURED_MEMORY, load_v06


ROOT = Path(__file__).resolve().parents[2]


def test_v06_changes_only_raw_preparation_lifecycle_and_identity() -> None:
    v05 = load_v05(ROOT)
    v06 = load_v06(ROOT)
    changed = {
        "schema_version",
        "status",
        "authorized_at",
        "authorized_scope",
        "protocol",
        "run_id",
        "raw_cuda_graph",
        "pre_capture_warmup",
        "recovery_v06",
        "current_authorization",
        "advance_only_to",
        "semantic_sha256",
    }
    assert {key: value for key, value in v06.items() if key not in changed} == {
        key: value for key, value in v05.items() if key not in changed
    }
    assert v06["run_id"] == "acr-v5d-real-tensor-feasibility-v06"
    assert v06["pre_capture_warmup"] == PRE_CAPTURE_WARMUP
    assert v06["recovery_v06"]["v05_measured_memory"] == V05_MEASURED_MEMORY
    assert v06["memory"]["peak_reserved_bytes_max"] == 23 * 1024**3


def test_v06_raw_delta_is_exact_and_transition_is_unchanged() -> None:
    v05 = load_v05(ROOT)
    v06 = load_v06(ROOT)
    delta = {
        key: value
        for key, value in v06["raw_cuda_graph"].items()
        if v05["raw_cuda_graph"].get(key) != value
    }
    assert delta == {
        "warmup_order": ["wrist", "downstream"],
        "all_warmups_before_any_capture": True,
        "inter_capture_warmups": 0,
        "empty_cache_calls": 0,
    }
    assert v06["transition_revalidation"] == v05["transition_revalidation"]
    assert v06["raw_cuda_graph"]["shared_private_pool"] is True


def test_v06_authorization_stops_before_gpu_selection() -> None:
    config = load_v06(ROOT)
    authorization = config["current_authorization"]
    assert authorization["pre_gpu_implementation"] is True
    assert authorization["cuda_hidden_verification"] is True
    for key in (
        "gpu_inspection",
        "gpu_selection",
        "model_loading",
        "model_queries",
        "cuda_compile_capture_or_timing",
        "simulator_use",
        "protected_outcome_access",
        "manuscript_changes",
    ):
        assert authorization[key] is False
    assert config["advance_only_to"] == "EXPLICIT_USER_COORDINATION_BEFORE_V06_GPU_SELECTION"
