from __future__ import annotations

import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/acr/v5_c_cpu_executor_freeze.json"
RESEARCH = ROOT / "docs/ACR_V5_C_EXECUTOR_RESEARCH_AND_DESIGN.md"
PROTOCOL = ROOT / "docs/ACR_V5_C_CPU_EXECUTOR_PROTOCOL.md"


def freeze() -> dict[str, object]:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_v5_c_preserves_the_exact_v5_b_selection() -> None:
    selected = freeze()["selected_method"]
    assert selected == {
        "method_id": "v5-a100-b40",
        "controller_version": "acr-isolated-controller-v1",
        "policy": "sa-acr",
        "scene_threshold": 0.30046895424836606,
        "translation_threshold": 0.685919037527938,
        "horizon": 1,
        "hard_reuse_cap": 0.4,
        "minimum_query_index": 2,
        "post_reuse_latch": True,
        "cache_age_consistency": True,
        "v5_b_semantic_sha256": (
            "8a9f15b818b58ed2868d4b1123a222a4c062507161ab7de911d8d233f3b1efec"
        ),
    }


def test_required_reuse_wall_ratios_reconcile() -> None:
    values = freeze()["feasibility"]
    refresh = values["prior_refresh_wall_ratio_vs_bfr"]
    target = values["future_weighted_wall_ratio_target"]
    point = values["v5_b_reuse_point"]
    lower = values["v5_b_reuse_lower_95"]
    point_required = (target - (1.0 - point) * refresh) / point
    lower_required = (target - (1.0 - lower) * refresh) / lower
    assert values["required_reuse_wall_ratio_at_point"] == pytest.approx(point_required)
    assert values["required_reuse_wall_ratio_at_lower_95"] == pytest.approx(lower_required)
    assert values["measurement_claim"] is False


def test_executor_contract_has_two_fresh_cores_and_complete_key() -> None:
    config = freeze()
    assert config["identities"] == {
        "reference_executor": "acr-reuse-executor-eager-v1",
        "static_executor": "acr-reuse-executor-static-v1",
        "integration": "ir-sa-acr-static-executor-v1",
    }
    assert set(config["cores"]) == {"wrist_visual_core", "downstream_action_core"}
    assert config["cores"]["wrist_visual_core"]["scene_encoder_calls"] == 0
    assert config["cores"]["downstream_action_core"]["fresh_each_reuse"] is True
    fields = set(config["compatibility_key_fields"])
    assert {
        "checkpoint_id",
        "instruction_sha256",
        "prompt_input_shape",
        "dtype",
        "device",
        "wrist_shape",
        "cached_scene_shape",
        "proprioception_shape",
        "model_training_state",
        "use_film",
        "use_diffusion",
    } <= fields
    assert len(fields) == len(config["compatibility_key_fields"])


def test_lifecycle_and_failure_rules_are_fail_closed() -> None:
    config = freeze()
    assert config["lifecycle"] == ["UNPREPARED", "PREPARED", "ACTIVE", "INVALIDATED"]
    assert config["reason_codes"] == {
        "prelaunch": "executor-unavailable",
        "postlaunch": "executor-failure",
    }
    failure = config["failure_policy"]
    assert failure["prelaunch_forces_eager_refresh"] is True
    assert failure["prelaunch_observes_reuse"] is False
    assert failure["postlaunch_invalidates_executor"] is True
    assert failure["postlaunch_invalidates_scene_cache"] is True
    assert failure["postlaunch_observes_controller"] is False
    assert failure["postlaunch_retry"] is False
    assert failure["restore_patched_methods_on_exit"] is True


def test_v5_c_is_cpu_only_and_does_not_claim_performance() -> None:
    config = freeze()
    caps = config["resource_caps"]
    assert caps["gpu_count"] == 0
    assert caps["model_queries"] == 0
    assert caps["simulator_episodes"] == 0
    assert caps["downloads"] == 0
    assert caps["new_task_outcomes"] == 0
    assert all(config["excluded"].values())
    assert config["advance_only_to"] == "V5_D_PROTOCOL_PREPARATION"


def test_docs_freeze_boundary_and_reject_unsafe_whole_query_capture() -> None:
    research = RESEARCH.read_text(encoding="utf-8")
    protocol = PROTOCOL.read_text(encoding="utf-8")
    assert "COMPLETE BEFORE V5-C IMPLEMENTATION" in research
    assert "FROZEN BEFORE V5-C IMPLEMENTATION" in protocol
    assert "complete `predict_action` function is therefore rejected" in research
    assert "Wrist visual core" in research
    assert "Downstream action core" in research
    assert ".cpu().detach().numpy()" in research
    assert "no partial failure may silently retry" in protocol
    assert "torch.compile`, CUDA graph capture/replay" in protocol
