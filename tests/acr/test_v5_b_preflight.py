from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/acr/v5_b_output_blind_preflight.json"


def load_script(name: str) -> ModuleType:
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ANALYZER = load_script("analyze_acr_v5_b.py")
VERIFIER = load_script("verify_acr_v5_b_result.py")


def config() -> dict[str, object]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_v5_b_freezes_six_exact_isolated_reuse_candidates() -> None:
    frozen = config()
    assert frozen["status"] == "FROZEN_BEFORE_V5_B_CANDIDATE_OUTPUTS"
    assert frozen["method"] == {
        "name": "IR-SA-ACR",
        "controller_version": "acr-isolated-controller-v1",
        "policy": "sa-acr",
        "horizon": 1,
        "minimum_query_index": 2,
        "post_reuse_latch": True,
        "cache_age_consistency": True,
        "gripper_transition_veto": True,
        "direction_reversal_veto": False,
    }
    candidates = frozen["candidates"]
    assert len(candidates) == frozen["threshold_family"]["candidate_count"] == 6
    assert [item["candidate_id"] for item in candidates] == [
        "v5-a100-b35",
        "v5-a100-b40",
        "v5-a150-b35",
        "v5-a150-b40",
        "v5-a200-b35",
        "v5-a200-b40",
    ]


def test_thresholds_follow_the_frozen_extrapolation_formula() -> None:
    frozen = config()
    family = frozen["threshold_family"]
    for candidate in frozen["candidates"]:
        level = candidate["level"]
        expected_scene = family["scene_low"] + level * (family["scene_high"] - family["scene_low"])
        expected_translation = family["translation_low"] + level * (
            family["translation_high"] - family["translation_low"]
        )
        assert candidate["scene_threshold"] == expected_scene
        assert candidate["translation_threshold"] == expected_translation


def test_input_is_outcome_sealed_and_protected_populations_remain_closed() -> None:
    frozen = config()
    assert frozen["input"]["outcome_fields_allowed"] is False
    assert frozen["input"]["trace_records"] == 1773
    assert frozen["input"]["episodes"] == 100
    assert frozen["protected"]["success_fields"] == "SEALED"
    assert frozen["protected"]["goal_states_0_9_seed_0"] == "UNOPENED"
    assert set(frozen["resource_caps"].values()) >= {0}
    assert frozen["resource_caps"]["gpu_count"] == 0
    assert frozen["resource_caps"]["new_task_outcomes"] == 0


def test_trace_schema_rejects_outcome_or_extra_fields() -> None:
    record = {key: None for key in ANALYZER.TRACE_KEYS}
    record["schema_version"] = "acr.fr-trace-query.v1"
    record["semantic_sha256"] = ANALYZER.semantic_sha256(record)
    ANALYZER.verify_trace_schema(record)
    record["success"] = True
    try:
        ANALYZER.verify_trace_schema(record)
    except RuntimeError as error:
        assert "schema changed" in str(error)
    else:  # pragma: no cover - safety assertion
        raise AssertionError("Outcome-bearing trace was accepted")


def test_synthetic_replay_enforces_isolation_and_prefix_cap() -> None:
    scene = (0.0,) * 1024
    action = (0.0,) * 56
    queries = tuple(
        ANALYZER.ReplayQuery(
            episode_id="episode-0",
            query_index=index,
            scene_representation=scene,
            normalized_eef_position=(0.0, 0.0, 0.0),
            action_chunk=action,
        )
        for index in range(20)
    )
    candidate = {
        "candidate_id": "synthetic",
        "level": 1.0,
        "scene_threshold": 1.0,
        "translation_threshold": 1.0,
        "hard_reuse_cap": 0.4,
    }
    first = ANALYZER.replay_candidate({"episode-0": queries}, candidate)
    second = ANALYZER.replay_candidate({"episode-0": queries}, candidate)
    assert ANALYZER.canonical_bytes(first) == ANALYZER.canonical_bytes(second)
    assert first["maximum_reuse_streak"] == 1
    assert first["maximum_prefix_reuse"] <= 0.4
    assert first["prefix_cap_violations"] == 0
    assert first["gripper_transition_reuses"] == 0
    assert first["isolation_state_mismatches"] == 0
    assert first["invariant_failures"] == 0
    assert first["post_reuse_refreshes"] > 0


def test_independent_verifier_recomputes_selection_and_rejects_tampering() -> None:
    frozen = config()
    candidate = frozen["candidates"][0]
    result_candidate = {
        **candidate,
        "episodes": 100,
        "queries": 1773,
        "reuse_rate": 0.36,
        "maximum_reuse_streak": 1,
        "prefix_cap_violations": 0,
        "gripper_transition_reuses": 0,
        "isolation_state_mismatches": 0,
        "invariant_failures": 0,
        "post_reuse_refreshes": 100,
        "reuse_rate_interval": {"lower_95": 0.31, "median": 0.36, "upper_95": 0.39},
        "logical_visual_reduction_point": 0.18,
        "logical_visual_reduction_interval": {
            "lower_95": 0.155,
            "median": 0.18,
            "upper_95": 0.195,
        },
    }
    checks = {
        "population": True,
        "maximum_streak": True,
        "prefix_cap": True,
        "gripper_transition": True,
        "isolation_state": True,
        "invariants": True,
        "post_reuse_refresh": True,
        "reuse_point": True,
        "reuse_lower_95": True,
        "logical_visual_point": True,
        "logical_visual_lower_95": True,
    }
    result_candidate.update({"gates": checks, "eligible": True})
    ineligible_checks = {key: False for key in checks}
    ineligible = [
        {
            **item,
            "episodes": 0,
            "queries": 0,
            "reuse_rate": 0.0,
            "maximum_reuse_streak": 2,
            "prefix_cap_violations": 1,
            "gripper_transition_reuses": 1,
            "isolation_state_mismatches": 1,
            "invariant_failures": 1,
            "post_reuse_refreshes": 0,
            "reuse_rate_interval": {"lower_95": 0.0, "median": 0.0, "upper_95": 0.0},
            "logical_visual_reduction_point": 0.0,
            "logical_visual_reduction_interval": {
                "lower_95": 0.0,
                "median": 0.0,
                "upper_95": 0.0,
            },
            "gates": ineligible_checks,
            "eligible": False,
        }
        for item in frozen["candidates"][1:]
    ]
    result = {
        "schema_version": "acr.v5b-result.v1",
        "candidate_count": 6,
        "replay_repetitions": 2,
        "replay_byte_identical": True,
        "input_manifest": {
            key: frozen["input"][key]
            for key in (
                "trace_records",
                "trace_artifact_bytes",
                "ordered_path_content_sha256",
            )
        },
        "candidates": [result_candidate, *ineligible],
        "eligible_candidate_ids": [candidate["candidate_id"]],
        "selected_candidate_id": candidate["candidate_id"],
        "disposition": "ADVANCE_TO_V5_C_PROTOCOL",
        "resources": {
            "gpu_count": 0,
            "model_queries": 0,
            "simulator_episodes": 0,
            "simulator_resets": 0,
            "downloads": 0,
            "new_task_outcomes": 0,
        },
        "protected": frozen["protected"],
    }
    result["semantic_sha256"] = VERIFIER.semantic_sha256(result)
    assert VERIFIER.verify(frozen, result) == []
    result["selected_candidate_id"] = None
    assert "result semantic hash mismatch" in VERIFIER.verify(frozen, result)
    assert "selected candidate mismatch" in VERIFIER.verify(frozen, result)
