from __future__ import annotations

import json
from pathlib import Path

import pytest

from savr.brace.b3 import (
    B3ProtocolError,
    QueryLedger,
    cycle_schedule,
    empirical_quantile,
    allowed_project_status,
    load_config_file,
    planned_query_count,
    profile_speed_gate,
    semantic_sha256,
    summarize_timings,
    validate_config,
)
from savr.brace.b3_openvla import (
    SourceTracker,
    action_record,
    compare_actions,
    deterministic_inputs,
    midpoint_proprio_state,
)


ROOT = Path(__file__).resolve().parents[2]


def load_config() -> dict:
    return json.loads((ROOT / "configs/brace/b3_physical_v1.json").read_text())


def test_frozen_b3_configuration_and_resource_boundary():
    config = load_config()
    validate_config(config)
    assert planned_query_count(config) == 388
    assert config["resource_caps"]["model_query_hard_cap"] == 420
    assert config["resource_caps"]["gpu_count"] == 1
    assert config["resource_caps"]["simulator_outcomes"] == 0
    assert config["authorization"]["b4_authorized"] is False
    changed = json.loads(json.dumps(config))
    changed["resource_caps"]["model_query_hard_cap"] = 481
    with pytest.raises(B3ProtocolError, match="semantic hash"):
        validate_config(changed)


def test_v02_recovery_overlay_changes_only_identity_and_guard_record():
    base = load_config()
    recovered = load_config_file(ROOT, Path("configs/brace/b3_physical_v2_recovery.json"))
    assert recovered["run_id"] == "brace-b3-physical-v02"
    assert recovered["recovery"]["prior_run_id"] == base["run_id"]
    for key in base:
        if key not in {"run_id", "semantic_sha256"}:
            assert recovered[key] == base[key]
    raw = b"?? tmp/a file.pdf\0?? results/brace-b3-physical-v02/launch.json\0"
    assert allowed_project_status(raw, recovered["run_id"])
    assert not allowed_project_status(raw + b"?? manuscript/paper.tex\0", recovered["run_id"])


def test_v03_recovery_overlay_changes_only_identity_and_normalization_record():
    base = load_config()
    recovered = load_config_file(ROOT, Path("configs/brace/b3_physical_v3_recovery.json"))
    assert recovered["run_id"] == "brace-b3-physical-v03"
    assert recovered["recovery"] == {
        "attempt": 3,
        "prior_run_id": "brace-b3-physical-v02",
        "correction": "evaluator_resolved_unnorm_key_for_proprio_fixture_only",
    }
    for key in base:
        if key not in {"run_id", "semantic_sha256"}:
            assert recovered[key] == base[key]


def test_profile_grid_is_nested_asymmetric_and_query_accounted():
    config = load_config()
    profiles = {profile["profile_id"]: profile for profile in config["profiles"]}
    assert set(profiles) == {"P1-S25", "P1-S50", "P2-D25", "P2-D50"}
    assert all(not any(profile["wrist_budgets"]) for profile in profiles.values() if profile["family"] == "P1")
    assert all(
        max(profile["wrist_budgets"]) < max(profile["scene_budgets"])
        for profile in profiles.values()
        if profile["family"] == "P2"
    )
    schedule = cycle_schedule(config)
    assert len(schedule) == 4 * 3 * 6
    assert schedule == cycle_schedule(config)
    assert len(set(schedule)) == len(schedule)


def test_query_ledger_cannot_borrow_or_exceed_method_allocation():
    ledger = QueryLedger(5, {"p0": 2, "p1": 3})
    ledger.consume("p0", 2)
    with pytest.raises(B3ProtocolError, match="allocation"):
        ledger.consume("p0")
    ledger.consume("p1", 3)
    assert ledger.record() == {
        "used": {"p0": 2, "p1": 3},
        "total": 5,
        "hard_cap": 5,
    }


def test_quantiles_and_conjunctive_speed_gate_known_answers():
    assert empirical_quantile([1, 2, 3, 4], 0.5) == 2.5
    assert summarize_timings([1, 2, 3, 4])["p99"] == pytest.approx(3.97)
    accepted = profile_speed_gate(
        p0_accelerated_ms=[100] * 6,
        accelerated_ms=[85] * 6,
        p0_cycle_ms=[200] * 6,
        contract_cycle_ms=[180] * 6,
        minimum_accelerated=0.10,
        minimum_cycle=0.08,
    )
    assert accepted["passed"] is True
    rejected = profile_speed_gate(
        p0_accelerated_ms=[100] * 6,
        accelerated_ms=[89] * 6,
        p0_cycle_ms=[200] * 6,
        contract_cycle_ms=[190] * 6,
        minimum_accelerated=0.10,
        minimum_cycle=0.08,
    )
    assert rejected["passed"] is False


def test_semantic_hash_rejects_any_profile_or_exclusion_change():
    config = load_config()
    assert semantic_sha256(config) == config["semantic_sha256"]
    changed = json.loads(json.dumps(config))
    changed["profiles"][0]["scene_budgets"][0] += 1
    assert semantic_sha256(changed) != config["semantic_sha256"]
    changed = json.loads(json.dumps(config))
    changed["comparators"]["gated_vla_cache"] = "executed"
    assert semantic_sha256(changed) != config["semantic_sha256"]


def test_deterministic_real_model_inputs_and_action_parity_helpers():
    import numpy as np

    inputs = deterministic_inputs(np)
    assert set(inputs) == {"input-a", "input-b", "input-c"}
    assert all(scene.shape == wrist.shape == (256, 256, 3) for scene, wrist in inputs.values())
    assert np.array_equal(inputs["input-a"][0][14:], inputs["input-c"][0][14:])
    actions = np.zeros((8, 7), dtype=np.float32)
    record = action_record(actions, np)
    assert record["shape"] == [8, 7] and len(record["sha256"]) == 64
    assert compare_actions(
        actions,
        actions.copy(),
        np=np,
        rtol=1e-5,
        atol=1e-6,
        exact_gripper=True,
    )["passed"]

    class FakeModel:
        norm_stats = {
            "libero_object_no_noops": {
                "proprio": {"q01": [-1.0] * 8, "q99": [1.0] * 8}
            }
        }

    class FakeConfig:
        unnorm_key = "libero_object_no_noops"

    assert np.array_equal(midpoint_proprio_state(FakeModel(), FakeConfig(), np), np.zeros(8))


def test_source_tracker_enforces_mixed_source_ages_and_nested_updates():
    config = load_config()
    profile = next(item for item in config["profiles"] if item["profile_id"] == "P2-D25")
    tracker = SourceTracker(anchor_query=0)
    ordered = tuple(range(1, 97))
    tracker.advance(
        1,
        ordered_positions=ordered,
        profile=profile,
        pruning_layers=(2, 6, 9, 11),
    )
    assert tracker.sources[0][0] == 1
    assert tracker.sources[2][0] == 0
    assert tracker.sources[2][30] == 1
    assert len(tracker.digest()) == 64
    with pytest.raises(B3ProtocolError, match="order"):
        tracker.advance(
            3,
            ordered_positions=ordered,
            profile=profile,
            pruning_layers=(2, 6, 9, 11),
        )
