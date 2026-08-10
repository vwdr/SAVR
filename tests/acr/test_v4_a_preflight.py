from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/acr/v4_a_diagnosis_preflight.json"


def _config() -> dict[str, object]:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_v4_a_is_cpu_only_and_protects_unopened_populations() -> None:
    config = _config()
    caps = config["resource_caps"]
    assert caps == {
        "gpu_count": 0,
        "model_queries": 0,
        "simulator_episodes": 0,
        "downloads": 0,
        "artifact_bytes": 536870912,
    }
    protected = config["protected"]
    assert protected["goal_states_0_9_seed_0"] == "UNOPENED"
    assert protected["all_suite_states_10_49_seeds_7_17_27"] == "UNOPENED"
    assert protected["manuscript_changes"] is False


def test_v4_a_candidate_family_is_small_and_fully_declared() -> None:
    family = _config()["controller_family"]
    assert family["interpolation_fractions"] == [0.5, 0.75, 1.0]
    assert family["transition_policies"] == [
        "gripper_only",
        "gripper_or_translation_direction_reversal",
    ]
    assert family["candidate_count"] == (
        len(family["interpolation_fractions"]) * len(family["transition_policies"])
    )
    assert family["minimum_query_index"] == 2
    assert family["horizon"] == 2
    assert family["hard_reuse_cap"] == 0.4
    assert family["task_or_state_identity_allowed"] is False


def test_v4_a_uncertainty_selection_and_stop_rules_are_frozen() -> None:
    config = _config()
    bootstrap = config["bootstrap"]
    assert bootstrap == {
        "unit": "episode",
        "seed": 4102026,
        "resamples": 10000,
        "confidence": 0.95,
        "outlier_deletion": False,
    }
    gates = config["controller_gates"]
    assert gates["replay_reuse_point_min"] == 0.35
    assert gates["maximum_reuse_streak"] == 1
    assert gates["gripper_transition_reuses_max"] == 0
    assert gates["predicted_visual_cuda_reduction_point_min"] == 0.12
    assert config["executor_decision_tree"][-1] == "stop_before_implementation"
    assert config["complete_method_gates"]["required_reuse_wall_ratio_feasibility_floor"] == 0.9
