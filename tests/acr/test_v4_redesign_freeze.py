from __future__ import annotations

import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
FREEZE_PATH = ROOT / "configs/acr/v4_redesign_freeze.json"
PROTOCOL_PATH = ROOT / "docs/ACR_V4_REDESIGN_PROTOCOL.md"


def _freeze() -> dict[str, object]:
    return json.loads(FREEZE_PATH.read_text(encoding="utf-8"))


def test_v4_freeze_preserves_v3_d_observations_and_break_even() -> None:
    freeze = _freeze()
    observed = freeze["v3_d_observed"]
    assert isinstance(observed, dict)
    assert observed["bfr_successes"] == 67
    assert observed["v3_successes"] == 67
    assert observed["episodes_per_policy"] == 70
    assert observed["v3_scene_reuse_rate"] == pytest.approx(0.2524344569288389)
    assert observed["v3_visual_cuda_reduction_vs_bfr"] == pytest.approx(0.08461770497265919)
    assert observed["v3_query_wall_ratio_vs_bfr"] == pytest.approx(1.0022598778791063)

    observed_slope = observed["v3_visual_cuda_reduction_vs_bfr"] / observed["v3_scene_reuse_rate"]
    estimated_break_even = 0.10 / observed_slope
    assert observed["estimated_reuse_rate_to_touch_ten_percent_visual_gate"] == (
        pytest.approx(estimated_break_even)
    )


def test_v4_promotion_margins_are_stronger_than_v3_d_gates() -> None:
    margins = _freeze()["promotion_margins"]
    assert isinstance(margins, dict)
    assert margins["realized_scene_reuse_min"] >= 0.35
    assert margins["visual_cuda_reduction_vs_bfr_min"] > 0.10
    assert margins["query_wall_ratio_vs_bfr_max"] < 1.00
    assert margins["query_wall_ratio_vs_sequential_fr_max"] <= 0.95
    assert margins["refresh_wall_ratio_vs_bfr_max"] <= 1.01


def test_v4_phase_budgets_and_protected_populations_are_frozen() -> None:
    freeze = _freeze()
    phases = freeze["phases"]
    assert isinstance(phases, dict)
    for phase_name in ("v4_a", "v4_b"):
        assert phases[phase_name]["gpu_count"] == 0
        assert phases[phase_name]["model_queries"] == 0
        assert phases[phase_name]["simulator_episodes"] == 0
    assert phases["v4_c"]["model_queries"] == 96
    assert phases["v4_c"]["simulator_episodes"] == 0
    assert phases["v4_d"]["episode_attempts"] == 200
    assert phases["v4_e"]["episode_attempts"] == 300

    confirmation = freeze["independent_confirmation_population"]
    protected = freeze["protected_populations"]
    assert confirmation["status"] == "protected_unopened"
    assert confirmation["suite"] == "libero_goal"
    assert protected["all_four_suites_state_ids"] == list(range(10, 50))
    assert protected["primary_seed"] == 7
    assert protected["reserve_seeds"] == [17, 27]


def test_v4_protocol_is_planning_only_and_requires_both_changes() -> None:
    freeze = _freeze()
    protocol = PROTOCOL_PATH.read_text(encoding="utf-8")
    assert freeze["status"] == "FROZEN_PLANNING_V4_A_UNAUTHORIZED"
    assert freeze["manuscript_changes_authorized"] is False
    assert freeze["downloads_allowed"] is False
    assert "V4-A NOT YET AUTHORIZED" in protocol
    assert "Controller contribution" in protocol
    assert "Executor contribution" in protocol
    assert "V4 controller with the V3 executor" in protocol
    assert "V3 controller with the V4 executor" in protocol
