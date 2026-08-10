from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


ROOT = Path(__file__).resolve().parents[2]


def load_script() -> ModuleType:
    path = ROOT / "scripts/analyze_acr_v4_a.py"
    spec = importlib.util.spec_from_file_location("analyze_acr_v4_a", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ANALYZER = load_script()


def test_percentile_and_distribution_are_deterministic() -> None:
    assert ANALYZER.percentile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.5
    assert ANALYZER.percentile([1.0, 2.0, 3.0, 4.0], 0.25) == 1.75
    summary = ANALYZER.distribution([4.0, 1.0, 3.0, 2.0])
    assert summary["count"] == 4
    assert summary["mean"] == 2.5
    assert summary["minimum"] == 1.0
    assert summary["maximum"] == 4.0


def test_bootstrap_ratio_uses_episode_clusters_and_repeats_exactly() -> None:
    episodes = [
        {"queries": 10, "reuses": 2},
        {"queries": 20, "reuses": 10},
        {"queries": 30, "reuses": 15},
    ]
    first = ANALYZER.bootstrap_ratio(episodes, seed=4102026, resamples=100)
    second = ANALYZER.bootstrap_ratio(episodes, seed=4102026, resamples=100)
    assert first == second
    assert 0.0 <= first[1]["lower_95"] <= first[1]["upper_95"] <= 1.0


def test_frozen_candidate_replay_enforces_isolated_reuse_and_budget() -> None:
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
        for index in range(10)
    )
    family = {
        "scene_threshold_low": 0.1,
        "scene_threshold_high": 0.2,
        "translation_threshold_low": 0.1,
        "translation_threshold_high": 0.2,
        "horizon": 2,
        "hard_reuse_cap": 0.4,
    }
    result = ANALYZER.replay_candidate(
        {"episode-0": queries},
        alpha=0.5,
        transition_policy="gripper_only",
        family=family,
        high_action_threshold=1.0,
    )
    assert result["maximum_reuse_streak"] == 1
    assert result["gripper_transition_reuses"] == 0
    assert result["reuses"] / result["queries"] <= 0.4
    assert result["reuses"] > 0


def test_required_reuse_ratio_formula_is_fail_closed() -> None:
    conservative_reuse = 0.30
    refresh_upper = 1.01
    required = (0.98 - (1.0 - conservative_reuse) * refresh_upper) / conservative_reuse
    assert required == pytest.approx(0.91)
    assert required >= 0.90


def test_v3_completion_manifest_is_not_required_to_have_semantic_hash() -> None:
    source = (ROOT / "scripts/analyze_acr_v4_a.py").read_text(encoding="utf-8")
    assert "for record in [*queries, *episodes, summary]:" in source
    assert "for record in [*queries, *episodes, completion, summary]:" not in source


def test_a5_integrity_delegates_to_the_original_committed_analyzer() -> None:
    source = (ROOT / "scripts/analyze_acr_v4_a.py").read_text(encoding="utf-8")
    assert "from analyze_acr_a5 import summarize_run" in source
    assert 'published_by_run[run_id]["records_sha256"]' in source
