from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/acr/v5_isolated_reuse_freeze.json"


def freeze() -> dict[str, object]:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_v5_freezes_explicit_isolated_reuse_semantics() -> None:
    controller = freeze()["controller"]
    assert controller["controller_version"] == "acr-isolated-controller-v1"
    assert controller["required_horizon"] == 1
    assert controller["maximum_consecutive_reuses"] == 1
    assert controller["minimum_completed_refreshes_between_reuses"] == 1
    assert controller["post_reuse_latch"] is True
    assert controller["cache_age_consistency_check"] is True
    assert controller["legacy_controller_unchanged"] is True


def test_v5_is_cpu_only_and_protects_all_unopened_populations() -> None:
    config = freeze()
    assert config["resource_caps"] == {
        "gpu_count": 0,
        "model_queries": 0,
        "simulator_episodes": 0,
        "downloads": 0,
        "new_outcomes": 0,
        "artifact_bytes": 268435456,
    }
    assert config["protected"]["goal_states_0_9_seed_0"] == "UNOPENED"
    assert config["protected"]["all_suite_states_10_49_seeds_7_17_27"] == "UNOPENED"
    assert config["protected"]["manuscript_changes"] is False


def test_v5_does_not_select_thresholds_or_authorize_experiments() -> None:
    future = freeze()["future_work"]
    assert set(future.values()) == {False}
    preserved = freeze()["preserved"]
    assert preserved["warmup_queries"] == [0, 1]
    assert all(value is True for key, value in preserved.items() if key != "warmup_queries")
