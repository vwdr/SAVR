from __future__ import annotations

from savr.cac.c1h import (
    cumulative_decision,
    frozen_schedule,
    stage1_decision,
    summarize,
    worst_case_query_count,
)


def conditions(population: str = "headroom_stage1") -> list[dict[str, object]]:
    return [
        {
            "population": population,
            "condition_id": f"c{index:03d}",
            "suite": f"suite{index // 30}",
            "task_index": index // 3,
            "task_id": f"task{index // 3}",
            "initial_state_id": index % 3,
            "seed": 7,
        }
        for index in range(120)
    ]


def terminal(dense_successes: int, cache_successes: int) -> list[dict[str, object]]:
    rows = []
    for index, condition in enumerate(conditions()):
        for arm, count in (("dense", dense_successes), ("D62", cache_successes)):
            rows.append(
                {
                    **condition,
                    "arm": arm,
                    "status": "completed",
                    "success": index < count,
                }
            )
    return rows


def test_schedule_is_paired_balanced_and_deterministic() -> None:
    first = frozen_schedule(conditions(), population="headroom_stage1", seed=20260905)
    second = frozen_schedule(conditions(), population="headroom_stage1", seed=20260905)
    assert first == second
    assert len(first) == 240
    assert sum(row["pair_position"] == 0 and row["arm"] == "dense" for row in first) == 60
    assert sum(row["pair_position"] == 0 and row["arm"] == "D62" for row in first) == 60


def test_stage1_gate_proceeds_on_declared_repair_headroom() -> None:
    value = summarize(terminal(96, 72), expected_conditions=120)
    assert value["dense_minus_D62_points"] == 20.0
    assert stage1_decision(value) == "proceed"


def test_stage1_gate_stops_on_collapsed_cache() -> None:
    value = summarize(terminal(96, 48), expected_conditions=120)
    assert stage1_decision(value) == "stop"


def test_stage1_gate_extends_only_the_ambiguity_region() -> None:
    value = summarize(terminal(96, 90), expected_conditions=120)
    assert stage1_decision(value) == "extend"


def test_cumulative_gate_is_mechanical() -> None:
    assert cumulative_decision(summarize(terminal(96, 84), expected_conditions=120)) == "proceed"
    assert cumulative_decision(summarize(terminal(96, 93), expected_conditions=120)) == "stop"


def test_query_cap_covers_both_stages_and_controls_without_slack_error() -> None:
    horizons = {
        "libero_spatial": 220,
        "libero_object": 280,
        "libero_goal": 300,
        "libero_10": 520,
    }
    assert worst_case_query_count(
        horizons, conditions_per_suite_per_stage=30, stages=1
    ) == 9962
    assert worst_case_query_count(
        horizons, conditions_per_suite_per_stage=30, stages=2
    ) == 19922
