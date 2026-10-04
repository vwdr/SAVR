from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from savr.pair.p4b import (
    cvar_improvement,
    expected_counts,
    repeat_slots,
    served_indices,
    slot_spec,
    spearman,
    validate_config,
)


ROOT = Path(__file__).resolve().parents[2]


def test_p4b_population_horizons_repeats_and_accounting_are_exact():
    specs = [slot_spec(task, slot) for task in range(40) for slot in range(7)]
    assert expected_counts() == {
        "anchors": 280,
        "all_fresh_controls": 40,
        "structured_contracts": 240,
        "structured_horizon1": 80,
        "structured_horizon2": 80,
        "structured_horizon4": 80,
        "exact_repeats": 24,
        "intervention_records": 776,
        "feature_records": 560,
        "contract_records": 304,
        "scheduled_model_calls": 1776,
        "planned_model_calls": 1784,
    }
    assert len(repeat_slots()) == 24
    by_suite_horizon = Counter(
        (spec.task_index // 10, spec.horizon) for spec in specs if spec.repeat
    )
    assert set(by_suite_horizon.values()) == {2}
    assert len(by_suite_horizon) == 12


def test_p4b_config_is_frozen_and_locally_valid_without_remote_indexes():
    config = json.loads((ROOT / "configs/pair/p4b_confirmatory_v1.json").read_text())
    validate_config(config, ROOT, require_large_inputs=False)


def test_p4b_service_is_exactly_balanced_within_suite_horizon_cells():
    suites = np.repeat(["a", "b", "c", "d"], 60)
    horizons = np.tile(np.repeat([1, 2, 4], 20), 4)
    scores = np.arange(240, dtype=float)
    selected = served_indices(scores, suites, horizons)
    assert len(selected) == 168
    cells = Counter((suites[index], horizons[index]) for index in selected)
    assert len(cells) == 12
    assert set(cells.values()) == {14}


def test_p4b_statistics_are_deterministic_and_conservative_when_proxy_tail_is_zero():
    suites = ["suite"] * 20
    horizons = [1] * 20
    observed = np.arange(20, dtype=float)
    scores = np.arange(20, dtype=float)
    value, router, proxy = cvar_improvement(
        observed, scores, scores[::-1], suites, horizons
    )
    assert len(router) == len(proxy) == 14
    assert value > 0
    assert spearman(scores, observed) == 1.0
    degenerate, _, _ = cvar_improvement(
        np.zeros(20), scores, scores, suites, horizons
    )
    assert degenerate == -1.0
