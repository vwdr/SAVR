from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("run_acr_v3_c", ROOT / "scripts/run_acr_v3_c.py")
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def test_v3_c_exact_frozen_schedule_and_caps():
    assert RUNNER.RUN_ID == "acr-v3c-correctness-latency-v01"
    assert RUNNER.QUERY_CAP == 64
    assert RUNNER.CORRECTNESS_QUERIES == 8
    assert RUNNER.WARMUP_QUERIES == 8
    assert RUNNER.TIMED_QUERIES == 48
    assert len(RUNNER.expected_query_labels()) == 64
    assert len(set(RUNNER.expected_query_labels())) == 64
    assert RUNNER.WALL_CAP_SECONDS == 3600
    assert RUNNER.ARTIFACT_CAP_BYTES == 512 * 1024**2
    assert RUNNER.REUSE_WEIGHT == 0.26055045871559634


def test_v3_c_configuration_semantic_hash_and_schedule_validate():
    config = json.loads((ROOT / RUNNER.CONFIG_RELATIVE).read_text())
    RUNNER.validate_gate_config(config)
    changed = json.loads(json.dumps(config))
    changed["timing"]["reuse_weight"] = 0.3
    with pytest.raises(RuntimeError, match="semantic hash"):
        RUNNER.validate_gate_config(changed)


def test_v3_c_query_budget_fails_before_query_65_and_rejects_duplicates():
    budget = RUNNER.QueryBudget()
    for index, label in enumerate(RUNNER.expected_query_labels()):
        assert budget.consume(label) == index
    with pytest.raises(RuntimeError, match="cap exceeded"):
        budget.consume("query-64")
    duplicate = RUNNER.QueryBudget()
    duplicate.consume("same")
    with pytest.raises(RuntimeError, match="unique"):
        duplicate.consume("same")


def test_v3_c_counterbalance_has_twelve_of_each_path_and_position_balance():
    observed = []
    positions = {path: [] for path in RUNNER.PATHS}
    for repetition in range(12):
        order = RUNNER.COUNTERBALANCE[repetition % 4]
        observed.extend(order)
        for position, path in enumerate(order):
            positions[path].append(position)
    assert {path: observed.count(path) for path in RUNNER.PATHS} == {
        path: 12 for path in RUNNER.PATHS
    }
    expected_positions = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
    assert all(sorted(values) == expected_positions for values in positions.values())


def test_v3_c_deterministic_inputs_are_distinct_contiguous_and_stable():
    first = RUNNER.deterministic_inputs(np, size=16)
    second = RUNNER.deterministic_inputs(np, size=16)
    assert set(first) == {"input-a", "input-b"}
    for label in first:
        for index in range(2):
            assert first[label][index].shape == (16, 16, 3)
            assert first[label][index].dtype == np.uint8
            assert first[label][index].flags.c_contiguous
            assert np.array_equal(first[label][index], second[label][index])
    assert not np.array_equal(first["input-a"][0], first["input-b"][0])
    assert not np.array_equal(first["input-a"][1], first["input-b"][1])


def timing_records(
    *,
    sequential_wall=100.0,
    bfr_wall=97.0,
    refresh_wall=98.0,
    reuse_wall=80.0,
    sequential_visual=100.0,
    refresh_visual=80.0,
    reuse_visual=40.0,
):
    wall = {
        "sequential-fr": sequential_wall,
        "batched-fr": bfr_wall,
        "v3-refresh": refresh_wall,
        "v3-reuse": reuse_wall,
    }
    visual = {
        "sequential-fr": sequential_visual,
        "batched-fr": 80.0,
        "v3-refresh": refresh_visual,
        "v3-reuse": reuse_visual,
    }
    return [
        {
            "kind": "timed",
            "path": path,
            "timing": {"wall_ms": wall[path], "visual_cuda_ms": visual[path]},
        }
        for path in RUNNER.PATHS
        for _ in range(12)
    ]


def test_v3_c_timing_gate_passes_only_when_all_six_conditions_pass():
    summary = RUNNER.summarize_timing(timing_records())
    gates = summary["gates"]
    assert gates["bfr_sequential_wall_ratio"] == pytest.approx(0.97)
    assert gates["v3_refresh_bfr_wall_ratio"] == pytest.approx(98 / 97)
    assert gates["v3_reuse_bfr_wall_ratio"] == pytest.approx(80 / 97)
    assert gates["v3_weighted_visual_cuda_reduction"] > 0.10
    assert gates["all_pass"] is True

    assert (
        RUNNER.summarize_timing(timing_records(bfr_wall=98.0001))["gates"][
            "bfr_sequential_pass"
        ]
        is False
    )
    assert (
        RUNNER.summarize_timing(timing_records(refresh_wall=98.9401))["gates"][
            "v3_refresh_bfr_pass"
        ]
        is False
    )
    assert (
        RUNNER.summarize_timing(timing_records(reuse_wall=95.0601))["gates"][
            "v3_reuse_bfr_pass"
        ]
        is False
    )
    weighted_failure = RUNNER.summarize_timing(
        timing_records(refresh_wall=98.94, reuse_wall=95.06)
    )
    assert weighted_failure["gates"]["v3_weighted_bfr_pass"] is False
    sequential_failure = RUNNER.summarize_timing(
        timing_records(bfr_wall=98.0, refresh_wall=99.96, reuse_wall=96.04)
    )
    assert sequential_failure["gates"]["v3_weighted_sequential_pass"] is False
    visual_failure = RUNNER.summarize_timing(
        timing_records(refresh_visual=95.0, reuse_visual=90.0)
    )
    assert visual_failure["gates"]["v3_weighted_visual_pass"] is False


def test_v3_c_summary_rejects_missing_timed_query():
    records = timing_records()
    records.pop()
    with pytest.raises(RuntimeError, match="twelve timed records"):
        RUNNER.summarize_timing(records)


def test_v3_c_exact_array_parity_rejects_one_ulp():
    baseline = np.array([[1.0, 2.0]], dtype=np.float32)
    proof = RUNNER.exact_array(baseline, baseline.copy(), np, "test")
    assert proof["equal"] is True
    changed = baseline.copy()
    changed[0, 0] = np.nextafter(changed[0, 0], np.float32(2.0))
    with pytest.raises(RuntimeError, match="exact parity failed"):
        RUNNER.exact_array(baseline, changed, np, "test")
