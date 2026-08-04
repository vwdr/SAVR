from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("run_acr_v2_c", ROOT / "scripts/run_acr_v2_c.py")
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)

RECOVERY_SPEC = importlib.util.spec_from_file_location(
    "run_acr_v2_c_recovery", ROOT / "scripts/run_acr_v2_c_recovery.py"
)
assert RECOVERY_SPEC is not None and RECOVERY_SPEC.loader is not None
RECOVERY = importlib.util.module_from_spec(RECOVERY_SPEC)
RECOVERY_SPEC.loader.exec_module(RECOVERY)


def test_v2_c_exact_frozen_schedule_and_caps():
    assert RUNNER.RUN_ID == "acr-v2c-correctness-latency-v01"
    assert RUNNER.QUERY_CAP == 48
    assert RUNNER.CORRECTNESS_QUERIES == 6
    assert RUNNER.WARMUP_QUERIES == 6
    assert RUNNER.TIMED_QUERIES == 36
    assert RUNNER.CORRECTNESS_QUERIES + RUNNER.WARMUP_QUERIES + RUNNER.TIMED_QUERIES == 48
    assert RUNNER.WALL_CAP_SECONDS == 3600
    assert RUNNER.ARTIFACT_CAP_BYTES == 512 * 1024**2
    assert RUNNER.REUSE_WEIGHT == 0.26055045871559634


def test_v2_c_configuration_semantic_hash_and_schedule_validate():
    config = json.loads((ROOT / RUNNER.CONFIG_RELATIVE).read_text())
    RUNNER.validate_gate_config(config)
    changed = json.loads(json.dumps(config))
    changed["timing"]["reuse_weight"] = 0.3
    with pytest.raises(RuntimeError, match="semantic hash"):
        RUNNER.validate_gate_config(changed)


def test_v2_c_query_budget_fails_before_query_49():
    budget = RUNNER.QueryBudget()
    for index in range(48):
        assert budget.consume(f"query-{index}") == index
    with pytest.raises(RuntimeError, match="cap exceeded"):
        budget.consume("query-48")
    assert len(budget.labels) == 48


def test_v2_c_recovery_uses_exactly_remaining_41_queries():
    assert RECOVERY.PARENT_QUERIES == 7
    assert RECOVERY.RECOVERY_QUERY_CAP == 41
    assert sum(RECOVERY.REMAINING_WARMUPS.values()) == 5
    assert RECOVERY.TIMED_QUERIES == 36
    assert RECOVERY.PARENT_QUERIES + RECOVERY.RECOVERY_QUERY_CAP == 48
    budget = RECOVERY.QueryBudget()
    for index in range(41):
        assert budget.consume(f"recovery-{index}") == index
    with pytest.raises(RuntimeError, match="cap exceeded"):
        budget.consume("recovery-41")


def test_v2_c_recovery_semantic_hash_and_corrected_counts_validate():
    config = json.loads((ROOT / RECOVERY.RECOVERY_CONFIG).read_text())
    RECOVERY.validate_recovery_config(config)
    assert config["corrected_component_counts"] == {
        "upstream-fr": {"siglip": 2, "dinov2": 2, "projector": 1},
        "dual-path-refresh": {"siglip": 2, "dinov2": 2, "projector": 1},
        "dual-path-reuse": {"siglip": 1, "dinov2": 1, "projector": 1},
    }
    changed = json.loads(json.dumps(config))
    changed["recovery_queries"]["total"] = 40
    with pytest.raises(RuntimeError, match="semantic hash"):
        RECOVERY.validate_recovery_config(changed)


def test_v2_c_counterbalance_has_twelve_of_each_path():
    observed = []
    for repetition in range(12):
        observed.extend(RUNNER.COUNTERBALANCE[repetition % 3])
    assert len(observed) == 36
    assert {path: observed.count(path) for path in RUNNER.PATHS} == {
        path: 12 for path in RUNNER.PATHS
    }


def timing_records(*, fr=100.0, refresh=104.0, reuse=70.0):
    wall = {
        "upstream-fr": fr,
        "dual-path-refresh": refresh,
        "dual-path-reuse": reuse,
    }
    records = []
    for path in RUNNER.PATHS:
        for repetition in range(12):
            records.append(
                {
                    "kind": "timed",
                    "path": path,
                    "timing": {
                        "wall_ms": wall[path],
                        "visual_cuda_ms": wall[path] / 10,
                    },
                }
            )
    return records


def test_v2_c_timing_gate_passes_only_all_three_thresholds():
    passing = RUNNER.summarize_timing(timing_records())
    assert passing["gates"]["refresh_wall_ratio"] == pytest.approx(1.04)
    assert passing["gates"]["reuse_wall_ratio"] == pytest.approx(0.7)
    assert passing["gates"]["weighted_expected_wall_ratio"] < 0.98
    assert passing["gates"]["all_pass"] is True

    refresh_failure = RUNNER.summarize_timing(timing_records(refresh=105.0001))
    assert refresh_failure["gates"]["refresh_pass"] is False
    assert refresh_failure["gates"]["all_pass"] is False
    reuse_failure = RUNNER.summarize_timing(timing_records(reuse=98.0001))
    assert reuse_failure["gates"]["reuse_pass"] is False
    weighted_failure = RUNNER.summarize_timing(timing_records(refresh=100.0, reuse=93.0))
    assert weighted_failure["gates"]["refresh_pass"] is True
    assert weighted_failure["gates"]["reuse_pass"] is True
    assert weighted_failure["gates"]["weighted_pass"] is False


def test_v2_c_summary_rejects_missing_timed_queries():
    records = timing_records()
    records.pop()
    with pytest.raises(RuntimeError, match="twelve timed records"):
        RUNNER.summarize_timing(records)


def test_v2_c_exact_array_parity_rejects_one_ulp():
    baseline = np.array([[1.0, 2.0]], dtype=np.float32)
    proof = RUNNER.exact_array(baseline, baseline.copy(), np, "test")
    assert proof["equal"] is True
    changed = baseline.copy()
    changed[0, 0] = np.nextafter(changed[0, 0], np.float32(2.0))
    with pytest.raises(RuntimeError, match="exact parity failed"):
        RUNNER.exact_array(baseline, changed, np, "test")
