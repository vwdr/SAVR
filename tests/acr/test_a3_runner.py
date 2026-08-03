from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "run_acr_correctness", ROOT / "scripts/run_acr_correctness.py"
)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def test_a3_frozen_limits_and_identity():
    assert RUNNER.RUN_ID == "acr-a3-correctness-none-v01"
    assert RUNNER.PLANNED_QUERIES == 12
    assert RUNNER.PLANNED_QUERIES <= RUNNER.QUERY_CAP == 16
    assert RUNNER.ARTIFACT_CAP_BYTES == 512 * 1024**2
    assert RUNNER.WALL_CAP_SECONDS == 3600


def test_query_budget_fails_before_seventeenth_query():
    budget = RUNNER.QueryBudget(16)
    for index in range(16):
        assert budget.consume(f"query-{index}") == index
    with pytest.raises(RuntimeError, match="cap exceeded"):
        budget.consume("query-16")
    assert len(budget.attempts) == 16


def test_synthetic_inputs_are_deterministic_and_camera_isolated():
    first = RUNNER.synthetic_images(np, size=32)
    second = RUNNER.synthetic_images(np, size=32)
    assert all(np.array_equal(left, right) for left, right in zip(first, second))
    scene, wrist, scene_variant, wrist_variant = first
    assert scene.shape == wrist.shape == (32, 32, 3)
    assert not np.array_equal(scene, scene_variant)
    assert not np.array_equal(wrist, wrist_variant)
    assert np.array_equal(scene, first[0]) and np.array_equal(wrist, first[1])


def test_exact_array_proof_is_bitwise_and_rejects_tolerance():
    baseline = np.array([[1.0, 2.0]], dtype=np.float32)
    proof = RUNNER.exact_array_proof(baseline, baseline.copy(), np, "test")
    assert proof["equal"] is True
    changed = baseline.copy()
    changed[0, 0] = np.nextafter(changed[0, 0], np.float32(2.0))
    with pytest.raises(RuntimeError, match="exact parity failed"):
        RUNNER.exact_array_proof(baseline, changed, np, "test")
