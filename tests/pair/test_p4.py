from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from savr.pair.p4 import atomic_group, expected_counts, slot_spec, validate_config
from savr.pair.p4_openvla import arbitrary_group_profile
from savr.pair.types import AtomicGroup, Camera


ROOT = Path(__file__).resolve().parents[2]


def test_p4_frozen_schedule_and_accounting():
    specs = [slot_spec(task, slot) for task in range(40) for slot in range(10)]
    counts = expected_counts()
    assert counts == {
        "anchors": 400,
        "train_anchors": 320,
        "calibration_anchors": 80,
        "controls": 80,
        "atomic": 120,
        "structured": 200,
        "repeats": 32,
        "intervention_records": 2120,
        "contract_summaries": 832,
        "base_noncontrol_feature_rows": 800,
        "scheduled_model_calls": 3520,
        "planned_model_calls": 3528,
    }
    assert sum(spec.repeat and spec.category == "atomic" for spec in specs) == 12
    assert sum(spec.repeat and spec.category == "structured" for spec in specs) == 20


def test_p4_config_and_atomic_groups_are_deterministic():
    config = json.loads((ROOT / "configs/pair/p4_pilot_v1.json").read_text())
    validate_config(config, ROOT, require_large_inputs=False)
    left = atomic_group("libero_object", "task", 3, 20260829)
    right = atomic_group("libero_object", "task", 3, 20260829)
    assert left == right


def test_p4_arbitrary_groups_form_a_nested_suffix_safe_profile():
    class FakeTorch:
        long = np.int64

        @staticmethod
        def as_tensor(values, *, device, dtype):
            del device
            return np.asarray(values, dtype=dtype)

    groups = (
        AtomicGroup(Camera.WRIST, 4, 11),
        AtomicGroup(Camera.PRIMARY, 0, 2),
        AtomicGroup(Camera.PRIMARY, 1, 6),
    )
    positions, proportions, mapping = arbitrary_group_profile(
        groups, FakeTorch, device="cpu"
    )
    assert positions.shape == (48,)
    assert proportions == (1 / 3, 2 / 3, 2 / 3, 1.0)
    assert len(set(positions.tolist())) == 48
    assert mapping[(Camera.WRIST, 4)] == 11
