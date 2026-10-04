from __future__ import annotations

import numpy as np
import pytest

from savr.pair.features import RouterFeatures, build_group_feature, fixed_instruction_projection
from savr.pair.types import AtomicGroup, Camera


@pytest.fixture
def tensor_bank_factory():
    def build(query: int, width: int = 3):
        result = {}
        for layer in (2, 6, 9, 11):
            base = np.arange(2 * 256 * width, dtype=np.float64).reshape(2, 256, width)
            result[layer] = base + query * 10_000 + layer * 100
        return result

    return build


@pytest.fixture
def router_features_factory():
    def build(*, source_age: int = 1, profile_id: str = "D37_BAL", value: float = 0.1):
        group = build_group_feature(
            group=AtomicGroup(Camera.PRIMARY, 0, 2),
            profile_id=profile_id,
            horizon=2,
            source_age=source_age,
            current_raw_tile=np.full((4, 4, 3), value),
            source_raw_tile=np.zeros((4, 4, 3)),
            current_projected_tile=np.full((16, 4), value),
            source_projected_tile=np.zeros((16, 4)),
            current_proprio=np.full(8, value),
            source_proprio=np.zeros(8),
            current_previous_action=np.full(7, value),
            source_previous_action=np.zeros(7),
            previous_salience=value,
            gripper_transition=False,
            remaining=1,
            source_mixture_count=1,
        )
        return RouterFeatures((group,), fixed_instruction_projection(np.full(16, value)))

    return build
