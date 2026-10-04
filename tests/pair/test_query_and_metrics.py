from __future__ import annotations

import numpy as np
import pytest

from savr.pair.metrics import normalized_l1_regret
from savr.pair.query_alignment import (
    AlignedQuery,
    TrajectorySpec,
    assert_split_inheritance,
    build_query_index,
)
from savr.pair.types import PairValidationError, QueryKey, Split


def test_query_index_uses_exact_eight_action_chunks_and_excludes_terminal_padding():
    rows = build_query_index([TrajectorySpec("libero_object", "task-0", "demo-0", 19, Split.TRAIN)])
    assert [row.key.step for row in rows] == [0, 8]
    assert rows[0].action_indices == tuple(range(8))
    assert rows[0].next_step == 8 and rows[1].next_step is None
    assert max(rows[-1].action_indices) == 15


def test_query_index_rejects_filtered_spacing_duplicate_identity_and_cross_split_rows():
    with pytest.raises(PairValidationError, match="eight actions"):
        AlignedQuery(
            QueryKey("suite", "task", "demo", 0, Split.TRAIN), tuple(range(8)), 7
        ).validate()
    with pytest.raises(PairValidationError, match="duplicate"):
        build_query_index(
            [
                TrajectorySpec("suite", "task", "demo", 8, Split.TRAIN),
                TrajectorySpec("suite", "task", "demo", 8, Split.TRAIN),
            ]
        )
    left = AlignedQuery(QueryKey("suite", "task", "demo", 0, Split.TRAIN), tuple(range(8)), None)
    right = AlignedQuery(
        QueryKey("suite", "task", "demo", 8, Split.CALIBRATION), tuple(range(8, 16)), None
    )
    with pytest.raises(PairValidationError, match="crosses"):
        assert_split_inheritance((left, right))


def test_normalized_l1_regret_excludes_padding_and_has_exact_known_answer():
    expert = np.zeros((8, 7))
    dense = np.zeros_like(expert)
    cached = np.zeros_like(expert)
    cached[:3] = 0.2
    cached[3:] = 99
    valid = np.array([True, True, True, False, False, False, False, False])
    result = normalized_l1_regret(expert, dense, cached, valid)
    assert result.valid_action_count == 21
    assert result.dense_expert_l1 == 0
    assert result.cached_expert_l1 == pytest.approx(0.2)
    assert result.signed_regret == pytest.approx(0.2)
    assert result.positive_regret == pytest.approx(0.2)
    assert result.action_max_abs_distortion == pytest.approx(0.2)


def test_normalized_l1_regret_fails_closed_on_shape_empty_mask_and_nonfinite():
    values = np.zeros((8, 7))
    with pytest.raises(PairValidationError, match="validity"):
        normalized_l1_regret(values, values, values, np.zeros(8, dtype=bool))
    with pytest.raises(PairValidationError, match="8x7"):
        normalized_l1_regret(values[:7], values[:7], values[:7], np.ones(7, dtype=bool))
    bad = values.copy()
    bad[0, 0] = np.nan
    with pytest.raises(PairValidationError, match="nonfinite"):
        normalized_l1_regret(bad, values, values, np.ones(8, dtype=bool))
