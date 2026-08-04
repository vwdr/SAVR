from __future__ import annotations

import hashlib

from scripts.analyze_acr_v2_diagnosis import (
    canonical_bytes,
    first_action_divergence,
    first_reuse_summary,
    semantic_record,
)


def query(index: int, *, refresh: bool, action: str, reversal: bool = False) -> dict:
    return {
        "query_index": index,
        "decision": {
            "scene_refresh": refresh,
            "scene_score": 0.2,
            "scene_threshold": 0.4,
            "translation_score": 0.3,
            "translation_threshold": 0.6,
        },
        "inputs": {"action_sha256": action, "direction_reversal": reversal},
    }


def test_first_reuse_summary_preserves_diagnostic_signals() -> None:
    rows = [
        query(0, refresh=True, action="a"),
        query(1, refresh=True, action="b"),
        query(2, refresh=False, action="c", reversal=True),
        query(3, refresh=False, action="d"),
    ]
    assert first_reuse_summary(rows) == {
        "query_index": 2,
        "direction_reversal": True,
        "scene_threshold_ratio": 0.5,
        "translation_threshold_ratio": 0.5,
        "reuse_query_indices": [2, 3],
        "direction_reversal_reuses": 1,
    }


def test_first_action_divergence_localizes_first_reuse() -> None:
    candidate = [
        query(0, refresh=True, action="a"),
        query(1, refresh=True, action="b"),
        query(2, refresh=False, action="changed"),
    ]
    full_refresh = [
        query(0, refresh=True, action="a"),
        query(1, refresh=True, action="b"),
        query(2, refresh=True, action="original"),
    ]
    assert first_action_divergence(candidate, full_refresh) == {
        "comparable_queries": 3,
        "first_reuse_query_index": 2,
        "first_action_mismatch_query_index": 2,
        "pre_reuse_action_hashes_match": True,
        "first_reuse_action_hash_differs": True,
    }


def test_semantic_hash_excludes_itself_and_is_deterministic() -> None:
    payload = {"b": 2, "a": [1, 3]}
    expected = hashlib.sha256(canonical_bytes(payload)).hexdigest()
    assert semantic_record(payload) == {**payload, "semantic_sha256": expected}
    assert semantic_record(payload) == semantic_record({"a": [1, 3], "b": 2})
