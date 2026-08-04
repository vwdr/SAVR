from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.analyze_acr_v3_feasibility import canonical_bytes, derive, semantic_record


ROOT = Path(__file__).resolve().parents[2]


def load_v2() -> dict:
    return json.loads(
        (ROOT / "reports/runtime/acr_v2_c_recovery.json").read_text(encoding="utf-8")
    )


def test_scene_skip_only_cannot_reach_the_frozen_two_percent_gate() -> None:
    result = derive(
        load_v2(),
        source_sha256="9e4d8e0034c4410dbc35f4e4a2b987eda4a0645ab603c3479da05a97bd6f1ae6",
    )
    ceiling = result["scene_skip_only_ceiling"]
    assert ceiling["weighted_wall_ratio"] == pytest.approx(0.9837978208612804)
    assert ceiling["maximum_weighted_wall_reduction"] == pytest.approx(
        0.016202179138719575
    )
    assert ceiling["target_reachable"] is False


def test_v3_requires_a_refresh_acceleration_and_batched_fr_ablation() -> None:
    result = derive(
        load_v2(),
        source_sha256="9e4d8e0034c4410dbc35f4e4a2b987eda4a0645ab603c3479da05a97bd6f1ae6",
    )
    requirement = result["redesign_requirement"]
    assert requirement["must_accelerate_refresh_queries"] is True
    assert requirement["maximum_refresh_wall_ratio_if_reuse_is_ideal"] == pytest.approx(
        0.9948639891578217
    )
    assert requirement["required_ablation"] == "batched_full_refresh"


def test_semantic_record_is_canonical_and_excludes_its_hash() -> None:
    payload = {"z": [2, 1], "a": {"b": True}}
    expected = hashlib.sha256(canonical_bytes(payload)).hexdigest()
    assert semantic_record(payload) == {**payload, "semantic_sha256": expected}
    assert semantic_record(payload) == semantic_record({"a": {"b": True}, "z": [2, 1]})


def test_derive_rejects_changed_v2_semantic_identity() -> None:
    changed = load_v2()
    changed["result_semantic_sha256"] = "changed"
    with pytest.raises(RuntimeError, match="semantic identity"):
        derive(changed, source_sha256="unused")


def test_tracked_feasibility_record_is_reproducible() -> None:
    tracked = json.loads(
        (ROOT / "reports/runtime/acr_v3_feasibility.json").read_text(encoding="utf-8")
    )
    derived = derive(
        load_v2(),
        source_sha256="9e4d8e0034c4410dbc35f4e4a2b987eda4a0645ab603c3479da05a97bd6f1ae6",
    )
    assert derived == tracked


def test_v3_freeze_semantic_hash_excludes_itself() -> None:
    freeze = json.loads((ROOT / "configs/acr/v3_freeze.json").read_text(encoding="utf-8"))
    observed = freeze.pop("semantic_sha256")
    assert hashlib.sha256(canonical_bytes(freeze)).hexdigest() == observed
    gate = freeze["correctness_latency_gate"]
    assert gate["maximum_model_queries"] == 64
    assert gate["timing_paths"] == [
        "upstream-sequential-fr",
        "batched-fr",
        "v3-refresh",
        "v3-reuse",
    ]
