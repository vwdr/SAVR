from __future__ import annotations

import json
from pathlib import Path

import pytest

from savr.acr.candidates import (
    FRTraceQuery,
    derive_candidates,
    derive_candidates_bytes,
    select_replay_option,
)
from savr.acr.records import (
    AttemptIdentity,
    ImmutableRecordStore,
    decode_float_sequence,
    encode_float_sequence,
    reconcile_episode_counts,
    reconcile_run,
    validate_record,
)
from savr.acr.statistics import (
    PairedCounts,
    PairedObservation,
    exact_mcnemar_pvalue,
    holm_adjust,
    newcombe_paired_interval,
    paired_counts,
    planned_sample_size,
    stratified_paired_bootstrap,
    wilson_interval,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def action(gripper=0.0):
    return tuple(item for _ in range(8) for item in (0, 0, 0, 0, 0, 0, gripper))


def trace():
    records = []
    for episode in ("episode-a", "episode-b"):
        for query in range(5):
            scene = [0.0] * (32 * 32)
            scene[query] = query / 10
            records.append(
                FRTraceQuery(
                    episode_id=episode,
                    query_index=query,
                    scene_representation=tuple(scene),
                    normalized_eef_position=(query / 20, 0, 0),
                    action_chunk=action(),
                )
            )
    return tuple(records)


def test_attempt_identity_and_monotonic_recovery(tmp_path):
    identity = AttemptIdentity(
        "acr-a4-object-fr-v01", "upstream-fr", "libero-object", 2, 3, 0, 0
    )
    assert identity.value.endswith("task-02/state-03/seed-0/attempt-0000")
    assert identity.query_id(7).endswith("query-000007")
    assert identity.episode_id.endswith("episode")
    store = ImmutableRecordStore(tmp_path)
    pairing = identity.value.rsplit("/attempt-", 1)[0]
    assert store.next_attempt_index(pairing) == 0
    store.write_once(identity.value, {"status": "interrupted"})
    assert store.next_attempt_index(pairing) == 1
    with pytest.raises(FileExistsError):
        store.write_once(identity.value, {"status": "completed"})
    assert json.loads((tmp_path / identity.value / "record.json").read_text())["status"] == "interrupted"


def test_compact_float_encoding_is_exact_bounded_and_tamper_evident():
    values = (0.0, -1.25, 3.141592653589793, 1e100)
    encoded = encode_float_sequence(values).as_record()
    assert decode_float_sequence(encoded) == values
    encoded["raw_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash mismatch"):
        decode_float_sequence(encoded)


def valid_query_record():
    sha = "0" * 64
    revision = "0" * 40
    attempt = "acr-a3-correctness-v01/factorized-fr/libero-object/task-00/state-00/seed-0/attempt-0000"
    return {
        "schema_version": "acr.query.v1",
        "run_id": "acr-a3-correctness-v01",
        "attempt_id": attempt,
        "query_id": f"{attempt}/query-000000",
        "phase": "A3",
        "policy": "factorized-fr",
        "suite": "libero_object",
        "task_id": 0,
        "initial_state_id": 0,
        "seed": 0,
        "query_index": 0,
        "environment_step": 0,
        "status": "completed",
        "error": None,
        "decision": {
            "scene_refresh": True,
            "refresh_reasons": ["policy"],
            "cache_age_before": None,
            "cache_age_after": 0,
            "reference_query_index": 0,
            "scene_score": None,
            "scene_threshold": None,
            "translation_score": None,
            "translation_threshold": None,
            "gripper_transition_veto": None,
            "horizon": None,
            "reuse_count_before": 0,
            "query_count_before": 0,
            "hard_reuse_cap": None,
        },
        "inputs": {
            "scene_image_sha256": sha,
            "wrist_image_sha256": sha,
            "proprio_sha256": sha,
            "action_sha256": sha,
            "context_sha256": sha,
        },
        "camera_work": {
            "scene_siglip_calls": 1,
            "scene_dinov2_calls": 1,
            "scene_projector_calls": 1,
            "wrist_siglip_calls": 1,
            "wrist_dinov2_calls": 1,
            "wrist_projector_calls": 1,
            "visual_token_count": 512,
            "token_order": "scene-wrist",
            "dtype": "bfloat16",
            "device": "cuda:0",
            "downstream_calls": 1,
        },
        "timing": {
            "inclusive": True,
            "controller_wall_ms": 0.1,
            "cache_concat_wall_ms": 0.1,
            "scene_visual_cuda_ms": 1.0,
            "wrist_visual_cuda_ms": 1.0,
            "total_visual_cuda_ms": 2.0,
            "downstream_cuda_ms": 1.0,
            "query_cuda_ms": 3.0,
            "query_wall_ms": 4.0,
        },
        "provenance": {
            "configuration_sha256": sha,
            "savr_revision": revision,
            "openvla_oft_revision": revision,
            "checkpoint_revision": revision,
            "recorded_at_utc": "2026-08-02T00:00:00Z",
        },
    }


def test_frozen_query_schema_accepts_valid_and_rejects_mutation():
    schema = json.loads((REPOSITORY_ROOT / "schemas/acr_query.schema.json").read_text())
    record = valid_query_record()
    validate_record(record, schema)
    record["unexpected"] = True
    with pytest.raises(ValueError, match="violates schema"):
        validate_record(record, schema)


def valid_counts():
    return {
        "queries": 3,
        "scene_refreshes": 2,
        "scene_reuses": 1,
        "wrist_refreshes": 3,
        "scene_siglip_calls": 2,
        "scene_dinov2_calls": 2,
        "scene_projector_calls": 2,
        "wrist_siglip_calls": 3,
        "wrist_dinov2_calls": 3,
        "wrist_projector_calls": 3,
        "downstream_calls": 3,
    }


def test_record_reconciliation_success_and_failures():
    reconcile_episode_counts(valid_counts())
    reconcile_run(scheduled_attempts=3, terminal_episodes=2, failures=1)
    for key in valid_counts():
        broken = valid_counts()
        broken[key] += 1
        with pytest.raises(ValueError):
            reconcile_episode_counts(broken)
    with pytest.raises(ValueError):
        reconcile_run(scheduled_attempts=3, terminal_episodes=1, failures=1)


def test_candidate_derivation_is_byte_identical_and_exactly_three():
    first = derive_candidates_bytes(trace())
    second = derive_candidates_bytes(trace())
    assert first == second
    payload = derive_candidates(trace())
    assert [candidate["configuration_id"] for candidate in payload["candidates"]] == [
        "acr-t25-h2-b30",
        "acr-t50-h4-b55",
        "acr-t70-h8-b75",
    ]
    assert all(candidate["trace_sha256"] == payload["trace_sha256"] for candidate in payload["candidates"])


def test_candidate_ties_choose_lower_reuse_then_lower_quantile():
    options = (
        (0.30, 0.500, 1.0, 1.0),
        (0.20, 0.900, 2.0, 2.0),
        (0.20, 0.600, 3.0, 3.0),
    )
    assert select_replay_option(options, 0.25) == options[2]


def test_newcombe_matches_published_method_10_example():
    interval = newcombe_paired_interval(PairedCounts(36, 12, 2, 0))
    assert interval.lower == pytest.approx(0.0569, abs=0.0001)
    assert interval.upper == pytest.approx(0.3404, abs=0.0001)


def test_statistics_boundaries_and_determinism():
    counts = paired_counts([True, True, False, False], [True, False, True, False])
    assert counts == PairedCounts(1, 1, 1, 1)
    assert counts.risk_difference == 0
    assert exact_mcnemar_pvalue(counts) == 1
    assert wilson_interval(0, 10).lower == 0
    observations = [
        PairedObservation("suite/task-a", 1, 0),
        PairedObservation("suite/task-a", 0, 0),
        PairedObservation("suite/task-b", 0, 1),
        PairedObservation("suite/task-b", 1, 1),
    ]
    first = stratified_paired_bootstrap(observations, resamples=500, seed=7)
    second = stratified_paired_bootstrap(observations, resamples=500, seed=7)
    assert first == second
    assert holm_adjust([0.01, 0.04, 0.03]) == pytest.approx((0.03, 0.06, 0.06))


def test_power_gate_rounds_to_strata_and_stops_above_cap():
    planned = planned_sample_size(0, 200)
    assert planned is not None and planned % 40 == 0 and planned <= 1600
    assert planned_sample_size(200, 200) is None
