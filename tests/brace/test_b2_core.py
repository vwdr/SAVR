from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from savr.brace.anchor import gate_identity, semantic_salience, sidecar_attention
from savr.brace.baseline import (
    FrameHistory,
    corrected_evaluator_source,
    correction_manifest,
    execute_p0_or_profile,
    propagate_episode_error,
)
from savr.brace.cache_adapter import (
    clone_dynamic_cache,
    position_preserving_index_update,
    transactional_cache_configuration,
)
from savr.brace.ledger import SourceLedger, SourceRecord, SourceRingBuffer
from savr.brace.patch_change import patch_change_scores
from savr.brace.profiles import (
    build_nested_reuse_sets,
    validate_exact_source_eligibility,
    validate_profile,
    validate_profile_grid,
)
from savr.brace.records import (
    assert_duplicate_arm_identity,
    experimental_execution_schedule,
    freeze_intent_record,
    validate_intent_record,
)
from savr.brace.runtime import ProfileRuntime
from savr.brace.sequence_map import derive_sequence_map
from savr.brace.types import (
    B2ValidationError,
    CacheIdentity,
    Contract,
    ContractMode,
    ContractState,
    LayerBudget,
    Profile,
    ProfileFamily,
)


SHA = "a" * 64
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def runtime_map():
    roles = ["special", "scene", "text", "wrist", "proprio", "scene", "action", "wrist"]
    return derive_sequence_map(roles, patch_ids={1: 0, 5: 1, 3: 0, 7: 1})


def source(query: int, *, proprio=(0.0, 0.0), action=(0.0, 0.0)):
    return SourceRecord.create(
        query=query,
        scene_image_sha256=SHA,
        wrist_image_sha256="b" * 64,
        normalized_proprio=proprio,
        prompt_sha256="c" * 64,
        action_mask_sha256="d" * 64,
        sequence_map_sha256=runtime_map().semantic_sha256,
        previous_action_summary=action,
        rng_sha256="e" * 64,
        configuration_sha256="f" * 64,
        counters={"query": query},
    )


def identity(sequence_sha: str):
    return CacheIdentity(
        model_sha256=SHA,
        checkpoint_sha256="b" * 64,
        sequence_map_sha256=sequence_sha,
        preprocessing_sha256="c" * 64,
        episode_id="episode-1",
        anchor_query=0,
        profile_id="P1-test",
        dtype="float32",
        device="cpu",
    )


def profile(family=ProfileFamily.P2):
    return Profile(
        profile_id=f"{family.value}-test",
        family=family,
        budgets=(LayerBudget(1, 1, 0), LayerBudget(3, 2, 1)),
        scene_change_limit=0.2,
        wrist_change_limit=0.1,
        scene_max_age=4,
        wrist_max_age=2,
    )


def test_runtime_sequence_map_uses_roles_not_fixed_offsets_and_rejects_bad_patch_maps():
    mapping = runtime_map()
    assert mapping.scene_positions == (1, 5)
    assert mapping.wrist_positions == (3, 7)
    assert mapping.action_positions == (6,)
    assert mapping.nonvisual_positions == (0, 2, 4, 6)
    with pytest.raises(B2ValidationError, match="exactly"):
        derive_sequence_map(mapping.roles, patch_ids={1: 0, 5: 1, 3: 0})
    with pytest.raises(B2ValidationError, match="contiguous"):
        derive_sequence_map(mapping.roles, patch_ids={1: 0, 5: 2, 3: 0, 7: 1})


def test_patch_change_known_answers_zero_norm_and_nonfinite_rejection():
    zeros = np.zeros((2, 2, 2))
    ones = np.ones((2, 2, 2))
    equal = patch_change_scores(
        zeros, zeros, lower=0, upper=1, l1_weight=0.5, cosine_weight=0.5
    )
    assert np.array_equal(equal, np.zeros(2))
    one_zero = patch_change_scores(
        zeros, ones, lower=0, upper=1, l1_weight=0.5, cosine_weight=0.5
    )
    assert np.array_equal(one_zero, np.ones(2))
    bad = ones.copy()
    bad[0, 0, 0] = np.nan
    with pytest.raises(B2ValidationError, match="nonfinite"):
        patch_change_scores(bad, ones, lower=0, upper=1, l1_weight=1, cosine_weight=0)


def test_sidecar_attention_matches_manual_softmax_and_runtime_semantic_spans():
    query = np.array([[[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, -0.5]]]])
    key = query.copy()
    probabilities = sidecar_attention(query, key, scale=1.0)
    manual = np.exp(query @ np.swapaxes(key, -1, -2))
    manual /= manual.sum(axis=-1, keepdims=True)
    assert np.allclose(probabilities, manual)
    scores = semantic_salience(
        [probabilities], instruction_positions=(0,), action_positions=(2,), visual_positions=(1, 3)
    )
    assert scores.shape == (2,)
    assert np.all((0 <= scores) & (scores <= 1))
    gate = gate_identity(
        anchor_query=2,
        model_sha256=SHA,
        sequence_map_sha256="b" * 64,
        backend="sdpa",
        selected_layers=(3,),
    )
    assert gate.anchor_query == 2 and len(gate.semantic_sha256) == 64
    assert gate.age(5) == 3
    with pytest.raises(B2ValidationError, match="predate"):
        gate.age(1)


def test_profiles_enforce_p1_wrist_rule_nondecending_budget_and_grid_cap():
    p1 = profile(ProfileFamily.P1)
    p1 = Profile(**{**p1.__dict__, "budgets": (LayerBudget(1, 1, 0), LayerBudget(3, 2, 0))})
    validate_profile(p1)
    with pytest.raises(B2ValidationError, match="P1"):
        validate_profile(profile(ProfileFamily.P1))
    decreasing = Profile(**{**profile().__dict__, "budgets": (LayerBudget(1, 2, 1), LayerBudget(3, 1, 1))})
    with pytest.raises(B2ValidationError, match="nondecreasing"):
        validate_profile(decreasing)
    many = tuple(Profile(**{**p1.__dict__, "profile_id": f"p-{index}"}) for index in range(7))
    with pytest.raises(B2ValidationError, match="one to six"):
        validate_profile_grid(many)


def test_nested_sets_use_complete_layer_suffix_and_never_reintroduce_tokens():
    mapping = runtime_map()
    test_profile = profile()
    visual = mapping.scene_positions + mapping.wrist_positions
    eligible = {layer: {position: True for position in visual} for layer in (1, 3)}
    eligible[3][1] = False
    changes = {
        1: {1: 0.01, 5: 0.2, 3: 0.01, 7: 0.2},
        3: {1: 0.01, 5: 0.2, 3: 0.01, 7: 0.2},
    }
    reuse = build_nested_reuse_sets(
        test_profile, mapping, eligible=eligible, change_scores=changes
    )
    assert 1 not in reuse[1]
    assert reuse[1] <= reuse[3]
    assert len(reuse[1] & set(mapping.scene_positions)) == 1
    assert len(reuse[3] & set(mapping.scene_positions)) == 1
    assert len(reuse[3] & set(mapping.wrist_positions)) == 1


def test_actual_source_change_scores_are_required_for_mixed_source_cache():
    sources = {(1, 1): 0, (1, 3): 2}
    with pytest.raises(B2ValidationError, match="actual source"):
        validate_exact_source_eligibility(
            source_queries=sources, source_change_scores={(1, 1, 0): 0.0, (1, 3, 0): 0.0}
        )
    validate_exact_source_eligibility(
        source_queries=sources, source_change_scores={(1, 1, 0): 0.0, (1, 3, 2): 0.0}
    )


def test_contract_state_expiry_abort_lock_and_reset_semantics():
    contract = Contract("p", ProfileFamily.P1, 2)
    active = ContractState.anchor().start(contract)
    assert active.advance(query=1).mode is ContractMode.CONTRACT
    assert active.advance(query=1).advance(query=2).mode is ContractMode.ANCHOR
    assert active.advance(query=1, abort_reason="drift").mode is ContractMode.ANCHOR
    experimental = ContractState.anchor().start(contract, experimental=True)
    locked = experimental.advance(query=1, abort_reason="drift")
    assert locked.mode is ContractMode.EXPERIMENT_FR_LOCK
    assert locked.advance(query=2).mode is ContractMode.ANCHOR
    with pytest.raises(B2ValidationError):
        Contract("p", ProfileFamily.P3, 1).validate()


def test_p0_through_p4_runtime_and_episode_reset_semantics():
    for family in ProfileFamily:
        runtime = ProfileRuntime.reset(family, "episode-a")
        if family in (ProfileFamily.P1, ProfileFamily.P2):
            runtime, anchor = runtime.step(0, contract_active=False)
            runtime, reused = runtime.step(1, contract_active=True)
            assert anchor.dense_executed and reused.accelerated_executed
        elif family is ProfileFamily.P4:
            runtime, first = runtime.step(0)
            runtime, second = runtime.step(1)
            assert first.action_discarded and not first.accelerated_executed
            assert second.action_discarded and second.accelerated_executed
            assert second.cache_source_query == 0
        else:
            runtime, transition = runtime.step(0)
            assert transition.dense_executed
    reset = ProfileRuntime.reset(ProfileFamily.P4, "episode-b")
    assert reset.last_query == -1 and reset.pending_dense_query is None


def test_source_ring_refuses_live_eviction_and_preserves_immutable_records():
    ring = SourceRingBuffer(2)
    first, second, third = source(0), source(1), source(2)
    ring.add(first)
    ring.add(second)
    with pytest.raises(B2ValidationError, match="live"):
        ring.add(third, live_sources={0, 1})
    ring.add(third, live_sources={1})
    assert ring.queries() == (1, 2)


def test_ledger_updates_sources_per_layer_token_and_checks_context_envelopes():
    mapping = runtime_map()
    keys = {(layer, position): f"{layer}{position}".ljust(64, "0") for layer in (1, 3) for position in mapping.scene_positions + mapping.wrist_positions}
    ledger = SourceLedger.dense(
        identity=identity(mapping.semantic_sha256),
        sequence_map=mapping,
        layers=(1, 3),
        source_record=source(0),
        gate_sha256="d" * 64,
        dtype="float32",
        shape=(1, 2),
        kv_digests=keys,
    )
    current_keys = {key: value.replace("0", "1") for key, value in keys.items()}
    updated = ledger.update(
        current_query=1,
        source_record=source(1, proprio=(0.01, 0), action=(0.01, 0)),
        reuse_sets={1: {1}, 3: {1, 3}},
        kv_digests=current_keys,
        profile_id="p",
        horizon=2,
        remaining=1,
    )
    assert updated.entries[(1, 1)].source_query == 0
    assert updated.entries[(1, 5)].source_query == 1
    assert updated.entries[(3, 3)].source_query == 0
    accepted, reason = updated.context_envelope(
        current_proprio=(0.02, 0),
        previous_proprio=(0.01, 0),
        current_action_summary=(0.02, 0),
        source_proprio_limit=0.1,
        step_proprio_limit=0.1,
        source_action_limit=0.1,
        gripper_transition=False,
    )
    assert accepted and reason is None
    assert updated.context_envelope(
        current_proprio=(0.02, 0),
        previous_proprio=(0.01, 0),
        current_action_summary=(0.02, 0),
        source_proprio_limit=0.1,
        step_proprio_limit=0.1,
        source_action_limit=0.1,
        gripper_transition=True,
    ) == (False, "unexpected_gripper_transition")


class FakeCache:
    def __init__(self):
        self.key_cache = [np.arange(6, dtype=float).reshape(1, 3, 2)]
        self.value_cache = [self.key_cache[0] + 10]
        self._seen_tokens = 3


def test_cache_clone_transaction_and_absolute_index_update_do_not_cross_mutate_arms():
    cache = FakeCache()
    cloned = clone_dynamic_cache(cache)
    cloned.key_cache[0][0, 0, 0] = 999
    assert cache.key_cache[0][0, 0, 0] == 0
    config = SimpleNamespace(mode="dense")
    with pytest.raises(RuntimeError):
        with transactional_cache_configuration(cache, config, {"mode": "reuse", "new": 1}) as arm:
            arm.value_cache[0][0, 0, 0] = -1
            cache._seen_tokens = 999
            raise RuntimeError("arm failed")
    assert cache._seen_tokens == 3 and config.mode == "dense" and not hasattr(config, "new")
    updated = position_preserving_index_update(
        cache.key_cache[0], np.array([[[100.0, 101.0]]]), [1]
    )
    assert np.array_equal(updated[0, 0], cache.key_cache[0][0, 0])
    assert np.array_equal(updated[0, 1], [100, 101])


def test_corrected_vla_cache_frame_history_error_propagation_and_p0_identity():
    official = (
        "prefix\n"
        + corrected_evaluator_source.__globals__["OFFICIAL_PREVIOUS_BLOCK"]
        + corrected_evaluator_source.__globals__["OFFICIAL_ACTION_BLOCK"]
        + corrected_evaluator_source.__globals__["OFFICIAL_ERROR_BLOCK"]
        + "suffix\n"
    )
    corrected = corrected_evaluator_source(official)
    manifest = correction_manifest(official)
    assert manifest["previous_cache_source_fixed"] and manifest["episode_errors_propagate"]
    assert "raise" in corrected
    history = FrameHistory()
    assert history.attach("current-0", "wrist-0") == ("current-0", "wrist-0")
    history.commit_cache_source("current-0", "wrist-0")
    assert history.attach("current-1", "wrist-1") == ("current-0", "wrist-0")
    with pytest.raises(RuntimeError):
        propagate_episode_error(lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    calls = []
    assert execute_p0_or_profile(
        enabled=False,
        dense_forward=lambda: calls.append("dense") or "same",
        accelerated_forward=lambda: calls.append("reuse") or "different",
    ) == "same"
    assert calls == ["dense"]


def test_intent_records_require_overlap_exclude_outcomes_and_lock_fr_after_abort():
    raw = {
        "run_id": "r",
        "episode_group": "g",
        "branch_id": "b",
        "assigned_contract": "p-h2",
        "inclusion_probability": 0.5,
        "assignment_probability": 0.5,
        "arm_order": ["FR", "CONTRACT"],
        "configuration_sha256": SHA,
    }
    record = freeze_intent_record(raw)
    validate_intent_record(record)
    with pytest.raises(B2ValidationError, match="nonzero"):
        freeze_intent_record({**raw, "assignment_probability": 0})
    with pytest.raises(B2ValidationError, match="forbidden"):
        freeze_intent_record({**raw, "success": True})
    assert experimental_execution_schedule(horizon=4, abort_at=1) == (
        "CONTRACT", "FR", "FR", "FR"
    )
    with pytest.raises(B2ValidationError, match="discordant"):
        assert_duplicate_arm_identity(("a",), ("b",))


def test_frozen_b2_configuration_and_runner_preserve_phase_boundary():
    from savr.brace.b1 import semantic_sha256

    configs = []
    for name in ("b2_correctness_v1.json", "b2_correctness_v2.json"):
        config = json.loads((REPOSITORY_ROOT / f"configs/brace/{name}").read_text())
        supplied = config.pop("semantic_sha256")
        assert semantic_sha256(config) == supplied
        configs.append(config)
        assert config["contracts"]["horizons"] == [1, 2, 4]
        assert config["contracts"]["maximum_base_profiles"] == 6
        assert config["b3_proposal"]["maximum_balanced_real_model_queries"] == 480
        assert config["b3_proposal"]["requires_separate_authorization"] is True
        caps = config["resource_caps"]
        assert caps["cuda_visible"] is False
        assert caps["model_queries"] == caps["policy_outcomes"] == caps["simulator_steps"] == 0
        assert caps["model_checkpoint_dataset_downloads_allowed"] is False
    v1, v2 = configs
    for key in ("stacks", "comparators", "contracts", "b3_proposal", "resource_caps"):
        assert v2[key] == v1[key]
    assert v2["recovery"]["supersedes_run_id"] == "brace-b2-correctness-v01"
    assert v2["recovery"]["scientific_gates_changed"] is False
    runner = (REPOSITORY_ROOT / "scripts/run_brace_b2.py").read_text()
    assert 'CONFIG_RELATIVE = Path("configs/brace/b2_correctness_v2.json")' in runner
    assert '"CUDA_VISIBLE_DEVICES": ""' in runner
    assert "initialize_model" not in runner
    assert "nvidia-smi" not in runner
