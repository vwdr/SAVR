from __future__ import annotations

import json
from pathlib import Path

import pytest

from savr.brace.b3 import QueryLedger, cycle_schedule, planned_query_count
from savr.brace.b3 import B3ProtocolError
from savr.brace.b3_openvla import (
    SDPASidecarTap,
    SourceTracker,
    ordered_profile_positions,
    patch_change_scores,
)


ROOT = Path(__file__).resolve().parents[2]


def config() -> dict:
    return json.loads((ROOT / "configs/brace/b3_physical_v1.json").read_text())


def torch_module():
    return pytest.importorskip("torch")


def test_real_torch_sdpa_sidecar_captures_all_layers_without_changing_output():
    torch = torch_module()
    torch.manual_seed(7)
    query = torch.randn(1, 2, 8, 4)
    key = torch.randn(1, 2, 8, 4)
    value = torch.randn(1, 2, 8, 4)
    reference = torch.nn.functional.scaled_dot_product_attention(query, key, value)
    with SDPASidecarTap(torch, (6, 15, 24)) as tap:
        outputs = [
            torch.nn.functional.scaled_dot_product_attention(query, key, value)
            for _ in range(32)
        ]
    assert tap.calls == 32 and set(tap.captured) == {6, 15, 24}
    assert all(torch.equal(reference, output) for output in outputs)
    salience = tap.salience(
        instruction_positions=(4, 5),
        action_positions=(6, 7),
        visual_positions=(0, 1, 2, 3),
    )
    assert tuple(salience.shape) == (4,)
    assert torch.isfinite(salience).all()
    assert bool(((0 <= salience) & (salience <= 1)).all())


def test_real_torch_sdpa_sidecar_fails_closed_on_decoder_call_drift():
    torch = torch_module()
    query = torch.zeros(1, 1, 2, 4)
    original = torch.nn.functional.scaled_dot_product_attention
    with pytest.raises(B3ProtocolError, match="31 decoder calls"):
        with SDPASidecarTap(torch, (6, 15, 24)):
            for _ in range(31):
                torch.nn.functional.scaled_dot_product_attention(query, query, query)
    assert torch.nn.functional.scaled_dot_product_attention is original


def test_all_profiles_order_real_tensors_and_complete_mixed_source_horizons():
    torch = torch_module()
    salience = torch.linspace(0, 1, 512)
    scene_change = torch.zeros(256)
    wrist_change = torch.zeros(256)
    pruning_layers = tuple(config()["model"]["pruning_layers"])
    for profile in config()["profiles"]:
        wrist_offsets = (0, int(profile["wrist_budgets"][-1])) if profile["family"] == "P2" else (0,)
        ordered_by_offset = {}
        for wrist_offset in wrist_offsets:
            ordered, proportions = ordered_profile_positions(
                profile,
                scene_change=scene_change,
                wrist_change=wrist_change,
                salience=salience,
                torch_module=torch,
                wrist_offset=wrist_offset,
            )
            expected = int(profile["scene_budgets"][-1]) + int(profile["wrist_budgets"][-1])
            assert ordered.device.type == "cpu"
            assert len(ordered) == expected == len(set(ordered.tolist()))
            assert all(1 <= int(position) <= 512 for position in ordered)
            assert proportions == sorted(proportions) and proportions[-1] == pytest.approx(1.0)
            ordered_by_offset[wrist_offset] = tuple(int(value) for value in ordered.tolist())
        tracker = SourceTracker(anchor_query=0)
        for query in range(1, 5):
            wrist_offset = 0
            if profile["family"] == "P2" and query > int(profile["wrist_max_age"]):
                wrist_offset = int(profile["wrist_budgets"][-1])
            tracker.advance(
                query,
                ordered_positions=ordered_by_offset[wrist_offset],
                profile=profile,
                pruning_layers=pruning_layers,
            )
        tracker.validate(4, profile)


def test_profile_ordering_fails_closed_when_change_gate_has_too_few_candidates():
    torch = torch_module()
    profile = next(item for item in config()["profiles"] if item["profile_id"] == "P1-S50")
    with pytest.raises(B3ProtocolError, match="enough suffix-eligible tokens"):
        ordered_profile_positions(
            profile,
            scene_change=torch.ones(256),
            wrist_change=torch.zeros(256),
            salience=torch.linspace(0, 1, 512),
            torch_module=torch,
        )


def test_patch_change_real_tensors_support_zero_and_full_dynamic_range():
    torch = torch_module()
    source = torch.zeros((1, 3, 224, 224))
    unchanged = patch_change_scores(
        source,
        source,
        torch_module=torch,
        epsilon=1e-8,
        weights=(0.5, 0.5),
    )
    changed_image = torch.ones_like(source)
    changed = patch_change_scores(
        changed_image,
        source,
        torch_module=torch,
        epsilon=1e-8,
        weights=(0.5, 0.5),
    )
    assert torch.equal(unchanged, torch.zeros(256))
    assert tuple(changed.shape) == (256,)
    assert torch.isfinite(changed).all() and bool((changed > 0).all())


def test_complete_model_free_schedule_consumes_each_frozen_allocation_once():
    cfg = config()
    allocations = {
        "core_fr": int(cfg["measurement"]["core_fr_queries"]),
        "cache_suite": int(cfg["measurement"]["cache_p0_queries"])
        + int(cfg["measurement"]["attention_parity_queries"])
        + int(cfg["measurement"]["corrected_vla_cache_queries"])
        + int(cfg["measurement"]["clean_profile_queries"]),
        "vla_adp": int(cfg["measurement"]["vla_adp_queries"]),
        "vla_pruner": int(cfg["measurement"]["vla_pruner_queries"]),
    }
    ledger = QueryLedger(int(cfg["resource_caps"]["model_query_hard_cap"]), allocations)
    for method, count in allocations.items():
        ledger.consume(method, count)
    assert ledger.total == planned_query_count(cfg) == 388
    schedule = cycle_schedule(cfg)
    assert sum(1 + horizon for _, horizon, _ in schedule) == 240
    assert len(schedule) == 72


def test_every_frozen_profile_cycle_has_valid_deterministic_provenance():
    torch = torch_module()
    cfg = config()
    profiles = {profile["profile_id"]: profile for profile in cfg["profiles"]}
    pruning_layers = tuple(cfg["model"]["pruning_layers"])
    scene_change = torch.zeros(256)
    wrist_change = torch.zeros(256)
    salience = torch.linspace(0, 1, 512)
    final_digests = []
    for profile_id, horizon, repetition in cycle_schedule(cfg):
        profile = profiles[profile_id]
        tracker = SourceTracker(anchor_query=0)
        for query in range(1, horizon + 1):
            wrist_offset = 0
            if profile["family"] == "P2" and query > int(profile["wrist_max_age"]):
                wrist_offset = int(profile["wrist_budgets"][-1])
            ordered, _ = ordered_profile_positions(
                profile,
                scene_change=scene_change,
                wrist_change=wrist_change,
                salience=salience,
                torch_module=torch,
                wrist_offset=wrist_offset,
            )
            tracker.advance(
                query,
                ordered_positions=tuple(int(value) for value in ordered.tolist()),
                profile=profile,
                pruning_layers=pruning_layers,
            )
        final_digests.append((profile_id, horizon, repetition, tracker.digest()))
    assert len(final_digests) == 72
    by_contract: dict[tuple[str, int], set[str]] = {}
    for profile_id, horizon, _, digest in final_digests:
        by_contract.setdefault((profile_id, horizon), set()).add(digest)
    assert all(len(digests) == 1 for digests in by_contract.values())
