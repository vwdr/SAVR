from __future__ import annotations

import numpy as np
import pytest

from savr.pair.interventions import (
    GroupSourceLedger,
    InterventionMask,
    TensorSourceRing,
    apply_intervention,
    profile_mask,
    tile_patch_indices,
)
from savr.pair.types import AtomicGroup, Camera, PairValidationError, ProfileSpec, SourcePointer


def frozen_profile() -> ProfileSpec:
    return ProfileSpec("D37_BAL", (32, 64, 96, 128), (16, 32, 48, 64))


def test_profile_mask_exactly_realizes_nested_tile_budgets():
    mask = profile_mask(
        frozen_profile(),
        {Camera.PRIMARY: tuple(range(16)), Camera.WRIST: tuple(reversed(range(16)))},
    )
    assert len(mask.groups) == 12
    for layer, primary, wrist in zip(
        (2, 6, 9, 11), (32, 64, 96, 128), (16, 32, 48, 64), strict=True
    ):
        assert sum(mask.reuses(layer, Camera.PRIMARY, tile) for tile in range(16)) * 16 == primary
        assert sum(mask.reuses(layer, Camera.WRIST, tile) for tile in range(16)) * 16 == wrist


def test_profiles_masks_and_tiles_fail_closed_on_nonnesting_duplicates_and_off_by_one():
    with pytest.raises(PairValidationError, match="not nested"):
        ProfileSpec("bad", (32, 16, 96, 128), (0, 0, 0, 0)).validate()
    with pytest.raises(PairValidationError, match="multiple onset"):
        InterventionMask(
            (AtomicGroup(Camera.PRIMARY, 0, 2), AtomicGroup(Camera.PRIMARY, 0, 6))
        ).validate()
    with pytest.raises(PairValidationError, match="outside"):
        tile_patch_indices(16)
    with pytest.raises(PairValidationError, match="permutation"):
        profile_mask(
            frozen_profile(),
            {Camera.PRIMARY: tuple(range(15)), Camera.WRIST: tuple(range(16))},
        )


def test_all_fresh_reproduces_dense_and_camera_positions_never_swap(tensor_bank_factory):
    dense = tensor_bank_factory(0)
    ledger = GroupSourceLedger.dense(0, dense)
    current = tensor_bank_factory(1)
    fresh = apply_intervention(current, ledger, InterventionMask.fresh())
    assert all(np.array_equal(fresh[layer], current[layer]) for layer in current)

    mask = InterventionMask((AtomicGroup(Camera.PRIMARY, 0, 2),))
    stale = apply_intervention(current, ledger, mask)
    indices = tile_patch_indices(0)
    assert np.array_equal(stale[2][0, indices], dense[2][0, indices])
    assert np.array_equal(stale[2][1], current[2][1])
    assert np.array_equal(stale[6][0, indices], dense[6][0, indices])


def test_recursive_ledger_resolves_actual_mixed_age_sources(tensor_bank_factory):
    q0, q1, q2 = tensor_bank_factory(0), tensor_bank_factory(1), tensor_bank_factory(2)
    ledger = GroupSourceLedger.dense(0, q0)
    first_mask = InterventionMask((AtomicGroup(Camera.PRIMARY, 0, 2),))
    ledger = ledger.advance(1, q1, first_mask)
    second_mask = InterventionMask(
        (AtomicGroup(Camera.PRIMARY, 0, 2), AtomicGroup(Camera.PRIMARY, 1, 2))
    )
    stale = apply_intervention(q2, ledger, second_mask)
    tile0, tile1 = tile_patch_indices(0), tile_patch_indices(1)
    assert np.array_equal(stale[11][0, tile0], q0[11][0, tile0])
    assert np.array_equal(stale[11][0, tile1], q1[11][0, tile1])
    ledger = ledger.advance(2, q2, second_mask)
    assert ledger.entries[(11, Camera.PRIMARY, 0)] == SourcePointer(0, 2)
    assert ledger.entries[(11, Camera.PRIMARY, 1)] == SourcePointer(1, 1)
    assert len(ledger.live_sources()) == 3


def test_missing_source_wrong_shape_and_live_ring_eviction_fail_closed(tensor_bank_factory):
    q0, q1, q2 = tensor_bank_factory(0), tensor_bank_factory(1), tensor_bank_factory(2)
    ledger = GroupSourceLedger.dense(0, q0, ring_capacity=2)
    ledger = ledger.advance(1, q1, InterventionMask((AtomicGroup(Camera.PRIMARY, 0, 2),)))
    with pytest.raises(PairValidationError, match="live source"):
        ledger.advance(2, q2, InterventionMask.fresh())

    broken = {layer: value.copy() for layer, value in q0.items()}
    broken[2] = broken[2][:, :255]
    with pytest.raises(PairValidationError, match="shape"):
        GroupSourceLedger.dense(0, broken)

    ring = TensorSourceRing()
    ring.add(0, q0)
    entries = {
        (layer, camera, tile): SourcePointer(
            3 if (layer, camera, tile) == (2, Camera.PRIMARY, 0) else 0, 0
        )
        for layer in (2, 6, 9, 11)
        for camera in Camera
        for tile in range(16)
    }
    with pytest.raises(PairValidationError, match="chronologically"):
        GroupSourceLedger(current_query=0, entries=entries, sources=ring)


def test_abort_resets_all_sources_and_clone_isolation(tensor_bank_factory):
    q0, q1 = tensor_bank_factory(0), tensor_bank_factory(1)
    ledger = GroupSourceLedger.dense(0, q0)
    clone = ledger.clone()
    clone.sources.get(0, 2)[0, 0, 0] = -999
    assert ledger.sources.get(0, 2)[0, 0, 0] != -999
    reset = ledger.abort(1, q1)
    assert reset.current_query == 1 and reset.live_sources() == (1,)
    assert all(pointer == SourcePointer(1, 0) for pointer in reset.entries.values())
