from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from savr.pair.p3 import P3Profile
from savr.pair.p3_openvla import (
    PhysicalSourceTracker,
    PreparedQuery,
    _change_score,
    cache_sample_digest,
    capture_reused_cache,
    ordered_tile_profile,
    ordered_tile_profile_vectorized,
    protected_action_record,
    runtime_positions,
    verify_reused_cache,
)
from savr.pair.types import Camera, PairValidationError


torch = pytest.importorskip("torch")


def prepared(value: float = 0.0):
    action_mask = torch.zeros((1, 79), dtype=torch.bool)
    action_mask[:, 22:78] = True
    return PreparedQuery(
        input_embeddings=torch.zeros((1, 79, 8)),
        action_mask=action_mask,
        projected_patches=torch.full((1, 513, 4), value),
        attention_mask=torch.ones((1, 79), dtype=torch.long),
        normalized_proprio=torch.zeros((1, 8)),
        preprocessed_pixels=torch.full((1, 6, 224, 224), value),
        instruction_token_indices=(4, 5, 6),
    )


def profile():
    return P3Profile("D37_BAL_PT1", "D37_BAL", (32, 64, 96, 128), (16, 32, 48, 64), 1, 1)


def test_runtime_positions_are_dynamic_complete_and_camera_separated():
    positions = runtime_positions(prepared(), torch)
    assert positions["primary"] == tuple(range(1, 257))
    assert positions["wrist"] == tuple(range(257, 513))
    assert len(positions["action"]) == 56
    assert positions["action"] == tuple(range(534, 590))
    assert positions["instruction"] == (517, 518, 519)
    assert positions["placeholder"] == tuple(range(535, 591))
    assert not set(positions["primary"]) & set(positions["wrist"])


def test_physical_tracker_is_recursive_bounded_and_resettable():
    tracker = PhysicalSourceTracker(0, prepared(0))
    tracker.add_record(1, prepared(1))
    groups = {(Camera.PRIMARY, 0): 2, (Camera.WRIST, 1): 6}
    tracker.advance(1, groups)
    assert tracker.sources[(2, Camera.PRIMARY, 0)] == 0
    assert tracker.sources[(1, Camera.PRIMARY, 0)] == 1
    assert tracker.sources[(6, Camera.WRIST, 1)] == 0
    assert tracker.mixture_count() == 2 and len(tracker.digest()) == 64
    with pytest.raises(PairValidationError, match="contiguous"):
        tracker.advance(3, groups)


def test_tile_profile_is_nested_aligned_and_uses_actual_sources():
    tracker = PhysicalSourceTracker(0, prepared(0))
    current = prepared(1)
    tracker.add_record(1, current)
    ordered, proportions, groups, metadata = ordered_tile_profile(
        current=current,
        tracker=tracker,
        profile=profile(),
        previous_salience=torch.linspace(0, 1, 512),
        torch_module=torch,
    )
    assert len(ordered) == 192 and len(set(ordered.tolist())) == 192
    assert proportions == pytest.approx((48 / 192, 96 / 192, 144 / 192, 1.0))
    assert len(groups) == 12
    assert metadata["reused_primary_tiles"] == 8
    assert metadata["reused_wrist_tiles"] == 4


def test_cache_samples_and_exact_reused_positions_detect_mutation():
    cache = SimpleNamespace(
        key_cache=[torch.arange(1 * 2 * 600 * 3).reshape(1, 2, 600, 3).float() for _ in range(32)],
        value_cache=[
            torch.arange(1 * 2 * 600 * 3).reshape(1, 2, 600, 3).float() + 1 for _ in range(32)
        ],
    )
    assert cache_sample_digest(cache, torch) == cache_sample_digest(cache, torch)
    ordered = torch.arange(1, 193, dtype=torch.long)
    snapshots = capture_reused_cache(cache, ordered, profile(), torch)
    verify_reused_cache(cache, snapshots)
    cache.key_cache[2][..., 1, 0] += 1
    with pytest.raises(PairValidationError, match="changed"):
        verify_reused_cache(cache, snapshots)


def test_protected_action_record_never_persists_values():
    record = protected_action_record(np.zeros((8, 7), dtype=np.float32), np)
    assert set(record) == {"shape", "finite", "sha256"}
    bad = np.zeros((8, 7), dtype=np.float32)
    bad[0, 0] = np.nan
    with pytest.raises(PairValidationError, match="finite"):
        protected_action_record(bad, np)


def test_change_score_handles_zero_tiles_without_false_change_or_nan():
    zero = torch.zeros(4)
    one = torch.ones(4)
    assert _change_score(zero, zero, torch).item() == pytest.approx(0.0)
    assert torch.isfinite(_change_score(zero, one, torch))
    assert _change_score(zero, one, torch).item() > 0.0


@pytest.mark.parametrize("mixed_sources", [False, True])
def test_vectorized_profile_is_exactly_equivalent_to_legacy(mixed_sources):
    generator = torch.Generator().manual_seed(37)

    def random_prepared():
        value = prepared()
        value.projected_patches = torch.randn((1, 513, 4), generator=generator)
        value.preprocessed_pixels = torch.randn((1, 6, 224, 224), generator=generator)
        return value

    tracker = PhysicalSourceTracker(0, random_prepared())
    if mixed_sources:
        tracker.add_record(1, random_prepared())
        tracker.advance(
            1,
            {
                (Camera.PRIMARY, 0): 2,
                (Camera.PRIMARY, 5): 6,
                (Camera.WRIST, 3): 9,
                (Camera.WRIST, 10): 11,
            },
        )
    current = random_prepared()
    current_query = 2 if mixed_sources else 1
    tracker.add_record(current_query, current)
    salience = torch.randn(512, generator=generator)
    legacy = ordered_tile_profile(
        current=current,
        tracker=tracker,
        profile=profile(),
        previous_salience=salience,
        torch_module=torch,
    )
    vectorized = ordered_tile_profile_vectorized(
        current=current,
        tracker=tracker,
        profile=profile(),
        previous_salience=salience,
        torch_module=torch,
    )
    assert torch.equal(legacy[0], vectorized[0])
    assert legacy[1:] == vectorized[1:]


def test_vectorized_profile_preserves_legacy_tie_behavior():
    tracker = PhysicalSourceTracker(0, prepared(0))
    current = prepared(1)
    tracker.add_record(1, current)
    arguments = {
        "current": current,
        "tracker": tracker,
        "profile": profile(),
        "previous_salience": torch.zeros(512),
        "torch_module": torch,
    }
    legacy = ordered_tile_profile(**arguments)
    vectorized = ordered_tile_profile_vectorized(**arguments)
    assert torch.equal(legacy[0], vectorized[0])
    assert legacy[1:] == vectorized[1:]
