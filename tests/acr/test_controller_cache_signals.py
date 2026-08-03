from __future__ import annotations

from dataclasses import fields, replace

import pytest

from savr.acr.cache import SceneCacheCompatibilityError, SceneCacheMiss, SceneTokenCache
from savr.acr.controller import ACRController
from savr.acr.signals import scene_change_from_representations, transition_signal
from savr.acr.types import ACRConfiguration, ACRContext, ACRPolicy


class Tensor:
    def __init__(self, values=(1.0,), shape=(1, 2, 1), dtype="float32", device="cpu"):
        self.values = list(values)
        self.shape = shape
        self.dtype = dtype
        self.device = device
        self.grad = True

    def detach(self):
        return self

    def clone(self):
        return Tensor(self.values, self.shape, self.dtype, self.device)

    def requires_grad_(self, value):
        self.grad = value
        return self


def context(**changes):
    value = ACRContext(
        episode_id="episode",
        attempt_id="attempt",
        task_id="task",
        instruction_sha256="0" * 64,
        checkpoint_id="checkpoint",
        upstream_revision="revision",
        configuration_id="candidate",
        controller_version="acr-controller-v1",
        preprocessing_id="preprocessing",
        action_head_id="head",
        dtype="float32",
        device="cpu",
        patch_count=2,
    )
    return replace(value, **changes)


def configuration(**changes):
    value = ACRConfiguration(
        configuration_id="candidate",
        policy=ACRPolicy.SA_ACR,
        scene_threshold=0.2,
        translation_threshold=0.2,
        horizon=4,
        hard_reuse_cap=0.75,
    )
    return replace(value, **changes)


def flat_scene(value=0.0):
    return (value,) * (32 * 32)


def action(gripper=0.0, x=0.0):
    return tuple(item for _ in range(8) for item in (x, 0, 0, 0, 0, 0, gripper))


def begin_controller(config=None):
    config = config or configuration()
    controller = ACRController(config)
    controller.reset(context(configuration_id=config.configuration_id))
    return controller


def decide_and_observe(controller, *, scene=None, position=(0.0, 0.0, 0.0), cache=False, age=0, chunk=None):
    scene = flat_scene() if scene is None else scene
    chunk = action() if chunk is None else chunk
    decision = controller.decide(
        scene_representation=scene,
        normalized_eef_position=position,
        cache_available=cache,
        cache_age=age,
    )
    controller.observe(
        decision=decision,
        scene_representation=scene,
        normalized_eef_position=position,
        action_chunk=chunk,
    )
    return decision


def test_scene_patch_score_exact_boundary():
    reference = flat_scene()
    current = list(reference)
    for y in range(4):
        for x in range(4):
            current[y * 32 + x] = 0.8
    change = scene_change_from_representations(tuple(current), reference)
    assert len(change.patch_scores) == 64
    assert change.patch_scores[0] == pytest.approx(0.8)
    assert change.top_four_mean == pytest.approx(0.2)


@pytest.mark.parametrize(
    ("score", "refresh"),
    [(0.199999, False), (0.2, False), (0.200001, True)],
)
def test_scene_threshold_is_strict(score, refresh):
    controller = begin_controller()
    decide_and_observe(controller)
    decide_and_observe(controller, cache=True, age=0)
    current = list(flat_scene())
    for patch in range(4):
        for offset_y in range(4):
            for offset_x in range(4):
                y = (patch // 2) * 4 + offset_y
                x = (patch % 2) * 4 + offset_x
                current[y * 32 + x] = score
    decision = controller.decide(
        scene_representation=current,
        normalized_eef_position=(0, 0, 0),
        cache_available=True,
        cache_age=0,
    )
    assert ("scene-change" in decision.reasons) is refresh


@pytest.mark.parametrize(
    ("position", "refresh"),
    [((0.199999, 0, 0), False), ((0.2, 0, 0), False), ((0.200001, 0, 0), True)],
)
def test_translation_threshold_is_strict(position, refresh):
    controller = begin_controller()
    decide_and_observe(controller)
    decide_and_observe(controller, cache=True)
    decision = controller.decide(
        scene_representation=flat_scene(),
        normalized_eef_position=position,
        cache_available=True,
        cache_age=0,
    )
    assert ("translation" in decision.reasons) is refresh


def test_refresh_conditions_order_and_fail_closed():
    controller = begin_controller(configuration(horizon=1, hard_reuse_cap=0.1))
    first = controller.decide(
        scene_representation=None,
        normalized_eef_position=None,
        cache_available=False,
        cache_age=0,
    )
    assert first.reasons == ("cache", "invalid-signal", "warm-up", "hard-cap")


def test_warmup_transition_horizon_and_reference_update():
    controller = begin_controller()
    first = decide_and_observe(controller, position=(0, 0, 0), chunk=action(0))
    second = decide_and_observe(
        controller, position=(0.1, 0, 0), cache=True, chunk=action(1)
    )
    assert "warm-up" in first.reasons and "warm-up" in second.reasons
    third = controller.decide(
        scene_representation=flat_scene(),
        normalized_eef_position=(0.1, 0, 0),
        cache_available=True,
        cache_age=4,
    )
    assert third.reference_query_index == 1
    assert "gripper-transition" in third.reasons
    assert "horizon" in third.reasons


def test_mixed_gripper_and_reversal_diagnostics():
    mixed = list(action(0, x=-1))
    mixed[-1] = 1
    signal = transition_signal(tuple(mixed), action(0, x=1))
    assert signal.mixed_latest_gripper
    assert signal.gripper_veto
    assert signal.translation_direction_reversals == (True, False, False)


def test_hard_cap_equality_is_reuse_safe_and_above_forces():
    controller = begin_controller(configuration(hard_reuse_cap=0.5))
    decide_and_observe(controller)
    decide_and_observe(controller, cache=True)
    equality = controller.decide(
        scene_representation=flat_scene(),
        normalized_eef_position=(0, 0, 0),
        cache_available=True,
        cache_age=0,
    )
    assert "hard-cap" not in equality.reasons  # prospective 1/3 < 0.5
    controller.observe(
        decision=equality,
        scene_representation=flat_scene(),
        normalized_eef_position=(0, 0, 0),
        action_chunk=action(),
    )
    exact = controller.decide(
        scene_representation=flat_scene(),
        normalized_eef_position=(0, 0, 0),
        cache_available=True,
        cache_age=1,
    )
    assert "hard-cap" not in exact.reasons  # prospective 2/4 == 0.5
    controller.observe(
        decision=exact,
        scene_representation=flat_scene(),
        normalized_eef_position=(0, 0, 0),
        action_chunk=action(),
    )
    above = controller.decide(
        scene_representation=flat_scene(),
        normalized_eef_position=(0, 0, 0),
        cache_available=True,
        cache_age=2,
    )
    assert "hard-cap" in above.reasons


def test_policy_truth_table():
    factorized = begin_controller(
        ACRConfiguration("factorized", ACRPolicy.FACTORIZED_FR)
    )
    assert "policy" in factorized.decide(
        scene_representation=flat_scene(),
        normalized_eef_position=None,
        cache_available=True,
        cache_age=0,
    ).reasons
    periodic_config = ACRConfiguration(
        "periodic", ACRPolicy.SCENE_PERIODIC, period=3
    )
    periodic = begin_controller(periodic_config)
    decide_and_observe(periodic)
    decide_and_observe(periodic, cache=True)
    assert not periodic.decide(
        scene_representation=flat_scene(),
        normalized_eef_position=None,
        cache_available=True,
        cache_age=0,
    ).refresh


def test_cache_owns_tensor_and_invalidates_every_context_identity():
    cache = SceneTokenCache()
    original = Tensor(values=(7.0,), shape=(1, 2, 1))
    base = context()
    metadata = cache.store(context=base, tokens=original, refresh_query_index=0)
    original.values[0] = 9
    cached = cache.load(base, metadata)
    assert cached.values == [7.0]
    assert cached is not original and cached.grad is False
    replacements = {
        "episode_id": "other",
        "attempt_id": "other",
        "task_id": "other",
        "instruction_sha256": "1" * 64,
        "checkpoint_id": "other",
        "upstream_revision": "other",
        "configuration_id": "other",
        "controller_version": "other",
        "preprocessing_id": "other",
        "action_head_id": "other",
        "dtype": "float16",
        "device": "cuda:0",
        "patch_count": 3,
    }
    checked = {field.name for field in fields(ACRContext)} - {
        "image_order", "number_of_images", "center_crop"
    }
    assert set(replacements) == checked
    for name, changed in replacements.items():
        assert not cache.available(replace(base, **{name: changed}))
    cache.mark_reused()
    assert cache.age == 1
    cache.invalidate()
    assert cache.age == 0
    with pytest.raises(SceneCacheMiss):
        cache.load(base, metadata)


def test_cache_rejects_shape_dtype_device_mismatch():
    cache = SceneTokenCache()
    base = context()
    metadata = cache.store(
        context=base, tokens=Tensor(shape=(1, 2, 1)), refresh_query_index=0
    )
    for changed in (
        replace(metadata, shape=(1, 2, 2)),
        replace(metadata, dtype="float16"),
        replace(metadata, device="cuda:0"),
    ):
        with pytest.raises(SceneCacheCompatibilityError):
            cache.load(base, changed)
