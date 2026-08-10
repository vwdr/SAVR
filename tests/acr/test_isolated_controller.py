from __future__ import annotations

import random
from dataclasses import replace

import pytest

from savr.acr.controller import ACRController
from savr.acr.isolated_controller import (
    ISOLATED_CONTROLLER_VERSION,
    IsolatedACRController,
)
from savr.acr.types import ACRConfiguration, ACRContext, ACRPolicy


def configuration(**changes: object) -> ACRConfiguration:
    value = ACRConfiguration(
        configuration_id="isolated",
        policy=ACRPolicy.SA_ACR,
        scene_threshold=1.0,
        translation_threshold=1.0,
        horizon=1,
        hard_reuse_cap=0.75,
        controller_version=ISOLATED_CONTROLLER_VERSION,
    )
    return replace(value, **changes)


def context(**changes: object) -> ACRContext:
    value = ACRContext(
        episode_id="episode",
        attempt_id="attempt",
        task_id="task",
        instruction_sha256="0" * 64,
        checkpoint_id="checkpoint",
        upstream_revision="revision",
        configuration_id="isolated",
        controller_version=ISOLATED_CONTROLLER_VERSION,
        preprocessing_id="preprocessing",
        action_head_id="head",
        dtype="float32",
        device="cpu",
        patch_count=1,
    )
    return replace(value, **changes)


def scene(value: float = 0.0) -> tuple[float, ...]:
    return (value,) * (32 * 32)


def action(value: float = 0.0, *, gripper: float = 0.0) -> tuple[float, ...]:
    return tuple(item for _ in range(8) for item in (value, 0.0, 0.0, 0.0, 0.0, 0.0, gripper))


def begin(config: ACRConfiguration | None = None) -> IsolatedACRController:
    config = config or configuration()
    controller = IsolatedACRController(config)
    controller.reset(
        context(
            configuration_id=config.configuration_id,
            controller_version=config.controller_version,
        )
    )
    return controller


def complete(
    controller: ACRController,
    *,
    cache_available: bool,
    cache_age: int,
    current_scene: tuple[float, ...] | None = None,
    position: tuple[float, float, float] | None = (0.0, 0.0, 0.0),
    current_action: tuple[float, ...] | None = None,
):
    current_scene = scene() if current_scene is None else current_scene
    current_action = action() if current_action is None else current_action
    decision = controller.decide(
        scene_representation=current_scene,
        normalized_eef_position=position,
        cache_available=cache_available,
        cache_age=cache_age,
    )
    controller.observe(
        decision=decision,
        scene_representation=current_scene,
        normalized_eef_position=position,
        action_chunk=current_action,
    )
    return decision


def warm_to_first_reuse(controller: IsolatedACRController) -> None:
    assert complete(controller, cache_available=False, cache_age=0).refresh
    assert complete(controller, cache_available=True, cache_age=0).refresh
    assert not complete(controller, cache_available=True, cache_age=0).refresh


def test_configuration_rejects_legacy_identity_horizon_and_policy() -> None:
    with pytest.raises(ValueError, match="identity"):
        IsolatedACRController(configuration(controller_version="acr-controller-v1"))
    with pytest.raises(ValueError, match="horizon 1"):
        IsolatedACRController(configuration(horizon=2))
    with pytest.raises(ValueError, match="state-aware"):
        IsolatedACRController(
            ACRConfiguration(
                configuration_id="visual",
                policy=ACRPolicy.SCENE_VISUAL,
                scene_threshold=1.0,
                horizon=1,
                hard_reuse_cap=0.5,
                controller_version=ISOLATED_CONTROLLER_VERSION,
            )
        )


def test_stable_trace_is_refresh_reuse_refresh_and_never_reuse_reuse() -> None:
    controller = begin()
    cache_available = False
    cache_age = 0
    previous_reuse = False
    reuses = 0
    for _ in range(40):
        decision = complete(
            controller,
            cache_available=cache_available,
            cache_age=cache_age,
        )
        current_reuse = not decision.refresh
        assert not (previous_reuse and current_reuse)
        previous_reuse = current_reuse
        reuses += int(current_reuse)
        cache_available = True
        cache_age = 0 if decision.refresh else cache_age + 1
        assert reuses / controller.query_index <= controller.configuration.hard_reuse_cap


def test_post_reuse_latch_forces_refresh_even_if_cache_age_is_falsely_zero() -> None:
    controller = begin()
    warm_to_first_reuse(controller)
    decision = controller.decide(
        scene_representation=scene(),
        normalized_eef_position=(0.0, 0.0, 0.0),
        cache_available=True,
        cache_age=0,
    )
    assert decision.refresh
    assert "post-reuse-refresh" in decision.reasons
    assert "isolation-state-mismatch" in decision.reasons


def test_cache_age_mismatch_forces_refresh_without_prior_reuse() -> None:
    controller = begin()
    complete(controller, cache_available=False, cache_age=0)
    decision = controller.decide(
        scene_representation=scene(),
        normalized_eef_position=(0.0, 0.0, 0.0),
        cache_available=True,
        cache_age=1,
    )
    assert decision.refresh
    assert "isolation-state-mismatch" in decision.reasons


def test_gripper_transition_and_context_identity_remain_fail_closed() -> None:
    controller = begin()
    complete(controller, cache_available=False, cache_age=0, current_action=action(gripper=0.0))
    complete(controller, cache_available=True, cache_age=0, current_action=action(gripper=1.0))
    transition = controller.decide(
        scene_representation=scene(),
        normalized_eef_position=(0.0, 0.0, 0.0),
        cache_available=True,
        cache_age=0,
    )
    assert transition.refresh
    assert "gripper-transition" in transition.reasons

    mismatch = IsolatedACRController(configuration())
    with pytest.raises(ValueError, match="identities differ"):
        mismatch.reset(context(configuration_id="different"))


def test_unobserved_refresh_cannot_clear_post_reuse_latch() -> None:
    controller = begin()
    warm_to_first_reuse(controller)
    first = controller.decide(
        scene_representation=scene(),
        normalized_eef_position=(0.0, 0.0, 0.0),
        cache_available=True,
        cache_age=1,
    )
    second = controller.decide(
        scene_representation=scene(),
        normalized_eef_position=(0.0, 0.0, 0.0),
        cache_available=True,
        cache_age=1,
    )
    assert first.refresh and second.refresh
    assert controller.refresh_required_after_reuse
    assert "post-reuse-refresh" in second.reasons


def test_observe_rejects_a_forged_consecutive_reuse() -> None:
    controller = begin()
    warm_to_first_reuse(controller)
    forced = controller.decide(
        scene_representation=scene(),
        normalized_eef_position=(0.0, 0.0, 0.0),
        cache_available=True,
        cache_age=1,
    )
    forged = replace(forced, refresh=False, reasons=())
    with pytest.raises(RuntimeError, match="consecutive reuse"):
        controller.observe(
            decision=forged,
            scene_representation=scene(),
            normalized_eef_position=(0.0, 0.0, 0.0),
            action_chunk=action(),
        )
    assert controller.refresh_required_after_reuse


def test_reset_clears_latch_and_completed_refresh_count() -> None:
    controller = begin()
    warm_to_first_reuse(controller)
    assert controller.snapshot().refresh_required_after_reuse
    assert controller.snapshot().completed_refreshes == 2
    controller.reset(context(episode_id="next", attempt_id="next"))
    snapshot = controller.snapshot()
    assert not snapshot.refresh_required_after_reuse
    assert snapshot.completed_refreshes == 0
    assert snapshot.completed_queries == 0


def test_deterministic_adversarial_trace_preserves_streak_and_prefix_cap() -> None:
    controller = begin(configuration(hard_reuse_cap=0.4))
    rng = random.Random(5102026)
    cache_available = False
    cache_age = 0
    previous_reuse = False
    reuses = 0
    for _ in range(200):
        invalid = rng.random() < 0.12
        current_scene = None if invalid else scene(rng.uniform(0.0, 0.01))
        position = None if invalid else (rng.uniform(0.0, 0.01), 0.0, 0.0)
        decision = controller.decide(
            scene_representation=current_scene,
            normalized_eef_position=position,
            cache_available=cache_available,
            cache_age=cache_age,
        )
        current_reuse = not decision.refresh
        assert not (previous_reuse and current_reuse)
        if invalid:
            assert decision.refresh
            assert "invalid-signal" in decision.reasons
        controller.observe(
            decision=decision,
            scene_representation=current_scene,
            normalized_eef_position=position,
            action_chunk=action(rng.uniform(-0.01, 0.01)),
        )
        previous_reuse = current_reuse
        reuses += int(current_reuse)
        assert reuses / controller.query_index <= 0.4
        cache_available = True
        cache_age = 0 if decision.refresh else cache_age + 1


def test_legacy_horizon_two_semantics_remain_unchanged() -> None:
    config = ACRConfiguration(
        configuration_id="legacy",
        policy=ACRPolicy.SA_ACR,
        scene_threshold=1.0,
        translation_threshold=1.0,
        horizon=2,
        hard_reuse_cap=0.75,
    )
    legacy = ACRController(config)
    legacy.reset(
        context(
            configuration_id="legacy",
            controller_version="acr-controller-v1",
        )
    )
    assert complete(legacy, cache_available=False, cache_age=0).refresh
    assert complete(legacy, cache_available=True, cache_age=0).refresh
    assert not complete(legacy, cache_available=True, cache_age=0).refresh
    assert not complete(legacy, cache_available=True, cache_age=1).refresh
