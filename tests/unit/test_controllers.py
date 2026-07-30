from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from savr.cache import CacheContext  # noqa: E402
from savr.controllers import (  # noqa: E402
    FullRefreshController,
    PeriodicRefreshController,
    StateAwareVisualRefreshController,
    VisualOnlyRefreshController,
)


BLACK_IMAGES = {
    "third_person": [[[0, 0, 0]]],
    "wrist": [[[0, 0, 0]]],
}
WHITE_THIRD_IMAGES = {
    "third_person": [[[255, 255, 255]]],
    "wrist": [[[0, 0, 0]]],
}
WHITE_WRIST_IMAGES = {
    "third_person": [[[0, 0, 0]]],
    "wrist": [[[255, 255, 255]]],
}
STATE = [0.5] * 8
ACTION = [[0.5, 0.5] for _ in range(2)]
CONTEXT = CacheContext("episode-1", "task-0", "checkpoint", "config")


class ControllerTests(unittest.TestCase):
    def test_full_refresh_always_refreshes(self) -> None:
        controller = FullRefreshController()
        controller.reset(CONTEXT)
        for index in range(3):
            decision = controller.decide(
                images={},
                state=None,
                cache_available=index > 0,
                cache_age=index,
            )
            self.assertTrue(decision.refresh)
            self.assertIn("full_refresh", decision.triggers)
            controller.observe(
                decision=decision,
                images={},
                state=None,
                action_chunk=[],
            )

    def test_periodic_refresh_cadence_is_query_based(self) -> None:
        expected = {
            1: [True, True, True, True, True, True, True],
            2: [True, False, True, False, True, False, True],
            3: [True, False, False, True, False, False, True],
        }
        for period, expected_outcomes in expected.items():
            with self.subTest(period=period):
                controller = PeriodicRefreshController(period=period)
                controller.reset(CONTEXT)
                outcomes = []
                for index in range(7):
                    decision = controller.decide(
                        images={},
                        state=None,
                        cache_available=index > 0,
                        cache_age=0,
                    )
                    outcomes.append(decision.refresh)
                    controller.observe(
                        decision=decision,
                        images={},
                        state=None,
                        action_chunk=[],
                    )
                self.assertEqual(outcomes, expected_outcomes)

    def test_visual_only_ignores_state_and_action(self) -> None:
        controller = VisualOnlyRefreshController(
            image_threshold=0.1,
            max_reuse_horizon=4,
        )
        controller.reset(CONTEXT)
        first = controller.decide(
            images=BLACK_IMAGES,
            state=None,
            cache_available=False,
            cache_age=0,
        )
        controller.observe(
            decision=first,
            images=BLACK_IMAGES,
            state=None,
            action_chunk=None,
        )
        second = controller.decide(
            images=BLACK_IMAGES,
            state=float("nan"),
            cache_available=True,
            cache_age=0,
        )
        self.assertFalse(second.refresh)
        self.assertIsNone(second.state_score)
        self.assertIsNone(second.action_score)

    def test_visual_only_reacts_to_either_camera(self) -> None:
        for changed_images in (WHITE_THIRD_IMAGES, WHITE_WRIST_IMAGES):
            controller = VisualOnlyRefreshController(
                image_threshold=0.1,
                max_reuse_horizon=4,
            )
            controller.reset(CONTEXT)
            first = controller.decide(
                images=BLACK_IMAGES,
                state=STATE,
                cache_available=False,
                cache_age=0,
            )
            controller.observe(
                decision=first,
                images=BLACK_IMAGES,
                state=STATE,
                action_chunk=ACTION,
            )
            changed = controller.decide(
                images=changed_images,
                state=STATE,
                cache_available=True,
                cache_age=0,
            )
            self.assertTrue(changed.refresh)
            self.assertIn("image_change", changed.triggers)

    def test_horizon_allows_exact_number_of_reuses(self) -> None:
        controller = VisualOnlyRefreshController(
            image_threshold=1.0,
            max_reuse_horizon=2,
        )
        controller.reset(CONTEXT)
        first = controller.decide(
            images=BLACK_IMAGES,
            state=STATE,
            cache_available=False,
            cache_age=0,
        )
        controller.observe(
            decision=first,
            images=BLACK_IMAGES,
            state=STATE,
            action_chunk=ACTION,
        )
        reuse_one = controller.decide(
            images=BLACK_IMAGES,
            state=STATE,
            cache_available=True,
            cache_age=0,
        )
        controller.observe(
            decision=reuse_one,
            images=BLACK_IMAGES,
            state=STATE,
            action_chunk=ACTION,
        )
        reuse_two = controller.decide(
            images=BLACK_IMAGES,
            state=STATE,
            cache_available=True,
            cache_age=1,
        )
        controller.observe(
            decision=reuse_two,
            images=BLACK_IMAGES,
            state=STATE,
            action_chunk=ACTION,
        )
        refresh = controller.decide(
            images=BLACK_IMAGES,
            state=STATE,
            cache_available=True,
            cache_age=2,
        )
        self.assertFalse(reuse_one.refresh)
        self.assertFalse(reuse_two.refresh)
        self.assertTrue(refresh.refresh)
        self.assertIn("max_reuse_horizon", refresh.triggers)

    def test_savr_action_history_warmup_and_trigger(self) -> None:
        controller = StateAwareVisualRefreshController(
            image_threshold=1.0,
            state_threshold=1.0,
            action_threshold=0.1,
            max_reuse_horizon=8,
            state_q01=[0.0] * 8,
            state_q99=[1.0] * 8,
            action_q01=[0.0, 0.0],
            action_q99=[1.0, 1.0],
        )
        controller.reset(CONTEXT)

        first = controller.decide(
            images=BLACK_IMAGES,
            state=STATE,
            cache_available=False,
            cache_age=0,
        )
        self.assertIn("action_history_warmup", first.triggers)
        controller.observe(
            decision=first,
            images=BLACK_IMAGES,
            state=STATE,
            action_chunk=[[0.0, 0.0]],
        )

        second = controller.decide(
            images=BLACK_IMAGES,
            state=STATE,
            cache_available=True,
            cache_age=0,
        )
        self.assertIn("action_history_warmup", second.triggers)
        controller.observe(
            decision=second,
            images=BLACK_IMAGES,
            state=STATE,
            action_chunk=[[1.0, 1.0]],
        )

        third = controller.decide(
            images=BLACK_IMAGES,
            state=STATE,
            cache_available=True,
            cache_age=0,
        )
        self.assertTrue(third.refresh)
        self.assertIn("action_change", third.triggers)
        self.assertAlmostEqual(third.action_score or 0.0, 2.0)

    def test_savr_image_state_and_overlap_truth_table(self) -> None:
        def primed_controller():
            controller = StateAwareVisualRefreshController(
                image_threshold=0.1,
                state_threshold=0.1,
                action_threshold=0.1,
                max_reuse_horizon=8,
                state_q01=[0.0] * 8,
                state_q99=[1.0] * 8,
                action_q01=[0.0, 0.0],
                action_q99=[1.0, 1.0],
            )
            controller.reset(CONTEXT)
            for action in ([[0.0, 0.0]], [[0.0, 0.0]]):
                decision = controller.decide(
                    images=BLACK_IMAGES,
                    state=[0.0] * 8,
                    cache_available=controller.query_index > 0,
                    cache_age=0,
                )
                controller.observe(
                    decision=decision,
                    images=BLACK_IMAGES,
                    state=[0.0] * 8,
                    action_chunk=action,
                )
            return controller

        no_trigger = primed_controller().decide(
            images=BLACK_IMAGES,
            state=[0.0] * 8,
            cache_available=True,
            cache_age=0,
        )
        self.assertFalse(no_trigger.refresh)

        image_only = primed_controller().decide(
            images=WHITE_THIRD_IMAGES,
            state=[0.0] * 8,
            cache_available=True,
            cache_age=0,
        )
        self.assertEqual(image_only.triggers, ("image_change",))

        state_only = primed_controller().decide(
            images=BLACK_IMAGES,
            state=[1.0] * 8,
            cache_available=True,
            cache_age=0,
        )
        self.assertEqual(state_only.triggers, ("state_change",))

        overlap = primed_controller().decide(
            images=WHITE_THIRD_IMAGES,
            state=[1.0] * 8,
            cache_available=True,
            cache_age=8,
        )
        self.assertEqual(
            overlap.triggers,
            ("image_change", "max_reuse_horizon", "state_change"),
        )

    def test_invalid_savr_state_forces_refresh(self) -> None:
        controller = StateAwareVisualRefreshController(
            image_threshold=1.0,
            state_threshold=1.0,
            action_threshold=1.0,
            max_reuse_horizon=8,
            state_q01=[0.0] * 8,
            state_q99=[1.0] * 8,
            action_q01=[0.0],
            action_q99=[1.0],
        )
        controller.reset(CONTEXT)
        first = controller.decide(
            images=BLACK_IMAGES,
            state=STATE,
            cache_available=False,
            cache_age=0,
        )
        controller.observe(
            decision=first,
            images=BLACK_IMAGES,
            state=STATE,
            action_chunk=[[0.0]],
        )
        invalid = controller.decide(
            images=BLACK_IMAGES,
            state=[float("nan")] * 8,
            cache_available=True,
            cache_age=0,
        )
        self.assertTrue(invalid.refresh)
        self.assertIn("invalid_state", invalid.triggers)

    def test_invalid_image_state_and_action_shapes_force_refresh(self) -> None:
        def controller() -> StateAwareVisualRefreshController:
            value = StateAwareVisualRefreshController(
                image_threshold=1.0,
                state_threshold=1.0,
                action_threshold=1.0,
                max_reuse_horizon=8,
                state_q01=[0.0] * 8,
                state_q99=[1.0] * 8,
                action_q01=[0.0, 0.0],
                action_q99=[1.0, 1.0],
            )
            value.reset(CONTEXT)
            return value

        for invalid_images in (
            {},
            {"third_person": [[[float("nan")]]], "wrist": [[[0.0]]]},
            {"third_person": [[[0.0]]]},
        ):
            with self.subTest(images=invalid_images):
                value = controller()
                first = value.decide(
                    images=BLACK_IMAGES,
                    state=STATE,
                    cache_available=False,
                    cache_age=0,
                )
                value.observe(
                    decision=first,
                    images=BLACK_IMAGES,
                    state=STATE,
                    action_chunk=ACTION,
                )
                invalid = value.decide(
                    images=invalid_images,
                    state=STATE,
                    cache_available=True,
                    cache_age=0,
                )
                self.assertTrue(invalid.refresh)
                self.assertIn("invalid_image", invalid.triggers)

        for invalid_state in (None, [0.0] * 7, [float("inf")] * 8):
            with self.subTest(state=invalid_state):
                value = controller()
                first = value.decide(
                    images=BLACK_IMAGES,
                    state=STATE,
                    cache_available=False,
                    cache_age=0,
                )
                value.observe(
                    decision=first,
                    images=BLACK_IMAGES,
                    state=STATE,
                    action_chunk=ACTION,
                )
                invalid = value.decide(
                    images=BLACK_IMAGES,
                    state=invalid_state,
                    cache_available=True,
                    cache_age=0,
                )
                self.assertTrue(invalid.refresh)
                self.assertIn("invalid_state", invalid.triggers)

        value = controller()
        for action in ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]):
            decision = value.decide(
                images=BLACK_IMAGES,
                state=STATE,
                cache_available=value.query_index > 0,
                cache_age=0,
            )
            value.observe(
                decision=decision,
                images=BLACK_IMAGES,
                state=STATE,
                action_chunk=action,
            )
        invalid_action = value.decide(
            images=BLACK_IMAGES,
            state=STATE,
            cache_available=True,
            cache_age=0,
        )
        self.assertTrue(invalid_action.refresh)
        self.assertIn("invalid_action_history", invalid_action.triggers)

    def test_savr_overlap_can_include_every_dynamic_trigger(self) -> None:
        controller = StateAwareVisualRefreshController(
            image_threshold=0.1,
            state_threshold=0.1,
            action_threshold=0.1,
            max_reuse_horizon=2,
            state_q01=[0.0] * 8,
            state_q99=[1.0] * 8,
            action_q01=[0.0, 0.0],
            action_q99=[1.0, 1.0],
        )
        controller.reset(CONTEXT)
        for action in ([[0.0, 0.0]], [[1.0, 1.0]]):
            decision = controller.decide(
                images=BLACK_IMAGES,
                state=[0.0] * 8,
                cache_available=controller.query_index > 0,
                cache_age=0,
            )
            controller.observe(
                decision=decision,
                images=BLACK_IMAGES,
                state=[0.0] * 8,
                action_chunk=action,
            )
        overlap = controller.decide(
            images=WHITE_WRIST_IMAGES,
            state=[1.0] * 8,
            cache_available=True,
            cache_age=2,
        )
        self.assertEqual(
            overlap.triggers,
            (
                "image_change",
                "max_reuse_horizon",
                "state_change",
                "action_change",
            ),
        )


if __name__ == "__main__":
    unittest.main()
