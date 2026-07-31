from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from savr.signals import (  # noqa: E402
    SignalValidationError,
    action_transition,
    grouped_action_change,
    grouped_state_change,
    patch_image_change,
    prepare_image_representations,
)


def image(value: int = 0) -> list[list[list[int]]]:
    return [[[value, value, value] for _ in range(32)] for _ in range(32)]


class SAVR2SignalTests(unittest.TestCase):
    def test_patch_signal_detects_local_change_without_camera_averaging(self) -> None:
        black = image()
        changed = image()
        for row in range(4):
            for column in range(4):
                changed[row][column] = [255, 255, 255]
        reference = prepare_image_representations(
            {"full_image": black, "wrist_image": black}
        )
        result = patch_image_change(
            {"full_image": black, "wrist_image": changed},
            reference,
        )
        self.assertEqual(len(result.per_camera["wrist_image"].patch_scores), 64)
        self.assertAlmostEqual(result.per_camera["wrist_image"].top_k_mean, 0.25)
        self.assertAlmostEqual(result.per_camera["wrist_image"].global_mean, 1 / 64)
        self.assertEqual(result.per_camera["full_image"].top_k_mean, 0.0)

    def test_grouped_state_scores_remain_independent(self) -> None:
        result = grouped_state_change(
            [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0] * 8,
            [0.0] * 8,
            [1.0] * 8,
        )
        self.assertAlmostEqual(result.scores["translation"], 2.0)
        self.assertEqual(result.scores["orientation"], 0.0)
        self.assertEqual(result.scores["gripper"], 0.0)

    def test_grouped_action_scores_use_all_eight_actions(self) -> None:
        newer = [[1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0] for _ in range(8)]
        older = [[0.0] * 7 for _ in range(8)]
        result = grouped_action_change(newer, older, [0.0] * 7, [1.0] * 7)
        self.assertAlmostEqual(result.scores["translation"], 2.0)
        self.assertEqual(result.scores["rotation"], 0.0)
        self.assertAlmostEqual(result.scores["gripper"], 2.0)

    def test_gripper_transition_and_direction_reversal_are_exact(self) -> None:
        older = [[-0.2, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0] for _ in range(8)]
        newer = [[0.2, 0.3, 0.0, 0.0, 0.0, 0.0, float(index >= 4)] for index in range(8)]
        result = action_transition(newer, older)
        self.assertTrue(result.mixed_latest_gripper)
        self.assertTrue(result.final_gripper_changed)
        self.assertTrue(result.gripper_veto)
        self.assertEqual(result.translation_direction_reversals, (True, False, False))

    def test_invalid_patch_and_action_shapes_fail_closed(self) -> None:
        reference = prepare_image_representations({"full_image": image()})
        with self.assertRaises(SignalValidationError):
            patch_image_change(
                {"full_image": image()},
                reference,
                grid_size=7,
            )
        with self.assertRaises(SignalValidationError):
            grouped_action_change([[0.0] * 7], [[0.0] * 7], [0.0] * 7, [1.0] * 7)


if __name__ == "__main__":
    unittest.main()
