from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from savr.signals import (  # noqa: E402
    SignalValidationError,
    action_change,
    image_change,
    prepare_image_representations,
    state_change,
)
import savr.signals as signals_module  # noqa: E402


class SignalTests(unittest.TestCase):
    def test_two_camera_image_change_uses_refresh_reference(self) -> None:
        black = [[[0, 0, 0] for _ in range(2)] for _ in range(2)]
        white = [[[255, 255, 255] for _ in range(2)] for _ in range(2)]
        reference = prepare_image_representations(
            {"third_person": black, "wrist": black}
        )
        result = image_change(
            {"third_person": white, "wrist": black},
            reference,
        )
        self.assertEqual(result.per_camera["third_person"], 1.0)
        self.assertEqual(result.per_camera["wrist"], 0.0)
        self.assertEqual(result.mean, 0.5)

    def test_state_change_uses_q01_q99_normalized_rms(self) -> None:
        score = state_change(
            [10.0] * 8,
            [0.0] * 8,
            [0.0] * 8,
            [10.0] * 8,
        )
        self.assertAlmostEqual(score, 2.0)

    def test_action_change_flattens_chunks_by_action_dimension(self) -> None:
        score = action_change(
            [[10.0, 10.0], [10.0, 10.0]],
            [[0.0, 0.0], [0.0, 0.0]],
            [0.0, 0.0],
            [10.0, 10.0],
        )
        self.assertAlmostEqual(score, 2.0)

    def test_invalid_signal_data_is_rejected(self) -> None:
        with self.assertRaises(SignalValidationError):
            state_change(
                [float("nan")] * 8,
                [0.0] * 8,
                [0.0] * 8,
                [1.0] * 8,
            )
        with self.assertRaises(SignalValidationError):
            prepare_image_representations({"third_person": [[[-1, 0, 0]]]})

    def test_dependency_free_image_path_matches_contract(self) -> None:
        original_numpy = signals_module._np
        signals_module._np = None
        try:
            black = [[[0, 0, 0]]]
            white = [[[255, 255, 255]]]
            reference = prepare_image_representations(
                {"third_person": black, "wrist": black}
            )
            result = image_change(
                {"third_person": white, "wrist": black},
                reference,
            )
            self.assertEqual(result.mean, 0.5)
        finally:
            signals_module._np = original_numpy


if __name__ == "__main__":
    unittest.main()
