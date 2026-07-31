from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from savr.calibration import (  # noqa: E402
    CalibrationQuery,
    SignalBounds,
    derive_savr_candidate,
    derive_vor_candidates,
    empirical_quantile,
    query_from_record,
    prepare_episodes,
    replay_episode,
    replay_prepared_episode,
    select_period,
)


BOUNDS = SignalBounds(
    state_q01=(0.0,) * 8,
    state_q99=(1.0,) * 8,
    action_q01=(0.0,) * 7,
    action_q99=(1.0,) * 7,
)


def trace(value: float, action: float | None = None) -> CalibrationQuery:
    return CalibrationQuery(
        images={
            "full_image": (value,) * (32 * 32 * 3),
            "wrist_image": (value,) * (32 * 32 * 3),
        },
        image_shapes={
            "full_image": (32, 32, 3),
            "wrist_image": (32, 32, 3),
        },
        state=(value,) * 8,
        actions=((value if action is None else action),) * 56,
    )


class CalibrationTests(unittest.TestCase):
    def test_query_record_validation(self) -> None:
        record = {
            "calibration_trace": {
                "images": {
                    "full_image": [0.0] * (32 * 32 * 3),
                    "wrist_image": [1.0] * (32 * 32 * 3),
                },
                "image_shapes": {
                    "full_image": [32, 32, 3],
                    "wrist_image": [32, 32, 3],
                },
                "state": [0.5] * 8,
                "actions": [0.25] * 56,
            }
        }
        parsed = query_from_record(record)
        self.assertEqual(len(parsed.images["full_image"]), 3072)
        self.assertEqual(len(parsed.actions), 56)

    def test_linear_quantile(self) -> None:
        self.assertEqual(empirical_quantile([0.0, 10.0], 0.0), 0.0)
        self.assertEqual(empirical_quantile([0.0, 10.0], 0.5), 5.0)
        self.assertEqual(empirical_quantile([0.0, 10.0], 1.0), 10.0)

    def test_savr_warmup_and_horizon_match_controller_semantics(self) -> None:
        episode = [trace(0.0) for _ in range(7)]
        replay = replay_episode(
            episode,
            image_threshold=1.0,
            state_threshold=1.0,
            action_threshold=1.0,
            max_reuse_horizon=2,
            bounds=BOUNDS,
        )
        self.assertEqual(
            replay.decisions,
            (True, True, False, False, True, False, False),
        )

    def test_image_reference_updates_only_on_refresh(self) -> None:
        episode = [trace(value) for value in (0.0, 0.1, 0.2, 0.3)]
        replay = replay_episode(
            episode,
            image_threshold=0.15,
            max_reuse_horizon=8,
            bounds=BOUNDS,
        )
        self.assertEqual(replay.decisions, (True, False, True, False))

    def test_precomputed_replay_matches_production_signal_replay(self) -> None:
        episode = [
            trace(value, action=action)
            for value, action in (
                (0.0, 0.0),
                (0.1, 0.0),
                (0.2, 0.5),
                (0.2, 0.5),
                (0.6, 0.2),
                (0.6, 0.2),
            )
        ]
        expected = replay_episode(
            episode,
            image_threshold=0.15,
            state_threshold=0.3,
            action_threshold=0.4,
            max_reuse_horizon=2,
            bounds=BOUNDS,
        )
        actual = replay_prepared_episode(
            prepare_episodes([episode], BOUNDS)[0],
            image_threshold=0.15,
            state_threshold=0.3,
            action_threshold=0.4,
            max_reuse_horizon=2,
        )
        self.assertEqual(actual, expected)

    def test_candidate_search_is_deterministic_and_conservative(self) -> None:
        episodes = [
            [trace(value, action=index / 10) for index, value in enumerate(values)]
            for values in (
                (0.0, 0.0, 0.0, 0.0, 0.0),
                (0.0, 0.1, 0.2, 0.3, 0.4),
            )
        ]
        first = derive_savr_candidate(
            episodes,
            bounds=BOUNDS,
            target_skip_rate=0.5,
            max_reuse_horizon=4,
            quantile_step=10,
        )
        second = derive_savr_candidate(
            episodes,
            bounds=BOUNDS,
            target_skip_rate=0.5,
            max_reuse_horizon=4,
            quantile_step=10,
        )
        self.assertEqual(first, second)
        self.assertLessEqual(first.simulated_skip_rate, 0.5)

    def test_vor_ranking_and_period_selection(self) -> None:
        episodes = [[trace(value) for value in (0.0, 0.1, 0.2, 0.3, 0.4)]]
        candidates = derive_vor_candidates(
            episodes,
            bounds=BOUNDS,
            target_refresh_rate=0.5,
            max_reuse_horizon=4,
            quantile_step=10,
        )
        self.assertLessEqual(
            abs(candidates[0].simulated_refreshes / 5 - 0.5),
            abs(candidates[-1].simulated_refreshes / 5 - 0.5),
        )
        self.assertEqual(select_period([5, 5], target_refresh_rate=0.4), (3, 0.4))


if __name__ == "__main__":
    unittest.main()
