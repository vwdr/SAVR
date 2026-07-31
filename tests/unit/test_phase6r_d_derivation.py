from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from derive_phase6r_d_candidates import (  # noqa: E402
    EpisodeTrace,
    QuerySignals,
    linear_quantile,
    replay,
)


def stable_signals(count: int) -> tuple[QuerySignals, ...]:
    result = []
    for index in range(count):
        result.append(
            QuerySignals(
                visual_lag1=(
                    {"full_image": 0.0, "wrist_image": 0.0}
                    if index >= 1
                    else None
                ),
                visual_lag2=(
                    {"full_image": 0.0, "wrist_image": 0.0}
                    if index >= 2
                    else None
                ),
                state=(
                    {"translation": 0.0, "orientation": 0.0, "gripper": 0.0}
                    if index >= 1
                    else None
                ),
                action=(
                    {"translation": 0.0, "rotation": 0.0, "gripper": 0.0}
                    if index >= 2
                    else None
                ),
                gripper_veto=False,
            )
        )
    return tuple(result)


THRESHOLDS = {
    "image": {"full_image": 0.0, "wrist_image": 0.0},
    "state": {"translation": 0.0, "orientation": 0.0, "gripper": 0.0},
    "action": {"translation": 0.0, "rotation": 0.0, "gripper": 0.0},
}


class Phase6RDDerivationTests(unittest.TestCase):
    def test_linear_quantile_matches_frozen_linear_rule(self) -> None:
        self.assertEqual(linear_quantile([0.0, 10.0], 0.0), 0.0)
        self.assertEqual(linear_quantile([0.0, 10.0], 0.5), 5.0)
        self.assertEqual(linear_quantile([0.0, 10.0], 1.0), 10.0)

    def test_replay_enforces_warmup_prefix_budget_and_isolated_reuse(self) -> None:
        episode = EpisodeTrace("episode", 0, 0, stable_signals(60))
        result = replay([episode], THRESHOLDS, 0.15)
        self.assertLessEqual(result["skip_rate"], 0.15)
        self.assertEqual(result["reuses"], 9)

    def test_gripper_veto_breaks_stable_fresh_recovery(self) -> None:
        signals = list(stable_signals(20))
        original = signals[6]
        signals[6] = QuerySignals(
            visual_lag1=original.visual_lag1,
            visual_lag2=original.visual_lag2,
            state=original.state,
            action=original.action,
            gripper_veto=True,
        )
        episode = EpisodeTrace("episode", 0, 0, tuple(signals))
        vetoed = replay([episode], THRESHOLDS, 0.15)
        stable = replay([EpisodeTrace("stable", 0, 0, stable_signals(20))], THRESHOLDS, 0.15)
        self.assertGreater(
            vetoed["earliest_first_reuse_query"],
            stable["earliest_first_reuse_query"],
        )


if __name__ == "__main__":
    unittest.main()
