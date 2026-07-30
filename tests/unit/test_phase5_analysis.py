from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "analyze_phase5_smoke",
    ROOT / "scripts" / "analyze_phase5_smoke.py",
)
assert SPEC is not None and SPEC.loader is not None
ANALYZER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYZER)


class Phase5AnalysisTests(unittest.TestCase):
    def test_percentile_is_deterministic_and_interpolated(self) -> None:
        self.assertEqual(ANALYZER.percentile([], 95), 0.0)
        self.assertEqual(ANALYZER.percentile([3.0], 95), 3.0)
        self.assertEqual(ANALYZER.percentile([0.0, 10.0], 50), 5.0)

    def test_episode_matrix_rejects_reordering_and_count_mismatch(self) -> None:
        episodes = []
        for state_id, policy in ANALYZER.SCHEDULE:
            episodes.append(
                {
                    "episode_id": f"{state_id}-{policy}",
                    "initial_state_id": state_id,
                    "policy": policy,
                    "status": "completed",
                    "query_count": 2,
                    "refresh_count": 1,
                    "reuse_count": 1,
                    "skipped_refresh_count": 1,
                    "refresh_rate": 0.5,
                    "trajectory_sha256": "a" * 64,
                }
            )
        ANALYZER.validate_episode_matrix(episodes)

        episodes[0], episodes[1] = episodes[1], episodes[0]
        with self.assertRaisesRegex(RuntimeError, "Episode order differs"):
            ANALYZER.validate_episode_matrix(episodes)

        episodes[0], episodes[1] = episodes[1], episodes[0]
        episodes[0]["reuse_count"] = 2
        with self.assertRaisesRegex(RuntimeError, "Episode count mismatch"):
            ANALYZER.validate_episode_matrix(episodes)


if __name__ == "__main__":
    unittest.main()
