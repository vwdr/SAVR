from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import analyze_phase6s_a as analysis  # noqa: E402


class Phase6SAAnalysisTests(unittest.TestCase):
    def test_search_grid_and_failure_set_are_frozen(self) -> None:
        self.assertEqual(analysis.WRIST_CAPS, (0.300, 0.325, 0.350, 0.375, 0.400))
        self.assertEqual(len(analysis.FAILED_EPISODES), 4)
        self.assertIn("savr2-b10_task_08_state_02", analysis.FAILED_EPISODES)

    def test_canonical_hash_is_order_independent_for_mappings(self) -> None:
        self.assertEqual(
            analysis.sha256_json({"a": 1, "b": 2}),
            analysis.sha256_json({"b": 2, "a": 1}),
        )


if __name__ == "__main__":
    unittest.main()
