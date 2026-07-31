from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from savr.analysis.statistics import (  # noqa: E402
    PairedBinaryCounts,
    paired_binary_counts,
    paired_noninferiority_sample_size,
    planning_power_result,
    power_sensitivity,
    wilson_upper,
)


class StatisticsTests(unittest.TestCase):
    def test_paired_counts_and_difference(self) -> None:
        counts = paired_binary_counts(
            {"a": True, "b": True, "c": False, "d": False},
            {"a": True, "b": False, "c": True, "d": False},
        )
        self.assertEqual(
            counts,
            PairedBinaryCounts(1, 1, 1, 1),
        )
        self.assertEqual(counts.success_difference, 0.0)
        self.assertEqual(counts.discordance_rate, 0.5)

    def test_wilson_upper_is_conservative_for_zero_events(self) -> None:
        upper = wilson_upper(0, 100)
        self.assertGreater(upper, 0.0)
        self.assertLess(upper, 0.05)

    def test_frozen_power_formula(self) -> None:
        required = paired_noninferiority_sample_size(0.03)
        self.assertEqual(required, 789)
        self.assertGreater(
            paired_noninferiority_sample_size(0.04),
            required,
        )

    def test_planning_uses_wilson_upper_and_balanced_rounding(self) -> None:
        result = planning_power_result(PairedBinaryCounts(97, 2, 1, 0))
        self.assertGreaterEqual(
            result["planning_discordance_rate"],
            result["observed_discordance_rate"],
        )
        self.assertGreaterEqual(result["recommended_sample_size"], 1200)
        self.assertEqual(result["recommended_sample_size"] % 400, 0)

    def test_sensitivity_grid_is_complete(self) -> None:
        rows = power_sensitivity()
        self.assertEqual(len(rows), 20)
        self.assertEqual(rows[0]["discordance_rate"], 0.01)
        self.assertEqual(rows[-1]["discordance_rate"], 0.10)


if __name__ == "__main__":
    unittest.main()
