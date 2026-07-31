from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "analyze_phase6r_a_diagnosis",
    ROOT / "scripts" / "analyze_phase6r_a_diagnosis.py",
)
assert SPEC is not None and SPEC.loader is not None
ANALYZER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYZER)


class Phase6RADiagnosisTests(unittest.TestCase):
    def test_percentile_interpolates(self) -> None:
        self.assertIsNone(ANALYZER.percentile([], 0.5))
        self.assertEqual(ANALYZER.percentile([0.0, 10.0], 0.5), 5.0)
        self.assertEqual(ANALYZER.percentile([3.0], 0.95), 3.0)

    def test_maximum_reuse_streak(self) -> None:
        records = [
            {"refresh": True},
            {"refresh": False},
            {"refresh": False},
            {"refresh": True},
            {"refresh": False},
        ]
        self.assertEqual(ANALYZER.maximum_reuse_streak(records), 2)

    def test_position_bins(self) -> None:
        self.assertEqual(ANALYZER.position_bin(0, 10), "early")
        self.assertEqual(ANALYZER.position_bin(4, 10), "middle")
        self.assertEqual(ANALYZER.position_bin(8, 10), "late")

    def test_episode_identifier_parser(self) -> None:
        self.assertEqual(
            ANALYZER.episode_pairing_from_id("savr-s25-h2_task_09_state_03"),
            (9, 3),
        )


if __name__ == "__main__":
    unittest.main()
