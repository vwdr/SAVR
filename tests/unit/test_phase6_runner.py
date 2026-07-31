from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

SPEC = importlib.util.spec_from_file_location(
    "run_phase6_calibration",
    ROOT / "scripts" / "run_phase6_calibration.py",
)
assert SPEC and SPEC.loader
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class Phase6RunnerTests(unittest.TestCase):
    def test_frozen_fr_config(self) -> None:
        config = RUNNER.load_config(
            ROOT / "configs" / "calibration" / "phase6_fr_signals.json"
        )
        self.assertEqual(config["run_id"], "phase6-fr-signals-v1")
        self.assertEqual(config["settings"], [{"configuration_id": "fr", "policy": "FR"}])
        self.assertEqual(RUNNER.EXPECTED_PAIRINGS, 100)
        self.assertEqual(RUNNER.INITIAL_STATE_IDS, tuple(range(10)))

    def test_rejects_invalid_threshold(self) -> None:
        config = {
            "protocol": "PHASE6_CALIBRATION_PROTOCOL.md",
            "run_id": "bad",
            "settings": [
                {
                    "configuration_id": "bad",
                    "policy": "SAVR",
                    "image_threshold": -1,
                    "state_threshold": 0,
                    "action_threshold": 0,
                    "max_reuse_horizon": 2,
                }
            ],
            "wall_cap_seconds": 1,
            "artifact_cap_bytes": 1,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "image_threshold"):
                RUNNER.load_config(path)

    def test_component_invariants_follow_effective_decision(self) -> None:
        result = SimpleNamespace(
            decision=SimpleNamespace(refresh=False),
            cache_event="reuse",
        )
        timing = SimpleNamespace(
            component_counts={
                "vision_backbone": 0,
                "visual_projector": 0,
                "language_model": 1,
                "action_head": 1,
            }
        )
        RUNNER.assert_component_invariants(result, timing)
        timing.component_counts["vision_backbone"] = 1
        with self.assertRaisesRegex(RuntimeError, "Vision count"):
            RUNNER.assert_component_invariants(result, timing)

    def test_progress_is_machine_readable(self) -> None:
        records = {
            "a": {"status": "completed"},
            "b": {"status": "failed"},
        }
        progress = RUNNER.progress_summary(
            records=records,
            settings=[{"configuration_id": "fr", "policy": "FR"}],
            elapsed=10.0,
        )
        self.assertEqual(progress["expected"], 100)
        self.assertEqual(progress["terminal"], 2)
        self.assertEqual(progress["remaining"], 98)
        self.assertEqual(progress["estimated_remaining_seconds"], 490.0)


if __name__ == "__main__":
    unittest.main()
