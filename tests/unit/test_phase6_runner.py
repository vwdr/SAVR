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
sys.path.insert(0, str(ROOT / "src"))

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

    def test_phase6r_stage1_config_and_controllers_are_frozen(self) -> None:
        config = RUNNER.load_config(
            ROOT / "configs" / "calibration" / "phase6r_d_stage1.json"
        )
        self.assertEqual(config["protocol"], "PHASE6R_PROTOCOL_V1.md")
        self.assertEqual(config["initial_state_ids"], [0, 1, 2])
        self.assertEqual(
            [setting["configuration_id"] for setting in config["settings"]],
            ["savr2-b05", "savr2-b10", "savr2-b15"],
        )
        for setting in config["settings"]:
            controller = RUNNER.controller_for_setting(
                setting,
                state_statistics={"q01": [0.0] * 8, "q99": [1.0] * 8},
                action_statistics={"q01": [0.0] * 7, "q99": [1.0] * 7},
                controllers=SimpleNamespace(),
            )
            self.assertEqual(
                controller.configuration.configuration_id,
                setting["configuration_id"],
            )

    def test_stage1_progress_uses_thirty_pairings_per_candidate(self) -> None:
        progress = RUNNER.progress_summary(
            records={},
            settings=[{}, {}, {}],
            elapsed=0.0,
            expected_pairings=30,
        )
        self.assertEqual(progress["expected"], 90)

    def test_savr2_episode_invariants_enforce_prefix_and_isolation(self) -> None:
        setting = {"policy": "SAVR2", "skip_budget": 0.15}
        valid = [{"refresh": True} for _ in range(6)] + [{"refresh": False}]
        RUNNER.assert_savr2_episode_invariants(valid, setting)
        with self.assertRaisesRegex(RuntimeError, "consecutive"):
            RUNNER.assert_savr2_episode_invariants(
                valid + [{"refresh": False}], setting
            )
        with self.assertRaisesRegex(RuntimeError, "prefix"):
            RUNNER.assert_savr2_episode_invariants(
                [{"refresh": True} for _ in range(5)] + [{"refresh": False}],
                {"policy": "SAVR2", "skip_budget": 0.10},
            )

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
