from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import run_phase6r_c_correctness as runner  # noqa: E402


class Phase6RCCorrectnessRunnerTests(unittest.TestCase):
    def test_real_model_plan_stays_inside_frozen_bounds(self) -> None:
        self.assertEqual(runner.PLANNED_QUERIES, 10)
        self.assertLessEqual(runner.PLANNED_QUERIES, runner.QUERY_CAP)
        self.assertEqual(runner.QUERY_CAP, 20)
        self.assertEqual(runner.ARTIFACT_CAP_BYTES, 256 * 1024**2)
        self.assertEqual(runner.MAX_WALL_SECONDS, 45 * 60)
        self.assertEqual(runner.RECOVERY_PLANNED_QUERIES, 8)
        self.assertLessEqual(
            runner.PLANNED_QUERIES + runner.RECOVERY_PLANNED_QUERIES,
            runner.QUERY_CAP,
        )
        self.assertEqual(
            runner.RECOVERY_TRACE_SHA256,
            "ff9f4bfc004b861260e36d61c5eab641356a9c27c25f7ceccf511e04dd687a63",
        )

    def test_schemas_accept_savr2_without_removing_existing_policies(self) -> None:
        run_schema = json.loads(
            (ROOT / "schemas" / "run_manifest.schema.json").read_text(encoding="utf-8")
        )
        episode_schema = json.loads(
            (ROOT / "schemas" / "episode_result.schema.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            set(run_schema["properties"]["policy"]["enum"]),
            {"FR", "PR", "VOR", "SAVR", "SAVR2", "SAVR3", "MIXED"},
        )
        self.assertEqual(
            set(episode_schema["properties"]["policy"]["enum"]),
            {"FR", "PR", "VOR", "SAVR", "SAVR2", "SAVR3"},
        )


if __name__ == "__main__":
    unittest.main()
