from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class BootstrapTests(unittest.TestCase):
    def test_schemas_are_valid_json_objects(self) -> None:
        for name in ("episode_result.schema.json", "run_manifest.schema.json"):
            data = json.loads((ROOT / "schemas" / name).read_text(encoding="utf-8"))
            self.assertEqual(data["type"], "object")
            self.assertTrue(data["required"])

    def test_manuscript_gap_is_explicit(self) -> None:
        text = (ROOT / "manuscript" / "README.md").read_text(encoding="utf-8")
        self.assertIn("not available", text)
        self.assertIn("Do not reconstruct or invent", text)

    def test_project_status_does_not_claim_results(self) -> None:
        text = (ROOT / "PROJECT_STATUS.md").read_text(encoding="utf-8")
        self.assertIn("no empirical claim is supported yet", text)


if __name__ == "__main__":
    unittest.main()
