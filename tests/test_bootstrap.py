from __future__ import annotations

import hashlib
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

    def test_manuscript_source_is_recorded(self) -> None:
        text = (ROOT / "manuscript" / "README.md").read_text(encoding="utf-8")
        manuscript = ROOT / "manuscript" / (
            "State-Aware Visual Refresh for Efficient VLA Inference.tex"
        )
        self.assertTrue(manuscript.is_file())
        digest = hashlib.sha256(manuscript.read_bytes()).hexdigest()
        self.assertEqual(
            digest,
            "4a0fe130f1cbc5557f77a518dcb65a703a647b1c4b8091499d8bfd8e10ab6e4f",
        )
        self.assertIn("4a0fe130f1cbc5557f77a518dcb65a703a647b1c4b8091499d8bfd8e10ab6e4f", text)
        self.assertIn("Do not edit the manuscript", text)

    def test_project_status_does_not_claim_results(self) -> None:
        text = (ROOT / "PROJECT_STATUS.md").read_text(encoding="utf-8")
        self.assertIn("no empirical claim is supported yet", text)


if __name__ == "__main__":
    unittest.main()
