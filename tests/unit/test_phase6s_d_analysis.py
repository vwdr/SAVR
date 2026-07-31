from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "analyze_phase6s_d", ROOT / "scripts" / "analyze_phase6s_d.py"
)
assert SPEC and SPEC.loader
ANALYZER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYZER)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class Phase6SDAnalysisTests(unittest.TestCase):
    def make_run(self, root: Path, *, fail_last: bool = False) -> None:
        write_json(root / "manifest.json", {"status": "completed", "policy": "SAVR3", "run_id": "phase6s-d-validation-v1"})
        write_json(root / "run_summary.json", {"status": "completed", "checkpoint_restored": True, "unexpected_new_checkpoint_files": []})
        global_query = 0
        for task in range(10):
            for state in range(3, 10):
                episode_id = f"savr3-rv-w375-b15_task_{task:02d}_state_{state:02d}"
                success = not (fail_last and task == 9 and state == 9)
                write_json(
                    root / "episodes" / f"{episode_id}.json",
                    {"episode_id": episode_id, "configuration_id": "savr3-rv-w375-b15", "policy": "SAVR3", "task": f"libero_spatial:{task}", "initial_state_id": state, "status": "completed", "success": success, "query_count": 7},
                )
                for index in range(7):
                    reuse = index == 6
                    count = 0 if reuse else 1
                    write_json(
                        root / "queries" / f"query_{global_query:08d}.json",
                        {"episode_id": episode_id, "episode_query_index": index, "refresh": not reuse, "decision": {"policy": "SAVR3", "translation_direction_reversals": [False, False, False]}, "timing": {"component_counts": {"vision_backbone": count, "visual_projector": count, "language_model": 1, "action_head": 1}}},
                    )
                    global_query += 1

    def test_positive_gate_requires_all_frozen_conditions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_run(root)
            result = ANALYZER.analyze(root)
            self.assertTrue(result["positive_method_result"])
            self.assertEqual(result["successes"], 70)
            self.assertEqual(result["reuses"], 70)

    def test_one_failure_makes_result_negative(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_run(root, fail_last=True)
            result = ANALYZER.analyze(root)
            self.assertFalse(result["positive_method_result"])
            self.assertFalse(result["gates"]["successes_70"])


if __name__ == "__main__":
    unittest.main()
