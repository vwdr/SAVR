"""Population-selection regression tests, requiring no model or accelerator."""
import copy
import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("baseline_worker", ROOT / "scripts/run_openvla_original_baseline.py")
WORKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(WORKER)


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.suites = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
        self.rows = [dict(suite=s, population="headroom_stage1", initial_state_id=0,
                          task_id=f"task_{i:02}", seed=7, condition_id=f"{s}-{i}")
                     for s in self.suites for i in reversed(range(10))]

    def test_all_tasks_original_only_sorted(self):
        result = WORKER.select_conditions(self.rows, self.suites)
        self.assertEqual(len(result), 40)
        self.assertTrue(all(r["arm_order"] == ["original"] for r in result))
        self.assertEqual([r["task_id"] for r in result[:10]], [f"task_{i:02}" for i in range(10)])
        self.assertTrue(all("arm_order" not in r for r in self.rows))

    def test_missing_task_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "ten unique"):
            WORKER.select_conditions(self.rows[1:], self.suites)

    def test_duplicate_task_rejected(self):
        self.rows[1]["task_id"] = self.rows[0]["task_id"]
        with self.assertRaisesRegex(RuntimeError, "ten unique"):
            WORKER.select_conditions(self.rows, self.suites)

    def test_duplicate_condition_rejected(self):
        self.rows[1]["condition_id"] = self.rows[0]["condition_id"]
        with self.assertRaisesRegex(RuntimeError, "duplicate condition"):
            WORKER.select_conditions(self.rows, self.suites)

    def test_wrong_seed_rejected(self):
        self.rows[0]["seed"] = 8
        with self.assertRaisesRegex(RuntimeError, "seed"):
            WORKER.select_conditions(self.rows, self.suites)

    def test_other_states_and_protected_populations_excluded(self):
        extras = copy.deepcopy(self.rows)
        for row in extras:
            row["population"] = "protected"
        other_states = copy.deepcopy(self.rows)
        for row in other_states:
            row["initial_state_id"] = 1
        self.assertEqual(WORKER.select_conditions(self.rows + extras + other_states, self.suites),
                         WORKER.select_conditions(self.rows, self.suites))


if __name__ == "__main__":
    unittest.main()
