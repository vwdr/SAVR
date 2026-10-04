"""Synthetic completion, tamper, and reconciliation tests; no experimental data."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("attention_analysis", ROOT / "scripts/analyze_openvla_original_baseline.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="attention-analysis-test-", dir=ROOT / "tmp")
        self.root = Path(self.temp.name)
        suites = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
        conditions = [dict(condition_id=str(i), suite=suites[i // 10], initial_state_id=0, seed=7,
                           task_id=f"task_{i}", arm_order=["original"])
                      for i in range(40)]
        self.config = {"schema_version": "openvla-original-baseline-40task-v1", "suites": suites, "episode_conditions": conditions}
        self.config_path = self.root / "config.json"
        self.dump("config.json", self.config)
        self.dump("launch.json", {"config": self.config,
                                   "config_sha256": hashlib.sha256(self.config_path.read_bytes()).hexdigest()})
        rows = [{"observation_id": str(i), "suite": suites[i // 2],
                 "parity_max_abs": {"actions": 0.0}, "restoration_max_abs": {"actions": 0.0},
                 "original_masks": [dict(layer=j, effective_mode="original", reference_is_causal=False) for j in range(32)],
                 "causal_masks": [dict(layer=j, effective_mode="causal_control", reference_is_causal=False) for j in range(32)],
                 "causal_vs_original_normalized_max_abs": 0.2} for i in range(8)]
        self.offline = {"complete": True, "queries": 32, "records": rows}
        self.episodes = {"complete": True, "records": [dict(c, mode=mode, success=mode == "original",
                                                            executed_steps=8, policy_queries=1, initial_state_sha256="a"*64)
                                                       for c in conditions for mode in c["arm_order"]]}
        self.summary = {"complete": True, "episode_count": 40, "model_queries": 72,
                        "peak_aggregate_gpu_memory_mib": 16000, "elapsed_seconds": 100,
                        "authenticated_files_unchanged": True, "checkpoint_unchanged": True, "training_performed": False,
                        "successes": {"original": 40}}
        self.refresh()

    def tearDown(self):
        self.temp.cleanup()

    def dump(self, name, value):
        (self.root / name).write_text(json.dumps(value))

    def refresh(self):
        self.dump("reference_check.json", self.offline)
        self.dump("episodes.json", self.episodes)
        self.dump("loaded_runtime.json", {"transformers_version": "4.40.1", "attention_classes": ["LlamaSdpaAttention"] * 32, "checkpoint_metadata_mutation_disabled": True, "tensorflow_gpu_disabled": True})
        self.summary["artifact_sha256"] = {n: hashlib.sha256((self.root / n).read_bytes()).hexdigest()
                                            for n in ("reference_check.json", "episodes.json", "loaded_runtime.json")}
        self.dump("worker_summary.json", self.summary)

    def test_complete_fixture_reconciles(self):
        result = MODULE.reconcile(self.root, self.config_path)
        self.assertEqual(result["successes"]["original"], 40)
        self.assertFalse(result["positive_method_result"])

    def test_no_summary_blocks_analysis(self):
        (self.root / "worker_summary.json").unlink()
        with self.assertRaises(FileNotFoundError):
            MODULE.reconcile(self.root, self.config_path)

    def test_incomplete_summary_blocks_analysis(self):
        self.summary["complete"] = False
        self.refresh()
        with self.assertRaisesRegex(ValueError, "not complete"):
            MODULE.reconcile(self.root, self.config_path)

    def test_technical_stop_blocks_analysis(self):
        self.dump("technical_stop.json", {"complete": False})
        with self.assertRaisesRegex(ValueError, "technical stop"):
            MODULE.reconcile(self.root, self.config_path)

    def test_tamper_is_rejected(self):
        self.dump("episodes.json", {"complete": True, "records": []})
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            MODULE.reconcile(self.root, self.config_path)

    def test_missing_episode_is_rejected(self):
        self.episodes["records"].pop()
        self.refresh()
        with self.assertRaisesRegex(ValueError, "episode count"):
            MODULE.reconcile(self.root, self.config_path)

    def test_duplicate_pair_is_rejected(self):
        self.episodes["records"][1] = copy.deepcopy(self.episodes["records"][0])
        self.refresh()
        with self.assertRaisesRegex(ValueError, "schedule"):
            MODULE.reconcile(self.root, self.config_path)

    def test_query_accounting_is_checked(self):
        self.summary["model_queries"] = 49
        self.refresh()
        with self.assertRaisesRegex(ValueError, "query counts"):
            MODULE.reconcile(self.root, self.config_path)

    def test_parity_failure_is_rejected(self):
        self.offline["records"][0]["parity_max_abs"]["actions"] = 0.1
        self.refresh()
        with self.assertRaisesRegex(ValueError, "reference check"):
            MODULE.reconcile(self.root, self.config_path)

    def test_wrong_success_summary_is_rejected(self):
        self.summary["successes"]["original"] = 7
        self.refresh()
        with self.assertRaisesRegex(ValueError, "success aggregation"):
            MODULE.reconcile(self.root, self.config_path)

    def test_invalid_state_hash_is_rejected(self):
        self.episodes["records"][0]["initial_state_sha256"] = "not-a-hash"
        self.refresh()
        with self.assertRaisesRegex(ValueError, "state hash"):
            MODULE.reconcile(self.root, self.config_path)

    def test_runtime_contract_is_checked(self):
        self.dump("loaded_runtime.json", {"transformers_version": "4.47.0"})
        self.summary["artifact_sha256"]["loaded_runtime.json"] = hashlib.sha256((self.root / "loaded_runtime.json").read_bytes()).hexdigest()
        self.dump("worker_summary.json", self.summary)
        with self.assertRaisesRegex(ValueError, "runtime contract"):
            MODULE.reconcile(self.root, self.config_path)


if __name__ == "__main__":
    unittest.main()

