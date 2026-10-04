"""CPU tests of frozen qualification and fail-closed reconciliation."""
import copy
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import run_specprune_qualification as worker


def fixture():
    cfg = dict(schema_version="specprune-real-qualification-v1", modes=list(worker.MODES), caps=dict(worker.CAPS),
               tolerance=1e-6, observations=8, reference_config=worker.BASE_CONFIG, reference_config_sha256=worker.BASE_SHA,
               output_root="results/specprune-real-qualification-v01", compression_preset="source-default-coarse-first-query",
               automatic_retry=False, observation_ids=list(range(8)))
    summary = dict(complete=True, model_queries=40, checkpoint_unchanged=True, authenticated_files_unchanged=True,
                   training_performed=False, simulator_episodes=0, automatic_retry=False, positive_method_result=False,
                   peak_aggregate_gpu_memory_mib=16000, elapsed_seconds=100)
    rows = []
    for i in range(8):
        modes = {}
        for mode in worker.MODES:
            item = dict(layers=32, finite=True)
            if mode in ("disabled", "keep_all", "restored"):
                item["parity_max_abs"] = dict(hidden=0., normalized=0., actions=0.)
            if mode in ("disabled", "keep_all", "compressed"):
                item["positions"] = list(range(605)) if mode != "compressed" else [0] + list(range(200, 605))
                item["action_positions"] = list(range(547, 603))
            modes[mode] = item
        rows.append(dict(observation_id=i, mode_order=list(worker.MODES), modes=modes, full_tokens=605))
    return cfg, summary, rows


class QualificationTests(unittest.TestCase):
    def test_valid_complete_fixture(self):
        self.assertTrue(worker.reconcile(*fixture()))

    def test_contract_drift(self):
        for field, value in (("tolerance", 1e-3), ("observations", 7), ("automatic_retry", True), ("modes", ["native"])):
            cfg, _, _ = fixture(); cfg[field] = value
            with self.assertRaises(ValueError): worker.validate_contract(cfg)

    def test_summary_must_be_complete_and_within_caps(self):
        for field, value in (("complete", False), ("model_queries", 39), ("checkpoint_unchanged", False),
                             ("peak_aggregate_gpu_memory_mib", 23552), ("elapsed_seconds", 1800),
                             ("training_performed", True), ("simulator_episodes", 1)):
            cfg, summary, rows = fixture(); summary[field] = value
            with self.assertRaises(ValueError): worker.reconcile(cfg, summary, rows)

    def test_parity_nan_and_excess_rejected(self):
        for value in (float("nan"), 1.1e-6, -1.):
            cfg, summary, rows = fixture(); rows[0]["modes"]["disabled"]["parity_max_abs"]["hidden"] = value
            with self.assertRaises(ValueError): worker.reconcile(cfg, summary, rows)

    def test_missing_and_duplicate_observations(self):
        cfg, summary, rows = fixture()
        with self.assertRaises(ValueError): worker.reconcile(cfg, summary, rows[:-1])
        rows[1]["observation_id"] = 0
        with self.assertRaises(ValueError): worker.reconcile(cfg, summary, rows)

    def test_protected_positions_cannot_disappear(self):
        cfg, summary, rows = fixture()
        rows[0]["modes"]["compressed"]["positions"].remove(547)
        with self.assertRaises(ValueError): worker.reconcile(cfg, summary, rows)

    def test_shifted_readout_rejected(self):
        cfg, summary, rows = fixture()
        rows[0]["modes"]["compressed"]["action_positions"] = list(range(548, 604))
        with self.assertRaises(ValueError): worker.reconcile(cfg, summary, rows)

    def test_modes_layers_and_compression_accounting(self):
        cfg, summary, rows = fixture()
        rows[0]["modes"]["compressed"]["positions"] = list(range(605))
        with self.assertRaises(ValueError): worker.reconcile(cfg, summary, rows)
        cfg, summary, rows = fixture(); rows[0]["modes"]["native"]["layers"] = 31
        with self.assertRaises(ValueError): worker.reconcile(cfg, summary, rows)
        cfg, summary, rows = fixture(); rows[0]["mode_order"].reverse()
        with self.assertRaises(ValueError): worker.reconcile(cfg, summary, rows)


if __name__ == "__main__":
    unittest.main()
