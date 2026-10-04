"""Fail-closed reconciliation of the bounded live bridge check."""
import copy
import importlib.util
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import run_specprune_episode_qualification as worker


class QualificationTests(unittest.TestCase):
    def fixture(self):
        config = dict(schema_version="specprune-episode-qualification-v1", caps=worker.CAPS.copy(),
            tolerance=1e-6, output_root="results/specprune-episode-qualification-v01",
            offline_calls=56, frame_indices=[0,1], episode_modes=["dense", "compressed"],
            condition_id=worker.CONDITION, initial_state_sha256=worker.STATE_SHA,
            automatic_retry=False, observation_ids=list(range(8)))
        errors = dict(hidden=0., normalized=0., actions=0.)
        offline = [dict(observation_id=i, calls=7, compressed_query_counts=[1,2,1],
            all_original_layers=True, persistent_indices_and_confidence=True,
            dense_parity=[errors.copy(),errors.copy()], reset_parity=errors.copy()) for i in range(8)]
        episodes = [dict(mode=mode, condition_id=worker.CONDITION,
            initial_state_sha256=worker.STATE_SHA, executed_steps=78, policy_queries=10,
            success=True, all_original_layers=True, controller_enabled=mode=="compressed")
            for mode in config["episode_modes"]]
        summary = dict(complete=True, offline_calls=56, model_queries=76, elapsed_seconds=100,
            peak_aggregate_gpu_memory_mib=16000, checkpoint_unchanged=True,
            authenticated_files_unchanged=True, automatic_retry=False,
            training_performed=False, positive_method_result=False)
        return config, summary, dict(offline=offline,episodes=episodes)

    def test_valid_fixture(self):
        self.assertTrue(worker.reconcile(*self.fixture()))

    def test_frozen_limits_cannot_change(self):
        for key, value in (("offline_calls", 55), ("automatic_retry", True),
                           ("frame_indices", [0,2]), ("tolerance", .1)):
            config,summary,records=self.fixture(); config[key]=value
            with self.assertRaises(ValueError): worker.reconcile(config,summary,records)

    def test_missing_duplicate_or_reordered_observations(self):
        for change in (lambda r:r.pop(), lambda r:r.__setitem__(1, r[0]), lambda r:r.reverse()):
            c,s,r=self.fixture(); change(r["offline"])
            with self.assertRaises(ValueError): worker.reconcile(c,s,r)

    def test_parity_nan_or_reset_failure(self):
        for key in ("dense_parity", "reset_parity"):
            for value in (float("nan"), .001):
                c,s,r=self.fixture()
                item=r["offline"][0][key]
                (item[0] if isinstance(item,list) else item)["actions"]=value
                with self.assertRaises(ValueError): worker.reconcile(c,s,r)

    def test_history_or_layer_failure(self):
        for key,value in (("compressed_query_counts",[1,2,3]),
                          ("persistent_indices_and_confidence",False), ("all_original_layers",False)):
            c,s,r=self.fixture(); r["offline"][0][key]=value
            with self.assertRaises(ValueError): worker.reconcile(c,s,r)

    def test_dense_must_match_reference_but_compressed_success_is_not_a_gate(self):
        c,s,r=self.fixture(); r["episodes"][1]["success"]=False
        self.assertTrue(worker.reconcile(c,s,r))
        r["episodes"][0]["executed_steps"]=79
        with self.assertRaises(ValueError): worker.reconcile(c,s,r)

    def test_wrong_state_or_controller(self):
        for key,value in (("initial_state_sha256","wrong"), ("controller_enabled",False)):
            c,s,r=self.fixture(); r["episodes"][1][key]=value
            with self.assertRaises(ValueError): worker.reconcile(c,s,r)

    def test_query_and_resource_accounting(self):
        for key,value in (("model_queries",77),("elapsed_seconds",1800),
                          ("peak_aggregate_gpu_memory_mib",23552),("complete",False)):
            c,s,r=self.fixture(); s[key]=value
            with self.assertRaises(ValueError): worker.reconcile(c,s,r)


if __name__ == "__main__": unittest.main()
