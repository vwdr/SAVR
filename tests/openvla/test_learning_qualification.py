import copy
import importlib.util
from pathlib import Path
import sys
import unittest

SCRIPTS = Path(__file__).resolve().parents[2]/'scripts'
sys.path.insert(0, str(SCRIPTS))
import run_current_frame_learning_qualification as worker


class ReconcileTests(unittest.TestCase):
    def valid(self):
        c = dict(observation_ids=[str(i) for i in range(8)])
        s = dict(complete=True, model_queries=80, optimizer_steps=128, episodes=0,
            positive_method_result=False, automatic_retry=False, authenticated_files_unchanged=True,
            elapsed_seconds=200, peak_aggregate_gpu_memory_mib=17000, artifact_bytes_before_summary=100000000)
        rows = [dict(sample_id=i+':'+str(f), frame=f, input_unchanged=True, feature_commands_equal=True,
            zero_action_only_equal=True, zero_visual_equal=True, record_roundtrip_equal=True,
            no_backbone_gradients=True) for i in c['observation_ids'] for f in (0, 1)]
        fits = [dict(use_visual=v, steps=64, initial_l1=.1, final_l1=.01, reload_equal=True,
            no_backbone_gradients=True, nonzero_output_gradient=True, all_updates_finite=True) for v in (False, True)]
        return c, s, rows, fits

    def test_exact_complete_contract(self):
        self.assertTrue(worker.reconcile(*self.valid()))

    def test_missing_and_misordered_frames(self):
        c, s, r, f = self.valid()
        with self.assertRaises(ValueError): worker.reconcile(c, s, r[:-1], f)
        with self.assertRaises(ValueError): worker.reconcile(c, s, r[::-1], f)

    def test_failed_parity_nonfinite_or_no_fit_rejected(self):
        for field in ('feature_commands_equal', 'zero_visual_equal', 'record_roundtrip_equal'):
            c, s, r, f = self.valid(); r[0][field] = False
            with self.assertRaises(ValueError): worker.reconcile(c, s, r, f)
        for value in (.1, float('nan'), -1):
            c, s, r, f = self.valid(); f[0]['final_l1'] = value
            with self.assertRaises(ValueError): worker.reconcile(c, s, r, f)

    def test_resource_and_step_mismatch_rejected(self):
        for key, value in dict(model_queries=79, optimizer_steps=127, peak_aggregate_gpu_memory_mib=23552).items():
            c, s, r, f = self.valid(); s[key] = value
            with self.assertRaises(ValueError): worker.reconcile(c, s, r, f)


if __name__ == '__main__': unittest.main()
