from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"scripts"))
import run_dense_precision_trace as worker

class TraceTests(unittest.TestCase):
    def fixture(self):
        c=dict(schema_version="dense-precision-trace-v1",caps=worker.CAPS.copy(),
            output_root="results/dense-precision-trace-v01",episode_modes=["native","bridge"],
            tolerance=1e-6,condition_id=worker.CONDITION,automatic_retry=False)
        s=dict(complete=True,model_queries=40,elapsed_seconds=90,peak_aggregate_gpu_memory_mib=16000,
            checkpoint_unchanged=True,authenticated_files_unchanged=True,training_performed=False,
            automatic_retry=False,positive_method_result=False)
        r=dict(episodes=[dict(mode=m,success=True,executed_steps=78,policy_queries=10) for m in c["episode_modes"]],
            traces=[dict(mode=m,query=q+1,observation_sha256=str(q),command_sha256=str(q),
                command_bytes_equal=True,layers=[32,32],errors=dict(hidden=0,normalized=0,raw_actions=0))
                for m in c["episode_modes"] for q in range(10)])
        return c,s,r
    def test_valid(self):self.assertTrue(worker.reconcile(*self.fixture()))
    def test_contract_and_limits(self):
        c,s,r=self.fixture();c["automatic_retry"]=True
        with self.assertRaises(ValueError):worker.reconcile(c,s,r)
        c,s,r=self.fixture();s["model_queries"]=41
        with self.assertRaises(ValueError):worker.reconcile(c,s,r)
    def test_one_step_difference_is_not_relaxed(self):
        c,s,r=self.fixture();r["episodes"][1]["executed_steps"]=79
        with self.assertRaises(ValueError):worker.reconcile(c,s,r)
    def test_command_bytes_and_trace_are_gates(self):
        for key,value in (("command_bytes_equal",False),("command_sha256","different"),
                          ("observation_sha256","different")):
            c,s,r=self.fixture();r["traces"][0][key]=value
            with self.assertRaises(ValueError):worker.reconcile(c,s,r)
    def test_nan_and_hidden_difference_rejected(self):
        for value in (float("nan"),.01):
            c,s,r=self.fixture();r["traces"][0]["errors"]["hidden"]=value
            with self.assertRaises(ValueError):worker.reconcile(c,s,r)

if __name__=="__main__":unittest.main()
