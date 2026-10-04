import ast
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
import run_fixed_compression_qualification as worker


def fixture():
    c=dict(observation_ids=[str(i) for i in range(8)],worker_sha256='a'*64)
    s=dict(complete=True,model_queries=112,episodes=0,training_performed=False,automatic_retry=False,
        positive_method_result=False,checkpoint_unchanged=True,authenticated_files_unchanged=True,
        worker_sha256='a'*64,elapsed_seconds=150.,peak_aggregate_gpu_memory_mib=16000)
    rows=[dict(observation_id=i,frame=f,input_unchanged=True,full_tokens=605,dense_errors=dict(hidden=0.,normalized=0.,raw=0.),
        dense_command_equal=True,modes=[dict(budget=b,finite=True,hooks_neutral=True,
        positions=sorted([0,*worker.stratified_visual_positions(b),*range(513,605)]),
        layer_lengths=[605-512+b]*32,original_positions_all_layers=True,no_cache_all_layers=True,state_query=1)
        for b in (512,384,256)]) for i in c['observation_ids'] for f in (0,1)]
    return c,s,rows


class QualificationTests(unittest.TestCase):
    def test_complete(self):self.assertTrue(worker.reconcile(*fixture()))

    def test_reject_corrupt_counts_parity_positions_and_resources(self):
        mutations=[lambda s,r:s.update(model_queries=111),lambda s,r:s.update(episodes=1),
            lambda s,r:s.update(elapsed_seconds=float('nan')),lambda s,r:s.update(peak_aggregate_gpu_memory_mib=23552),
            lambda s,r:r.pop(),lambda s,r:r.reverse(),lambda s,r:r[0].update(dense_command_equal=False),
            lambda s,r:r[0]['dense_errors'].update(raw=1e-5),lambda s,r:r[0]['dense_errors'].update(raw=float('nan')),
            lambda s,r:r[0]['modes'][1].update(hooks_neutral=False),lambda s,r:r[0]['modes'][1]['positions'].pop(),
            lambda s,r:r[0]['modes'][2].update(layer_lengths=[605]*32),lambda s,r:r[0]['modes'][0].update(no_cache_all_layers=False)]
        for mutate in mutations:
            c,s,r=fixture();mutate(s,r)
            with self.assertRaises(ValueError):worker.reconcile(c,s,r)

    def test_preflight_before_model_or_gpu(self):
        tree=ast.parse(Path(worker.__file__).read_text())
        main=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='main')
        calls=[(ast.unparse(x.func),x.lineno) for x in ast.walk(main) if isinstance(x,ast.Call)]
        line=lambda name:next(n for k,n in calls if k==name)
        self.assertLess(line('load_offline_inputs'),line('base.snapshot'))
        self.assertLess(line('load_offline_inputs'),line('run'))


if __name__=='__main__':unittest.main()
