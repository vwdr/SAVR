import ast
import copy
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_contemporary_reference_recovery01 as worker


def fixture():
    config = dict(observation_ids=[str(i) for i in range(8)], worker_sha256='a' * 64)
    summary = dict(complete=True, model_queries=80, episodes=0, hook_neutrality_passed=True,
        checkpoint_unchanged=True, authenticated_files_unchanged=True, automatic_retry=False,
        training_performed=False, positive_method_result=False, worker_sha256='a' * 64,
        elapsed_seconds=120., peak_aggregate_gpu_memory_mib=16000)
    records = dict(episodes=[], timing=[], parity=[], hook_checks=[dict(observation_id=i, arm=arm, hooks_neutral=True)
        for i in config['observation_ids'] for arm in ('native', 'dense', 'specprune')])
    return config, summary, records


class RecoveryTests(unittest.TestCase):
    def test_hook_completeness(self):
        self.assertTrue(worker.reconcile_hooks(*fixture()))

    def test_hook_reject_partial_duplicate_false_and_resource(self):
        for mutate in (lambda s, r: s.update(model_queries=79), lambda s, r: s.update(episodes=1),
                       lambda s, r: r['hook_checks'].pop(), lambda s, r: r['hook_checks'].reverse(),
                       lambda s, r: r['hook_checks'][0].update(hooks_neutral=False),
                       lambda s, r: s.update(elapsed_seconds=float('nan')),
                       lambda s, r: s.update(peak_aggregate_gpu_memory_mib=23552)):
            c, s, r = fixture(); mutate(s, r)
            with self.assertRaises(ValueError): worker.reconcile_hooks(c, s, r)

    def test_actual_input_loading_precedes_gpu_and_model_dispatch(self):
        tree = ast.parse(Path(worker.__file__).read_text())
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
        calls = [(ast.unparse(n.func), n.lineno) for n in ast.walk(main) if isinstance(n, ast.Call)]
        line = lambda name: next(l for n, l in calls if n == name)
        self.assertLess(line('load_offline_inputs'), line('base.snapshot'))
        self.assertLess(line('load_offline_inputs'), line('run'))

    def test_only_offline_factory_in_fixed_trace_worker(self):
        tree = ast.parse(Path(worker.__file__).read_text())
        calls = [ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)]
        self.assertEqual(calls.count('OfflineCameraPair.capture'), 2)
        self.assertNotIn('CameraPair.capture', calls)


if __name__ == '__main__': unittest.main()
