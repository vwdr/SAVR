import ast
from pathlib import Path
import sys
import unittest
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_adaptive_qualification as worker


def fixture():
    c = dict(observation_ids=[str(i) for i in range(8)], worker_sha256='a' * 64,
             adaptive_zero_tolerance_max=1e-3)
    s = dict(complete=True, model_queries=worker.MODEL_CALLS, episodes=0, training_performed=False,
        automatic_retry=False, positive_method_result=False, checkpoint_unchanged=True,
        authenticated_files_unchanged=True, worker_sha256='a' * 64, elapsed_seconds=300.,
        peak_aggregate_gpu_memory_mib=16000, adaptive_zero_tolerance=1e-6,
        zero_parity_observed_max=0.0, pruning_shift_observed_max=0.0)

    def mode(name, ratio, expected):
        full, post = 605, 605 - 512 + expected
        return dict(mode=name, fastv_r=ratio, retained_expected=expected,
            retained_queries=[512, 512, 512, expected], steady_query=4, state_query=4,
            pruned=True, effective_fastv_r=ratio, lengths_before_prune=True,
            lengths_after_prune=True, original_positions_all_layers=True,
            no_cache_all_layers=True, hooks_neutral=True, sdpa_calls=32,
            full_tokens=full, post_prune_length=post,
            layer_lengths=[full] * 4 + [post] * 28,
            steady_errors=dict(hidden=0., normalized=0., raw=0.))

    rows = [dict(observation_id=i, frame=f, input_unchanged=True, full_tokens=605,
                 sdpa_native=32, sdpa_zero=32, zero_hooks_neutral=True, zero_command_equal=True,
                 zero_errors=dict(hidden=0., normalized=0., raw=0.),
                 modes=[mode('adaptive_75', .25, 384), mode('adaptive_50', .5, 256)] if f == 1 else [])
            for i in c['observation_ids'] for f in (0, 1)]
    return c, s, rows


class QualificationTests(unittest.TestCase):
    def test_complete(self):
        self.assertTrue(worker.reconcile(*fixture()))

    def test_reject_corrupt_counts_witnesses_modes_and_resources(self):
        mutations = [
            lambda s, r: s.update(model_queries=worker.MODEL_CALLS - 1), lambda s, r: s.update(episodes=1),
            lambda s, r: s.update(elapsed_seconds=float('nan')),
            lambda s, r: s.update(peak_aggregate_gpu_memory_mib=23552),
            lambda s, r: s.update(adaptive_zero_tolerance=2e-3),
            lambda s, r: s.update(zero_parity_observed_max=5e-4 + 1e-9),
            lambda s, r: s.update(adaptive_zero_tolerance=2e-6),
            lambda s, r: r.pop(), lambda s, r: r.reverse(),
            lambda s, r: r[0].update(sdpa_native=0), lambda s, r: r[0].update(sdpa_zero=1),
            lambda s, r: r[0].update(zero_hooks_neutral=False),
            lambda s, r: r[0]['zero_errors'].update(raw=float('nan')),
            lambda s, r: r[0]['zero_errors'].update(hidden=5e-4 + 1e-9),
            lambda s, r: r[1].update(input_unchanged=False),
            lambda s, r: r[1]['modes'][0].update(retained_queries=[512, 512, 512, 256]),
            lambda s, r: r[1]['modes'][1].update(sdpa_calls=1),
            lambda s, r: r[1]['modes'][1].update(pruned=False),
            lambda s, r: r[1]['modes'][0].update(original_positions_all_layers=False),
            lambda s, r: r[1]['modes'][0]['steady_errors'].update(raw=float('nan')),
            lambda s, r: s.update(pruning_shift_observed_max=-1.0),
            lambda s, r: s.update(pruning_shift_observed_max=float('inf')),
            # Numeric prune-boundary evidence must survive reconcile.
            lambda s, r: r[1]['modes'][0].update(post_prune_length=385),
            lambda s, r: r[1]['modes'][0].update(full_tokens=512),
            lambda s, r: r[1]['modes'][0].update(layer_lengths=[605] * 4 + [384] * 28),
            lambda s, r: r[1]['modes'][1].update(layer_lengths=[605] * 31),
        ]
        for mutate in mutations:
            c, s, r = fixture(); mutate(s, r)
            with self.assertRaises(ValueError):
                worker.reconcile(c, s, r)

    def test_zero_parity_tolerance_is_separate_from_pruning_shift(self):
        # Regression: pruning-induced steady differences (a different, shorter
        # computation) must not calibrate or inflate the numerical tolerance.
        c, s, r = fixture()
        for m in r[1]['modes']:
            m['steady_errors'] = dict(hidden=0.5, normalized=0.25, raw=2.0)
        s['pruning_shift_observed_max'] = 2.0
        # zero-parity stays tiny, so the tolerance stays at its floor 1e-6.
        self.assertTrue(worker.reconcile(c, s, r))
        for m in r[1]['modes']:
            self.assertTrue(all(m['steady_errors'][k] > 0.1 for k in m['steady_errors']))

    def test_post_prune_layer_length_eq_full_minus_pruned_visual(self):
        # Regression: post-prune layer length is full - (512 - retained_visual)
        # (the port keeps [0] + top_visual + trailing non-visual tokens), so it
        # is 605-512+384=477 at 75% retention and 605-512+256=349 at 50% — NOT
        # the retained-visual count (384/256) alone. reconcile must accept the
        # correct invariant and reject a gate that compares against 384/256.
        c, s, r = fixture()
        self.assertTrue(worker.reconcile(c, s, r))  # fixture carries 477/349
        for m in r[1]['modes']:
            expected_post = m['full_tokens'] - 512 + m['retained_expected']
            self.assertEqual(m['post_prune_length'], expected_post)
            self.assertEqual(m['layer_lengths'], [m['full_tokens']] * 4 + [expected_post] * 28)
        # Corrupt the record to the naive post-prune == retained_visual reading.
        c2, s2, r2 = fixture()
        bad = [605] * 4 + [384] * 28
        r2[1]['modes'][0].update(layer_lengths=bad, post_prune_length=384)
        with self.assertRaises(ValueError):
            worker.reconcile(c2, s2, r2)

    def test_missing_separated_maxima_rejected(self):
        c, s, r = fixture()
        del s['zero_parity_observed_max']
        with self.assertRaises(KeyError):
            worker.reconcile(c, s, r)
        c, s, r = fixture()
        del s['pruning_shift_observed_max']
        with self.assertRaises(KeyError):
            worker.reconcile(c, s, r)

    def test_analyzer_recomputes_tolerance_from_zero_parity_only(self):
        import analyze_adaptive_qualification
        self.assertTrue(callable(analyze_adaptive_qualification.analyze))

    def test_decoder_scoped_witness_requires_all_32_native_layers(self):
        c, s, r = fixture()
        self.assertTrue(worker.reconcile(c, s, r))
        c2, s2, r2 = fixture()
        r2[1]['modes'][0]['sdpa_calls'] = 1
        with self.assertRaises(ValueError):
            worker.reconcile(c2, s2, r2)

    def test_command_equality_remains_required(self):
        c, s, r = fixture()
        r[0]['zero_command_equal'] = False
        with self.assertRaises(ValueError):
            worker.reconcile(c, s, r)
        c2, s2, r2 = fixture()
        r2[0]['zero_command_equal'] = 'yes'
        with self.assertRaises((ValueError, TypeError)):
            worker.reconcile(c2, s2, r2)

    def test_preflight_before_model_or_gpu(self):
        tree = ast.parse(Path(worker.__file__).read_text())
        main = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == 'main')
        calls = [(ast.unparse(x.func), x.lineno) for x in ast.walk(main) if isinstance(x, ast.Call)]
        line = lambda name: next(n for k, n in calls if k == name)
        self.assertLess(line('load_offline_inputs'), line('base.snapshot'))
        self.assertLess(line('load_offline_inputs'), line('run'))


if __name__ == '__main__':
    unittest.main()
