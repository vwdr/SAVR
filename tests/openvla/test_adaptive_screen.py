import ast
import json
from pathlib import Path
import sys
import unittest
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from savr.openvla.adaptive_screen_contract import (ARMS, CAPS, TRIAGE, episode_slots, timing_slots,
                                                   expected_retained, reconcile)
from analyze_adaptive_screen import analyze

SECONDS = dict(_512=1.0, _384=.8)
WARM_ADAPTIVE_SECONDS = .95
STEADY_ADAPTIVE_SECONDS = .70


def seconds_for(slot):
    if slot['arm'] in ('512', '384'):
        return SECONDS.get('_' + slot['arm'])
    return WARM_ADAPTIVE_SECONDS if slot['regime'] == 'warm' else STEADY_ADAPTIVE_SECONDS


def fixture():
    ref = json.loads((ROOT / 'configs/openvla/original_baseline_40task_v1.json').read_text())
    ids = [str(i) for i in range(8)]
    c = dict(schema_version='adaptive-screen-v1', launch_ready=True, caps=CAPS.copy(),
        automatic_retry=False, arms=list(ARMS), tolerance=1e-6, adaptive_zero_tolerance_max=1e-3,
        output_root='results/adaptive-screen-v03', episode_slots=episode_slots(ref),
        timing_slots=timing_slots(ids), observation_ids=ids, worker_sha256='a' * 64, triage=TRIAGE.copy())
    r = dict(episodes=[], timing=[], parity=[])
    for slot in c['episode_slots']:
        arm = slot['arm']
        retained = 512 if arm == 'native' else expected_retained(arm, 1)
        r['episodes'].append(dict(slot_id=slot['slot_id'], condition_id=slot['condition']['condition_id'],
            arm=arm, suite=slot['condition']['suite'], success=True, policy_queries=1, executed_steps=1,
            replans=0, discarded_actions=0, remaining_actions=7, controller_enabled=False, episode_seconds=2.,
            simulator_seconds=.2, controller_seconds=.1, nonquery_preparation_seconds=.1,
            queries=[dict(query=1, seconds=.5, retained_visual_tokens=retained, precise=False)]))
        if arm == 'native':
            r['parity'].append(dict(slot_id=slot['slot_id'], query=1, pair='native-vs-512',
                command_bytes_equal=True, layers=[32, 32], errors=dict(hidden=0., normalized=0., raw=0.),
                sdpa_calls=32, shadow_sdpa_calls=32, observation_sha256='a' * 64, command_sha256='b' * 64))
            r['parity'].append(dict(slot_id=slot['slot_id'], query=1, pair='native-vs-adaptive-zero',
                command_bytes_equal=True, layers=[32, 32], errors=dict(hidden=0., normalized=0., raw=0.),
                sdpa_calls=32, shadow_sdpa_calls=32, observation_sha256='a' * 64, command_sha256='b' * 64))
    for t in c['timing_slots']:
        r['timing'].append(dict(t, seconds=seconds_for(t), query=t['position'], precise=False,
            retained_visual_tokens=expected_retained(t['arm'], t['position'])))
    s = dict(complete=True, training_performed=False, automatic_retry=False, positive_method_result=False,
        checkpoint_unchanged=True, authenticated_files_unchanged=True, worker_sha256='a' * 64,
        elapsed_seconds=300., peak_aggregate_gpu_memory_mib=16000, episodes=168, model_queries=424,
        adaptive_zero_tolerance=1e-6, adaptive_zero_tolerance_max=1e-3)
    return c, ref, s, r


class ScreenTests(unittest.TestCase):
    def test_exact_counts_schedule_and_resource_bound(self):
        c, ref, s, r = fixture()
        self.assertTrue(reconcile(c, ref, s, r, c['observation_ids']))
        self.assertEqual(len(c['episode_slots']), 168)
        self.assertEqual(len(c['timing_slots']), 240)
        self.assertEqual(sum(t['warmup'] for t in c['timing_slots']), 16)
        self.assertEqual(240 + 8 * 3 + 160 * 1, 424)
        for arm in ARMS:
            self.assertEqual(sum(x['arm'] == arm for x in c['episode_slots']), 40)
            self.assertEqual(sum(x['arm'] == arm and not x['warmup'] for x in c['timing_slots']), 56)
        self.assertEqual(sum(x['arm'] == 'native' for x in c['episode_slots']), 8)
        for t in (x for x in c['timing_slots'] if x['arm'] in ('adaptive_75', 'adaptive_50')):
            self.assertEqual(t['regime'], 'warm' if t['position'] <= 3 else 'steady')

    def test_mutated_schedule_counts_resources_parity_and_queue_rejected(self):
        mutations = [lambda c, s, r: c['episode_slots'].reverse(),
            lambda c, s, r: c['timing_slots'].pop(),
            lambda c, s, r: s.update(model_queries=423),
            lambda c, s, r: s.update(adaptive_zero_tolerance=1.5e-3),
            lambda c, s, r: c.update(adaptive_zero_tolerance_max=1e-2),
            lambda c, s, r: c['triage'].update(preference=['adaptive_50']),
            lambda c, s, r: r['episodes'].pop(),
            lambda c, s, r: r['timing'].pop(),
            lambda c, s, r: s.update(elapsed_seconds=float('nan')),
            lambda c, s, r: r['episodes'][1].update(remaining_actions=6),
            lambda c, s, r: r['parity'][0].update(command_bytes_equal=False),
            lambda c, s, r: r['parity'][1].update(shadow_sdpa_calls=1),
            lambda c, s, r: r['parity'][0].update(sdpa_calls=0),
            lambda c, s, r: r['parity'][0].update(layers=[32, 0]),
            lambda c, s, r: r['parity'][1].update(layers=[32, 0]),
            lambda c, s, r: r['parity'][1].update(errors=dict(hidden=2e-3, normalized=0., raw=0.)),
            lambda c, s, r: r['episodes'][1]['queries'][0].update(retained_visual_tokens=99)]
        for mutate in mutations:
            c, ref, s, r = fixture(); mutate(c, s, r)
            with self.assertRaises(ValueError):
                reconcile(c, ref, s, r, c['observation_ids'])

    def test_evidence_only_no_automatic_selection(self):
        a = analyze(*fixture(), draws=100)
        self.assertIsNone(a['selected_arm'])
        self.assertEqual(a['decision'], 'evidence_recorded_no_automatic_selection')
        self.assertFalse(a['positive_method_result'])
        self.assertTrue(a['technical_reconciliation_passed'])
        eligible = {x['arm'] for x in a['candidates'] if x['eligible']}
        self.assertTrue({'adaptive_75', 'adaptive_50'} <= eligible)
        self.assertIn('controlled_mean_query_reduction', a['candidates'][0])

    def test_baseline_and_control_concerns_reported(self):
        c, ref, s, r = fixture(); r['episodes'][0]['success'] = False
        a = analyze(c, ref, s, r, draws=100)
        self.assertTrue(a['baseline_concern']); self.assertIsNone(a['selected_arm'])
        c, ref, s, r = fixture()
        for e in [x for x in r['episodes'] if x['arm'] == '512'][1:3]:
            e['success'] = False
        self.assertTrue(analyze(c, ref, s, r, draws=100)['baseline_concern'])

    def test_worker_preflight_before_gpu(self):
        tree = ast.parse((ROOT / 'scripts/run_adaptive_screen.py').read_text())
        main = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == 'main')
        calls = [(ast.unparse(x.func), x.lineno) for x in ast.walk(main) if isinstance(x, ast.Call)]
        line = lambda name: next(n for k, n in calls if k == name)
        self.assertLess(line('load_offline_inputs'), line('base.snapshot'))
        self.assertLess(line('load_offline_inputs'), line('run'))

    def test_timing_regime_means_reported_separately_from_all_query_mean(self):
        c, ref, s, r = fixture(); a = analyze(c, ref, s, r, draws=100)
        for arm in ('adaptive_75', 'adaptive_50'):
            by = a['timing_seconds'][arm]
            self.assertLess(by['steady']['mean'], by['warm']['mean'])
            self.assertLess(by['all']['mean'], 1.0)


if __name__ == '__main__':
    unittest.main()
