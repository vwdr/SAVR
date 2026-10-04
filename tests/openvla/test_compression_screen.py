import ast
import copy
import json
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from savr.openvla.compression_screen_contract import CAPS,episode_slots,timing_slots,reconcile
from analyze_fixed_compression_screen import analyze


def fixture():
    ref=json.loads((ROOT/'configs/openvla/original_baseline_40task_v1.json').read_text());ids=[str(i) for i in range(8)]
    c=dict(schema_version='fixed-compression-screen-v1',launch_ready=True,caps=CAPS.copy(),automatic_retry=False,
        budgets=[512,384,256],tolerance=1e-6,output_root='results/fixed-compression-screen-v01',
        episode_slots=episode_slots(ref),timing_slots=timing_slots(ids),observation_ids=ids,worker_sha256='a'*64,
        triage=dict(minimum_mean_time_reduction=.1,maximum_success_count_loss=6,dense_minimum_successes=39,preference=[384,256]))
    r=dict(episodes=[],timing=[],parity=[])
    for slot in c['episode_slots']:
        budget=512 if slot['arm']=='native' else int(slot['arm'])
        r['episodes'].append(dict(slot_id=slot['slot_id'],condition_id=slot['condition']['condition_id'],
            arm=slot['arm'],suite=slot['condition']['suite'],success=True,policy_queries=1,executed_steps=1,
            replans=0,discarded_actions=0,remaining_actions=7,controller_enabled=False,episode_seconds=2.,
            simulator_seconds=.2,controller_seconds=.1,nonquery_preparation_seconds=.1,
            queries=[dict(query=1,seconds=1.,retained_visual_tokens=budget,precise=False)]))
        if slot['arm']=='native':r['parity'].append(dict(slot_id=slot['slot_id'],query=1,
            command_bytes_equal=True,layers=[32,32],errors=dict(hidden=0.,normalized=0.,raw=0.),
            observation_sha256='a'*64,command_sha256='b'*64))
    for t in c['timing_slots']:r['timing'].append(dict(t,seconds=1. if t['arm']=='512' else .8,
        retained_visual_tokens=int(t['arm']),query=t['frame']+1,precise=False))
    s=dict(complete=True,training_performed=False,automatic_retry=False,positive_method_result=False,
        checkpoint_unchanged=True,authenticated_files_unchanged=True,worker_sha256='a'*64,
        elapsed_seconds=300.,peak_aggregate_gpu_memory_mib=16000,episodes=128,model_queries=340)
    return c,ref,s,r


class ScreenTests(unittest.TestCase):
    def test_exact_counts_schedule_and_resource_bound(self):
        c,ref,s,r=fixture();self.assertTrue(reconcile(c,ref,s,r,c['observation_ids']))
        self.assertEqual(len(c['episode_slots']),128);self.assertEqual(len(c['timing_slots']),204)
        self.assertEqual(sum(t['warmup'] for t in c['timing_slots']),12)
        self.assertEqual(3*10*(28+35+38+65)+4*(28+35+38+65)+204,5848)
        for arm in ('512','384','256'):
            self.assertEqual(sum(x['arm']==arm for x in c['episode_slots']),40)
            self.assertEqual(sum(x['arm']==arm and not x['warmup'] for x in c['timing_slots']),64)

    def test_mutated_schedule_counts_resources_parity_and_queue_rejected(self):
        mutations=[lambda c,s,r:c['episode_slots'].reverse(),lambda c,s,r:c['timing_slots'].pop(),
            lambda c,s,r:s.update(model_queries=339),lambda c,s,r:r['episodes'].pop(),
            lambda c,s,r:r['timing'].pop(),lambda c,s,r:s.update(elapsed_seconds=float('nan')),
            lambda c,s,r:r['episodes'][1].update(remaining_actions=6),
            lambda c,s,r:r['parity'][0].update(command_bytes_equal=False),
            lambda c,s,r:r['episodes'][1]['queries'][0].update(retained_visual_tokens=99),
            lambda c,s,r:c['triage'].update(maximum_success_count_loss=7)]
        for mutate in mutations:
            c,ref,s,r=fixture();mutate(c,s,r)
            with self.assertRaises(ValueError):reconcile(c,ref,s,r,c['observation_ids'])

    def test_prefer_gentler_when_both_eligible(self):
        a=analyze(*fixture(),draws=100)
        self.assertEqual(a['selected_budget'],384);self.assertFalse(a['positive_method_result'])

    def test_success_gate_and_no_outcome_retry(self):
        c,ref,s,r=fixture()
        for arm in ('384','256'):
            rows=[e for e in r['episodes'] if e['arm']==arm]
            for e in rows[:7]:e['success']=False
        a=analyze(c,ref,s,r,draws=100)
        self.assertIsNone(a['selected_budget']);self.assertTrue(a['technical_reconciliation_passed'])

    def test_timing_gate(self):
        c,ref,s,r=fixture()
        for t in r['timing']:
            if t['arm']=='384':t['seconds']=.95
        self.assertEqual(analyze(c,ref,s,r,draws=100)['selected_budget'],256)

    def test_baseline_and_control_concerns_block_selection(self):
        c,ref,s,r=fixture()
        r['episodes'][0]['success']=False
        a=analyze(c,ref,s,r,draws=100);self.assertTrue(a['baseline_concern']);self.assertIsNone(a['selected_budget'])
        c,ref,s,r=fixture()
        for e in [x for x in r['episodes'] if x['arm']=='512'][1:3]:e['success']=False
        self.assertTrue(analyze(c,ref,s,r,draws=100)['baseline_concern'])

    def test_worker_preflight_before_gpu(self):
        tree=ast.parse((ROOT/'scripts/run_fixed_compression_screen.py').read_text())
        main=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='main')
        calls=[(ast.unparse(x.func),x.lineno) for x in ast.walk(main) if isinstance(x,ast.Call)]
        line=lambda name:next(n for k,n in calls if k==name)
        self.assertLess(line('load_offline_inputs'),line('base.snapshot'));self.assertLess(line('load_offline_inputs'),line('run'))


if __name__=='__main__':unittest.main()
