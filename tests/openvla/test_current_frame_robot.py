"""CPU-only checks of frozen robot schedules, reconciliation and interpretation."""
import ast
from collections import Counter
import copy
import json
import math
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from savr.openvla.current_frame_robot_contract import (
    ARMS,CAPS,GATES,episode_slots,timing_slots,validate,qualification_pass,reconcile,decision)
from savr.openvla.contemporary_contract import SUITES,HORIZONS
from analyze_current_frame_robot_pilot import analyze


def fixture():
    ref=json.loads((ROOT/'configs/openvla/original_baseline_40task_v1.json').read_text())
    ledger=[json.loads(x) for x in (ROOT/'reports/cac_c0_recovery01/simulator_populations_v1.jsonl').read_text().splitlines()]
    ids=[str(i) for i in range(8)]
    slots=episode_slots(ledger,ref)
    c=dict(schema_version='current-frame-robot-pilot-v1',caps=CAPS.copy(),gates=GATES.copy(),
           automatic_retry=False,arms=list(ARMS),output_root='results/current-frame-robot-pilot-v01',
           observation_ids=ids,episode_slots=slots,timing_slots=timing_slots(ids),
           initial_state_sha256={s['condition']['condition_id']:'a'*64 for s in slots})
    r=dict(episodes=[],timing=[],parity=[],qualification=[])
    for slot in slots:
        arm=slot['arm'];budget=512 if arm in ('native','dense') else 384
        r['episodes'].append(dict(slot_id=slot['slot_id'],condition_id=slot['condition']['condition_id'],
            arm=arm,suite=slot['condition']['suite'],success=True,policy_queries=1,executed_steps=1,
            replans=0,discarded_actions=0,remaining_actions=7,controller_enabled=False,episode_seconds=2.,
            simulator_seconds=.2,controller_seconds=.1,nonquery_preparation_seconds=.1,
            queries=[dict(query=1,seconds=1.,retained_visual_tokens=budget,precise=False)]))
        if arm=='native':r['parity'].append(dict(slot_id=slot['slot_id'],query=1,
            command_bytes_equal=True,layers=[32,32],errors=dict(hidden=0.,normalized=0.,raw=0.),
            observation_sha256='a'*64,command_sha256='b'*64))
    for slot in c['timing_slots']:
        r['timing'].append(dict(slot,seconds=1. if slot['arm']=='dense' else .8,
            retained_visual_tokens=512 if slot['arm']=='dense' else 384,query=slot['frame']+1,precise=False))
    for identity in ids:
        for frame in (0,1):
            r['qualification'].append(dict(observation_id=identity,frame=frame,
                **{k:True for k in ('input_unchanged','dense_command_equal','zero_command_equal',
                                    'action_only_command_equal','visual_command_equal','features_equal','no_gradients')},
                dense_errors=dict(hidden=0.,normalized=0.,raw=0.)))
    s=dict(complete=True,training_performed=False,automatic_retry=False,positive_method_result=False,
        checkpoint_unchanged=True,authenticated_files_unchanged=True,elapsed_seconds=1000.,
        peak_aggregate_gpu_memory_mib=16000,episodes=488,model_queries=864)
    return c,ledger,ref,s,r


class RobotPilotTests(unittest.TestCase):
    def test_real_metadata_schedule_counts_and_balance(self):
        c,ledger,ref,s,r=fixture();validate(c,ledger,ref,c['observation_ids'])
        self.assertTrue(reconcile(c,s,r))
        primary=[x for x in c['episode_slots'] if x['role']=='primary']
        self.assertEqual(Counter(x['arm'] for x in primary),{a:120 for a in ARMS})
        self.assertEqual(Counter(x['condition']['initial_state_id'] for x in primary),{1:160,2:160,3:160})
        self.assertEqual(Counter(x['arm'] for x in primary[::4]),{a:30 for a in ARMS})
        self.assertEqual(len(c['initial_state_sha256']),124)
        self.assertEqual(sum(x['warmup'] for x in c['timing_slots']),16)
        self.assertEqual(Counter(x['arm'] for x in c['timing_slots'] if not x['warmup']),{a:64 for a in ARMS})
        worst=sum(math.ceil(HORIZONS[x['condition']['suite']]/8)*(2 if x['arm']=='native' else 1)
                  for x in c['episode_slots'])+96+272
        self.assertEqual(worst,20952);self.assertLess(worst,CAPS['model_calls'])

    def test_schedule_hash_and_gate_mutations_rejected(self):
        mutations=[lambda c:c['episode_slots'].reverse(),lambda c:c['timing_slots'].pop(),
            lambda c:c['gates'].update(visual_gain_over_compressed=2),
            lambda c:c['caps'].update(episodes=489),lambda c:c.update(automatic_retry=True),
            lambda c:c['initial_state_sha256'].pop(next(iter(c['initial_state_sha256']))),
            lambda c:c['initial_state_sha256'].update({next(iter(c['initial_state_sha256'])):'x'*64})]
        for change in mutations:
            c,ledger,ref,_,_=fixture();change(c)
            with self.assertRaises(ValueError):validate(c,ledger,ref,c['observation_ids'])

    def test_population_missing_duplicate_and_locked_state_rejected(self):
        c,ledger,ref,_,_=fixture()
        chosen=next(x for x in ledger if x['population']=='headroom_stage1' and x['initial_state_id']==1)
        bad=[x for x in ledger if x!=chosen]
        with self.assertRaises(ValueError):episode_slots(bad,ref)
        with self.assertRaises(ValueError):episode_slots(ledger+[chosen],ref)
        bad=copy.deepcopy(ledger)
        next(x for x in bad if x==chosen)['initial_state_id']=6
        with self.assertRaises(ValueError):episode_slots(bad,ref)

    def test_integration_flags_and_errors_fail_closed(self):
        c,_,_,_,r=fixture()
        qualification_pass(r['qualification'],c['observation_ids'])
        for key in ('input_unchanged','dense_command_equal','zero_command_equal',
                    'action_only_command_equal','visual_command_equal','features_equal','no_gradients'):
            rows=copy.deepcopy(r['qualification']);rows[0][key]=False
            with self.assertRaises(ValueError):qualification_pass(rows,c['observation_ids'])
        for value in (1e-5,float('nan'),-1.):
            rows=copy.deepcopy(r['qualification']);rows[0]['dense_errors']['hidden']=value
            with self.assertRaises(ValueError):qualification_pass(rows,c['observation_ids'])
        with self.assertRaises(ValueError):qualification_pass(r['qualification'][:-1],c['observation_ids'])

    def test_completed_evidence_mutations_fail_closed(self):
        mutations=[lambda c,s,r:s.update(complete=False),lambda c,s,r:s.update(model_queries=863),
            lambda c,s,r:s.update(positive_method_result=True),lambda c,s,r:s.update(elapsed_seconds=57600),
            lambda c,s,r:s.update(peak_aggregate_gpu_memory_mib=23552),
            lambda c,s,r:r['episodes'].pop(),lambda c,s,r:r['timing'].pop(),
            lambda c,s,r:r['parity'].pop(),lambda c,s,r:r['parity'][0]['errors'].clear(),
            lambda c,s,r:r['parity'][0].update(command_sha256='not-a-hash'),
            lambda c,s,r:r['parity'][0].update(command_bytes_equal=False),
            lambda c,s,r:r['episodes'][1].update(remaining_actions=6),
            lambda c,s,r:r['episodes'][1].update(success=1),
            lambda c,s,r:r['episodes'][1].update(controller_enabled=True),
            lambda c,s,r:r['episodes'][1]['queries'][0].update(retained_visual_tokens=256),
            lambda c,s,r:r['episodes'][1]['queries'][0].update(seconds=3.),
            lambda c,s,r:r['timing'][0].update(seconds=float('nan'))]
        for change in mutations:
            c,_,_,s,r=fixture();change(c,s,r)
            with self.assertRaises(ValueError):reconcile(c,s,r)

    def test_gates_preliminary_only_and_individually_binding(self):
        wins=dict(dense=116,compressed=110,action_only=112,visual=115)
        suites={su:dict(compressed=27,visual=28) for su in SUITES}
        a=decision(wins,8,suites,.12)
        self.assertTrue(a['positive_development_signal']);self.assertFalse(a['paper_ready'])
        self.assertFalse(a['confirmatory_result'])
        for k,v in [('dense',107),('dense',118),('compressed',113),('action_only',115)]:
            bad=dict(wins);bad[k]=v
            self.assertFalse(decision(bad,8,suites,.12)['positive_development_signal'])
        self.assertFalse(decision(wins,7,suites,.12)['positive_development_signal'])
        self.assertFalse(decision(wins,8,suites,.09)['positive_development_signal'])
        suites[SUITES[0]]['compressed']=31
        self.assertFalse(decision(wins,8,suites,.12)['positive_development_signal'])

    def test_analyzer_ceiling_not_positive_and_pair_accounting(self):
        c,_,_,s,r=fixture();a=analyze(c,s,r)
        self.assertEqual(a['successes'],{arm:120 for arm in ARMS})
        self.assertFalse(a['decision']['positive_development_signal'])
        self.assertAlmostEqual(a['controlled_reductions']['visual'],.2)
        self.assertTrue(all(x['observation_command_traces_equal'] for x in a['native_controls']))
        self.assertTrue(all(x['net_successes']==0 and x['candidate_only']==0 and x['reference_only']==0 for x in a['comparisons']))
        for arm,count in [('dense',4),('compressed',10),('action_only',8),('visual',5)]:
            rows=[e for e in r['episodes'] if e['arm']==arm]
            for e in rows[:count]:e['success']=False
        a=analyze(c,s,r)
        self.assertTrue(a['decision']['positive_development_signal'])
        compare=next(x for x in a['comparisons'] if (x['candidate'],x['reference'])==('visual','compressed'))
        self.assertEqual(compare['net_successes'],5);self.assertEqual(compare['candidate_only'],5)
        self.assertEqual(compare['reference_only'],0)
        self.assertAlmostEqual(compare['success_difference'],5/120)
        self.assertEqual(a['controlled_policy_query_seconds']['visual']['n'],64)

    def test_source_compiles_and_worker_has_qualification_before_timing_before_episodes(self):
        files=['scripts/run_current_frame_robot_pilot.py','scripts/analyze_current_frame_robot_pilot.py',
               'scripts/freeze_current_frame_robot_states.py','src/savr/openvla/current_frame_robot_contract.py']
        for f in files:ast.parse((ROOT/f).read_text())
        source=(ROOT/files[0]).read_text()
        self.assertLess(source.index("qualification_pass(records['qualification']"),source.index("for slot in c['timing_slots']"))
        self.assertLess(source.index("for slot in c['timing_slots']"),source.index("for slot in c['episode_slots']"))
        self.assertNotIn('optimizer.step',source)
        self.assertNotIn('backward(',source)


if __name__=='__main__':unittest.main()
