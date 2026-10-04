import copy
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import torch
from savr.openvla.current_frame_corrector import CurrentFrameCorrector, teacher_imitation_loss
from savr.openvla.current_frame_training_plan import schedule, authorized_indices, offline_decision

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
import fit_current_frame_adapters as fit


def samples():
    return [dict(sample_id=f'{task}-{role}-{i}',learning_role=role,
            identity=dict(split='train',suite=f's{task//10}',task_id=task%10,
                          trajectory_id=f'{task}-{role}-{i//10}'))
            for task in range(40) for role,count in [('fit',80),('validation',20)] for i in range(count)]


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.samples=samples();cls.plan=schedule(cls.samples)

    def test_schedule_complete_deterministic_epochs(self):
        p=self.plan
        self.assertEqual(p,schedule(copy.deepcopy(self.samples)))
        self.assertEqual(len(p['epoch_orders']),10)
        self.assertEqual(sum(len(x)//16 for x in p['epoch_orders']),2000)
        for order in p['epoch_orders']:
            self.assertEqual(set(order),set(p['fit']))
        self.assertNotEqual(p['epoch_orders'][0],p['epoch_orders'][1])

    def test_shuffled_targets_within_task_deranged_fit_only(self):
        p=self.plan
        self.assertEqual(set(p['shuffled_targets'].values()),set(p['fit']))
        for source,target in p['shuffled_targets'].items():
            source=int(source);self.assertNotEqual(source,target)
            a,b=self.samples[source]['identity'],self.samples[target]['identity']
            self.assertEqual((a['suite'],a['task_id']),(b['suite'],b['task_id']))

    def test_bad_membership_and_trajectory_overlap_rejected(self):
        for change in ('duplicate','role','trajectory','split'):
            rows=copy.deepcopy(self.samples)
            if change=='duplicate': rows[1]['sample_id']=rows[0]['sample_id']
            if change=='role': rows[0]['learning_role']='validation'
            if change=='trajectory':rows[80]['identity']['trajectory_id']=rows[0]['identity']['trajectory_id']
            if change=='split': rows[0]['identity']['split']='locked'
            with self.assertRaises(ValueError): schedule(rows)

    def test_reader_role_and_validation_seal_guards_before_file_access(self):
        r=fit.Reader.__new__(fit.Reader);r.plan=self.plan;r.validation_ready=False
        with self.assertRaisesRegex(ValueError,'unauthorized'):r.get(self.plan['validation'][0],'fit')
        with self.assertRaisesRegex(ValueError,'checkpoint freeze'):r.get(self.plan['validation'][0],'validation')
        with self.assertRaises(ValueError):authorized_indices(self.plan,[],'fit')
        with self.assertRaises(ValueError):r.batch(self.plan['validation'][:2],'validation','cpu',shuffled=True)

    def test_shared_initialization_zero_and_identical_visual_control(self):
        dims=dict(input_dim=16,width=16,heads=4,blocks=2)
        states=fit.initial_states(**dims)
        for key,tensor in states[False].items():self.assertTrue(torch.equal(tensor,states[True][key]))
        again=fit.initial_states(**dims)
        self.assertEqual(fit.state_digest(states[True]),fit.state_digest(again[True]))
        self.assertEqual(float(states[True]['output.weight'].abs().sum()),0.)

    def test_actual_fp32_optimizer_and_final_reload(self):
        torch.manual_seed(7)
        dims=dict(input_dim=16,width=16,heads=4,blocks=2)
        states=fit.initial_states(**dims)
        x=dict(action_features=torch.randn(2,8,16),base_actions=torch.randn(2,8,7),
               current_visual=torch.randn(2,512,16),state=torch.randn(2,8),instruction=torch.randn(2,16))
        target=x['base_actions']+.1
        for visual in (False,True):
            model=CurrentFrameCorrector(use_visual=visual,**dims);model.load_state_dict(states[visual])
            optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=0,foreach=False)
            before=float(teacher_imitation_loss(model(**x),target))
            for _ in range(20):fit.update(model,x,target,optimizer)
            model.eval()
            with torch.no_grad(): prediction=model(**x)
            self.assertLess(float(teacher_imitation_loss(prediction,target)),before)
            with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as d:
                path=Path(d)/'final.pt';torch.save(model.state_dict(),path)
                loaded=CurrentFrameCorrector(use_visual=visual,**dims).eval()
                loaded.load_state_dict(torch.load(path,weights_only=True),strict=True)
                with torch.no_grad():self.assertTrue(torch.equal(prediction,loaded(**x)))

    def test_nonfinite_training_target_fails_closed(self):
        model=CurrentFrameCorrector(input_dim=16,width=16,heads=4)
        x=dict(action_features=torch.zeros(1,8,16),base_actions=torch.zeros(1,8,7),
               current_visual=torch.zeros(1,512,16),state=torch.zeros(1,8),instruction=torch.zeros(1,16))
        with self.assertRaises(ValueError):
            fit.update(model,x,torch.full((1,8,7),float('nan')),torch.optim.AdamW(model.parameters()))

    def test_offline_rule_does_not_claim_robot_efficacy(self):
        m=dict(base_l1=.1,visual_l1=.08,action_only_l1=.07,visual_shuffled_l1=.12)
        d=offline_decision(m)
        self.assertTrue(d['eligible_for_development_evaluation'])
        self.assertFalse(d['positive_method_result']);self.assertFalse(d['visual_better_than_action_only'])
        for bad in (.1,.12):
            self.assertFalse(offline_decision(dict(m,visual_l1=bad))['eligible_for_development_evaluation'])
        self.assertEqual(offline_decision(dict(m,base_l1=0))['reason'],'zero_teacher_discrepancy')

    def test_bad_aggregate_counts_and_nonfinite(self):
        with self.assertRaises(ValueError):fit.aggregate([])
        rows=[dict(sample_id=str(i),suite='s',**{a+'_l1':float('nan') for a in ('base',*fit.ARMS)}) for i in range(800)]
        with self.assertRaises(ValueError):fit.aggregate(rows)

    def test_independent_prediction_reconciliation(self):
        rows=[]
        for i in self.plan['validation']:
            sample=self.samples[i];identity=sample['identity']
            row=dict(sample_id=sample['sample_id'],suite=identity['suite'],task_id=identity['task_id'],
                     teacher=[[0.]*7 for _ in range(8)])
            for arm in ('base',*fit.ARMS):
                row[arm+'_prediction']=[[.1]*7 for _ in range(8)]
                row[arm+'_l1']=.1;row[arm+'_gripper_sign_disagreement']=1.
            rows.append(row)
        fit.validate_predictions(rows,self.samples,self.plan)
        rows[0]['visual_l1']=.2
        with self.assertRaises(ValueError):fit.validate_predictions(rows,self.samples,self.plan)


if __name__=='__main__':unittest.main()
