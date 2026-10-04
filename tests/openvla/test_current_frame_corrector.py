import unittest
import torch
from savr.openvla.current_frame_corrector import CurrentFrameCorrector,teacher_imitation_loss

class CorrectorTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(7)
    def inputs(self,grad=False):
        return dict(action_features=torch.randn(2,8,16,requires_grad=grad),
            base_actions=torch.randn(2,8,7,requires_grad=grad),
            current_visual=torch.randn(2,512,16,requires_grad=grad),
            state=torch.randn(2,8,requires_grad=grad),instruction=torch.randn(2,16,requires_grad=grad))
    def model(self,visual=True):return CurrentFrameCorrector(input_dim=16,width=16,heads=4,use_visual=visual)
    def test_zero_init_is_exact_base_and_batch_shape(self):
        for visual in (True,False):
            x=self.inputs();y=self.model(visual)(**x)
            self.assertTrue(torch.equal(y,x['base_actions']))
    def test_no_backward_into_backbone_or_teacher(self):
        x=self.inputs(True);m=self.model();target=torch.randn(2,8,7,requires_grad=True)
        teacher_imitation_loss(m(**x),target).backward()
        self.assertTrue(all(v.grad is None for v in x.values()))
        self.assertIsNone(target.grad)
        self.assertGreater(float(m.output.weight.grad.abs().sum()),0)
    def test_inference_tensors_can_train_without_backbone_graph(self):
        with torch.inference_mode():x=self.inputs()
        m=self.model();teacher_imitation_loss(m(**x),torch.ones(2,8,7)).backward()
        self.assertIsNotNone(m.output.weight.grad)
    def test_tiny_synthetic_fit_reduces_loss(self):
        m=self.model();x=self.inputs();target=x['base_actions']+.2
        optimizer=torch.optim.Adam(m.parameters(),lr=.005)
        before=float(teacher_imitation_loss(m(**x),target))
        for _ in range(20):
            optimizer.zero_grad();loss=teacher_imitation_loss(m(**x),target);loss.backward();optimizer.step()
        self.assertLess(float(teacher_imitation_loss(m(**x),target)),before*.5)
        self.assertGreater(float(m.visual_projection.weight.grad.abs().sum()),0)
    def test_visual_input_matters_after_output_is_nonzero(self):
        m=self.model();torch.nn.init.normal_(m.output.weight,std=.1)
        x=self.inputs();a=m(**x);x['current_visual']=x['current_visual']+2
        self.assertFalse(torch.allclose(a,m(**x)))
    def test_action_only_does_not_consume_visual_input(self):
        m=self.model(False);x=self.inputs();x['current_visual']=None
        self.assertEqual(m(**x).shape,(2,8,7))
        self.assertFalse(any('visual' in k or 'attention' in k for k in m.state_dict()))
    def test_invalid_features_and_labels(self):
        m=self.model();x=self.inputs();x['current_visual']=torch.zeros(2,32,16)
        with self.assertRaises(ValueError):m(**x)
        x=self.inputs();x['state'][0,0]=float('nan')
        with self.assertRaises(ValueError):m(**x)
        with self.assertRaises(ValueError):teacher_imitation_loss(torch.zeros(2,8,7),torch.zeros(2,7,7))

if __name__=='__main__':unittest.main()
