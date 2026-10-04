import unittest
from types import SimpleNamespace
import torch
from savr.openvla.current_frame_features import head_features, pack_features, CorrectedCurrentFrameQuery
from savr.openvla.current_frame_corrector import CurrentFrameCorrector


class FeatureTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)

    def test_head_capture_exact_and_removed_on_error(self):
        layer = torch.nn.Linear(4096, 7)
        head = SimpleNamespace(model=SimpleNamespace(fc2=layer))
        head.predict_action = lambda x: layer(x)
        x = torch.randn(1, 8, 4096)
        z, y = head_features(head, x)
        self.assertTrue(torch.equal(z, x))
        self.assertTrue(torch.equal(y, layer(x)))
        self.assertFalse(layer._forward_pre_hooks)
        with self.assertRaises(ValueError): head_features(head, x[:, :7])
        self.assertFalse(layer._forward_pre_hooks)

    def test_feature_order_precision_and_no_proprio_in_visual(self):
        patches = torch.zeros(1, 513, 4096, dtype=torch.bfloat16)
        patches[:, :256] = 1; patches[:, 256:512] = 2; patches[:, 512] = 99
        emb = torch.arange(10).view(1, 10, 1).expand(1, 10, 4096).to(torch.bfloat16)
        prepared = SimpleNamespace(projected_patches=patches, input_embeddings=emb,
            instruction_token_indices=(2, 3, 4), normalized_proprio=torch.ones(1, 8, dtype=torch.bfloat16))
        z = torch.ones(1, 8, 4096, dtype=torch.bfloat16)
        values = pack_features(prepared, z, torch.zeros(1, 8, 7, dtype=torch.bfloat16))
        self.assertEqual(values['current_visual'].max().item(), 2)
        self.assertTrue((values['instruction'] == 3).all())
        self.assertEqual(values['state'].dtype, torch.float32)
        with self.assertRaises(ValueError): pack_features(prepared, z.float(), torch.zeros(1, 8, 7))

    def test_zero_init_wrapper_preserves_actions_and_query_once(self):
        features = dict(action_features=torch.randn(8, 4096, dtype=torch.bfloat16),
            current_visual=torch.randn(512, 4096, dtype=torch.bfloat16),
            instruction=torch.randn(4096, dtype=torch.bfloat16), state=torch.randn(8),
            base_actions=torch.randn(8, 7))
        class Query:
            def __init__(self):
                self.features = features
                self.model = SimpleNamespace(_unnormalize_actions=lambda x, key: x * 2)
                self.cfg = SimpleNamespace(unnorm_key='test')
            def __call__(self, obs, text, previous, state):
                state.query += 1; self.last_query = dict(query=state.query)
        for visual in (False, True):
            state = SimpleNamespace(query=0)
            model = CurrentFrameCorrector(use_visual=visual).eval()
            wrapper = CorrectedCurrentFrameQuery(Query(), model)
            result = wrapper(None, '', None, state)
            self.assertTrue(torch.equal(torch.from_numpy(result), features['base_actions']*2))
            self.assertEqual(state.query, 1)


if __name__ == '__main__': unittest.main()
