"""Execute the real pinned SDPA layers, not boolean-only record fixtures."""
import unittest
import torch
from transformers import LlamaConfig, LlamaModel
from savr.openvla.official_semantics import derive_official_layout
from savr.openvla.fixed_compression import fixed_decoder_forward
from savr.openvla.adaptive_query import count_sdpa
from savr.openvla.adaptive_sdpa import adaptive_sdpa_forward
from savr.openvla.vlapruner_adaptive import AdaptiveSelectionConfig, AdaptiveQueryState


class AdaptiveSDPATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        cfg = LlamaConfig(hidden_size=8, intermediate_size=16, num_hidden_layers=32,
            num_attention_heads=2, num_key_value_heads=2, vocab_size=32, attention_dropout=0.)
        cfg._attn_implementation = 'sdpa'
        cls.model = LlamaModel(cfg).eval()
        mask = torch.zeros((1, 92), dtype=torch.bool)
        mask[:, 35:91] = True
        cls.layout = derive_official_layout(action_mask=mask, projected_tokens=513,
                                            instruction_token_indices=(2, 3))
        cls.emb = torch.randn(1, cls.layout.full_sequence_tokens, 8)

    @torch.inference_mode()
    def test_zero_and_startup_are_exact_native_sdpa(self):
        dense, pos = fixed_decoder_forward(self.model, self.emb, self.layout, 512)
        for ratio in (0., .25, .5):
            config = AdaptiveSelectionConfig(fastv_r=ratio)
            state = AdaptiveQueryState(config)
            state.reset()
            for _ in range(3):
                original = torch.nn.functional.scaled_dot_product_attention
                with count_sdpa() as counted:
                    out = adaptive_sdpa_forward(self.model, self.emb, self.layout, config, state)
                self.assertIs(torch.nn.functional.scaled_dot_product_attention, original)
                self.assertEqual(counted[0], 32)
                self.assertTrue(torch.equal(out.hidden, dense))
                self.assertTrue(torch.equal(out.positions, pos))
                self.assertEqual(out.retained_visual_tokens, 512)
                self.assertEqual([i for i, x in enumerate(out.attentions) if x is not None], [3, 15])

    @torch.inference_mode()
    def test_actual_fourth_query_prunes_and_history_resets(self):
        for ratio, budget in ((.25, 384), (.5, 256)):
            cfg = AdaptiveSelectionConfig(fastv_r=ratio)
            state = AdaptiveQueryState(cfg)
            state.reset()
            retained = []
            for _ in range(4):
                out = adaptive_sdpa_forward(self.model, self.emb, self.layout, cfg, state)
                retained.append(out.retained_visual_tokens)
            self.assertEqual(retained, [512, 512, 512, budget])
            self.assertEqual(out.hidden.shape[1], self.layout.full_sequence_tokens - 512 + budget)
            self.assertTrue(out.pruned)
            self.assertTrue(torch.isfinite(out.hidden).all())
            self.assertEqual(len(state.history), 3)
            state.reset()
            self.assertEqual(state.query, 0)
            self.assertEqual(len(state.history), 0)
            self.assertEqual(state.effective_prune_ratio(), 0.)


if __name__ == '__main__':
    unittest.main()
