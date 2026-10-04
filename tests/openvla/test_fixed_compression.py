import unittest
import torch
from transformers import LlamaConfig, LlamaModel
from savr.openvla.fixed_compression import fixed_decoder_forward
from savr.openvla.specprune import EpisodeState, SpecPruneSelection, pruned_decoder_forward
from savr.openvla.official_semantics import OfficialInferenceLayout, select_official_action_hidden
from savr.openvla.spatial_selection import stratified_visual_positions


def layout(prompt=34):
    return OfficialInferenceLayout(projected_tokens=513, input_tokens=prompt+58,
        full_sequence_tokens=prompt+571, prompt_tokens=prompt,
        placeholder_input_positions=tuple(range(prompt+1,prompt+57)),
        placeholder_multimodal_positions=tuple(range(prompt+514,prompt+570)),
        action_readout_positions=tuple(range(prompt+513,prompt+569)),
        instruction_input_positions=(2,3), instruction_multimodal_positions=(515,516),
        stop_position=prompt+570)


class FixedCompressionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1); torch.manual_seed(7)
        cfg=LlamaConfig(vocab_size=128, hidden_size=32, intermediate_size=64,
            num_hidden_layers=32,num_attention_heads=4,num_key_value_heads=4,max_position_embeddings=1024)
        cfg._attn_implementation='sdpa'
        self.model=LlamaModel(cfg).eval(); self.layout=layout()
        self.embeddings=torch.randn(1,self.layout.full_sequence_tokens,32)

    def test_all_tokens_match_qualified_direct_decoder_exactly(self):
        state=EpisodeState();state.reset('test')
        selector=SpecPruneSelection(self.layout,state,enabled=False)
        with torch.inference_mode():
            reference, pos=pruned_decoder_forward(self.model,self.embeddings,selector)
            actual, ids=fixed_decoder_forward(self.model,self.embeddings,self.layout,512)
        self.assertTrue(torch.equal(reference,actual));self.assertTrue(torch.equal(pos,ids))

    def test_actual_shorter_sequence_at_every_layer_and_original_positions(self):
        for budget in (384,256):
            seen=[]
            def observe(module,args,kwargs):
                seen.append((args[0].shape[1],kwargs['position_ids'].clone(),kwargs['use_cache'],kwargs['past_key_value']))
            handles=[x.register_forward_pre_hook(observe,with_kwargs=True) for x in self.model.layers]
            try:
                with torch.inference_mode():
                    hidden,ids=fixed_decoder_forward(self.model,self.embeddings,self.layout,budget)
            finally:
                for h in handles:h.remove()
            self.assertEqual(len(seen),32)
            self.assertTrue(all(n==self.layout.full_sequence_tokens-512+budget and torch.equal(p[0],ids)
                                and not cache and past is None for n,p,cache,past in seen))
            self.assertEqual(tuple(ids[(ids>=1)&(ids<=512)].tolist()),stratified_visual_positions(budget))
            self.assertTrue(torch.isfinite(hidden).all())
            self.assertEqual(select_official_action_hidden(hidden,self.layout,ids).shape,(1,56,32))

    def test_readout_and_protected_positions_for_varied_prompts(self):
        for prompt in (12,34,70):
            contract=layout(prompt); x=torch.randn(1,contract.full_sequence_tokens,32)
            with torch.inference_mode():
                _,ids=fixed_decoder_forward(self.model,x,contract,256)
            self.assertTrue({0,*range(513,contract.full_sequence_tokens)}.issubset(ids.tolist()))
            sentinel=ids.float().reshape(1,-1,1)
            actual=select_official_action_hidden(sentinel,contract,ids)
            self.assertEqual(actual.flatten().tolist(),list(contract.action_readout_positions))

    def test_bidirectional_future_influence(self):
        changed=self.embeddings.clone();changed[:,-1]=torch.randn(32)*5
        with torch.inference_mode():
            a,_=fixed_decoder_forward(self.model,self.embeddings,self.layout,256)
            b,_=fixed_decoder_forward(self.model,changed,self.layout,256)
        self.assertGreater(float((a[:,:-1]-b[:,:-1]).abs().max()),1e-6)

    def test_no_cross_query_persistence(self):
        with torch.inference_mode():
            a,_=fixed_decoder_forward(self.model,self.embeddings,self.layout,384)
            fixed_decoder_forward(self.model,self.embeddings*2,self.layout,256)
            b,_=fixed_decoder_forward(self.model,self.embeddings,self.layout,384)
        self.assertTrue(torch.equal(a,b))

    def test_reject_training_shape_layer_count_and_bad_budget(self):
        with self.assertRaises(ValueError):fixed_decoder_forward(self.model,self.embeddings,self.layout,256)
        with torch.inference_mode():
            for budget in (True,128,384.0):
                with self.assertRaises(ValueError):fixed_decoder_forward(self.model,self.embeddings,self.layout,budget)
            with self.assertRaises(ValueError):fixed_decoder_forward(self.model,self.embeddings[:,:-1],self.layout,256)
            with self.assertRaises(ValueError):fixed_decoder_forward(self.model,self.embeddings,self.layout,256,expected_layers=31)


if __name__=='__main__': unittest.main()
