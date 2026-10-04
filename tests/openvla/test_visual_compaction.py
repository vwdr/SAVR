"""CPU qualification, including a tiny random LLaMA on the original runtime.

No pretrained weights, task outcomes, GPU, or evidence of method efficacy.
"""
import unittest

from savr.openvla.official_semantics import (
    OfficialInferenceLayout, OpenVLASemanticError, select_official_action_hidden,
)
from savr.openvla.visual_compaction import VisualCompaction


def layout(prompt=34):
    # Independently construct the pinned layout: BOS, 513 inserted tokens,
    # prompt, 56 placeholders, stop. Readout precedes placeholders by one.
    placeholders = tuple(range(prompt + 1, prompt + 57))
    return OfficialInferenceLayout(
        projected_tokens=513, input_tokens=prompt + 58,
        full_sequence_tokens=prompt + 571, prompt_tokens=prompt,
        placeholder_input_positions=placeholders,
        placeholder_multimodal_positions=tuple(p + 513 for p in placeholders),
        action_readout_positions=tuple(range(prompt + 513, prompt + 569)),
        instruction_input_positions=(2, 3), instruction_multimodal_positions=(515, 516),
        stop_position=prompt + 570,
    )


class MapTests(unittest.TestCase):
    def test_keep_all_is_identity(self):
        dense = VisualCompaction.dense(layout())
        self.assertEqual(dense.retain_visual(range(1, 513)), dense)
        self.assertEqual(dense.offsets_from(dense), tuple(range(605)))

    def test_repeated_compaction_preserves_action_and_nonvisual_tokens(self):
        for prompt in (12, 34, 70):
            dense = VisualCompaction.dense(layout(prompt))
            first = dense.retain_visual(range(1, 513, 2))
            second = first.retain_visual((1, 129, 257, 385))
            self.assertEqual(tuple(second.absolute_positions[i] for i in second.action_offsets()),
                             dense.layout.action_readout_positions)
            self.assertTrue({0, *range(513, dense.layout.full_sequence_tokens)}.issubset(second.absolute_positions))
            self.assertEqual(tuple(first.absolute_positions[i] for i in second.offsets_from(first)),
                             second.absolute_positions)

    def test_tail_offset_is_not_the_official_readout(self):
        selected = VisualCompaction.dense(layout()).retain_visual((1, 257))
        self.assertNotEqual(selected.absolute_positions[-57:-1], selected.layout.action_readout_positions)
        self.assertEqual(selected.absolute_positions[-58:-2], selected.layout.action_readout_positions)

    def test_reject_removed_nonvisual_token(self):
        dense = VisualCompaction.dense(layout())
        for protected in (0, 513, 515, 547, 603, 604):
            with self.assertRaisesRegex(OpenVLASemanticError, "protected"):
                VisualCompaction(dense.layout, tuple(p for p in dense.absolute_positions if p != protected))

    def test_reject_invalid_visual_selection(self):
        dense = VisualCompaction.dense(layout())
        for values in ((1, 1, 257), (0, 257), (1, 513), (True, 257), (1.0, 257)):
            with self.assertRaises(OpenVLASemanticError):
                dense.retain_visual(values)

    def test_reject_camera_deletion(self):
        dense = VisualCompaction.dense(layout())
        for values in ((), (1,), (257,)):
            with self.assertRaisesRegex(OpenVLASemanticError, "both cameras"):
                dense.retain_visual(values)

    def test_no_token_resurrection(self):
        dense = VisualCompaction.dense(layout())
        selected = dense.retain_visual((1, 257))
        with self.assertRaisesRegex(OpenVLASemanticError, "resurrect"):
            selected.retain_visual((1, 2, 257))
        with self.assertRaisesRegex(OpenVLASemanticError, "resurrect"):
            dense.offsets_from(selected)

    def test_reject_cross_query_layout(self):
        with self.assertRaisesRegex(OpenVLASemanticError, "query layouts"):
            VisualCompaction.dense(layout(12)).offsets_from(VisualCompaction.dense(layout(34)))

    def test_reject_corrupt_position_map(self):
        dense = VisualCompaction.dense(layout())
        for positions in (tuple(reversed(dense.absolute_positions)), dense.absolute_positions + (604,),
                          (-1,) + dense.absolute_positions, dense.absolute_positions + (605,)):
            with self.assertRaisesRegex(OpenVLASemanticError, "sorted unique"):
                VisualCompaction(dense.layout, positions)


try:
    import torch
    from transformers import LlamaConfig, LlamaModel
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "requires existing Torch/Transformers project runtime")
class TensorTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        cfg = LlamaConfig(vocab_size=128, hidden_size=32, intermediate_size=64,
                          num_hidden_layers=2, num_attention_heads=4,
                          num_key_value_heads=4, max_position_embeddings=1024)
        cfg._attn_implementation = "sdpa"
        self.model = LlamaModel(cfg).eval()
        self.dense = VisualCompaction.dense(layout())
        self.embeddings = torch.randn(1, 605, 32)

    def test_dense_explicit_positions_match_native_dense(self):
        with torch.inference_mode():
            original = self.model(inputs_embeds=self.embeddings,
                                  attention_mask=torch.ones((1, 605), dtype=torch.long),
                                  use_cache=False).last_hidden_state
            mapped = self.model(**self.dense.fresh_decoder_arguments(torch, self.embeddings)).last_hidden_state
        self.assertTrue(torch.equal(original, mapped))

    def test_compact_real_runtime_and_readout(self):
        selected = self.dense.retain_visual(range(1, 513, 2))
        kwargs = selected.fresh_decoder_arguments(torch, self.embeddings)
        self.assertEqual(kwargs["inputs_embeds"].shape[1], 349)
        self.assertEqual(kwargs["position_ids"].tolist()[0], list(selected.absolute_positions))
        with torch.inference_mode():
            output = self.model(**kwargs)
        self.assertIsNone(output.past_key_values)
        self.assertTrue(torch.isfinite(output.last_hidden_state).all())
        hidden = select_official_action_hidden(output.last_hidden_state, selected.layout, kwargs["cache_position"])
        self.assertEqual(tuple(hidden.shape), (1, 56, 32))
        self.assertTrue(torch.equal(hidden, output.last_hidden_state[:, list(selected.action_offsets()), :]))

    def test_repeated_tensor_compaction_keeps_sentinels(self):
        hidden = torch.arange(605).reshape(1, 605, 1)
        first = self.dense.retain_visual(range(1, 513, 2))
        second = first.retain_visual((1, 257))
        reduced = second.compact_hidden(torch, first.compact_hidden(torch, hidden, self.dense), first)
        self.assertEqual(reduced.flatten().tolist(), list(second.absolute_positions))

    def test_reject_wrong_hidden_shape(self):
        with self.assertRaisesRegex(OpenVLASemanticError, "hidden sequence"):
            self.dense.compact_hidden(torch, self.embeddings[:, :-1], self.dense)

    def test_future_token_still_influences_earlier_tokens_after_compaction(self):
        selected = self.dense.retain_visual((1, 257))
        changed = self.embeddings.clone()
        changed[:, -1] = torch.randn(32) * 5
        with torch.inference_mode():
            first = self.model(**selected.fresh_decoder_arguments(torch, self.embeddings)).last_hidden_state
            second = self.model(**selected.fresh_decoder_arguments(torch, changed)).last_hidden_state
        self.assertGreater(float((first[:, :-1] - second[:, :-1]).abs().max()), 1e-6)


if __name__ == "__main__":
    unittest.main()
