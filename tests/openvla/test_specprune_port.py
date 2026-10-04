"""CPU-only equivalence against pinned, AST-isolated upstream functions.

No upstream loader or external import side effects are executed. Source methods
run on deterministic synthetic attention tensors and fake decoder layers.
"""
import ast
from contextlib import contextmanager
import hashlib
from pathlib import Path
from types import SimpleNamespace, ModuleType, MethodType
import sys
import unittest
from unittest.mock import patch
import numpy as np
import torch
from transformers import LlamaConfig, LlamaModel

from savr.openvla.specprune import (
    EpisodeState, SpecPruneSelection, NativeAttentionScores, low_change_indices,
    prior_global_indices, task_relevant_set, pruned_decoder_forward,
)
from savr.openvla.official_semantics import derive_official_layout, select_official_action_hidden

ROOT = Path(__file__).resolve().parents[2]
UP = ROOT / "reports/specprune_source_audit_v1/upstream"
EXTRA = ROOT / "reports/specprune_algorithm_port_v1/upstream/spec_prune_vla.py"


def view_as_blocks(image, block_shape):
    """Independent slicing oracle for the source's missing skimage dependency.

    Only the pinned RGB/14-pixel use is supported. This deliberately does not
    use the production reshape/transpose patchifier. No package is installed.
    """
    if image.shape != (224, 224, 3) or block_shape != (14, 14, 3):
        raise ValueError("unsupported test-oracle block shape")
    rows = [[image[r:r+14, c:c+14].copy() for c in range(0, 224, 14)]
            for r in range(0, 224, 14)]
    return np.asarray(rows)[:, :, None, :, :, :]


def layout(prompt=34):
    mask = torch.zeros((1, prompt + 58), dtype=torch.bool)
    mask[:, prompt+1:prompt+57] = True
    return derive_official_layout(action_mask=mask, projected_tokens=513, instruction_token_indices=(2, 3))


def extract(path, names, owner=None):
    tree = ast.parse(path.read_text())
    if owner:
        tree = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == owner)
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in functions} == set(names)
    for fn in functions:
        fn.decorator_list = []
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future] + functions, type_ignores=[]))
    namespace = dict(torch=torch, np=np, view_as_blocks=view_as_blocks,
                     StaticCache=type("StaticCache", (), {}), BaseModelOutputWithPast=SimpleNamespace)
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


@contextmanager
def source_constants():
    constants = ModuleType("experiments.robot.spec_prune_constants")
    exec(compile((UP / "spec_prune_constants.py").read_text(), "pinned_constants", "exec"), vars(constants))
    actions = ModuleType("prismatic.vla.constants")
    actions.ACTION_DIM, actions.NUM_ACTIONS_CHUNK = 7, 8
    with patch.dict(sys.modules, {constants.__name__: constants, actions.__name__: actions}):
        yield


def attention_for(positions, layer, uniform=False):
    n = len(positions)
    if uniform:
        return torch.full((1, 2, n, n), 1 / n)
    values = torch.sin(positions[:, None].float() * .13 + positions[None, :].float() * .017 + layer) * .8 + 1
    values = values / values.sum(-1, keepdim=True)
    return torch.stack([values, values], dim=0).unsqueeze(0)


class FakeLayer:
    def __init__(self, index, trace, uniform):
        self.layer_idx, self.trace, self.uniform = index, trace, uniform

    def __call__(self, hidden, **kwargs):
        ids = kwargs["cache_position"]
        self.trace.append(ids.clone())
        return hidden, attention_for(ids, self.layer_idx, self.uniform)


class PortTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.source = extract(UP / "modeling_llama.py", ["forward", "_as_token_tensor", "task_relevant_set"], "LlamaModel")
        cls.extra = extract(EXTRA, ["patchify", "calculate_patch_similarity", "preprocess_image_for_patchify", "get_similarity_indices", "vlm_layer_attn", "get_layer_attn_indices"])

    def test_extra_source_identity(self):
        raw = EXTRA.read_bytes()
        self.assertEqual(hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest(),
                         "44f1ef83e18a8817135df1836305fe094797a6e1")

    def test_patch_similarity_matches_source_both_views(self):
        rng = np.random.default_rng(7)
        current = rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)
        previous = current.copy()
        previous[28:56, 70:98] = 0
        for a, b in ((current, previous), (current, current), (np.zeros_like(current), np.zeros_like(current))):
            for wrist, topk, threshold in ((False, 236, .986), (True, 230, .98), (False, 240, .986), (True, 236, .98)):
                expected = self.extra["get_similarity_indices"](a, b, topk, sim_threshold=threshold, view="wrist" if wrist else "primary")
                actual = low_change_indices(a, b, top_k=topk, threshold=threshold, wrist=wrist)
                np.testing.assert_array_equal(actual, expected)

    def test_patch_invalid_shape_and_budget(self):
        image = np.ones((224, 224, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            low_change_indices(image[:200], image, top_k=24, threshold=.98)
        with self.assertRaises(ValueError):
            low_change_indices(image, image, top_k=0, threshold=.98)

    def oracle(self, item, low, previous, precise, uniform=False, enabled=True, dynamic=True):
        trace = []
        oracle = SimpleNamespace(config=SimpleNamespace(output_attentions=True, output_hidden_states=True, use_cache=False, use_return_dict=True),
                                 gradient_checkpointing=False, training=False,
                                 dynamic_importance_last_infer_step=None, dynamic_importance_episode_key=None,
                                 dynamic_importance_confidence={}, norm=lambda x: x)
        oracle.layers = [FakeLayer(i, trace, uniform) for i in range(32)]
        oracle._update_causal_mask = lambda *args: None
        oracle._as_token_tensor = self.source["_as_token_tensor"]
        oracle.task_relevant_set = MethodType(self.source["task_relevant_set"], oracle)
        with source_constants():
            out = self.source["forward"](oracle, SimpleNamespace(token_prune=enabled, dynamic_prune=dynamic),
                inputs_embeds=torch.arange(item.full_sequence_tokens).float().reshape(1, -1, 1),
                use_cache=False, output_attentions=True, output_hidden_states=True,
                precise_mode=precise, num_prompt=item.prompt_tokens, skip_layer_list=[],
                high_similarity_indices=[p for p in low if p < 257],
                high_similarity_indices_wrist=[p for p in low if p >= 257],
                prev_attn_indices=list(previous), infer_step=1, episode_id="test")
        return oracle, out, trace

    def test_full_32_layer_selection_matches_actual_source(self):
        for prompt, precise, uniform, enabled, dynamic in (
            (12, False, False, True, True), (34, False, False, True, True),
            (70, True, False, True, True), (34, False, True, True, True),
            (34, False, False, False, True), (34, True, False, True, False),
        ):
            with self.subTest(prompt=prompt, precise=precise, uniform=uniform, enabled=enabled, dynamic=dynamic):
                item = layout(prompt)
                low = tuple(range(1, 230)) + tuple(range(257, 470))
                previous = (12, 91, 205, 289, 397)
                oracle, out, trace = self.oracle(item, low, previous, precise, uniform, enabled, dynamic)
                state = EpisodeState()
                state.reset("test")
                state.precise, state.previous_indices = precise, previous
                selector = SpecPruneSelection(item, state, low, enabled=enabled, dynamic=dynamic)
                for i in range(32):
                    self.assertTrue(torch.equal(selector.positions, trace[i]), f"layer {i}")
                    attn = attention_for(selector.positions, i, uniform) if selector.needs_attention(i) else None
                    selector.after_layer(i, attn)
                self.assertEqual(selector.positions.tolist(), out.last_hidden_state.flatten().long().tolist())
                self.assertTrue(torch.equal(selector.scores, oracle.token_importance_scores))
                selector.finish()
                if enabled:
                    topk = (15, 12) if precise else (24, 19)
                    expected_global = torch.unique(torch.cat([
                        self.extra["get_layer_attn_indices"](out.attentions, topk[int(w)]//2, primary=not w, goal_layers=[14, 30]) for w in (False, True)]))
                    self.assertEqual(state.previous_indices, tuple(expected_global.tolist()))

    def test_first_prune_precedes_importance_update(self):
        state = EpisodeState(); state.reset(1)
        selector = SpecPruneSelection(layout(), state)
        for i in range(11):
            selector.after_layer(i, attention_for(selector.positions, i) if selector.needs_attention(i) else None)
        self.assertEqual(state.confidence, {})
        self.assertLess(len(selector.positions), 605)
        visual = (selector.positions > 0) & (selector.positions < 513)
        self.assertTrue(torch.equal(selector.scores[visual], torch.zeros(int(visual.sum()))))

    def test_resets_even_repeated_one_query_episode_id(self):
        state = EpisodeState(); state.reset("same")
        state.query = 1; state.precise = True; state.previous_indices = (4, 299)
        state.confidence[14] = torch.tensor(4.)
        state.reset("same")
        self.assertEqual((state.query, state.precise, state.previous_indices, state.confidence), (0, False, (), {}))

    def test_controller_thresholds_and_signed_z(self):
        state = EpisodeState(); state.reset(1)
        self.assertFalse(state.observe_executed_action(np.zeros(7), acting_steps=10, queue_remaining=4))
        self.assertFalse(state.precise)
        state.observe_executed_action(np.array([.35, 0, 0, 0, 0, 0, 0]), acting_steps=11, queue_remaining=4)
        self.assertFalse(state.precise)
        self.assertTrue(state.observe_executed_action(np.zeros(7), acting_steps=12, queue_remaining=3))
        self.assertTrue(state.precise)
        state.observe_executed_action(np.array([0, 0, .3, 0, 0, 0, 0]), acting_steps=13, queue_remaining=2)
        self.assertTrue(state.precise)
        state.observe_executed_action(np.array([0, 0, .3, 0, 0, 0, 0]), acting_steps=14, queue_remaining=0)
        self.assertFalse(state.precise)
        state.observe_executed_action(np.array([0, 0, -.7, 0, 0, 0, 0]), acting_steps=15, queue_remaining=4)
        self.assertTrue(state.precise)  # Preserves signed-z source condition.
        state.reset(2)
        state.observe_executed_action(np.ones(7), acting_steps=9, queue_remaining=0)
        np.testing.assert_array_equal(state.action_sum, np.zeros(7))
        state.observe_executed_action(np.ones(7), acting_steps=10, queue_remaining=0)
        np.testing.assert_array_equal(state.action_sum, np.ones(7))
        self.assertFalse(state.precise)

    def test_requires_reset_and_ordered_complete_query(self):
        state = EpisodeState()
        with self.assertRaises(ValueError): SpecPruneSelection(layout(), state)
        state.reset(1)
        selector = SpecPruneSelection(layout(), state)
        with self.assertRaises(ValueError): selector.after_layer(1)
        with self.assertRaises(ValueError): selector.finish()
        with self.assertRaises(ValueError): selector.after_layer(0, torch.zeros(1, 2, 6, 6))

    def test_frame_lookback_and_confidence_persist_only_within_episode(self):
        state = EpisodeState(); state.reset(1)
        self.assertEqual(state.previous_frame_index(10), -1)
        for query in range(2):
            selector = SpecPruneSelection(layout(), state)
            for i in range(32):
                selector.after_layer(i, attention_for(selector.positions, i + query) if selector.needs_attention(i) else None)
            selector.finish()
            if query == 0:
                confidence = {k: v.clone() for k, v in state.confidence.items()}
                self.assertEqual(state.previous_frame_index(10), -6)
                self.assertEqual(state.previous_frame_index(2), -1)
            else:
                self.assertEqual(set(confidence), set(state.confidence))
                self.assertTrue(all(torch.equal(value, state.confidence[k]) for k, value in confidence.items()))
        state.action_sum[:] = 0
        state.action_sum[0] = 6
        self.assertEqual(state.previous_frame_index(10), -5)
        self.assertTrue(np.array_equal(state.action_sum, np.zeros(7)))
        state.action_sum[0] = 99
        with self.assertRaises(ValueError): state.previous_frame_index(10)

    def test_capture_rejects_non_native_masks_and_dropout(self):
        q = torch.ones(1, 2, 8, 4)
        native = torch.nn.functional.scaled_dot_product_attention
        for kwargs in ({"dropout_p": .1}, {"attn_mask": torch.ones(8, 8, dtype=torch.bool)},
                       {"attn_mask": torch.ones(1, 1, 8, 8)}):
            with self.assertRaises(ValueError):
                with NativeAttentionScores():
                    torch.nn.functional.scaled_dot_product_attention(q, q, q, **kwargs)
            self.assertIs(torch.nn.functional.scaled_dot_product_attention, native)

    def test_attention_capture_is_neutral_and_matches_manual_scores(self):
        torch.manual_seed(7)
        q, k, v = [torch.randn(1, 4, 20, 8) for _ in range(3)]
        native = torch.nn.functional.scaled_dot_product_attention
        expected = native(q, k, v, is_causal=False)
        with NativeAttentionScores() as capture:
            output = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)
        self.assertTrue(torch.equal(expected, output))
        self.assertTrue(torch.equal(capture.scores, torch.softmax(q @ k.transpose(2, 3) / np.sqrt(8), -1)))
        self.assertIs(torch.nn.functional.scaled_dot_product_attention, native)
        with self.assertRaises(ValueError):
            with NativeAttentionScores():
                torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        self.assertIs(torch.nn.functional.scaled_dot_product_attention, native)

    def test_tiny_original_decoder_keep_all_and_pruning(self):
        torch.manual_seed(7)
        cfg = LlamaConfig(vocab_size=128, hidden_size=32, intermediate_size=64,
                          num_hidden_layers=32, num_attention_heads=4, num_key_value_heads=4,
                          max_position_embeddings=1024)
        cfg._attn_implementation = "sdpa"
        model = LlamaModel(cfg).eval()
        item = layout()
        embeddings = torch.randn(1, item.full_sequence_tokens, 32)
        with torch.inference_mode():
            native = model(inputs_embeds=embeddings, use_cache=False).last_hidden_state
            state = EpisodeState(); state.reset(1)
            dense, ids = pruned_decoder_forward(model, embeddings, SpecPruneSelection(item, state, enabled=False))
            self.assertTrue(torch.equal(native, dense))
            state.reset("keep-all-with-scores")
            scored, _ = pruned_decoder_forward(model, embeddings, SpecPruneSelection(item, state, enabled=True, dynamic=False))
            self.assertTrue(torch.equal(native, scored))
            state.reset(2)
            selector = SpecPruneSelection(item, state, tuple(range(1, 237)) + tuple(range(257, 487)))
            compressed, ids = pruned_decoder_forward(model, embeddings, selector)
            self.assertLess(compressed.shape[1], embeddings.shape[1])
            self.assertTrue(torch.isfinite(compressed).all())
            self.assertEqual(select_official_action_hidden(compressed, item, ids).shape, (1, 56, 32))
            restored = model(inputs_embeds=embeddings, use_cache=False).last_hidden_state
            self.assertTrue(torch.equal(native, restored))

    def test_bfloat16_native_keep_all_and_compressed(self):
        torch.manual_seed(7)
        cfg = LlamaConfig(vocab_size=128, hidden_size=32, intermediate_size=64,
                          num_hidden_layers=32, num_attention_heads=4, num_key_value_heads=4,
                          max_position_embeddings=1024)
        cfg._attn_implementation = "sdpa"
        model = LlamaModel(cfg).eval().to(dtype=torch.bfloat16)
        item = layout()
        embeddings = torch.randn(1, item.full_sequence_tokens, 32, dtype=torch.bfloat16)
        with torch.inference_mode():
            native = model(inputs_embeds=embeddings, use_cache=False).last_hidden_state
            state = EpisodeState(); state.reset(1)
            kept, _ = pruned_decoder_forward(model, embeddings, SpecPruneSelection(item, state, dynamic=False))
            self.assertTrue(torch.equal(native, kept))
            state.reset(2)
            compressed, ids = pruned_decoder_forward(model, embeddings, SpecPruneSelection(item, state))
            self.assertLess(len(ids), item.full_sequence_tokens)
            self.assertTrue(torch.isfinite(compressed).all())


if __name__ == "__main__":
    unittest.main()
