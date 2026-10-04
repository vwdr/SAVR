"""CPU-only verification of the VLA-Pruner adaptive port.

Two layers of checks:

1. **Selection parity against the pinned release source.** The selected
   methods of the vendored release file (``modeling_llama.py`` /
   ``modeling_prismatic.py`` at commit
   ``84d4b7192c77abf1585610e2f12393319b7ebff9``) are AST-isolated and run on
   deterministic synthetic tensors; the port must reproduce their indices and
   scores. The upstream source lives in the approved scratch directory
   (outside the repo; never committed); these tests skip when it is absent.

2. **Stack behavior.** Retained-token counts per retention arm, warm-start
   (first ``av_hist_w`` queries after a reset run dense), per-episode history
   reset, layer-15 EMA record, and dense/adaptive byte-parity on tiny
   Transformers 4.40.1 LlamaModel instances (32 SDPA layers, CPU).

No upstream loader, checkpoint, install, or GPU is used.
"""
import ast
from collections import deque
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace, MethodType
import unittest

import numpy as np
import torch
from transformers import LlamaConfig, LlamaModel

from savr.openvla.official_semantics import (
    derive_official_layout,
    select_official_action_hidden,
)
from savr.openvla.adaptive_query import count_sdpa
from savr.openvla.attention_diagnostic import LlamaAttentionDiagnostic
from savr.openvla.vlapruner_adaptive import (
    ACTION_DIM,
    AdaptiveForwardResult,
    AdaptiveQueryState,
    AdaptiveSelectionConfig,
    ActionAttentionHistory,
    build_fastv_config,
    fastv_pruning_indices,
    redundancy_minization,
    slice_range,
    vlapruner_image_indices,
    adaptive_decoder_forward,
)

ROOT = Path(__file__).resolve().parents[2]
_SOURCE_DIR = Path(
    os.environ.get(
        "VLA_PRUNER_SRC",
        "/private/var/folders/1k/ct_z8c1d3wn3jv4p559_b3dm0000gn/T/opencode/vla_pruner_src",
    )
)
UP_MODELING_LLAMA = _SOURCE_DIR / "src_openvla-oft_transformers_src_transformers_models_llama_modeling_llama.py"
UP_MODELING_PRISMATIC = _SOURCE_DIR / "src_openvla-oft_prismatic_extern_hf_modeling_prismatic.py"

UPSTREAM_LLAMA_SHA1 = "b245f276dcfe9895fed160ff53b95ab7c9caa1f1"
UPSTREAM_PRISMATIC_SHA1 = "b6222c12b5c3c1552a59a046f866f4c7ce4f2f09"


def layout(prompt: int = 34):
    mask = torch.zeros((1, prompt + 58), dtype=torch.bool)
    mask[:, prompt + 1 : prompt + 57] = True
    return derive_official_layout(
        action_mask=mask, projected_tokens=513, instruction_token_indices=(2, 3)
    )


def extract(path, names, owner=None):
    """AST-isolate named methods from an upstream source class."""
    tree = ast.parse(path.read_text())
    if owner:
        tree = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == owner)
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in functions} == set(names), f"missing upstream methods: {names}"
    for fn in functions:
        fn.decorator_list = []
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future] + functions, type_ignores=[]))
    namespace = dict(torch=torch, np=np)
    exec(compile(module, str(path), "exec"), namespace)
    return {name: namespace[name] for name in names}


def seeded_attention_avg(seq, seed, scale=None, uniform=False):
    """Deterministic (seq, seq) row-stochastic attention mean over heads."""
    rng = np.random.default_rng(seed)
    if uniform:
        values = np.full((seq, seq), 1.0 / seq)
    else:
        rows = np.sin(np.arange(seq)[:, None] * 0.13 + np.arange(seq)[None, :] * 0.017 + seed) * 0.8 + 1.0
        rows = rows + rng.standard_normal((seq, seq)) * 0.02
        values = rows / rows.sum(axis=1, keepdims=True)
    t = torch.tensor(values, dtype=torch.float64)
    if scale is not None:
        t = t.to(dtype=scale)
    return t


def seeded_embeddings(seq, dim, seed, dtype=torch.float32):
    rng = np.random.default_rng(seed)
    t = torch.tensor(rng.standard_normal((1, seq, dim)).astype(np.float32))
    return t.to(dtype=dtype)


def same_config_dict(cfg: AdaptiveSelectionConfig, layout_item, ratio, historical, device, dtype):
    """The exact config dict our ``build_fastv_config`` would build for a query."""
    return build_fastv_config(
        config=cfg,
        action_start=int(layout_item.action_readout_positions[0]),
        action_end=int(layout_item.action_readout_positions[0]) + 56,
        prune_ratio=ratio,
        historical_attention=historical,
    )


class SourceSelectionOracle:
    """AST-isolated selection methods bound to minimal state (upstream code)."""

    def __init__(self, methods: dict):
        self.ns = SimpleNamespace()
        for name, fn in methods.items():
            if name == "_slice_range":  # upstream @staticmethod
                setattr(self.ns, name, fn)
            else:
                setattr(self.ns, name, MethodType(fn, self.ns))

    def call(self, name, *args, **kwargs):
        return getattr(self.ns, name)(*args, **kwargs)


@unittest.skipUnless(
    UP_MODELING_LLAMA.exists() and UP_MODELING_PRISMATIC.exists(),
    "pinned VLA-Pruner source not present (approved scratch dir)",
)
class ReleasedSourceParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.llama = extract(
            UP_MODELING_LLAMA,
            [
                "_slice_range",
                "_visual_token_spans",
                "_fastv_scores",
                "_redundancy_minization",
                "_vlapruner_image_indices",
                "_fastv_pruning_indices",
            ],
            "LlamaModel",
        )
        cls.prismatic = extract(
            UP_MODELING_PRISMATIC,
            ["_build_historical_action_attention", "_update_action_attention_history"],
            "OpenVLAForActionPrediction",
        )
        cls.selection = SourceSelectionOracle(cls.llama)

    def test_pinned_source_hash(self):
        for path, expected in (
            (UP_MODELING_LLAMA, UPSTREAM_LLAMA_SHA1),
            (UP_MODELING_PRISMATIC, UPSTREAM_PRISMATIC_SHA1),
        ):
            raw = path.read_bytes()
            self.assertEqual(
                hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest(),
                expected,
                f"unexpected pinned source bytes in {path.name}",
            )

    def test_slice_range_matches(self):
        for start, end, limit in ((1, 513, 605), (-5, 700, 605), (300, 100, 605), (0, 0, 605)):
            self.assertEqual(slice_range(start, end, limit), self.selection.call("_slice_range", start, end, limit))

    def test_redundancy_minization_matches(self):
        torch.manual_seed(7)
        for n, keep, dtype in ((40, 30, torch.float32), (20, 12, torch.float32), (33, 20, torch.bfloat16), (10, 10, torch.float32)):
            features = torch.randn(n, 8, dtype=dtype)
            expected = self.selection.call("_redundancy_minization", features, keep)
            actual = redundancy_minization(features, keep)
            self.assertTrue(torch.equal(actual, expected), f"MMDP mismatch n={n} keep={keep}")
            self.assertEqual(actual.numel(), keep)
            self.assertEqual(len(torch.unique(actual)), keep)

    def test_selection_matches_released_source(self):
        item = layout(34)
        seq = item.full_sequence_tokens
        dtype_pairs = [(torch.float32, torch.float32), (torch.bfloat16, torch.bfloat16)]
        for ratio in (0.25, 0.5):
            cfg = AdaptiveSelectionConfig(fastv_r=ratio)
            for hist in (None, "guided"):
                for uniform in (False, True):
                    for attn_dtype, embed_dtype in dtype_pairs:
                        with self.subTest(ratio=ratio, hist=hist, uniform=uniform, dtype=attn_dtype):
                            attention_avg = seeded_attention_avg(seq, 7 + int(ratio * 100), attn_dtype, uniform)
                            inputs_embeds = seeded_embeddings(seq, 8, 3, embed_dtype)
                            device = attention_avg.device
                            historical = (
                                None
                                if hist is None
                                else torch.rand(512, dtype=torch.float32)
                            )
                            fastv = same_config_dict(cfg, item, ratio, historical, device, attn_dtype)
                            expected_keep, expected_info = self.selection.call(
                                "_fastv_pruning_indices",
                                attention_avg,
                                seq,
                                fastv,
                                inputs_embeds=inputs_embeds,
                            )
                            actual_keep, actual_info = fastv_pruning_indices(
                                attention_avg, seq, fastv, inputs_embeds=inputs_embeds
                            )
                            self.assertTrue(torch.equal(actual_keep, expected_keep), "keep_indices differ")
                            for key in (
                                "image_token_start_index",
                                "image_token_length",
                                "num_keep",
                                "kept_visual_tokens",
                                "mode",
                                "image_spans",
                            ):
                                self.assertEqual(actual_info[key], expected_info[key], f"info[{key}] differs")
                            for key in ("current_action_scores",):
                                if key in expected_info:
                                    self.assertTrue(
                                        torch.equal(actual_info[key], expected_info[key]),
                                        f"info[{key}] tensor differs",
                                    )
                            # Retained-token arithmetic (verified counts).
                            self.assertEqual(actual_info["kept_visual_tokens"], cfg.expected_retained_visual())

    def test_span_indices_match(self):
        item = layout(34)
        seq = item.full_sequence_tokens
        attention_avg = seeded_attention_avg(seq, 11, torch.float32, False)
        inputs_embeds = seeded_embeddings(seq, 8, 5, torch.float32)
        for ratio in (0.25, 0.5):
            cfg = AdaptiveSelectionConfig(fastv_r=ratio)
            historical = torch.rand(cfg.num_visual_tokens)
            fastv = same_config_dict(cfg, item, ratio, historical, attention_avg.device, torch.float32)
            for span in ((1, 257), (257, 513)):
                span_keep = int(round(256 * (1.0 - ratio)))
                expected, expected_si = self.selection.call(
                    "_vlapruner_image_indices",
                    attention_avg,
                    span[0],
                    span[1],
                    seq,
                    span_keep,
                    fastv,
                    inputs_embeds=inputs_embeds,
                )
                actual, actual_si = vlapruner_image_indices(
                    attention_avg, span[0], span[1], seq, span_keep, fastv, inputs_embeds=inputs_embeds
                )
                self.assertTrue(torch.equal(actual, expected), f"span {span} differs")
                for key in ("prefill_scores", "current_action_scores", "action_scores", "candidate_indices"):
                    if key in expected_si:
                        self.assertTrue(torch.equal(actual_si[key], expected_si[key]), f"span info {key} differs")

    def test_scores_are_raw_means_not_normalized(self):
        """The executed path must NOT min-max normalize or weight 0.5/0.5."""
        item = layout(34)
        seq = item.full_sequence_tokens
        action_start = int(item.action_readout_positions[0])
        action_end = action_start + 56
        attention_avg = torch.zeros(seq, seq, dtype=torch.float32)
        # Span 1: prefill rows vote 0..255 across the 256 columns; action rows
        # vote a different scale (500.0, with one 0.0) so any halving (0.5/0.5
        # weighting) or rescaling would change the recorded score values.
        attention_avg[:action_start, 1:257] = torch.arange(256, dtype=torch.float32)
        attention_avg[action_start:action_end, 1:257] = torch.tensor(
            [0.0] + [500.0] * 255, dtype=torch.float32
        )
        # Span 2: flat votes.
        attention_avg[:action_start, 257:513] = 1.0
        attention_avg[action_start:action_end, 257:513] = 2.0
        cfg = AdaptiveSelectionConfig(fastv_r=0.25)
        fastv = same_config_dict(cfg, item, 0.25, None, attention_avg.device, torch.float32)
        keep, info = fastv_pruning_indices(
            attention_avg, seq, fastv, inputs_embeds=seeded_embeddings(seq, 8, 2)
        )
        span1 = (1, 257)
        _, span1_info = vlapruner_image_indices(
            attention_avg, span1[0], span1[1], seq, 192, fastv, inputs_embeds=seeded_embeddings(seq, 8, 2)
        )
        expected_prefill = attention_avg[:action_start, 1:257].mean(dim=0)
        expected_action = attention_avg[action_start:action_end, 1:257].mean(dim=0)
        self.assertTrue(
            torch.allclose(span1_info["prefill_scores"].float(), expected_prefill, atol=1e-6),
            "prefill scores are not the raw row means",
        )
        self.assertTrue(
            torch.allclose(span1_info["current_action_scores"].float(), expected_action, atol=1e-6),
            "action scores are not the raw row means",
        )
        # Explicitly reject the dead-code behaviors: 0.5/0.5 weighting and
        # min-max normalization must not appear in the recorded values.
        self.assertFalse(
            torch.allclose(span1_info["current_action_scores"].float(), 0.5 * expected_action, atol=1e-6)
        )
        normalized = (expected_action - expected_action.min()) / (expected_action.max() - expected_action.min())
        self.assertFalse(
            torch.allclose(span1_info["current_action_scores"].float(), normalized, atol=1e-5)
        )
        self.assertEqual(tuple(info["image_spans"][0]), (1, 257, 192))
        self.assertEqual(info["kept_visual_tokens"], 384)

    def test_historical_builder_matches(self):
        item = layout(34)
        hist = deque(maxlen=3)
        for seed in (1, 2, 3):
            scores = torch.tensor(np.random.default_rng(seed).standard_normal(512), dtype=torch.float32)
            hist.append(scores)
        expected = self.prismatic["_build_historical_action_attention"](
            SimpleNamespace(use_temporal=True, av_hist=hist, av_decay=0.8), torch.device("cpu"), torch.float32
        )
        ours = ActionAttentionHistory(3, 0.8)
        for seed in (1, 2, 3):
            ours.append(torch.tensor(np.random.default_rng(seed).standard_normal(512), dtype=torch.float32))
        actual = ours.build_guided(torch.device("cpu"), torch.float32)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))
        # Not ready before the deque is full.
        self.assertIsNone(ActionAttentionHistory(3, 0.8).build_guided(torch.device("cpu"), torch.float32))

    def test_layer15_history_record_matches_dense_and_pruned(self):
        item = layout(34)
        action_start = int(item.action_readout_positions[0])
        action_end = action_start + 56
        visual_start, visual_end = 1, 513
        torch.manual_seed(7)
        layer15_full = torch.randn(1, 4, 605, 605, dtype=torch.float32)

        # Dense branch (kept_indices None).
        ours = AdaptiveQueryState(AdaptiveSelectionConfig(fastv_r=0.25))
        ours.reset()
        ours.record_layer15(layer15_full, None, None, item)
        oracle_hist = deque(maxlen=3)
        upstream_meta = dict(
            visual_token_start=visual_start,
            visual_token_end=visual_end,
            action_token_start=action_start,
            action_token_end=action_end,
            action_dim=ACTION_DIM,
            num_visual_tokens=512,
        )
        oracle_model = SimpleNamespace(
            use_temporal=True,
            vla_pruner_layer=15,
            vla_pruner_action_horizon=0,
            av_hist=oracle_hist,
            av_decay=0.8,
        )
        self.prismatic["_update_action_attention_history"](
            oracle_model, SimpleNamespace(attentions=(None,) * 15 + (layer15_full,), pruning_info={}), upstream_meta
        )
        self.assertTrue(torch.equal(ours.history._items[0], oracle_hist[0]))

        # Pruned branch: layer 15 runs on the shortened sequence.
        keep = torch.tensor([0] + list(range(1, 200)) + list(range(210, 300)) + list(range(513, 605))).sort()
        keep = keep.values
        layer15_pruned = torch.randn(1, 4, keep.numel(), keep.numel(), dtype=torch.float32)
        ours = AdaptiveQueryState(AdaptiveSelectionConfig(fastv_r=0.25))
        ours.reset()
        ours.record_layer15(layer15_pruned, keep, 3, item)
        oracle_hist2 = deque(maxlen=3)
        oracle_model2 = SimpleNamespace(
            use_temporal=True,
            vla_pruner_layer=15,
            vla_pruner_action_horizon=0,
            av_hist=oracle_hist2,
            av_decay=0.8,
        )
        self.prismatic["_update_action_attention_history"](
            oracle_model2,
            SimpleNamespace(attentions=(None,) * 15 + (layer15_pruned,), pruning_info={"kept_indices": keep, "pruning_layer": 3}),
            upstream_meta,
        )
        self.assertTrue(torch.equal(ours.history._items[0], oracle_hist2[0]))
        # Removed visual positions stay exactly zero; kept positions are nonzero.
        vec = ours.history._items[0]
        kept_visual_rel = keep[(keep >= visual_start) & (keep < visual_end)] - visual_start
        all_visual_rel = torch.arange(visual_end - visual_start, dtype=keep.dtype)
        pruned_visual = all_visual_rel[~torch.isin(all_visual_rel, kept_visual_rel)]
        self.assertGreater(pruned_visual.numel(), 0)
        self.assertTrue(torch.all(vec[pruned_visual] == 0))
        self.assertTrue(torch.all(vec[kept_visual_rel] != 0))


class AdaptiveBehaviorTests(unittest.TestCase):
    def test_config_validation(self):
        with self.assertRaises(RuntimeError):
            AdaptiveSelectionConfig(fastv_r=1.5)
        with self.assertRaises(RuntimeError):
            AdaptiveSelectionConfig(fastv_r=-0.1)
        with self.assertRaises(RuntimeError):
            AdaptiveSelectionConfig(fastv_r=0.25, vla_pruner_mode="bogus")
        self.assertEqual(AdaptiveSelectionConfig(fastv_r=0.0).expected_retained_visual(), 512)
        self.assertEqual(AdaptiveSelectionConfig(fastv_r=0.25).expected_retained_visual(), 384)
        self.assertEqual(AdaptiveSelectionConfig(fastv_r=0.5).expected_retained_visual(), 256)

    def test_first_three_queries_dense_then_pruned(self):
        cfg = AdaptiveSelectionConfig(fastv_r=0.25)
        state = AdaptiveQueryState(cfg)
        with self.assertRaises(RuntimeError):
            state.effective_prune_ratio()
        state.reset()
        # One forward appends one history entry; the effective ratio follows the
        # history length, exactly like the release's `_build_fastv_config`.
        self.assertEqual(state.effective_prune_ratio(), 0.0)
        state.history.append(torch.zeros(cfg.num_visual_tokens))
        self.assertEqual(state.effective_prune_ratio(), 0.0)
        state.history.append(torch.zeros(cfg.num_visual_tokens))
        self.assertEqual(state.effective_prune_ratio(), 0.0)
        state.history.append(torch.zeros(cfg.num_visual_tokens))
        self.assertEqual(state.effective_prune_ratio(), cfg.fastv_r)
        # Query counter is observation only; it never drives the ratio.
        state.query += 1
        self.assertEqual(state.effective_prune_ratio(), cfg.fastv_r)

    def test_history_reset_prevents_cross_episode_leakage(self):
        cfg = AdaptiveSelectionConfig(fastv_r=0.5)
        state = AdaptiveQueryState(cfg)
        state.reset()
        for _ in range(3):
            state.history.append(torch.zeros(512))
        self.assertEqual(len(state.history), 3)
        state.reset()
        self.assertEqual(len(state.history), 0)
        self.assertEqual(state.effective_prune_ratio(), 0.0)

    def test_guided_requires_full_history(self):
        hist = ActionAttentionHistory(3, 0.8)
        for seed in (1, 2):
            hist.append(torch.randn(512))
        self.assertIsNone(hist.build_guided(torch.device("cpu"), torch.float32))
        hist.append(torch.randn(512))
        guided = hist.build_guided(torch.device("cpu"), torch.float32)
        self.assertIsNotNone(guided)
        self.assertEqual(tuple(guided.shape), (512,))

    def test_non_temporal_mode_never_prunes_warm(self):
        cfg = AdaptiveSelectionConfig(fastv_r=0.25, vla_pruner_mode="semantic")
        state = AdaptiveQueryState(cfg)
        state.reset()
        self.assertEqual(state.effective_prune_ratio(), 0.25)


def tiny_model(dtype):
    torch.manual_seed(7)
    cfg = LlamaConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=32,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=1024,
    )
    cfg._attn_implementation = "sdpa"
    model = LlamaModel(cfg).eval().to(dtype=dtype)
    return model


class ModelLevelParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_dense_ratio_zero_byte_identical(self):
        item = layout(34)
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                model = tiny_model(dtype)
                embeddings = seeded_embeddings(item.full_sequence_tokens, 32, 9, dtype)
                with torch.inference_mode():
                    native = model(
                        inputs_embeds=embeddings, use_cache=False,
                        output_attentions=True, output_hidden_states=True, return_dict=True,
                    )
                    state = AdaptiveQueryState(AdaptiveSelectionConfig(fastv_r=0.25))
                    state.reset()
                    result = adaptive_decoder_forward(model, embeddings, item, state.config, state)
                    self.assertIsInstance(result, AdaptiveForwardResult)
                    self.assertFalse(result.pruned)
                    self.assertTrue(torch.equal(native.last_hidden_state, result.hidden))
                    self.assertEqual(len(native.attentions), 32)
                    for a, b in zip(native.attentions, result.attentions):
                        self.assertTrue(torch.equal(a, b), "attention tensor differs at a layer")
                    self.assertEqual(state.query, 1)

    def test_released_diagnostic_conflicts_with_adaptive_eager_path(self):
        """The released LlamaAttentionDiagnostic forbids output_attentions=True
        by design, while the adaptive decoder forward REQUIRES it (it executes
        the manual eager attention path that output_attentions=True forces in
        Transformers 4.40.1). Workers must therefore never install the released
        diagnostic around adaptive queries; the adaptive audit witness is the
        zero-SDPA call count (count_sdpa) plus official-capture values.
        Regression: S0 and S1 audited adaptive arms crashed with
        'attention-output fallback is prohibited' when wrapped in the released
        diagnostic."""
        item = layout(34)
        model = tiny_model(torch.float32)
        embeddings = seeded_embeddings(item.full_sequence_tokens, 32, 17, torch.float32)
        config = AdaptiveSelectionConfig(fastv_r=0.0)
        with torch.inference_mode():
            with self.assertRaisesRegex(RuntimeError, "attention-output fallback is prohibited"):
                state = AdaptiveQueryState(config)
                state.reset()
                with LlamaAttentionDiagnostic(torch, model.layers, 'original'):
                    adaptive_decoder_forward(model, embeddings, item, config, state)
            state = AdaptiveQueryState(config)
            state.reset()
            with count_sdpa() as sdpas:
                result = adaptive_decoder_forward(model, embeddings, item, config, state)
            self.assertIsInstance(result, AdaptiveForwardResult)
            self.assertEqual(sdpas[0], 0, "adaptive eager path must not call released SDPA")
            self.assertTrue(torch.isfinite(result.hidden).all())
            self.assertEqual(state.query, 1)

    def test_adaptive_counts_and_positions(self):
        item = layout(34)
        for dtype in (torch.float32, torch.bfloat16):
            for ratio, expected_visual in ((0.25, 384), (0.5, 256)):
                with self.subTest(dtype=dtype, ratio=ratio):
                    model = tiny_model(dtype)
                    embeddings = seeded_embeddings(item.full_sequence_tokens, 32, 11, dtype)
                    config = AdaptiveSelectionConfig(fastv_r=ratio)
                    state = AdaptiveQueryState(config)
                    state.reset()
                    with torch.inference_mode():
                        # Queries 1..3 run dense (warm start), query 4+ prunes.
                        warm = [adaptive_decoder_forward(model, embeddings, item, config, state) for _ in range(3)]
                        pruned = [adaptive_decoder_forward(model, embeddings, item, config, state) for _ in range(3)]
                    for r in warm:
                        self.assertFalse(r.pruned)
                        self.assertEqual(r.retained_visual_tokens, 512)
                        self.assertEqual(r.pruning_info["effective_fastv_r"], 0.0)
                        self.assertEqual(r.positions.numel(), item.full_sequence_tokens)
                    for r in pruned:
                        self.assertTrue(r.pruned)
                        self.assertEqual(r.retained_visual_tokens, expected_visual)
                        self.assertEqual(r.pruning_info["effective_fastv_r"], ratio)
                        self.assertEqual(r.pruning_info["pruning_layer"], 3)
                        # keep = [0] + selected visual + [513..605)
                        expected_len = 1 + expected_visual + (item.full_sequence_tokens - 513)
                        self.assertEqual(r.positions.numel(), expected_len)
                        self.assertEqual(int(r.positions[0]), 0)
                        self.assertTrue(int(r.positions[-1]) == item.full_sequence_tokens - 1)
                        self.assertTrue(bool((r.positions[1 : 1 + expected_visual] >= 1).all()))
                        self.assertTrue(bool((r.positions[1 : 1 + expected_visual] <= 512).all()))
                        visual_in_range = int(((r.positions >= 1) & (r.positions <= 512)).sum())
                        self.assertEqual(visual_in_range, expected_visual)
                        action_hidden = select_official_action_hidden(r.hidden, item, r.positions)
                        self.assertEqual(tuple(action_hidden.shape), (1, 56, 32))
                        self.assertTrue(torch.isfinite(action_hidden).all())
                        self.assertTrue(torch.isfinite(r.hidden).all())
                    self.assertEqual(state.query, 6)

    def test_ema_guides_query_four_span_scores(self):
        """Query 4+ engages the EMA-guided action scores (history = layer 15)."""
        item = layout(34)
        torch.manual_seed(7)
        model = tiny_model(torch.float32)
        embeddings = seeded_embeddings(item.full_sequence_tokens, 32, 13, torch.float32)
        config = AdaptiveSelectionConfig(fastv_r=0.25)
        state = AdaptiveQueryState(config)
        state.reset()
        with torch.inference_mode():
            for _ in range(3):
                adaptive_decoder_forward(model, embeddings, item, config, state)
        self.assertEqual(len(state.history), 3)
        self.assertEqual(state.effective_prune_ratio(), 0.25)  # pruning engages now
        items = list(state.history._items)
        expected = (
            1.0 * items[2] + 0.8 * items[1] + 0.64 * items[0]
        ) / (1.0 + 0.8 + 0.64)
        guided = state.history.build_guided(embeddings.device, embeddings.dtype)
        self.assertIsNotNone(guided)
        self.assertTrue(torch.allclose(guided.float(), expected.float(), atol=1e-6))

    def test_rejection_modes(self):
        item = layout(34)
        model = tiny_model(torch.float32)
        embeddings = seeded_embeddings(item.full_sequence_tokens, 32, 17, torch.float32)
        config = AdaptiveSelectionConfig(fastv_r=0.25)
        state = AdaptiveQueryState(config)
        with torch.inference_mode():
            with self.assertRaises(RuntimeError):
                adaptive_decoder_forward(model, embeddings, item, config, state)  # not reset
            state.reset()
            with self.assertRaises(ValueError):
                adaptive_decoder_forward(model, embeddings[:, :-1], item, config, state)  # wrong sequence length
            with self.assertRaises(ValueError):
                adaptive_decoder_forward(model, embeddings, item, config, state, expected_layers=28)


if __name__ == "__main__":
    unittest.main()