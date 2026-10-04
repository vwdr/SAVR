"""Adaptive query bridge for the frozen adaptive screen and qualification.

Mirrors the released VLA-Pruner-OFT preparation and action head exactly like
``fixed_compression.FixedCompressionQuery``, but replaces the decoder forward
with the adaptive selection path. The per-episode adaptive history lives
inside ``AdaptiveQueryState`` and is reset automatically when the shared
``EpisodeState`` starts a fresh episode (``query == 0``), which is how the
screen reuses one bridge instance across episodes.

``fastv_r=0`` retains all visual tokens. Legacy ``eager`` mode is preserved for
diagnosis. Recovery ``native_sdpa`` mode preserves native outputs while extracting
scores at layers 3 and 15. Decoder witnesses require zero or 32 SDPA calls,
respectively. Nested instrumentation is restored to its immediate caller.
"""
import contextlib
import numpy as np
import torch

from .official_semantics import (derive_official_layout, prepare_semantic_query,
                                 select_official_action_hidden)
from .vlapruner_adaptive import (AdaptiveQueryState, AdaptiveSelectionConfig,
                                 adaptive_decoder_forward)

@contextlib.contextmanager
def count_sdpa():
    """Count SDPA invocations within this explicit scope, preserving outer hooks."""
    counter = [0]
    original = torch.nn.functional.scaled_dot_product_attention

    def counted(*args, **kwargs):
        counter[0] += 1
        return original(*args, **kwargs)

    torch.nn.functional.scaled_dot_product_attention = counted
    try:
        yield counter
    finally:
        torch.nn.functional.scaled_dot_product_attention = original


class AdaptivePruneQuery:
    """One adaptive arm: released current-image preparation, adaptive forward."""

    def __init__(self, *, model, head, proprio, processor, cfg, utils,
                 instruction_indexer, fastv_r, action_horizon=0, attention_backend='eager'):
        if attention_backend not in ('eager', 'native_sdpa'):
            raise ValueError('unknown adaptive attention backend')
        self.attention_backend = attention_backend
        self.fastv_r = float(fastv_r)
        if not 0.0 <= self.fastv_r <= 1.0:
            raise ValueError("fastv_r must be a ratio in [0, 1]")
        self.model, self.head, self.proprio = model, head, proprio
        self.processor, self.cfg, self.utils = processor, cfg, utils
        self.instruction_indexer = instruction_indexer
        self.action_horizon = int(action_horizon)
        self.adaptive = AdaptiveQueryState(AdaptiveSelectionConfig(
            fastv_r=self.fastv_r, action_horizon=self.action_horizon))
        self.last_query = None

    def name(self):
        if self.fastv_r == 0.0:
            return "adaptive_zero"
        return f"adaptive_{int(round(100 * (1.0 - self.fastv_r)))}"

    @torch.inference_mode()
    def __call__(self, observation, instruction, previous, state):
        self.last_query = None
        if (not state.started or state.precise or state.previous_indices or state.confidence
                or any(m.training for m in (self.model, self.head, self.proprio))):
            raise ValueError("reset inference-only adaptive state required")
        if state.query == 0:
            self.adaptive.reset()
        preparation_calls = 0

        def prepare_once(images, cfg):
            nonlocal preparation_calls
            if cfg is not self.cfg or preparation_calls:
                raise ValueError("prepare current images exactly once")
            preparation_calls += 1
            return self.utils.prepare_images_for_vla(images, cfg)

        prepared = prepare_semantic_query(
            torch_module=torch, np_module=np, model=self.model, processor=self.processor,
            proprio_projector=self.proprio, prepare_images=prepare_once,
            normalize_proprio=self.utils.normalize_proprio, instruction_indexer=self.instruction_indexer,
            cfg=self.cfg, raw_scene=observation["full_image"].copy(),
            raw_wrist=observation["wrist_image"].copy(), raw_state=observation["state"].copy(),
            instruction=instruction)
        if preparation_calls != 1:
            raise ValueError("current preprocessing missing")
        layout = derive_official_layout(action_mask=prepared.action_mask, projected_tokens=513,
                                        instruction_token_indices=prepared.instruction_token_indices)
        embeddings, mask = self.model._build_multimodal_attention(
            prepared.input_embeddings * ~prepared.action_mask.unsqueeze(-1),
            prepared.projected_patches, prepared.attention_mask)
        if mask.ndim != 2 or not bool((mask == 1).all()):
            raise ValueError("only unpadded current queries supported")

        original = torch.nn.functional.scaled_dot_product_attention
        calls = [0]

        def witness(*args, **kwargs):
            calls[0] += 1
            return original(*args, **kwargs)

        torch.nn.functional.scaled_dot_product_attention = witness
        try:
            if self.attention_backend == 'native_sdpa':
                from .adaptive_sdpa import adaptive_sdpa_forward
                forward = adaptive_sdpa_forward
            else:
                forward = adaptive_decoder_forward
            final = forward(self.model.language_model.model, embeddings,
                            layout, self.adaptive.config, self.adaptive)
        finally:
            torch.nn.functional.scaled_dot_product_attention = original
        expected_calls = 32 if self.attention_backend == 'native_sdpa' else 0
        if calls[0] != expected_calls:
            raise ValueError('adaptive decoder SDPA count differs from selected backend')

        action_hidden = select_official_action_hidden(final.hidden, layout, final.positions)
        normalized = self.head.predict_action(action_hidden).reshape(8, 7).float().cpu().numpy()
        actions = np.asarray(self.model._unnormalize_actions(normalized, self.cfg.unnorm_key))
        if actions.shape != (8, 7) or not np.isfinite(normalized).all() or not np.isfinite(actions).all():
            raise ValueError("finite 8x7 action output required")
        state.query += 1
        self.last_query = dict(query=state.query, previous_frame=previous.step, precise=False,
                               retained_visual_tokens=final.retained_visual_tokens,
                               regime="uniform" if self.fastv_r == 0.0
                                      else ("warm" if state.query <= 3 else "steady"),
                               pruned=final.pruned,
                               effective_fastv_r=float(final.pruning_info.get("effective_fastv_r", 0.0)),
                               attention_layers=len(final.attentions), sdpa_calls=calls[0],
                               attention_backend=self.attention_backend)
        return actions
