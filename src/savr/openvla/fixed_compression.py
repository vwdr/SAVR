"""Current-frame, pre-layer-zero deletion. No stale states or learned correction."""
import numpy as np
import torch
from .official_semantics import prepare_semantic_query, derive_official_layout, select_official_action_hidden
from .spatial_selection import stratified_visual_positions
from .visual_compaction import VisualCompaction


def fixed_decoder_forward(decoder, embeddings, layout, budget, *, expected_layers=32):
    """Same original SDPA modules/readout, with original rotary position IDs.

    Both cameras were fully encoded before this call. Only visual positions
    are deleted, before the first decoder layer. Never allocate a past-K/V cache
    or retain layer outputs. The caller times mapping/validation as well.
    """
    if (torch.is_grad_enabled() or decoder.training or embeddings.ndim != 3
            or embeddings.shape[:2] != (1, layout.full_sequence_tokens)
            or len(decoder.layers) != expected_layers
            or any(type(x.self_attn).__name__ != "LlamaSdpaAttention" for x in decoder.layers)):
        raise ValueError("original inference-only SDPA decoder and full input required")
    dense = VisualCompaction.dense(layout)
    compact = dense.retain_visual(stratified_visual_positions(budget))
    # Avoid an extra identity copy for the 512-token control.
    hidden = embeddings if budget == 512 else compact.compact_hidden(torch, embeddings, dense)
    positions = torch.tensor(compact.absolute_positions, device=hidden.device, dtype=torch.long)
    for layer in decoder.layers:
        hidden = layer(hidden, attention_mask=None, position_ids=positions.unsqueeze(0),
                       past_key_value=None, output_attentions=False, use_cache=False,
                       cache_position=positions)[0]
    return decoder.norm(hidden), positions


class FixedCompressionQuery:
    """Released current-image preparation and head; one fixed budget, no history."""
    def __init__(self, *, model, head, proprio, processor, cfg, utils, instruction_indexer, budget):
        stratified_visual_positions(budget)  # Reject bool, unknown or fractional budgets.
        self.model, self.head, self.proprio = model, head, proprio
        self.processor, self.cfg, self.utils = processor, cfg, utils
        self.instruction_indexer, self.budget = instruction_indexer, budget
        self.last_query = None

    @torch.inference_mode()
    def __call__(self, observation, instruction, previous, state):
        self.last_query = None
        if (not state.started or state.precise or state.previous_indices or state.confidence
                or any(m.training for m in (self.model, self.head, self.proprio))):
            raise ValueError("reset inference-only fixed-budget state required")
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
        hidden, positions = fixed_decoder_forward(self.model.language_model.model, embeddings, layout, self.budget)
        action_hidden = select_official_action_hidden(hidden, layout, positions)
        normalized = self.head.predict_action(action_hidden).reshape(8, 7).float().cpu().numpy()
        actions = np.asarray(self.model._unnormalize_actions(normalized, self.cfg.unnorm_key))
        if actions.shape != (8, 7) or not np.isfinite(normalized).all() or not np.isfinite(actions).all():
            raise ValueError("finite 8x7 action output required")
        state.query += 1
        self.last_query = dict(query=state.query, previous_frame=previous.step, precise=False,
                               retained_visual_tokens=self.budget)
        return actions
