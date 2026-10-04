"""Backend-aligned VLA-Pruner selection adaptation, not an exact upstream reproduction.

Keep native bidirectional SDPA outputs and separately observe probabilities at
selection layer 3 and history layer 15. All extraction/selection costs belong
inside the timed query. No checkpoint, attention mask, or action readout changes.
"""
import torch
from .specprune import NativeAttentionScores
from .vlapruner_adaptive import AdaptiveForwardResult, build_fastv_config, fastv_pruning_indices


def adaptive_sdpa_forward(decoder, embeddings, layout, config, state, *, expected_layers=32):
    if (torch.is_grad_enabled() or decoder.training or not state.started
            or embeddings.shape[:2] != (1, layout.full_sequence_tokens)
            or len(decoder.layers) != expected_layers
            or any(type(l.self_attn).__name__ != 'LlamaSdpaAttention' for l in decoder.layers)):
        raise ValueError('original inference-only decoder and reset state required')
    if config.fastv_k != 3 or config.vla_pruner_layer != 15 or expected_layers != 32:
        raise ValueError('frozen 32-layer selection/history contract required')
    length = embeddings.shape[1]
    positions = torch.arange(length, device=embeddings.device)
    ratio = state.effective_prune_ratio()
    action_start = int(layout.action_readout_positions[0])
    fastv = build_fastv_config(config=config, action_start=action_start,
        action_end=action_start + len(layout.action_readout_positions), prune_ratio=ratio,
        historical_attention=state.historical_attention(embeddings.device, embeddings.dtype))
    hidden, kept, info, average = embeddings, None, None, None
    attentions = [None] * expected_layers
    for index, layer in enumerate(decoder.layers):
        kwargs = dict(attention_mask=None, position_ids=positions.unsqueeze(0),
            past_key_value=None, output_attentions=False, use_cache=False, cache_position=positions)
        if index in (3, 15):
            with NativeAttentionScores() as capture:
                hidden = layer(hidden, **kwargs)[0]
            scores = capture.scores
            attentions[index] = scores
        else:
            hidden = layer(hidden, **kwargs)[0]
            scores = None
        if index == 3 and ratio > 0:
            average = scores.mean(dim=1)[0]
            kept, info = fastv_pruning_indices(average, length, fastv, inputs_embeds=embeddings)
            hidden = hidden.index_select(1, kept)
            positions = kept
    hidden = decoder.norm(hidden)
    pruned = kept is not None and kept.numel() < length
    state.record_layer15(attentions[15], kept, 3 if pruned else None, layout)
    state.query += 1
    retained = int(((positions >= 1) & (positions <= 512)).sum())
    return AdaptiveForwardResult(hidden=hidden, positions=positions, attention_avg=average,
        score_info=info, pruning_info=dict(query=state.query, effective_fastv_r=ratio,
            original_seq_length=length, kept_indices=kept, pruning_layer=3 if pruned else None,
            score_layers=[3, 15], attention_backend='native_sdpa'), hidden_states=None,
        attentions=tuple(attentions), pruned=pruned, retained_visual_tokens=retained)
