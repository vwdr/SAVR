"""Actual hard-compressed features and downstream-only action correction.

The qualified fixed-compression implementation is not modified. This new path
must agree with it before it is used for training or closed-loop evaluation.
"""
import numpy as np
import torch
from .fixed_compression import FixedCompressionQuery, fixed_decoder_forward
from .official_semantics import prepare_semantic_query, derive_official_layout, select_official_action_hidden


def head_features(head, action_hidden):
    """Capture the input to the released head's final linear layer, not a tail slice."""
    captured = []
    layer = head.model.fc2
    if not isinstance(layer, torch.nn.Linear) or (layer.in_features, layer.out_features) != (4096, 7):
        raise ValueError("released 4096-to-7 final action layer required")
    def capture(module, args):
        if len(args) != 1 or tuple(args[0].shape) != (1, 8, 4096):
            raise ValueError("action-head penultimate layout changed")
        captured.append(args[0].detach())
    handle = layer.register_forward_pre_hook(capture)
    try:
        normalized = head.predict_action(action_hidden)
    finally:
        handle.remove()
    if len(captured) != 1 or tuple(normalized.shape) != (1, 8, 7):
        raise ValueError("exactly one eight-action head invocation required")
    return captured[0], normalized


def pack_features(prepared, z, normalized):
    """Store the same precision presented at deployment, including pooled language."""
    indices = prepared.instruction_token_indices
    if not indices or tuple(prepared.projected_patches.shape) != (1, 513, 4096):
        raise ValueError("two current views and one proprio token required")
    values = dict(
        action_features=z[0].detach(),
        current_visual=prepared.projected_patches[0, :512].detach(),
        instruction=prepared.input_embeddings[0, list(indices)].float().mean(0).to(torch.bfloat16).detach(),
        base_actions=normalized[0].float().detach(),
        state=prepared.normalized_proprio.reshape(8).float().detach(),
    )
    expected = dict(action_features=(8, 4096), current_visual=(512, 4096),
                    instruction=(4096,), base_actions=(8, 7), state=(8,))
    for name, tensor in values.items():
        dtype = torch.float32 if name in ('base_actions', 'state') else torch.bfloat16
        if tuple(tensor.shape) != expected[name] or tensor.dtype != dtype or not bool(torch.isfinite(tensor).all()):
            raise ValueError('invalid feature boundary: ' + name)
    return values


class CurrentFrameFeatureQuery(FixedCompressionQuery):
    """One current-frame query. Retained features are replaced on every call."""
    @torch.inference_mode()
    def __call__(self, observation, instruction, previous, state):
        self.last_query = None
        self.features = None
        if (not state.started or state.precise or state.previous_indices or state.confidence
                or any(m.training for m in (self.model, self.head, self.proprio))):
            raise ValueError("reset inference-only fixed-budget state required")
        calls = []
        def prepare_once(images, cfg):
            if cfg is not self.cfg or calls:
                raise ValueError("prepare current images exactly once")
            calls.append(True)
            return self.utils.prepare_images_for_vla(images, cfg)
        prepared = prepare_semantic_query(
            torch_module=torch, np_module=np, model=self.model, processor=self.processor,
            proprio_projector=self.proprio, prepare_images=prepare_once,
            normalize_proprio=self.utils.normalize_proprio, instruction_indexer=self.instruction_indexer,
            cfg=self.cfg, raw_scene=observation['full_image'].copy(),
            raw_wrist=observation['wrist_image'].copy(), raw_state=observation['state'].copy(),
            instruction=instruction)
        if len(calls) != 1:
            raise ValueError("current preprocessing missing")
        layout = derive_official_layout(action_mask=prepared.action_mask, projected_tokens=513,
                                       instruction_token_indices=prepared.instruction_token_indices)
        embeddings, mask = self.model._build_multimodal_attention(
            prepared.input_embeddings * ~prepared.action_mask.unsqueeze(-1),
            prepared.projected_patches, prepared.attention_mask)
        if mask.ndim != 2 or not bool((mask == 1).all()):
            raise ValueError("only unpadded current queries supported")
        hidden, positions = fixed_decoder_forward(self.model.language_model.model, embeddings, layout, self.budget)
        z, normalized = head_features(self.head, select_official_action_hidden(hidden, layout, positions))
        self.features = pack_features(prepared, z, normalized)
        actions = np.asarray(self.model._unnormalize_actions(
            self.features['base_actions'].cpu().numpy(), self.cfg.unnorm_key))
        if actions.shape != (8, 7) or not np.isfinite(actions).all():
            raise ValueError("finite eight-action chunk required")
        state.query += 1
        self.last_query = dict(query=state.query, previous_frame=previous.step, precise=False,
                               retained_visual_tokens=self.budget)
        return actions


class CorrectedCurrentFrameQuery:
    """Timing must include this entire wrapper, including feature extraction/copies."""
    def __init__(self, feature_query, corrector):
        self.feature_query, self.corrector = feature_query, corrector
        self.last_query = None

    @torch.inference_mode()
    def __call__(self, observation, instruction, previous, state):
        if self.corrector.training:
            raise ValueError("corrector must be in evaluation mode")
        self.feature_query(observation, instruction, previous, state)
        inputs = {k: v.unsqueeze(0) for k, v in self.feature_query.features.items()}
        normalized = self.corrector(**inputs)[0].float().cpu().numpy()
        actions = np.asarray(self.feature_query.model._unnormalize_actions(
            normalized, self.feature_query.cfg.unnorm_key))
        if actions.shape != (8, 7) or not np.isfinite(actions).all():
            raise ValueError("nonfinite corrected action")
        self.last_query = dict(self.feature_query.last_query)
        return actions
