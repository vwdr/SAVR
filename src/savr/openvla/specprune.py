"""Isolated same-checkpoint SpecPrune-OFT selection port (not benchmark-qualified).

Adapted from alexwhz-sjtu/SpecPrune-VLA, revision
8091adc4b574ce9008d49a1dc9a210f4eec314c1. Original notices are reproduced in
specprune_LICENSE.txt; original LLaMA code also carries Apache-2.0 notices.
No loader, installed runtime, pretrained weights or simulator is modified here.
Selection signals intentionally follow the pinned source, including its text
windows and placeholder-based importance, rather than silently redefining them
as the different released-checkpoint action-head readout positions.
"""
from dataclasses import dataclass, field
import math
import numpy as np
import torch


def low_change_indices(current, previous, *, top_k, threshold, wrist=False):
    """Pinned 224x224 RGB patch cosine/top-k; callers supply policy-prepared images."""
    def patches(image):
        array = np.asarray(image)
        if array.shape != (224, 224, 3) or array.dtype != np.uint8:
            raise ValueError("expected policy-prepared uint8 224x224 RGB images")
        return array.reshape(16, 14, 16, 14, 3).transpose(0, 2, 1, 3, 4).reshape(256, -1).astype(np.float32)
    if type(top_k) is not int or not 1 <= top_k <= 256:
        raise ValueError("positive patch budget required")
    a, b = patches(current), patches(previous)
    similarity = np.sum(a * b, axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-8)
    ids = np.where(similarity >= threshold)[0]
    if len(ids) == 0:
        return np.array([], dtype=int)
    k = min(top_k, len(ids))
    return ids[np.argpartition(similarity[ids], -k)[-k:]] + (257 if wrist else 1)


def task_relevant_set(attention, positions, prompt, topk, *, wrist=False, first_layer=False):
    """Preserve the source's distinct layer-zero versus later text windows."""
    offset = 257 if wrist else 1
    amap = attention.float().squeeze(0).mean(0)
    visual = (positions >= offset) & (positions < offset + 256)
    if first_layer:
        text = (positions >= 523) & (positions < 513 + prompt)
        relation = amap[text][:, visual]
    else:
        relation = amap[-57-prompt+10:-57, visual]
    k = min(max(int(topk), 0), int(visual.sum()))
    if k == 0 or relation.numel() == 0:
        return positions.new_empty(0)
    selected = relation.mean(0).reshape(-1).topk(k).indices
    return positions[visual][selected]


def prior_global_indices(layer_attention, positions, topk, *, wrist=False):
    """One source goal layer; caller unions layers 14 and 30.

    The source uses absolute text IDs [522,547), even for variable prompt length,
    and includes zero-scored removed patches in its 256-patch ranking. Preserve
    both here; history indices may reappear in the NEXT fresh query, not this one.
    """
    amap = layer_attention.float().squeeze(0).mean(0)
    offset = 257 if wrist else 1
    visual = (positions >= offset) & (positions < offset + 256)
    text = (positions >= 522) & (positions < 547)
    if not bool(text.any()):
        raise ValueError("source global-attention text window is empty")
    scores = amap[text][:, visual].mean(0)
    full = torch.zeros(256, device=scores.device)
    full[(positions[visual] - offset).long()] = scores
    values = full.cpu().numpy().astype(np.float32)
    ranked = sorted(range(256), key=lambda i: values[i], reverse=True)
    return positions.new_tensor(sorted(ranked[:topk])) + offset


@dataclass
class EpisodeState:
    """Caller must explicitly reset before EVERY episode, even repeated IDs."""
    episode_id: object = None
    started: bool = False
    query: int = 0
    precise: bool = False
    previous_mode: bool = True
    confidence: dict = field(default_factory=dict)
    previous_indices: tuple = ()
    action_sum: np.ndarray = field(default_factory=lambda: np.zeros(7, dtype=np.float64))

    def reset(self, episode_id):
        self.episode_id, self.started, self.query = episode_id, True, 0
        self.precise, self.previous_mode = False, True
        self.confidence.clear()
        self.previous_indices = ()
        self.action_sum = np.zeros(7, dtype=np.float64)

    def previous_frame_index(self, available_frames):
        """Source lookback on per-control-step prepared frames; consume action sum.

        The caller stores the paired, already preprocessed camera frames. This
        selector does not substitute the previous policy-query frame for the
        source's shorter control-step lookback.
        """
        if not self.started or type(available_frames) is not int or available_frames < 1:
            raise ValueError("episode frame history required")
        speed = np.linalg.norm(self.action_sum[:3])
        lookback = min(int((-16 / 3 * speed / 6 + 22 / 3)) + 3, 6)
        if lookback < 1:
            raise ValueError("source controller produced a nonpositive frame lookback")
        index = -1 if self.query == 0 or available_frames < lookback else -lookback
        self.action_sum.fill(0)
        return index

    def observe_executed_action(self, action, *, acting_steps, queue_remaining):
        """Source thresholds on the executed, gripper-processed action; return replan flag.

        This does not execute/skip simulator steps. The simulator must always
        execute its action, including before the controller's warm-up threshold.
        """
        action = np.asarray(action)
        if not self.started or action.shape != (7,) or not np.isfinite(action).all():
            raise ValueError("invalid controller state or action")
        if acting_steps < 10:
            return False
        self.action_sum += action
        if acting_steps == 10:
            return False
        velocity, xy, rot = np.linalg.norm(action[:3]), np.linalg.norm(action[:2]), np.linalg.norm(action[3:6])
        z = action[2]  # Source uses signed z, not its absolute magnitude.
        replan = False
        if self.precise:
            self.precise = not ((z > .2 or rot > .3) and queue_remaining == 0)
        else:
            self.precise = bool(velocity < .35 or (xy < .1 and z < .1 and rot < .2))
            replan = not self.previous_mode and self.precise and queue_remaining > 0
        self.previous_mode = self.precise
        return bool(replan)


class SpecPruneSelection:
    """A single fresh query's layerwise selection state; no hidden/K/V persistence."""
    def __init__(self, layout, episode, low_change=(), *, enabled=True, dynamic=True, device="cpu"):
        if not episode.started:
            raise ValueError("episode reset required")
        self.layout, self.episode = layout, episode
        self.enabled, self.dynamic = enabled, dynamic
        self.positions = torch.arange(layout.full_sequence_tokens, device=device)
        self.scores = torch.zeros(layout.full_sequence_tokens, device=device)
        self.low_change = self.positions.new_tensor(tuple(int(p) for p in low_change))
        self.previous = self.positions.new_tensor(episode.previous_indices)
        for ids in (self.low_change, self.previous):
            if bool(((ids < 1) | (ids > 512)).any()):
                raise ValueError("selection indices must be absolute visual IDs")
        self.full_retain = torch.ones(layout.full_sequence_tokens, dtype=torch.bool, device=device)
        self.early_high = self.positions.new_empty(0)
        self.global_sets = []
        self.trace = []
        self.next_layer = 0
        self.closed = False
        self.topk = (15, 12) if episode.precise else (24, 19)

    def needs_attention(self, layer):
        return self.enabled and (layer in (0, 1, 14, 30) or (self.dynamic and layer in (19, 24)))

    def after_layer(self, layer, attention=None):
        if self.closed or layer != self.next_layer:
            raise ValueError("layers must execute exactly once in order")
        before = self.positions
        if self.needs_attention(layer):
            if (attention is None or attention.ndim != 4 or attention.shape[0] != 1
                    or tuple(attention.shape[-2:]) != (len(before), len(before))
                    or not bool(torch.isfinite(attention).all())):
                raise ValueError("attention and absolute position map differ")
        if self.enabled and layer in (14, 30):
            for wrist, topk in ((False, self.topk[0] // 2), (True, self.topk[1] // 2)):
                self.global_sets.append(prior_global_indices(attention, before, topk, wrist=wrist))
        keep = torch.ones(len(before), dtype=torch.bool, device=before.device)
        if self.enabled and layer in (0, 1):
            selections = [task_relevant_set(attention, before, self.layout.prompt_tokens,
                          self.topk[int(wrist)] * (2 if layer == 0 else 1),
                          wrist=wrist, first_layer=layer == 0) for wrist in (False, True)]
            if layer == 0:
                self.early_high = torch.cat([selections[i][:self.topk[i]] for i in (0, 1)])
            self.full_retain[self.low_change] = False
            self.full_retain[torch.cat(selections)] = True
            self.full_retain[self.previous] = True
            if layer == 1:
                self.full_retain[self.early_high] = True
            keep = self.full_retain[before]
        if self.enabled and self.dynamic:
            if layer in (14, 19, 24):
                visual = (before >= 1) & (before < 513)
                # Source importance uses placeholders, independently of action-head readout.
                start = self.layout.full_sequence_tokens - 57
                action = (before >= start) & (before < self.layout.full_sequence_tokens - 1)
                if bool(visual.any() & action.any()):
                    amap = attention.float().squeeze(0).mean(0)
                    relation = amap[action][:, visual]
                    mean = relation.mean(0)
                    order = torch.sort(mean, descending=True).indices
                    ranks = torch.empty_like(order)
                    ranks[order] = torch.arange(len(order), device=order.device)
                    sigma = torch.sigmoid(-ranks.float())
                    weight = sigma / (sigma.sum() + 1e-8)
                    confidence = self.episode.confidence.get(layer)
                    if confidence is None:
                        probability = relation + 1e-8
                        probability = probability / probability.sum(-1, keepdim=True)
                        entropy = -(probability * torch.log(probability)).sum(-1).mean()
                        entropy = entropy / np.log(max(int(visual.sum()), 2))
                        confidence = (1 / (entropy + 1e-8)).detach()
                        self.episode.confidence[layer] = confidence
                    contribution = weight * confidence.to(weight)
                    self.scores[visual] = .8 * self.scores[visual] + .2 * contribution
            if layer in (10, 15, 20, 25):
                nonvisual = (before < 1) | (before >= 513)
                self.scores[nonvisual] = float("inf")
                order = torch.sort(self.scores, descending=True).indices
                number = min(len(before), max(int(.9 * len(before)), 60 + self.layout.prompt_tokens + 58))
                if number >= len(before):
                    self.dynamic = False
                keep = torch.zeros_like(keep)
                keep[order[:number]] = True
        after = before[keep]
        expected = torch.cat([before.new_tensor([0]), torch.arange(513, self.layout.full_sequence_tokens, device=before.device)])
        if not torch.equal(after[(after == 0) | (after >= 513)], expected):
            raise ValueError("pruning removed a required nonvisual state")
        self.positions, self.scores = after, self.scores[keep]
        self.trace.append(dict(layer=layer, before=len(before), after=len(after)))
        self.next_layer += 1
        return keep

    def finish(self, expected_layers=32):
        if self.closed or self.next_layer != expected_layers:
            raise ValueError("incomplete or repeated query completion")
        self.episode.previous_indices = tuple(torch.unique(torch.cat(self.global_sets)).cpu().tolist()) if self.global_sets else ()
        self.episode.query += 1
        self.closed = True


class NativeAttentionScores:
    """Capture scores without changing the native SDPA output, scoped to one layer.

    Only inference, unpadded, square current-query attention is supported here.
    The auxiliary matrix is released after that layer; its cost is not excluded
    from subsequent timing. No production performance claim is made.
    """
    def __init__(self):
        self.original = None
        self.scores = None
        self.calls = 0

    def __enter__(self):
        if self.original is not None:
            raise ValueError("score capture already active")
        self.original = torch.nn.functional.scaled_dot_product_attention
        torch.nn.functional.scaled_dot_product_attention = self.capture
        return self

    def capture(self, query, key, value, *args, **kwargs):
        if args or kwargs.get("is_causal", False) or kwargs.get("dropout_p", 0) != 0:
            raise ValueError("unexpected native attention semantics")
        if query.shape[-2] != key.shape[-2] or query.shape[0] != 1 or self.calls:
            raise ValueError("expected one square batch-one attention call")
        mask = kwargs.get("attn_mask")
        if mask is not None and (mask.dtype == torch.bool or not bool((mask == 0).all())):
            raise ValueError("only unpadded bidirectional attention is supported")
        logits = torch.matmul(query, key.transpose(2, 3))
        logits = logits / math.sqrt(query.shape[-1]) if kwargs.get("scale") is None else logits * kwargs["scale"]
        self.scores = torch.softmax(logits, dim=-1, dtype=torch.float32).to(query.dtype)
        self.calls += 1
        return self.original(query, key, value, **kwargs)

    def __exit__(self, kind, value, tb):
        if self.original is not None:
            torch.nn.functional.scaled_dot_product_attention = self.original
            self.original = None
        if kind is None and self.calls != 1:
            raise ValueError("native SDPA was not called once")


def pruned_decoder_forward(decoder, embeddings, selector):
    """Run original decoder modules with true deletion and original rotary IDs.

    No checkpoint loading, action decoding, simulator loop, or GPU dispatch.
    This qualification implementation stores no past K/V across queries.
    """
    if torch.is_grad_enabled() or decoder.training or embeddings.shape[:2] != (1, selector.layout.full_sequence_tokens):
        raise ValueError("expected inference-only full current-query embeddings")
    if any(type(layer.self_attn).__name__ != "LlamaSdpaAttention" for layer in decoder.layers):
        raise ValueError("original SDPA decoder required")
    hidden = embeddings
    for index, layer in enumerate(decoder.layers):
        arguments = dict(attention_mask=None, position_ids=selector.positions.unsqueeze(0),
                         past_key_value=None, output_attentions=False, use_cache=False,
                         cache_position=selector.positions)
        if selector.needs_attention(index):
            with NativeAttentionScores() as capture:
                hidden = layer(hidden, **arguments)[0]
            attention = capture.scores
        else:
            hidden = layer(hidden, **arguments)[0]
            attention = None
        keep = selector.after_layer(index, attention)
        if not bool(keep.all()):
            hidden = hidden[:, keep, :]
    hidden = decoder.norm(hidden)
    selector.finish(expected_layers=len(decoder.layers))
    return hidden, selector.positions
