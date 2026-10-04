"""Local translation of the released VLA-Pruner-OFT adaptive selection path.

Port decision and source pinning: `reports/VLAPRUNER_COMPARATOR_AUDIT_AND_SCREEN_SPEC_V2.md`
(pinned release commit `84d4b7192c77abf1585610e2f12393319b7ebff9`, MIT). This
module re-implements only the *selection mechanics* of the executed VLA-Pruner
inference path that the audit pinned at code level:

- prune after layer index ``fastv_k`` (layer 3); score = head-mean attention at
  that layer; semantic rows = full prefill block ``[0, action_start)``; action
  rows = the 56 action-readout rows; no min-max normalization and no
  semantic/action weighting on the executed path;
- per camera span (256 tokens each), top-k of the prefill and action scores,
  union, and when the union exceeds the budget, MMDP greedy over the *LLM input
  embeddings* of the candidates (cosine distance, model dtype, first pick by
  second-nearest distance, then max-min);
- ``position_ids = keep_indices`` after pruning and a fresh causal mask, so
  absolute rotary positions are preserved;
- per-query EMA action history captured at layer ``vla_pruner_layer`` (15),
  deque ``av_hist_w`` (3) with decay ``av_decay`` (0.8), replacing the current
  action scores from query >= 4 per episode; the first ``av_hist_w`` queries
  after each episode reset run fully dense (``fastv_r`` forced 0). This is the
  release's *executed* behavior (the normalization/weighting helpers in
  ``vla_pruner_utils.py`` are dead code in the release).

This is a model-level comparator on our pinned runtime (Transformers 4.40.1,
LlamaSdpaAttention, Torch 2.2.0), not the upstream loader and not an upstream
install. Like the SpecPrune port, it is not itself a claim: parity with the
AST-isolated pinned source is verified CPU-only in
``tests/openvla/test_vlapruner_adaptive.py``, and dense/adaptive parity gates
must pass before any measured run.

API shape follows the existing ``fixed_compression.py`` comparator: the caller
embeds the full current query, then ``adaptive_decoder_forward`` runs the
decoder; ``AdaptiveQueryState`` carries the per-episode history and is reset by
the harness between episodes.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Optional

import torch

from .official_semantics import OfficialInferenceLayout, OpenVLASemanticError

# OpenVLA-OFT fixed action-head geometry (same values the release constants use).
ACTION_DIM = 7
NUM_ACTIONS_CHUNK = 8

TEMPORAL_MODES = frozenset({"action", "semantic_action", "prefill_action", "vla_pruner"})


@dataclass(frozen=True)
class AdaptiveSelectionConfig:
    """Pinned VLA-Pruner-OFT tunables (release values unless noted).

    ``fastv_r`` is a *pruning ratio*: retention = 1 - fastv_r per camera span.
    The released OFT scripts evaluate 0.75/0.875/0.9275; our matched arms use
    0.25 (75% retention) and 0.5 (50% retention).
    """

    fastv_k: int = 3
    fastv_r: float = 0.25
    vla_pruner_layer: int = 15
    av_hist_w: int = 3
    av_decay: float = 0.8
    vla_pruner_mode: str = "semantic_action"
    image_token_start_index: int = 1
    patches_per_image: int = 256
    num_images: int = 2
    action_horizon: int = 0
    action_dim: int = ACTION_DIM
    num_actions_chunk: int = NUM_ACTIONS_CHUNK
    return_attentions: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= self.fastv_r <= 1.0:
            raise OpenVLASemanticError("fastv_r must be a pruning ratio in [0, 1]")
        if self.vla_pruner_mode not in {"semantic", "prefill", "semantic_action",
                                        "prefill_action", "action", "vla_pruner"}:
            raise OpenVLASemanticError("unsupported vla_pruner_mode")
        if self.image_token_start_index < 1 or self.patches_per_image < 1 or self.num_images < 1:
            raise OpenVLASemanticError("image geometry must be positive")

    @property
    def num_visual_tokens(self) -> int:
        return self.patches_per_image * self.num_images

    @property
    def use_temporal(self) -> bool:
        return self.vla_pruner_mode in TEMPORAL_MODES

    def expected_retained_visual(self) -> int:
        """Released arithmetic: per-span round(span_len * (1 - fastv_r)), summed."""
        span_len = self.patches_per_image
        return sum(max(0, min(span_len, int(round(span_len * (1.0 - self.fastv_r)))))
                   for _ in range(self.num_images))


# ---------------------------------------------------------------------------
# Pure selection translation (mechanical mirror of the pinned source methods).
# ---------------------------------------------------------------------------


def slice_range(start: int, end: int, limit: int) -> tuple[int, int]:
    """Mirror of the release static ``_slice_range``."""
    start = max(0, min(int(start), limit))
    end = max(start, min(int(end), limit))
    return start, end


def visual_token_spans(
    image_start: int,
    image_end: int,
    seq_length: int,
    config: AdaptiveSelectionConfig,
) -> list[tuple[int, int]]:
    """Mirror of the release ``_visual_token_spans`` (patches per image)."""
    if config.patches_per_image <= 0 or config.num_images <= 1:
        return [(image_start, image_end)]
    spans: list[tuple[int, int]] = []
    for image_idx in range(config.num_images):
        span_start = image_start + image_idx * config.patches_per_image
        span_end = min(span_start + config.patches_per_image, image_end)
        span_start, span_end = slice_range(span_start, span_end, seq_length)
        if span_end > span_start:
            spans.append((span_start, span_end))
    return spans or [(image_start, image_end)]


def redundancy_minization(
    visual_feature_vectors: torch.Tensor,
    num_keep: int,
    cosine_matrix: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Mirror of the release ``_redundancy_minization`` (MMDP greedy, exact)."""
    if len(visual_feature_vectors) <= num_keep:
        return torch.arange(len(visual_feature_vectors), device=visual_feature_vectors.device)
    if cosine_matrix is None:
        norm_matrix = visual_feature_vectors / visual_feature_vectors.norm(
            dim=1, keepdim=True
        ).clamp_min(1e-6)
        cosine_similarity = torch.mm(norm_matrix, norm_matrix.t())
        cosine_matrix = 1.0 - cosine_similarity
    selected = torch.empty(num_keep, dtype=torch.long, device=visual_feature_vectors.device)
    for i in range(num_keep):
        if i == 0:
            distances = cosine_matrix
        else:
            chosen = torch.index_select(selected, 0, torch.arange(0, i, device=cosine_matrix.device))
            distances = torch.index_select(cosine_matrix, 0, chosen)
        if i == 0:
            scores = torch.topk(distances, 2, dim=0, largest=False).values[1, :]
        else:
            scores = torch.min(distances, dim=0).values
        selected[i] = torch.argmax(scores)
    return selected


def fastv_scores(
    attention_avg: torch.Tensor,
    image_start: int,
    image_end: int,
    seq_length: int,
    fastv_config: dict,
) -> torch.Tensor:
    """Mirror of the release ``_fastv_scores`` (mode fallback, unused default)."""
    if fastv_config.get("fastv_attention_source") == "last":
        return attention_avg[seq_length - 1, image_start:image_end]
    if bool(fastv_config.get("use_text_vision_selection", False)):
        text_start = fastv_config.get("text_token_start", image_end)
        text_end = fastv_config.get("text_token_end", seq_length - 1)
        text_start, text_end = slice_range(text_start, text_end, seq_length)
        if text_end > text_start:
            return attention_avg[text_start:text_end, image_start:image_end].mean(dim=0)
    if bool(fastv_config.get("use_prefil_attention", True)):
        prefill_end = fastv_config.get("action_token_start", seq_length)
        prefill_start, prefill_end = slice_range(0, prefill_end, seq_length)
        if prefill_end > prefill_start:
            return attention_avg[prefill_start:prefill_end, image_start:image_end].mean(dim=0)
    default_rater_index = int(fastv_config.get("text_token_end", seq_length)) - 1
    rater_index = int(fastv_config.get("fastv_rater_index", default_rater_index))
    rater_index = max(0, min(rater_index, seq_length - 1))
    return attention_avg[rater_index, image_start:image_end]


def vlapruner_image_indices(
    attention_avg: torch.Tensor,
    image_start: int,
    image_end: int,
    seq_length: int,
    num_keep: int,
    fastv_config: dict,
    inputs_embeds: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, dict]:
    """Mirror of the release ``_vlapruner_image_indices`` (per-span selection)."""
    mode = fastv_config.get("vla_pruner_mode", "semantic_action")
    score_info: dict[str, Any] = {}

    prefill_scores: Optional[torch.Tensor] = None
    if mode in {"semantic", "prefill", "semantic_action", "prefill_action", "vla_pruner"}:
        prefill_end = fastv_config.get("action_token_start", seq_length)
        prefill_start, prefill_end = slice_range(0, prefill_end, seq_length)
        if prefill_end > prefill_start:
            prefill_scores = attention_avg[prefill_start:prefill_end, image_start:image_end].mean(dim=0)
            score_info["prefill_scores"] = prefill_scores.detach()

    action_scores: Optional[torch.Tensor] = None
    if mode in {"action", "semantic_action", "prefill_action", "vla_pruner"}:
        action_start = fastv_config.get("action_token_start", seq_length - 1)
        action_end = fastv_config.get("action_token_end", seq_length - 1)
        action_dim = max(1, int(fastv_config.get("action_dim", 1)))
        action_horizon = int(fastv_config.get("action_horizon", 0))
        if action_horizon > 0:
            action_end = min(int(action_end), int(action_start) + action_horizon * action_dim)
        action_start, action_end = slice_range(action_start, action_end, seq_length)
        if action_end > action_start:
            action_scores = attention_avg[action_start:action_end, image_start:image_end].mean(dim=0)
            score_info["current_action_scores"] = action_scores.detach()
            historical_attention = fastv_config.get("historical_attention")
            if bool(fastv_config.get("use_temporal", False)) and historical_attention is not None:
                visual_start = int(fastv_config.get("image_token_start_index", image_start))
                rel_start = max(0, image_start - visual_start)
                rel_end = rel_start + (image_end - image_start)
                historical = historical_attention.to(
                    device=attention_avg.device, dtype=attention_avg.dtype
                ).flatten()
                if historical.numel() >= rel_end:
                    action_scores = historical[rel_start:rel_end]
            score_info["action_scores"] = action_scores.detach()

    if prefill_scores is None and action_scores is None:
        scores = fastv_scores(attention_avg, image_start, image_end, seq_length, fastv_config)
        score_info["scores"] = scores.detach()
        return scores.topk(num_keep).indices + image_start, score_info
    if prefill_scores is None:
        assert action_scores is not None
        score_info["scores"] = action_scores.detach()
        return action_scores.topk(num_keep).indices + image_start, score_info
    if action_scores is None:
        score_info["scores"] = prefill_scores.detach()
        return prefill_scores.topk(num_keep).indices + image_start, score_info

    prefill_topk = prefill_scores.topk(num_keep).indices
    action_topk = action_scores.topk(num_keep).indices
    candidate_indices = torch.unique(torch.cat([prefill_topk, action_topk])).sort().values
    score_info["candidate_indices"] = candidate_indices.detach()

    if candidate_indices.numel() > num_keep and inputs_embeds is not None:
        visual_features = inputs_embeds[0, image_start:image_end]
        selected_features = visual_features.index_select(0, candidate_indices)
        final_indices = redundancy_minization(selected_features, num_keep)
        candidate_indices = candidate_indices.index_select(0, final_indices).sort().values

    return candidate_indices + image_start, score_info


def fastv_pruning_indices(
    attention_avg: torch.Tensor,
    seq_length: int,
    fastv_config: dict,
    inputs_embeds: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, dict]:
    """Mirror of the release ``_fastv_pruning_indices`` (whole-sequence selection)."""
    device = attention_avg.device
    image_start = int(fastv_config["image_token_start_index"])
    image_len = int(fastv_config["image_token_length"])
    image_start, image_end = slice_range(image_start, image_start + image_len, seq_length)
    image_len = image_end - image_start
    image_spans = visual_token_spans(image_start, image_end, seq_length, _config_from_dict(fastv_config))

    if image_len <= 0:
        keep_indices = torch.arange(seq_length, device=device)
        return keep_indices, {
            "scores": None,
            "image_token_start_index": image_start,
            "image_token_length": image_len,
            "num_keep": image_len,
            "mode": "none",
        }

    prune_ratio = float(fastv_config.get("fastv_r", 0.5))
    num_keep = int(round(image_len * (1.0 - prune_ratio)))
    num_keep = max(0, min(image_len, num_keep))
    if num_keep >= image_len:
        keep_indices = torch.arange(seq_length, device=device)
        return keep_indices, {
            "scores": None,
            "image_token_start_index": image_start,
            "image_token_length": image_len,
            "num_keep": image_len,
            "mode": "none",
            "image_spans": image_spans,
        }

    use_vla_pruner = bool(fastv_config.get("use_vla_pruner", False))
    top_image_indices: list[torch.Tensor] = []
    span_info: list[tuple[int, int, int]] = []
    if use_vla_pruner:
        score_info: dict[str, Any] = {}
        current_action_scores: list[torch.Tensor] = []
        for span_start, span_end in image_spans:
            span_len = span_end - span_start
            span_keep = int(round(span_len * (1.0 - prune_ratio)))
            span_keep = max(0, min(span_len, span_keep))
            span_indices, span_scores = vlapruner_image_indices(
                attention_avg, span_start, span_end, seq_length, span_keep, fastv_config, inputs_embeds
            )
            top_image_indices.append(span_indices)
            if span_scores.get("current_action_scores") is not None:
                current_action_scores.append(span_scores["current_action_scores"])
            span_info.append((span_start, span_end, span_keep))
        top_image_indices = torch.cat(top_image_indices).sort().values
        score_info["image_spans"] = span_info
        if current_action_scores:
            score_info["current_action_scores"] = torch.cat(current_action_scores).detach()
        mode = fastv_config.get("vla_pruner_mode", "semantic_action")
    else:
        span_scores: list[torch.Tensor] = []
        for span_start, span_end in image_spans:
            span_len = span_end - span_start
            span_keep = int(round(span_len * (1.0 - prune_ratio)))
            span_keep = max(0, min(span_len, span_keep))
            scores = fastv_scores(attention_avg, span_start, span_end, seq_length, fastv_config)
            top_image_indices.append(scores.topk(span_keep).indices + span_start)
            span_scores.append(scores.detach())
            span_info.append((span_start, span_end, span_keep))
        top_image_indices = torch.cat(top_image_indices).sort().values
        score_info = {"scores": span_scores, "image_spans": span_info}
        mode = "fastv"

    keep_indices = torch.cat(
        (
            torch.arange(image_start, device=device),
            top_image_indices,
            torch.arange(image_end, seq_length, device=device),
        )
    ).sort().values

    return keep_indices, {
        "image_token_start_index": image_start,
        "image_token_length": image_len,
        "num_keep": num_keep,
        "kept_visual_tokens": int(top_image_indices.numel()),
        "mode": mode,
        **score_info,
    }


def _config_from_dict(fastv_config: dict) -> AdaptiveSelectionConfig:
    """Small front-end to feed the config-backed pure helpers from a config dict.

    ``fastv_pruning_indices`` consumes a dict (mechanical mirror of the
    release); ``_visual_token_spans`` needs only geometry, taken from the dict.
    """
    return AdaptiveSelectionConfig(
        fastv_k=int(fastv_config.get("fastv_k", 3)),
        fastv_r=float(fastv_config.get("fastv_r", 0.5)),
        vla_pruner_layer=int(fastv_config.get("vla_pruner_layer", 15)),
        av_hist_w=max(1, int(fastv_config.get("av_hist_w", 3))),
        av_decay=float(fastv_config.get("av_decay", 0.8)),
        vla_pruner_mode=str(fastv_config.get("vla_pruner_mode", "semantic_action")),
        image_token_start_index=int(fastv_config.get("image_token_start_index", 1)),
        patches_per_image=int(fastv_config.get("patches_per_image", 0)),
        num_images=int(fastv_config.get("num_images", 1)),
        action_horizon=int(fastv_config.get("action_horizon", 0)),
        action_dim=int(fastv_config.get("action_dim", ACTION_DIM)),
        num_actions_chunk=int(fastv_config.get("num_actions_chunk", NUM_ACTIONS_CHUNK)),
        return_attentions=bool(fastv_config.get("return_attentions", True)),
    )


# ---------------------------------------------------------------------------
# Per-episode EMA history and query state.
# ---------------------------------------------------------------------------


class ActionAttentionHistory:
    """Deque of per-query action -> visual attention vectors (release semantics).

    Mirrors the release ``av_hist``: each entry is a detached fp32 CPU vector
    of length ``num_visual_tokens`` captured at layer ``vla_pruner_layer``
    after each forward (full-length during warm-start, mapped through
    ``kept_indices`` once pruning is active).
    """

    def __init__(self, maxlen: int, decay: float) -> None:
        self.maxlen = max(1, int(maxlen))
        self.decay = float(decay)
        self._items = deque(maxlen=self.maxlen)

    def __len__(self) -> int:
        return len(self._items)

    def reset(self) -> None:
        self._items.clear()

    def append(self, scores: torch.Tensor) -> None:
        self._items.append(scores.detach().float().cpu())

    def build_guided(self, device: Any, dtype: Any) -> Optional[torch.Tensor]:
        """Mirror of the release ``_build_historical_action_attention``."""
        if len(self._items) < self.maxlen:
            return None
        weights = torch.tensor(
            [float(self.decay) ** i for i in range(len(self._items))],
            dtype=torch.float32,
            device=device,
        )
        guided = torch.zeros_like(self._items[0], dtype=torch.float32, device=device)
        for weight, scores in zip(weights, reversed(list(self._items))):
            guided = guided + weight * scores.to(device=device, dtype=torch.float32)
        guided = guided / weights.sum().clamp_min(1e-8)
        return guided.to(dtype=dtype)


class AdaptiveQueryState:
    """Per-episode adaptive state: history, query counter, effective ratio.

    The harness must call ``reset()`` at the start of every episode (the
    release calls ``model.reset_av_history()`` per episode). Queries 1..w run
    dense because the history is not yet full; query >= w+1 prunes with the
    configured ratio and the EMA-guided action scores.
    """

    def __init__(self, config: AdaptiveSelectionConfig) -> None:
        self.config = config
        self.history = ActionAttentionHistory(config.av_hist_w, config.av_decay)
        self.query = 0
        self._started = False

    def reset(self) -> None:
        self.history.reset()
        self.query = 0
        self._started = True

    @property
    def started(self) -> bool:
        return self._started

    def effective_prune_ratio(self) -> float:
        """Mirror of ``_build_fastv_config``: 0.0 while the history is warming."""
        if not self._started:
            raise OpenVLASemanticError("adaptive query state must be reset before use")
        if self.config.use_temporal and len(self.history) < self.history.maxlen:
            return 0.0
        return self.config.fastv_r

    def historical_attention(self, device: Any, dtype: Any) -> Optional[torch.Tensor]:
        if not self.config.use_temporal:
            return None
        return self.history.build_guided(device, dtype)

    def record_layer15(
        self,
        layer15_attention: torch.Tensor,
        kept_indices: Optional[torch.Tensor],
        pruning_layer: Optional[int],
        layout: OfficialInferenceLayout,
    ) -> None:
        """Mirror of the release ``_update_action_attention_history``.

        ``layer15_attention`` is the raw layer-15 attention tensor
        (1, heads, q, k). When pruning already happened (``kept_indices`` and
        ``pruning_layer`` present and layer 15 is past the pruning layer), the
        pruned-length attention is mapped back to absolute visual indices and
        pruned positions stay zero; otherwise the full-length rows/columns are
        used. ``layout`` supplies the fixed visual/action boundaries.
        """
        if not self.config.use_temporal:
            return
        if int(layer15_attention.dim()) != 4 or int(layer15_attention.shape[0]) != 1:
            raise OpenVLASemanticError("layer-15 attention must be (1, heads, q, k)")

        visual_start = self.config.image_token_start_index
        visual_end = visual_start + self.config.num_visual_tokens
        action_start = int(layout.action_readout_positions[0])
        action_end = action_start + len(layout.action_readout_positions)

        attention_avg = layer15_attention.to(torch.float32).mean(dim=1)[0]
        if kept_indices is not None and pruning_layer is not None and self.config.vla_pruner_layer > int(pruning_layer):
            kept = kept_indices.to(attention_avg.device)
            action_mask = (kept >= action_start) & (kept < action_end)
            visual_mask = (kept >= visual_start) & (kept < visual_end)
            action_positions = torch.nonzero(action_mask, as_tuple=False).flatten()
            visual_positions = torch.nonzero(visual_mask, as_tuple=False).flatten()
            if action_positions.numel() == 0:
                return
            action_to_kept_visual = attention_avg.index_select(0, action_positions)
            if visual_positions.numel() > 0:
                action_to_kept_visual = action_to_kept_visual.index_select(1, visual_positions).mean(dim=0)
                action_scores = torch.zeros(
                    self.config.num_visual_tokens,
                    dtype=action_to_kept_visual.dtype,
                    device=action_to_kept_visual.device,
                )
                visual_rel = kept.index_select(0, visual_positions) - visual_start
                action_scores.index_copy_(0, visual_rel, action_to_kept_visual)
            else:
                action_scores = torch.zeros(
                    self.config.num_visual_tokens,
                    dtype=attention_avg.dtype,
                    device=attention_avg.device,
                )
        else:
            action_scores = attention_avg[action_start:action_end, visual_start:visual_end].mean(dim=0)

        if int(action_scores.numel()) == self.config.num_visual_tokens:
            self.history.append(action_scores)


# ---------------------------------------------------------------------------
# Model-level adaptive forward.
# ---------------------------------------------------------------------------


@dataclass
class AdaptiveForwardResult:
    hidden: torch.Tensor          # normed final hidden states (1, kept, D)
    positions: torch.Tensor       # 1-D absolute kept positions
    attention_avg: Optional[torch.Tensor]  # head-mean attention at the prune layer (pre-prune)
    score_info: Optional[dict]
    pruning_info: dict
    hidden_states: Optional[tuple]  # per-layer hidden states + final normed, like upstream
    attentions: tuple                # all-layer attention tensors (return_attentions)
    pruned: bool
    retained_visual_tokens: int


def adaptive_decoder_forward(
    decoder: Any,
    embeddings: torch.Tensor,
    layout: OfficialInferenceLayout,
    config: AdaptiveSelectionConfig,
    state: AdaptiveQueryState,
    *,
    expected_layers: int = 32,
) -> AdaptiveForwardResult:
    """Run the pinned VLA-Pruner-OFT decoder forward on one current query.

    Execution order mirrors the release: layers 0..fastv_k run at full length
    with attention computation; at layer ``fastv_k`` the head-mean attention
    selects ``keep_indices``; from layer ``fastv_k + 1`` on the hidden states
    run on the shortened sequence with ``position_ids = keep_indices`` (fresh
    causal mask, absolute rotary). Layers are run exactly like the vanilla
    forward (``attention_mask=None``; ``LlamaSdpaAttention`` computes the same
    causal mask), so a ratio-0 query is byte-identical to the native forward.
    The layer-15 attention record is appended to the per-episode history after
    the forward, mirroring the release ordering.
    """
    if (
        torch.is_grad_enabled()
        or decoder.training
        or embeddings.ndim != 3
        or embeddings.shape[:2] != (1, layout.full_sequence_tokens)
        or len(decoder.layers) != expected_layers
        or any(type(x.self_attn).__name__ != "LlamaSdpaAttention" for x in decoder.layers)
    ):
        raise ValueError("original inference-only SDPA decoder and full input required")
    if not state.started:
        raise OpenVLASemanticError("adaptive query state must be reset before use")

    seq_length = int(embeddings.shape[1])
    device = embeddings.device
    positions = torch.arange(seq_length, device=device, dtype=torch.long)
    prune_ratio = state.effective_prune_ratio()
    num_keep_total = int(round(config.num_visual_tokens * (1.0 - prune_ratio)))
    pruned_expected = prune_ratio > 0.0 and num_keep_total < config.num_visual_tokens

    fastv_layer = max(0, min(int(config.fastv_k), len(decoder.layers) - 1))
    action_start = int(layout.action_readout_positions[0])
    action_end = action_start + len(layout.action_readout_positions)
    if not (1 <= action_start < action_end <= seq_length - 1):
        raise OpenVLASemanticError("action rows do not lie inside the query sequence")

    fastv_config = build_fastv_config(
        config=config,
        action_start=action_start,
        action_end=action_end,
        prune_ratio=prune_ratio,
        historical_attention=state.historical_attention(device, embeddings.dtype),
    )

    hidden = embeddings
    hidden_states: list[torch.Tensor] = [embeddings]
    attentions: list[torch.Tensor] = []
    attention_avg: Optional[torch.Tensor] = None
    score_info: Optional[dict] = None
    keep_indices: Optional[torch.Tensor] = None
    pruning_layer: Optional[int] = None
    pruned = False
    retained_visual = config.num_visual_tokens
    cache_position = positions

    for layer_idx, decoder_layer in enumerate(decoder.layers):
        layer_outputs = decoder_layer(
            hidden,
            attention_mask=None,
            position_ids=positions.unsqueeze(0),
            past_key_value=None,
            output_attentions=True,
            use_cache=False,
            cache_position=cache_position,
        )
        layer_attention = layer_outputs[1]
        hidden = layer_outputs[0]
        if config.return_attentions:
            attentions.append(layer_attention)
        hidden_states.append(hidden)

        if (
            layer_idx == fastv_layer
            and pruned_expected
            and keep_indices is None
            and layer_attention is not None
        ):
            attention_avg = torch.mean(layer_attention, dim=1)[0]
            keep_indices, score_info = fastv_pruning_indices(
                attention_avg, seq_length, fastv_config, inputs_embeds=embeddings
            )
            if keep_indices.shape[0] < seq_length:
                hidden = hidden.index_select(1, keep_indices)
                positions = keep_indices
                cache_position = torch.arange(keep_indices.shape[0], device=device)
                pruning_layer = layer_idx
                pruned = True
                retained_visual = int(score_info["kept_visual_tokens"])

    hidden = decoder.norm(hidden)
    hidden_states.append(hidden)

    pruning_info = {
        "original_seq_length": seq_length,
        "pruned_indices": None,
        "kept_indices": keep_indices,
        "pruning_layer": pruning_layer,
        "mode": "none" if not (pruned or pruned_expected) else (
            score_info.get("mode") if score_info is not None else "vla_pruner"
        ),
    }
    if pruned:
        all_indices = torch.arange(seq_length, device=device)
        pruning_info["pruned_indices"] = all_indices[~torch.isin(all_indices, keep_indices)]

    final = AdaptiveForwardResult(
        hidden=hidden,
        positions=positions,
        attention_avg=attention_avg,
        score_info=score_info,
        pruning_info=pruning_info,
        hidden_states=tuple(hidden_states),
        attentions=tuple(attentions) if config.return_attentions else (),
        pruned=pruned,
        retained_visual_tokens=retained_visual,
    )

    # Release order: the action history is updated after the forward with the
    # layer-15 attention (from the returned attentions).
    if config.return_attentions and len(final.attentions) == expected_layers:
        layer_id = max(0, min(int(config.vla_pruner_layer), len(final.attentions) - 1))
        state.record_layer15(final.attentions[layer_id], keep_indices, pruning_layer, layout)

    state.query += 1
    final.pruning_info.update({"query": state.query, "effective_fastv_r": prune_ratio})
    return final


def build_fastv_config(
    *,
    config: AdaptiveSelectionConfig,
    action_start: int,
    action_end: int,
    prune_ratio: float,
    historical_attention: Optional[torch.Tensor],
) -> dict:
    """Mirror of the release ``_build_fastv_config`` (per query)."""
    return {
        "fastv_k": int(config.fastv_k),
        "fastv_r": float(prune_ratio),
        "vla_pruner_layer": int(config.vla_pruner_layer),
        "image_token_start_index": int(config.image_token_start_index),
        "image_token_length": int(config.num_visual_tokens),
        "historical_attention": historical_attention,
        "use_temporal": bool(config.use_temporal),
        "use_text_vision_selection": False,
        "use_prefil_attention": True,
        "fastv_attention_source": "prefill",
        "SparseVLM": False,
        "use_vla_pruner": True,
        "vla_pruner_mode": str(config.vla_pruner_mode),
        "semantic_weight": 0.5,
        "action_weight": 0.5,
        "text_token_start": int(config.image_token_start_index + config.num_visual_tokens),
        "text_token_end": int(action_start),
        "action_token_start": int(action_start),
        "action_token_end": int(action_end),
        "patches_per_image": int(config.patches_per_image),
        "num_images": int(config.num_images),
        "action_horizon": int(config.action_horizon),
        "action_dim": int(config.action_dim),
        "return_attentions": bool(config.return_attentions),
    }