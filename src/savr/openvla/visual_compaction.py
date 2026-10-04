"""Validated visual-only compaction for the authenticated OFT token layout.

This is shared compatibility infrastructure, not a SpecPrune implementation.
It does not choose tokens, change attention, retain stale K/V, or load a model.
The host-side checks are for qualification; their cost must not be hidden if
this implementation is subsequently used in measured inference.
"""

from dataclasses import dataclass
from typing import Any, Sequence

from .official_semantics import OfficialInferenceLayout, OpenVLASemanticError


@dataclass(frozen=True)
class VisualCompaction:
    layout: OfficialInferenceLayout
    absolute_positions: tuple[int, ...]

    def __post_init__(self) -> None:
        positions = self.absolute_positions
        total = self.layout.full_sequence_tokens
        if self.layout.projected_tokens != 513:
            raise OpenVLASemanticError("compaction requires the pinned two-camera layout")
        if (not positions or any(type(p) is not int for p in positions)
                or tuple(sorted(set(positions))) != positions
                or positions[0] < 0 or positions[-1] >= total):
            raise OpenVLASemanticError("positions must be sorted unique integer token IDs")
        protected = {0, *range(513, total)}
        if not protected.issubset(positions):
            raise OpenVLASemanticError("compaction removed a protected nonvisual token")
        if not any(1 <= p <= 256 for p in positions) or not any(257 <= p <= 512 for p in positions):
            raise OpenVLASemanticError("both cameras must retain current visual tokens")

    @classmethod
    def dense(cls, layout: OfficialInferenceLayout) -> "VisualCompaction":
        return cls(layout, tuple(range(layout.full_sequence_tokens)))

    def retain_visual(self, visual_positions: Sequence[int]) -> "VisualCompaction":
        values = tuple(visual_positions)
        if (any(type(p) is not int or not 1 <= p <= 512 for p in values)
                or len(set(values)) != len(values)):
            raise OpenVLASemanticError("selection must contain unique absolute visual token IDs")
        if not set(values).issubset(self.absolute_positions):
            raise OpenVLASemanticError("a later layer cannot resurrect a removed token")
        protected = tuple(p for p in self.absolute_positions if p == 0 or p >= 513)
        return VisualCompaction(self.layout, tuple(sorted(protected + values)))

    def offsets_from(self, previous: "VisualCompaction") -> tuple[int, ...]:
        if previous.layout != self.layout:
            raise OpenVLASemanticError("compaction cannot cross query layouts")
        offsets = {p: i for i, p in enumerate(previous.absolute_positions)}
        if not set(self.absolute_positions).issubset(offsets):
            raise OpenVLASemanticError("compaction cannot resurrect a removed token")
        return tuple(offsets[p] for p in self.absolute_positions)

    def action_offsets(self) -> tuple[int, ...]:
        offsets = {p: i for i, p in enumerate(self.absolute_positions)}
        return tuple(offsets[p] for p in self.layout.action_readout_positions)

    def compact_hidden(self, torch_module: Any, hidden: Any,
                       previous: "VisualCompaction") -> Any:
        if (hidden.ndim != 3 or hidden.shape[0] != 1
                or hidden.shape[1] != len(previous.absolute_positions)):
            raise OpenVLASemanticError("hidden sequence does not match the previous position map")
        offsets = self.offsets_from(previous)
        index = torch_module.tensor(offsets, device=hidden.device, dtype=torch_module.long)
        return hidden.index_select(1, index)

    def fresh_decoder_arguments(self, torch_module: Any, dense_embeddings: Any) -> dict:
        """Compact an unpadded current query before layer zero, with no K/V cache.

        A layer-adaptive comparator must also compact intermediate masks and
        report its final position map. This helper alone is not that comparator.
        """
        hidden = self.compact_hidden(torch_module, dense_embeddings, self.dense(self.layout))
        positions = torch_module.tensor(self.absolute_positions, device=hidden.device,
                                       dtype=torch_module.long)
        return dict(inputs_embeds=hidden, input_ids=None,
                    attention_mask=torch_module.ones((1, len(self.absolute_positions)),
                                                     device=hidden.device, dtype=torch_module.long),
                    position_ids=positions.unsqueeze(0), cache_position=positions,
                    past_key_values=None, use_cache=False, output_attentions=False,
                    output_hidden_states=True, return_dict=True)
