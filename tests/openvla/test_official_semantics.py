from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from savr.openvla.official_semantics import (
    OfficialActionHeadCapture,
    OfficialBoundaryCapture,
    OpenVLASemanticError,
    SemanticSDPASidecarTap,
    cache_fork_active_positions,
    derive_official_layout,
    derive_semantic_runtime_positions,
    select_official_action_hidden,
    structurally_aligned_dense_forward,
)


torch = pytest.importorskip("torch")


def mask_for_prompt(prompt_input_tokens: int) -> torch.Tensor:
    total = prompt_input_tokens + 56 + 1
    mask = torch.zeros((1, total), dtype=torch.bool)
    mask[:, prompt_input_tokens : prompt_input_tokens + 56] = True
    return mask


@pytest.mark.parametrize("prompt_input_tokens", [5, 12, 22, 37])
def test_layout_matches_released_prompt_count_formula(prompt_input_tokens: int) -> None:
    layout = derive_official_layout(
        action_mask=mask_for_prompt(prompt_input_tokens),
        projected_tokens=513,
        instruction_token_indices=(2, 3),
    )
    assert layout.prompt_tokens == prompt_input_tokens - 1
    assert layout.action_readout_positions == tuple(
        range(513 + prompt_input_tokens - 1, 513 + prompt_input_tokens - 1 + 56)
    )
    assert layout.placeholder_multimodal_positions[0] == layout.action_readout_positions[0] + 1
    assert layout.stop_position == layout.full_sequence_tokens - 1


def test_sentinel_hidden_proves_former_tail_slice_is_shifted() -> None:
    layout = derive_official_layout(
        action_mask=mask_for_prompt(22),
        projected_tokens=513,
        instruction_token_indices=(4, 5, 6),
    )
    hidden = torch.arange(layout.full_sequence_tokens, dtype=torch.float32).reshape(1, -1, 1)
    official = select_official_action_hidden(hidden, layout)
    former = hidden[:, -57:-1, :]
    assert torch.equal(official[:, 1:, :], former[:, :-1, :])
    assert not torch.equal(official, former)
    assert official[0, 0, 0].item() == layout.full_sequence_tokens - 58
    assert official[0, -1, 0].item() == layout.full_sequence_tokens - 3


def test_compacted_hidden_uses_absolute_position_map_without_tail_assumptions() -> None:
    layout = derive_official_layout(
        action_mask=mask_for_prompt(22),
        projected_tokens=513,
        instruction_token_indices=(4, 5, 6),
    )
    removed = set(range(1, 97)) | set(range(257, 321))
    positions = torch.tensor(
        [position for position in range(layout.full_sequence_tokens) if position not in removed],
        dtype=torch.long,
    )
    hidden = positions.float().reshape(1, -1, 1)
    selected = select_official_action_hidden(hidden, layout, positions)
    assert tuple(selected.flatten().tolist()) == tuple(
        float(position) for position in layout.action_readout_positions
    )


def test_compacted_hidden_rejects_missing_action_or_invalid_position_map() -> None:
    layout = derive_official_layout(
        action_mask=mask_for_prompt(22),
        projected_tokens=513,
        instruction_token_indices=(4, 5, 6),
    )
    positions = torch.arange(layout.full_sequence_tokens, dtype=torch.long)
    missing = positions[positions != layout.action_readout_positions[0]]
    with pytest.raises(OpenVLASemanticError, match="removed an official action"):
        select_official_action_hidden(
            torch.zeros((1, len(missing), 4)), layout, missing
        )
    duplicate = positions.clone()
    duplicate[1] = duplicate[0]
    with pytest.raises(OpenVLASemanticError, match="sorted unique"):
        select_official_action_hidden(
            torch.zeros((1, len(duplicate), 4)), layout, duplicate
        )


def test_cache_fork_position_map_contract_is_explicit() -> None:
    positions = torch.arange(12)
    assert cache_fork_active_positions(SimpleNamespace(attentions=(positions,))) is positions
    with pytest.raises(OpenVLASemanticError, match="one final position map"):
        cache_fork_active_positions(SimpleNamespace(attentions=None))
    with pytest.raises(OpenVLASemanticError, match="rank one"):
        cache_fork_active_positions(SimpleNamespace(attentions=(positions.unsqueeze(0),)))


def test_canonical_runtime_positions_use_instruction_only_and_official_action_states() -> None:
    positions = derive_semantic_runtime_positions(
        action_mask=mask_for_prompt(22),
        projected_tokens=513,
        instruction_token_indices=(4, 5, 6),
    )
    assert positions["primary"] == tuple(range(1, 257))
    assert positions["wrist"] == tuple(range(257, 513))
    assert positions["proprio"] == (513,)
    assert positions["instruction"] == (517, 518, 519)
    assert positions["action"] == tuple(range(534, 590))
    assert positions["placeholder"] == tuple(range(535, 591))
    assert positions["stop"] == (591,)


@pytest.mark.parametrize(
    "mutation, message",
    [
        ("short", "56 action placeholders"),
        ("gap", "contiguous"),
        ("bad_stop", "prompt/action/stop"),
    ],
)
def test_layout_rejects_malformed_action_masks(mutation: str, message: str) -> None:
    mask = mask_for_prompt(22)
    if mutation == "short":
        mask[:, 77] = False
    elif mutation == "gap":
        mask[:, 30] = False
        mask[:, 78] = True
    else:
        mask = torch.cat([mask, torch.zeros((1, 1), dtype=torch.bool)], dim=1)
    with pytest.raises(OpenVLASemanticError, match=message):
        derive_official_layout(
            action_mask=mask,
            projected_tokens=513,
            instruction_token_indices=(2, 3),
        )


def test_layout_rejects_invalid_instruction_positions_and_hidden_length() -> None:
    mask = mask_for_prompt(22)
    with pytest.raises(OpenVLASemanticError, match="instruction-only"):
        derive_official_layout(
            action_mask=mask,
            projected_tokens=513,
            instruction_token_indices=(2, 2),
        )
    layout = derive_official_layout(
        action_mask=mask,
        projected_tokens=513,
        instruction_token_indices=(2, 3),
    )
    with pytest.raises(OpenVLASemanticError, match="explicit active positions"):
        select_official_action_hidden(torch.zeros((1, layout.full_sequence_tokens - 1, 4)), layout)


class DummyHead:
    def __init__(self) -> None:
        self.inputs = []

    def predict_action(self, hidden):
        self.inputs.append(hidden)
        return hidden.sum(dim=-1)


def test_official_action_head_capture_is_exact_and_restores_class_method() -> None:
    head = DummyHead()
    original = head.predict_action
    hidden = torch.zeros((1, 56, 8))
    with OfficialActionHeadCapture(head) as capture:
        result = head.predict_action(hidden)
        assert result.shape == (1, 56)
    assert capture.calls == 1
    assert torch.equal(capture.exact_hidden(), hidden)
    assert torch.equal(capture.exact_output(), result)
    assert head.predict_action.__func__ is original.__func__
    assert "predict_action" not in vars(head)


def test_official_action_head_capture_restores_after_exception_and_rejects_reentry() -> None:
    head = DummyHead()
    original = head.predict_action
    capture = OfficialActionHeadCapture(head)
    with pytest.raises(RuntimeError, match="injected"):
        with capture:
            head.predict_action(torch.zeros((1, 56, 8)))
            raise RuntimeError("injected")
    assert head.predict_action.__func__ is original.__func__
    with OfficialActionHeadCapture(head) as repeated:
        head.predict_action(torch.zeros((1, 56, 8)))
        with pytest.raises(OpenVLASemanticError, match="more than once"):
            head.predict_action(torch.zeros((1, 56, 8)))
    assert repeated.calls == 2


def test_official_action_head_capture_supports_torch_modules() -> None:
    class ModuleHead(torch.nn.Module):
        def predict_action(self, hidden):
            return hidden.mean(dim=-1)

    head = ModuleHead()
    original = head.predict_action
    hidden = torch.ones((1, 56, 16))
    with OfficialActionHeadCapture(head) as capture:
        assert tuple(head.predict_action(hidden).shape) == (1, 56)
    assert torch.equal(capture.exact_hidden(), hidden)
    assert head.predict_action.__func__ is original.__func__
    assert "predict_action" not in vars(head)


def test_semantic_sidecar_is_neutral_and_restores_sdpa() -> None:
    original = torch.nn.functional.scaled_dot_product_attention
    query = torch.randn(1, 1, 2, 4)
    key = torch.randn(1, 1, 2, 4)
    value = torch.randn(1, 1, 2, 4)
    expected = original(query, key, value)
    with SemanticSDPASidecarTap(torch, (6, 15, 24)) as tap:
        observed = None
        for _ in range(32):
            observed = torch.nn.functional.scaled_dot_product_attention(query, key, value)
    assert tap.calls == 32 and set(tap.captured) == {6, 15, 24}
    assert torch.equal(observed, expected)
    assert torch.nn.functional.scaled_dot_product_attention is original


def test_semantic_sidecar_restores_after_exception() -> None:
    original = torch.nn.functional.scaled_dot_product_attention
    with pytest.raises(RuntimeError, match="injected"):
        with SemanticSDPASidecarTap(torch, (6, 15, 24)):
            raise RuntimeError("injected")
    assert torch.nn.functional.scaled_dot_product_attention is original


def test_full_official_boundary_capture_records_once_and_restores() -> None:
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(100, 4)

        def get_input_embeddings(self):
            return self.embedding

        def _process_action_masks(self, labels):
            return labels.bool()

        def _process_vision_features(self, pixels, language, use_film=False):
            return pixels + language

        def _process_proprio_features(self, projected, proprio, projector):
            return projected + proprio

        def _build_multimodal_attention(self, embeddings, projected, mask):
            return embeddings + projected, mask

    model = Model()
    head = DummyHead()
    originals = {
        name: getattr(model, name).__func__
        for name in (
            "_process_action_masks",
            "_process_vision_features",
            "_process_proprio_features",
            "_build_multimodal_attention",
        )
    }
    pixels = torch.ones((1, 56, 4))
    language = torch.full((1, 56, 4), 2.0)
    proprio = torch.full((1, 56, 4), 3.0)
    mask = torch.ones((1, 56), dtype=torch.long)
    with OfficialBoundaryCapture(model, head) as capture:
        input_ids = torch.arange(56).reshape(1, 56)
        input_embeddings = model.get_input_embeddings()(input_ids)
        action_mask = model._process_action_masks(mask)
        vision = model._process_vision_features(pixels, language, False)
        projected = model._process_proprio_features(vision, proprio, None)
        multimodal, observed_mask = model._build_multimodal_attention(language, projected, mask)
        head.predict_action(multimodal)
        assert torch.equal(observed_mask, mask)
    values = capture.exact_values()
    assert torch.equal(values["input_ids"], input_ids)
    assert torch.equal(values["input_embeddings"], input_embeddings)
    assert torch.equal(values["action_mask"], action_mask)
    assert torch.equal(values["pixel_values"], pixels)
    assert torch.equal(values["projected_with_proprio"], projected)
    assert torch.equal(values["multimodal_embeddings"], multimodal)
    assert torch.equal(values["action_hidden"], multimodal)
    assert torch.equal(values["normalized_actions"], multimodal.sum(dim=-1))
    for name, original in originals.items():
        assert getattr(model, name).__func__ is original
        assert name not in vars(model)


def test_full_official_boundary_capture_restores_every_method_on_exception() -> None:
    class Model:
        def __init__(self):
            self.embedding = torch.nn.Embedding(10, 2)

        def get_input_embeddings(self):
            return self.embedding

        def _process_action_masks(self, labels):
            return labels.bool()

        def _process_vision_features(self, pixels, language, use_film=False):
            raise RuntimeError("injected")

        def _process_proprio_features(self, projected, proprio, projector):
            return projected

        def _build_multimodal_attention(self, embeddings, projected, mask):
            return embeddings, mask

    model = Model()
    head = DummyHead()
    originals = {name: getattr(model, name).__func__ for name in (
        "_process_action_masks", "_process_vision_features", "_process_proprio_features", "_build_multimodal_attention"
    )}
    with pytest.raises(RuntimeError, match="injected"):
        with OfficialBoundaryCapture(model, head):
            model._process_vision_features(torch.zeros(1), torch.zeros(1), False)
    for name, original in originals.items():
        assert getattr(model, name).__func__ is original
        assert name not in vars(model)


@pytest.mark.parametrize(
    ("use_cache", "expects_cache"),
    [(None, False), (False, False), (True, True)],
)
def test_structurally_aligned_dense_forward_uses_official_states_and_cache_semantics(
    use_cache, expects_cache
) -> None:
    prompt_input_tokens = 22
    action_mask = mask_for_prompt(prompt_input_tokens)
    input_tokens = int(action_mask.shape[1])
    projected = torch.arange(513, dtype=torch.float32).reshape(1, 513, 1)
    embeddings = torch.arange(input_tokens, dtype=torch.float32).reshape(1, input_tokens, 1)

    class Language:
        def __init__(self) -> None:
            self.observed_use_cache = None

        def __call__(self, **kwargs):
            self.observed_use_cache = kwargs["use_cache"]
            return SimpleNamespace(
                hidden_states=(kwargs["inputs_embeds"],),
                past_key_values=("cache",) if kwargs["use_cache"] else None,
            )

    class Model:
        def __init__(self) -> None:
            self.language_model = Language()

        def _build_multimodal_attention(self, inputs, patches, mask):
            return (
                torch.cat([inputs[:, :1], patches, inputs[:, 1:]], dim=1),
                torch.cat(
                    [mask[:, :1], torch.ones((1, 513), dtype=mask.dtype), mask[:, 1:]],
                    dim=1,
                ),
            )

        def _unnormalize_actions(self, actions, key):
            assert key == "suite"
            return actions

    class Head:
        def predict_action(self, hidden):
            return hidden[..., 0]

    prepared = SimpleNamespace(
        action_mask=action_mask,
        projected_patches=projected,
        input_embeddings=embeddings,
        attention_mask=torch.ones((1, input_tokens), dtype=torch.long),
        instruction_token_indices=(4, 5, 6),
    )
    model = Model()
    result = structurally_aligned_dense_forward(
        torch_module=torch,
        np_module=np,
        model=model,
        action_head=Head(),
        cfg=SimpleNamespace(unnorm_key="suite"),
        prepared=prepared,
        use_cache=use_cache,
    )
    expected = result["multimodal_embeddings"][:, -58:-2, :]
    assert torch.equal(result["action_hidden"], expected)
    assert (result["cache"] is not None) is expects_cache
    assert model.language_model.observed_use_cache is use_cache
    assert result["actions"].shape == (8, 7)


def test_structurally_aligned_dense_forward_rejects_unknown_cache_mode() -> None:
    with pytest.raises(OpenVLASemanticError, match="None, false, or true"):
        structurally_aligned_dense_forward(
            torch_module=torch,
            np_module=np,
            model=None,
            action_head=None,
            cfg=None,
            prepared=None,
            use_cache="invalid",
        )
