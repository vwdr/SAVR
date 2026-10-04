"""Structurally derived OpenVLA-OFT action-readout semantics.

This module is independent of the historical BRACE/PAIR tail-slice helpers. It
defines the pinned released evaluator's regression-head positions from runtime
tensor structure and provides a reversible hook that captures the true official
action-head input for real-model parity tests.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence


class OpenVLASemanticError(RuntimeError):
    """Raised when runtime tensors do not satisfy the official policy contract."""


@dataclass(frozen=True)
class OfficialInferenceLayout:
    projected_tokens: int
    input_tokens: int
    full_sequence_tokens: int
    prompt_tokens: int
    placeholder_input_positions: tuple[int, ...]
    placeholder_multimodal_positions: tuple[int, ...]
    action_readout_positions: tuple[int, ...]
    instruction_input_positions: tuple[int, ...]
    instruction_multimodal_positions: tuple[int, ...]
    stop_position: int


@dataclass(frozen=True)
class SemanticPreparedQuery:
    """All custom-side tensors needed for independent official parity."""

    input_ids: Any
    input_embeddings: Any
    action_mask: Any
    language_embeddings: Any
    vision_output: Any
    projected_patches: Any
    attention_mask: Any
    normalized_proprio: Any
    preprocessed_pixels: Any
    instruction_token_indices: tuple[int, ...]


class SemanticSDPASidecarTap:
    """Reversible post-RoPE Q/K capture used only to prove sidecar neutrality."""

    def __init__(self, torch_module: Any, layers: Sequence[int]) -> None:
        self.torch = torch_module
        self.layers = frozenset(int(layer) for layer in layers)
        self.captured: dict[int, tuple[Any, Any, Any, float]] = {}
        self.calls = 0
        self._original: Any = None

    def __enter__(self) -> "SemanticSDPASidecarTap":
        if self._original is not None:
            raise OpenVLASemanticError("semantic sidecar cannot be nested")
        functional = self.torch.nn.functional
        self._original = functional.scaled_dot_product_attention

        def wrapped(query: Any, key: Any, value: Any, *args: Any, **kwargs: Any) -> Any:
            layer = self.calls
            self.calls += 1
            if layer in self.layers:
                mask = kwargs.get("attn_mask")
                if mask is None and args:
                    mask = args[0]
                scale = kwargs.get("scale")
                if scale is None:
                    scale = 1.0 / math.sqrt(int(query.shape[-1]))
                self.captured[layer] = (
                    query.detach(),
                    key.detach(),
                    mask.detach() if mask is not None else None,
                    float(scale),
                )
            return self._original(query, key, value, *args, **kwargs)

        functional.scaled_dot_product_attention = wrapped
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        original = self._original
        self._original = None
        if original is not None:
            self.torch.nn.functional.scaled_dot_product_attention = original
        if exc_type is None and (
            self.calls != 32 or set(self.captured) != set(self.layers)
        ):
            raise OpenVLASemanticError(
                "semantic sidecar did not observe exactly 32 decoder layers"
            )


def _true_positions(mask: Any) -> tuple[int, ...]:
    if getattr(mask, "ndim", None) != 2 or int(mask.shape[0]) != 1:
        raise OpenVLASemanticError("action mask must be rank two with batch size one")
    values = mask[0].nonzero(as_tuple=False).flatten().tolist()
    return tuple(int(value) for value in values)


def derive_official_layout(
    *,
    action_mask: Any,
    projected_tokens: int,
    instruction_token_indices: Sequence[int],
) -> OfficialInferenceLayout:
    """Derive the released evaluator's exact action-head and instruction spans."""

    projected = int(projected_tokens)
    input_tokens = int(action_mask.shape[1])
    if projected != 513:
        raise OpenVLASemanticError("pinned two-camera visual/proprio count must be 513")
    placeholders = _true_positions(action_mask)
    if len(placeholders) != 56:
        raise OpenVLASemanticError("pinned regression policy requires 56 action placeholders")
    if placeholders != tuple(range(placeholders[0], placeholders[0] + 56)):
        raise OpenVLASemanticError("action placeholders must be contiguous")
    if placeholders[0] <= 1 or placeholders[-1] != input_tokens - 2:
        raise OpenVLASemanticError("prompt/action/stop token order changed")

    prompt_tokens = placeholders[0] - 1
    placeholder_multimodal = tuple(projected + value for value in placeholders)
    readout = tuple(value - 1 for value in placeholder_multimodal)
    instruction_input = tuple(int(value) for value in instruction_token_indices)
    if (
        not instruction_input
        or len(set(instruction_input)) != len(instruction_input)
        or tuple(sorted(instruction_input)) != instruction_input
        or min(instruction_input) <= 0
        or max(instruction_input) >= placeholders[0] - 1
    ):
        raise OpenVLASemanticError("instruction-only token positions are invalid")
    instruction_multimodal = tuple(projected + value for value in instruction_input)
    full_sequence = projected + input_tokens
    stop_position = full_sequence - 1
    if (
        readout[0] != projected + prompt_tokens
        or readout[-1] != placeholder_multimodal[-1] - 1
        or readout != tuple(range(readout[0], readout[0] + 56))
        or max(readout) >= stop_position
        or set(readout) & set(instruction_multimodal)
    ):
        raise OpenVLASemanticError("official action-readout positions are inconsistent")
    return OfficialInferenceLayout(
        projected_tokens=projected,
        input_tokens=input_tokens,
        full_sequence_tokens=full_sequence,
        prompt_tokens=prompt_tokens,
        placeholder_input_positions=placeholders,
        placeholder_multimodal_positions=placeholder_multimodal,
        action_readout_positions=readout,
        instruction_input_positions=instruction_input,
        instruction_multimodal_positions=instruction_multimodal,
        stop_position=stop_position,
    )


def select_official_action_hidden(
    last_hidden: Any,
    layout: OfficialInferenceLayout,
    active_positions: Any = None,
) -> Any:
    """Select official action states from dense or explicitly compacted output.

    The pinned VLA-Cache fork returns the final absolute ``cache_position``
    vector alongside its output.  When visual tokens are reused, the hidden
    sequence is compacted, so absolute action positions must be mapped through
    that vector instead of being treated as dense tensor offsets.
    """

    if getattr(last_hidden, "ndim", None) != 3 or int(last_hidden.shape[0]) != 1:
        raise OpenVLASemanticError("language-model hidden-state layout changed")
    if active_positions is None:
        if int(last_hidden.shape[1]) != layout.full_sequence_tokens:
            raise OpenVLASemanticError(
                "compacted hidden states require explicit active positions"
            )
        selected = last_hidden[:, list(layout.action_readout_positions), :]
    else:
        if (
            getattr(active_positions, "ndim", None) != 1
            or int(active_positions.shape[0]) != int(last_hidden.shape[1])
        ):
            raise OpenVLASemanticError("active-position layout does not match hidden states")
        observed = tuple(int(value) for value in active_positions.detach().cpu().tolist())
        if (
            len(set(observed)) != len(observed)
            or tuple(sorted(observed)) != observed
            or not observed
            or observed[0] < 0
            or observed[-1] >= layout.full_sequence_tokens
        ):
            raise OpenVLASemanticError("active positions are not a sorted unique layout subset")
        offsets = {position: offset for offset, position in enumerate(observed)}
        if not all(position in offsets for position in layout.action_readout_positions):
            raise OpenVLASemanticError("cache compaction removed an official action state")
        index = active_positions.new_tensor(
            [offsets[position] for position in layout.action_readout_positions],
            dtype=active_positions.dtype,
        )
        selected = last_hidden.index_select(1, index)
    if tuple(selected.shape[:2]) != (1, 56):
        raise OpenVLASemanticError("official action-head input is not 1x56")
    return selected


def cache_fork_active_positions(output: Any) -> Any:
    """Extract the pinned cache fork's final hidden-to-absolute-position map."""

    attentions = getattr(output, "attentions", None)
    if not isinstance(attentions, tuple) or len(attentions) != 1:
        raise OpenVLASemanticError("cache fork did not expose one final position map")
    positions = attentions[0]
    if getattr(positions, "ndim", None) != 1:
        raise OpenVLASemanticError("cache fork final position map is not rank one")
    return positions


def derive_semantic_runtime_positions(
    *,
    action_mask: Any,
    projected_tokens: int,
    instruction_token_indices: Sequence[int],
) -> dict[str, tuple[int, ...]]:
    """Return the canonical multimodal position map used by cache sidecars."""

    layout = derive_official_layout(
        action_mask=action_mask,
        projected_tokens=projected_tokens,
        instruction_token_indices=instruction_token_indices,
    )
    if projected_tokens != 513:
        raise OpenVLASemanticError("semantic position map requires 513 projected tokens")
    positions = {
        "primary": tuple(range(1, 257)),
        "wrist": tuple(range(257, 513)),
        "proprio": (513,),
        "prompt": tuple(
            projected_tokens + index
            for index in range(1, layout.placeholder_input_positions[0])
        ),
        "instruction": layout.instruction_multimodal_positions,
        "action": layout.action_readout_positions,
        "placeholder": layout.placeholder_multimodal_positions,
        "stop": (layout.stop_position,),
    }
    visual = positions["primary"] + positions["wrist"]
    if (
        len(visual) != 512
        or len(set(visual)) != 512
        or positions["proprio"] != (513,)
        or len(positions["action"]) != 56
        or len(positions["placeholder"]) != 56
        or positions["action"][0] != positions["placeholder"][0] - 1
        or positions["action"][-1] != positions["placeholder"][-1] - 1
        or not positions["instruction"]
        or set(positions["instruction"]) & set(positions["action"])
    ):
        raise OpenVLASemanticError("canonical semantic runtime positions are invalid")
    return positions


class OfficialActionHeadCapture:
    """Reversibly capture the exact tensor received by the official action head."""

    def __init__(self, action_head: Any) -> None:
        self.action_head = action_head
        self.calls = 0
        self.hidden: Any = None
        self.output: Any = None
        self._bound_original: Any = None
        self._had_instance_attribute = False
        self._instance_original: Any = None

    def __enter__(self) -> "OfficialActionHeadCapture":
        if self._bound_original is not None:
            raise OpenVLASemanticError("official action-head capture cannot be nested")
        instance_values = vars(self.action_head)
        self._had_instance_attribute = "predict_action" in instance_values
        self._instance_original = instance_values.get("predict_action")
        self._bound_original = getattr(self.action_head, "predict_action")

        def wrapped(hidden: Any, *args: Any, **kwargs: Any) -> Any:
            self.calls += 1
            if self.calls != 1:
                raise OpenVLASemanticError("official action head was invoked more than once")
            if getattr(hidden, "ndim", None) != 3 or tuple(hidden.shape[:2]) != (1, 56):
                raise OpenVLASemanticError("official action head received an invalid hidden tensor")
            self.hidden = hidden.detach().clone()
            output = self._bound_original(hidden, *args, **kwargs)
            self.output = output.detach().clone()
            return output

        setattr(self.action_head, "predict_action", wrapped)
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self._had_instance_attribute:
            setattr(self.action_head, "predict_action", self._instance_original)
        else:
            delattr(self.action_head, "predict_action")

    def exact_hidden(self) -> Any:
        if self.calls != 1 or self.hidden is None or self.output is None:
            raise OpenVLASemanticError("official action head was not captured exactly once")
        return self.hidden

    def exact_output(self) -> Any:
        self.exact_hidden()
        return self.output


class _MethodPatch:
    """Patch one instance method and restore class/instance lookup exactly."""

    def __init__(self, instance: Any, name: str, replacement: Any) -> None:
        self.instance = instance
        self.name = name
        self.replacement = replacement
        self.had_instance_attribute = False
        self.instance_original: Any = None
        self.bound_original: Any = None

    def install(self) -> Any:
        values = vars(self.instance)
        self.had_instance_attribute = self.name in values
        self.instance_original = values.get(self.name)
        self.bound_original = getattr(self.instance, self.name)
        setattr(self.instance, self.name, self.replacement(self.bound_original))
        return self.bound_original

    def restore(self) -> None:
        if self.had_instance_attribute:
            setattr(self.instance, self.name, self.instance_original)
        else:
            delattr(self.instance, self.name)


class OfficialBoundaryCapture:
    """Capture every material tensor boundary from one released official call."""

    def __init__(self, model: Any, action_head: Any) -> None:
        self.model = model
        self.action_capture = OfficialActionHeadCapture(action_head)
        self.calls = {
            "embedding": 0,
            "action_mask": 0,
            "vision": 0,
            "proprio": 0,
            "multimodal": 0,
        }
        self.values: dict[str, Any] = {}
        self._patches: list[_MethodPatch] = []

    @staticmethod
    def _clone(value: Any) -> Any:
        return value.detach().clone() if value is not None else None

    def _vision_wrapper(self, original: Any) -> Any:
        def wrapped(pixel_values: Any, language_embeddings: Any = None, use_film: bool = False):
            self.calls["vision"] += 1
            if self.calls["vision"] != 1:
                raise OpenVLASemanticError("official vision boundary ran more than once")
            self.values["pixel_values"] = self._clone(pixel_values)
            self.values["language_embeddings"] = self._clone(language_embeddings)
            output = original(pixel_values, language_embeddings, use_film)
            self.values["vision_output"] = self._clone(output)
            return output

        return wrapped

    def _embedding_wrapper(self, original: Any) -> Any:
        def wrapped(input_ids: Any, *args: Any, **kwargs: Any):
            self.calls["embedding"] += 1
            if self.calls["embedding"] != 1:
                raise OpenVLASemanticError("official token embedding ran more than once")
            self.values["input_ids"] = self._clone(input_ids)
            output = original(input_ids, *args, **kwargs)
            self.values["input_embeddings"] = self._clone(output)
            return output

        return wrapped

    def _action_mask_wrapper(self, original: Any) -> Any:
        def wrapped(labels: Any, *args: Any, **kwargs: Any):
            self.calls["action_mask"] += 1
            if self.calls["action_mask"] != 1:
                raise OpenVLASemanticError("official action-mask boundary ran more than once")
            output = original(labels, *args, **kwargs)
            self.values["action_mask"] = self._clone(output)
            return output

        return wrapped

    def _proprio_wrapper(self, original: Any) -> Any:
        def wrapped(projected: Any, proprio: Any, projector: Any):
            self.calls["proprio"] += 1
            if self.calls["proprio"] != 1:
                raise OpenVLASemanticError("official proprio boundary ran more than once")
            self.values["pre_proprio_projected"] = self._clone(projected)
            self.values["normalized_proprio"] = self._clone(proprio)
            output = original(projected, proprio, projector)
            self.values["projected_with_proprio"] = self._clone(output)
            return output

        return wrapped

    def _multimodal_wrapper(self, original: Any) -> Any:
        def wrapped(input_embeddings: Any, projected: Any, attention_mask: Any):
            self.calls["multimodal"] += 1
            if self.calls["multimodal"] != 1:
                raise OpenVLASemanticError("official multimodal boundary ran more than once")
            self.values["masked_input_embeddings"] = self._clone(input_embeddings)
            self.values["multimodal_projected"] = self._clone(projected)
            self.values["input_attention_mask"] = self._clone(attention_mask)
            output_embeddings, output_mask = original(input_embeddings, projected, attention_mask)
            self.values["multimodal_embeddings"] = self._clone(output_embeddings)
            self.values["multimodal_attention_mask"] = self._clone(output_mask)
            return output_embeddings, output_mask

        return wrapped

    def __enter__(self) -> "OfficialBoundaryCapture":
        if self._patches:
            raise OpenVLASemanticError("official boundary capture cannot be nested")
        embedding = self.model.get_input_embeddings()
        specifications = (
            (embedding, "forward", self._embedding_wrapper),
            (self.model, "_process_action_masks", self._action_mask_wrapper),
            (self.model, "_process_vision_features", self._vision_wrapper),
            (self.model, "_process_proprio_features", self._proprio_wrapper),
            (self.model, "_build_multimodal_attention", self._multimodal_wrapper),
        )
        try:
            for instance, name, wrapper in specifications:
                patch = _MethodPatch(instance, name, wrapper)
                patch.install()
                self._patches.append(patch)
            self.action_capture.__enter__()
        except BaseException:
            for patch in reversed(self._patches):
                patch.restore()
            self._patches.clear()
            raise
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        try:
            self.action_capture.__exit__(exc_type, exc, traceback)
        finally:
            for patch in reversed(self._patches):
                patch.restore()
            self._patches.clear()

    def exact_values(self) -> dict[str, Any]:
        if self.calls != {
            "embedding": 1,
            "action_mask": 1,
            "vision": 1,
            "proprio": 1,
            "multimodal": 1,
        }:
            raise OpenVLASemanticError("official tensor boundaries were not captured exactly once")
        values = dict(self.values)
        values["action_hidden"] = self.action_capture.exact_hidden()
        values["normalized_actions"] = self.action_capture.exact_output()
        required = {
            "input_ids",
            "input_embeddings",
            "action_mask",
            "pixel_values",
            "language_embeddings",
            "vision_output",
            "pre_proprio_projected",
            "normalized_proprio",
            "projected_with_proprio",
            "masked_input_embeddings",
            "multimodal_projected",
            "input_attention_mask",
            "multimodal_embeddings",
            "multimodal_attention_mask",
            "action_hidden",
            "normalized_actions",
        }
        if set(values) != required:
            raise OpenVLASemanticError("official tensor-boundary capture is incomplete")
        return values


def prepare_semantic_query(
    *,
    torch_module: Any,
    np_module: Any,
    model: Any,
    processor: Any,
    proprio_projector: Any,
    prepare_images: Any,
    normalize_proprio: Any,
    instruction_indexer: Any,
    cfg: Any,
    raw_scene: Any,
    raw_wrist: Any,
    raw_state: Any,
    instruction: str,
) -> SemanticPreparedQuery:
    """Recreate the pinned dense input path while exposing every boundary."""

    torch = torch_module
    if torch.is_grad_enabled():
        raise OpenVLASemanticError("semantic query preparation requires gradients disabled")
    images = prepare_images([raw_scene, raw_wrist], cfg)
    if len(images) != 2:
        raise OpenVLASemanticError("semantic parity requires exactly two prepared images")
    prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"
    primary = processor(prompt, images[0]).to("cuda:0", dtype=torch.bfloat16)
    wrist = processor(prompt, images[1]).to("cuda:0", dtype=torch.bfloat16)
    pixels = torch.cat([primary["pixel_values"], wrist["pixel_values"]], dim=1)
    input_ids = primary["input_ids"]
    original_input_ids = input_ids.clone()
    attention_mask = primary["attention_mask"]
    instruction_indices = tuple(
        int(value)
        for value in instruction_indexer(
            processor.tokenizer, prompt, instruction, original_input_ids
        )
    )
    if not torch.all(input_ids[:, -1] == 29871):
        input_ids = torch.cat(
            [
                input_ids,
                torch.tensor([[29871]], device=input_ids.device, dtype=input_ids.dtype),
            ],
            dim=1,
        )
    labels = input_ids.clone()
    labels[:] = -100
    input_ids, attention_mask = model._prepare_input_for_action_prediction(
        input_ids, attention_mask
    )
    labels = model._prepare_labels_for_action_prediction(labels, input_ids)
    embeddings = model.get_input_embeddings()(input_ids)
    action_mask = model._process_action_masks(labels)
    language = embeddings[~action_mask].reshape(
        embeddings.shape[0], -1, embeddings.shape[2]
    )
    vision = model._process_vision_features(pixels, language, False)
    stats = model.norm_stats[cfg.unnorm_key]["proprio"]
    normalized_np = normalize_proprio(np_module.asarray(raw_state).copy(), stats)
    normalized = torch.as_tensor(
        normalized_np, device="cuda:0", dtype=vision.dtype
    ).reshape(1, -1)
    projected = model._process_proprio_features(
        vision, normalized, proprio_projector
    )
    if (
        tuple(projected.shape) != (1, 513, 4096)
        or tuple(action_mask.shape) != tuple(embeddings.shape[:2])
        or int(action_mask.sum().item()) != 56
    ):
        raise OpenVLASemanticError("semantic prepared-query tensor layout changed")
    derive_official_layout(
        action_mask=action_mask,
        projected_tokens=int(projected.shape[1]),
        instruction_token_indices=instruction_indices,
    )
    return SemanticPreparedQuery(
        input_ids=input_ids,
        input_embeddings=embeddings,
        action_mask=action_mask,
        language_embeddings=language,
        vision_output=vision,
        projected_patches=projected,
        attention_mask=attention_mask,
        normalized_proprio=normalized,
        preprocessed_pixels=pixels,
        instruction_token_indices=instruction_indices,
    )


def structurally_aligned_dense_forward(
    *,
    torch_module: Any,
    np_module: Any,
    model: Any,
    action_head: Any,
    cfg: Any,
    prepared: Any,
    use_cache: bool | None,
    tap_factory: Any = None,
) -> dict[str, Any]:
    """Run a dense custom forward with the exact official regression readout."""

    if use_cache not in (None, False, True):
        raise OpenVLASemanticError("dense parity use_cache must be None, false, or true")
    layout = derive_official_layout(
        action_mask=prepared.action_mask,
        projected_tokens=int(prepared.projected_patches.shape[1]),
        instruction_token_indices=prepared.instruction_token_indices,
    )
    masked = prepared.input_embeddings * ~prepared.action_mask.unsqueeze(-1)
    multimodal, multimodal_mask = model._build_multimodal_attention(
        masked, prepared.projected_patches, prepared.attention_mask
    )
    if int(multimodal.shape[1]) != layout.full_sequence_tokens:
        raise OpenVLASemanticError("custom multimodal length differs from official layout")
    arguments = {
        "input_ids": None,
        "attention_mask": multimodal_mask,
        "position_ids": None,
        "past_key_values": None,
        "inputs_embeds": multimodal,
        "labels": None,
        "use_cache": use_cache,
        "output_attentions": False,
        "output_hidden_states": True,
        "return_dict": True,
    }
    tap = tap_factory() if tap_factory is not None else None
    if tap is None:
        output = model.language_model(**arguments)
    else:
        with tap:
            output = model.language_model(**arguments)
    hidden = select_official_action_hidden(output.hidden_states[-1], layout)
    normalized = action_head.predict_action(hidden).reshape(8, 7)
    normalized_np = normalized.float().cpu().detach().numpy()
    actions = np_module.asarray(model._unnormalize_actions(normalized_np, cfg.unnorm_key))
    if (
        normalized_np.shape != (8, 7)
        or actions.shape != (8, 7)
        or not np_module.isfinite(normalized_np).all()
        or not np_module.isfinite(actions).all()
    ):
        raise OpenVLASemanticError("structurally aligned dense action is invalid")
    return {
        "layout": layout,
        "masked_input_embeddings": masked,
        "multimodal_embeddings": multimodal,
        "multimodal_attention_mask": multimodal_mask,
        "action_hidden": hidden,
        "normalized_actions": normalized_np,
        "actions": actions,
        "cache": output.past_key_values,
        "tap": tap,
    }
