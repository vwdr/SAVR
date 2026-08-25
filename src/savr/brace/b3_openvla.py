"""Injected PyTorch/OpenVLA helpers for BRACE-B3.

This module imports no numerical or GPU package at import time.  The B3 worker
injects its pinned stack so CPU-only protocol tests remain side-effect free.
"""

from __future__ import annotations

import hashlib
import math
import time
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from savr.brace.b3 import B3ProtocolError


def deterministic_inputs(np: Any, size: int = 256) -> dict[str, tuple[Any, Any]]:
    y, x = np.indices((size, size))
    scene = np.stack(((x + y) % 256, x % 256, y % 256), axis=-1).astype(np.uint8)
    wrist = np.stack(
        ((2 * x + y) % 256, (x + 3 * y) % 256, (255 - x) % 256), axis=-1
    ).astype(np.uint8)
    scene_b = np.ascontiguousarray(scene[:, ::-1, (2, 0, 1)])
    wrist_b = np.ascontiguousarray(wrist[::-1, :, (1, 2, 0)])
    scene_c = scene.copy()
    wrist_c = wrist.copy()
    scene_c[:14, :14] = 255 - scene_c[:14, :14]
    wrist_c[-14:, -14:] = 255 - wrist_c[-14:, -14:]
    return {
        "input-a": (scene, wrist),
        "input-b": (scene_b, wrist_b),
        "input-c": (scene_c, wrist_c),
    }


def midpoint_proprio_state(model: Any, cfg: Any, np: Any) -> Any:
    """Resolve the evaluator-selected normalization key and return a safe midpoint."""

    if not hasattr(cfg, "unnorm_key") or cfg.unnorm_key not in model.norm_stats:
        raise B3ProtocolError("B3 evaluator did not resolve the checkpoint normalization key")
    stats = model.norm_stats[cfg.unnorm_key]["proprio"]
    state = (np.asarray(stats["q01"], dtype=np.float64) + np.asarray(stats["q99"], dtype=np.float64)) / 2
    if state.shape != (8,) or not np.isfinite(state).all():
        raise B3ProtocolError("B3 pinned proprioception statistics changed")
    return state


def array_sha256(value: Any, np: Any) -> str:
    return hashlib.sha256(np.asarray(value).tobytes()).hexdigest()


def action_record(value: Any, np: Any) -> dict[str, Any]:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (8, 7) or not np.isfinite(array).all():
        raise B3ProtocolError("B3 action output must be finite [8,7]")
    return {
        "shape": [8, 7],
        "sha256": array_sha256(array, np),
        "values": array.tolist(),
        "gripper_decisions": [bool(item > 0) for item in array[:, -1]],
    }


def compare_actions(
    reference: Any,
    candidate: Any,
    *,
    np: Any,
    rtol: float,
    atol: float,
    exact_gripper: bool,
) -> dict[str, Any]:
    left = np.asarray(reference, dtype=np.float32)
    right = np.asarray(candidate, dtype=np.float32)
    metadata = left.shape == right.shape == (8, 7)
    finite = bool(np.isfinite(left).all() and np.isfinite(right).all())
    close = metadata and finite and bool(np.allclose(left, right, rtol=rtol, atol=atol))
    gripper = bool(np.array_equal(left[:, -1] > 0, right[:, -1] > 0))
    return {
        "passed": close and (gripper or not exact_gripper),
        "finite": finite,
        "gripper_exact": gripper,
        "maximum_absolute_difference": (
            float(np.max(np.abs(left - right))) if metadata else None
        ),
        "reference_sha256": array_sha256(left, np),
        "candidate_sha256": array_sha256(right, np),
    }


class SDPASidecarTap(AbstractContextManager["SDPASidecarTap"]):
    """Capture post-RoPE Q/K while still calling the original SDPA primitive."""

    def __init__(self, torch_module: Any, layers: Sequence[int]) -> None:
        self.torch = torch_module
        self.layers = frozenset(int(layer) for layer in layers)
        self.captured: dict[int, tuple[Any, Any, Any, float]] = {}
        self.calls = 0
        self._original: Any = None

    def __enter__(self) -> "SDPASidecarTap":
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
                self.captured[layer] = (query.detach(), key.detach(), mask, float(scale))
            return self._original(query, key, value, *args, **kwargs)

        functional.scaled_dot_product_attention = wrapped
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.torch.nn.functional.scaled_dot_product_attention = self._original
        if exc_type is None and (self.calls != 32 or set(self.captured) != set(self.layers)):
            raise B3ProtocolError(
                f"B3 dense sidecar observed {self.calls} decoder calls and layers "
                f"{sorted(self.captured)}, expected 32 and {sorted(self.layers)}"
            )

    def salience(
        self,
        *,
        instruction_positions: Sequence[int],
        action_positions: Sequence[int],
        visual_positions: Sequence[int],
    ) -> Any:
        torch = self.torch
        families = []
        first_query = next(iter(self.captured.values()))[0]
        device = first_query.device
        for positions in (instruction_positions, action_positions):
            per_layer = []
            index = torch.as_tensor(tuple(positions), device=device, dtype=torch.long)
            visual = torch.as_tensor(tuple(visual_positions), device=device, dtype=torch.long)
            for query, key, mask, scale in self.captured.values():
                selected_query = query.index_select(2, index)
                logits = torch.matmul(selected_query.float(), key.float().transpose(-1, -2)) * scale
                if mask is not None:
                    selected_mask = mask
                    if selected_mask.ndim == 4 and selected_mask.shape[-2] != 1:
                        selected_mask = selected_mask.index_select(-2, index)
                    logits = logits + selected_mask.float()
                probabilities = torch.softmax(logits, dim=-1)
                per_layer.append(probabilities.index_select(-1, visual).mean(dim=(1, 2))[0])
            score = torch.stack(per_layer).mean(dim=0)
            order = torch.argsort(torch.argsort(score))
            families.append(order.float() / max(1, int(score.numel()) - 1))
        return torch.maximum(families[0], families[1])


@dataclass
class PreparedQuery:
    input_embeddings: Any
    action_mask: Any
    projected_patches: Any
    attention_mask: Any
    prompt_tokens: int
    normalized_proprio: Any
    preprocessed_pixels: Any


def prepare_query(
    *,
    torch_module: Any,
    np: Any,
    model: Any,
    processor: Any,
    proprio_projector: Any,
    prepare_images: Any,
    normalize_proprio: Any,
    cfg: Any,
    raw_scene: Any,
    raw_wrist: Any,
    raw_state: Any,
    instruction: str,
) -> PreparedQuery:
    torch = torch_module
    images = prepare_images([raw_scene, raw_wrist], cfg)
    prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"
    primary = processor(prompt, images[0]).to("cuda:0", dtype=torch.bfloat16)
    wrist = processor(prompt, images[1]).to("cuda:0", dtype=torch.bfloat16)
    pixel_values = torch.cat([primary["pixel_values"], wrist["pixel_values"]], dim=1)
    input_ids = primary["input_ids"]
    attention_mask = primary["attention_mask"]
    if not torch.all(input_ids[:, -1] == 29871):
        input_ids = torch.cat(
            [input_ids, torch.tensor([[29871]], device=input_ids.device, dtype=input_ids.dtype)],
            dim=1,
        )
    prompt_tokens = int(input_ids.shape[-1]) - 1
    labels = input_ids.clone()
    labels[:] = -100
    input_ids, attention_mask = model._prepare_input_for_action_prediction(
        input_ids, attention_mask
    )
    labels = model._prepare_labels_for_action_prediction(labels, input_ids)
    input_embeddings = model.get_input_embeddings()(input_ids)
    action_mask = model._process_action_masks(labels)
    language_embeddings = input_embeddings[~action_mask].reshape(
        input_embeddings.shape[0], -1, input_embeddings.shape[2]
    )
    projected = model._process_vision_features(pixel_values, language_embeddings, False)
    stats = model.norm_stats[cfg.unnorm_key]["proprio"]
    normalized_np = normalize_proprio(np.asarray(raw_state).copy(), stats)
    normalized = torch.as_tensor(
        normalized_np, device="cuda:0", dtype=projected.dtype
    ).reshape(1, -1)
    projected = model._process_proprio_features(projected, normalized, proprio_projector)
    if (
        tuple(projected.shape) != (1, 513, 4096)
        or tuple(input_embeddings.shape) != (1, 79, 4096)
        or tuple(action_mask.shape) != (1, 79)
        or prompt_tokens != 21
    ):
        raise B3ProtocolError("B3 pinned multimodal tensor layout changed")
    return PreparedQuery(
        input_embeddings=input_embeddings,
        action_mask=action_mask,
        projected_patches=projected,
        attention_mask=attention_mask,
        prompt_tokens=prompt_tokens,
        normalized_proprio=normalized,
        preprocessed_pixels=pixel_values,
    )


def runtime_positions(prepared: PreparedQuery, torch_module: Any) -> dict[str, tuple[int, ...]]:
    torch = torch_module
    action_input = tuple(
        int(index) for index in torch.nonzero(prepared.action_mask[0], as_tuple=False).flatten().tolist()
    )
    action = tuple(513 + index for index in action_input if index > 0)
    nonaction_input = tuple(
        index
        for index in range(1, int(prepared.action_mask.shape[1]) - 1)
        if index not in action_input
    )
    instruction = tuple(513 + index for index in nonaction_input)
    positions = {
        "scene": tuple(range(1, 257)),
        "wrist": tuple(range(257, 513)),
        "proprio": (513,),
        "instruction": instruction,
        "action": action,
    }
    complete = set().union(*(set(value) for value in positions.values()))
    if len(complete) != 590 or min(complete) != 1 or max(complete) != 590:
        raise B3ProtocolError("B3 runtime position map is incomplete or overlapping")
    return positions


def dense_or_cached_forward(
    *,
    torch_module: Any,
    np: Any,
    model: Any,
    action_head: Any,
    cfg: Any,
    prepared: PreparedQuery,
    past_key_values: Any,
    capture_layers: Sequence[int] = (),
) -> dict[str, Any]:
    torch = torch_module
    masked = prepared.input_embeddings * ~prepared.action_mask.unsqueeze(-1)
    multimodal, multimodal_mask = model._build_multimodal_attention(
        masked, prepared.projected_patches, prepared.attention_mask
    )
    if tuple(multimodal.shape) != (1, 592, 4096):
        raise B3ProtocolError("B3 full multimodal sequence shape changed")
    tap = SDPASidecarTap(torch, capture_layers) if capture_layers else None
    torch.cuda.synchronize()
    event_start = torch.cuda.Event(enable_timing=True)
    event_end = torch.cuda.Event(enable_timing=True)
    wall_start = time.perf_counter()
    event_start.record()
    if tap is None:
        output = model.language_model(
            input_ids=None,
            attention_mask=multimodal_mask,
            position_ids=None,
            past_key_values=past_key_values,
            inputs_embeds=multimodal,
            labels=None,
            use_cache=True,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )
    else:
        with tap:
            output = model.language_model(
                input_ids=None,
                attention_mask=multimodal_mask,
                position_ids=None,
                past_key_values=past_key_values,
                inputs_embeds=multimodal,
                labels=None,
                use_cache=True,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
    last_hidden = output.hidden_states[-1]
    hidden = last_hidden[:, -57:-1, :]
    normalized = action_head.predict_action(hidden).reshape(8, 7)
    normalized_cpu = normalized.float().cpu().detach().numpy()
    actions = model._unnormalize_actions(normalized_cpu, cfg.unnorm_key)
    event_end.record()
    torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - wall_start) * 1000
    cuda_ms = float(event_start.elapsed_time(event_end))
    return {
        "actions": actions,
        "normalized_actions": normalized_cpu,
        "cache": output.past_key_values,
        "tap": tap,
        "wall_ms": wall_ms,
        "cuda_ms": cuda_ms,
        "active_sequence_length": int(last_hidden.shape[1]),
    }


def patch_change_scores(
    current: Any,
    source: Any,
    *,
    torch_module: Any,
    epsilon: float,
    weights: Sequence[float],
) -> Any:
    torch = torch_module
    if tuple(current.shape) != tuple(source.shape) or current.ndim != 4:
        raise B3ProtocolError("B3 preprocessed image tensors changed shape")
    camera = current[:, :3].float()
    old = source[:, :3].float()
    camera = camera.unfold(2, 14, 14).unfold(3, 14, 14).permute(0, 2, 3, 1, 4, 5)
    old = old.unfold(2, 14, 14).unfold(3, 14, 14).permute(0, 2, 3, 1, 4, 5)
    camera = camera.reshape(256, -1)
    old = old.reshape(256, -1)
    span = torch.maximum(camera.max(), old.max()) - torch.minimum(camera.min(), old.min())
    denominator = torch.clamp(span * camera.shape[1], min=epsilon)
    l1 = torch.clamp((camera - old).abs().sum(dim=1) / denominator, 0, 1)
    current_norm = torch.linalg.vector_norm(camera, dim=1)
    old_norm = torch.linalg.vector_norm(old, dim=1)
    both_zero = (current_norm <= epsilon) & (old_norm <= epsilon)
    cosine = (camera * old).sum(dim=1) / torch.clamp(current_norm * old_norm, min=epsilon)
    cosine = torch.where(both_zero, torch.ones_like(cosine), cosine)
    cosine_distance = torch.clamp((1 - cosine) / 2, 0, 1)
    score = float(weights[0]) * l1 + float(weights[1]) * cosine_distance
    if not bool(torch.isfinite(score).all()):
        raise B3ProtocolError("B3 patch-change score became nonfinite")
    return score


def ordered_profile_positions(
    profile: Mapping[str, Any],
    *,
    scene_change: Any,
    wrist_change: Any,
    salience: Any,
    torch_module: Any,
    scene_offset: int = 0,
    wrist_offset: int = 0,
) -> tuple[Any, list[float]]:
    torch = torch_module
    scene_salience = salience[:256]
    wrist_salience = salience[256:]
    protected_scene = set(
        torch.topk(scene_salience, int(profile["protected_scene_tokens"])).indices.tolist()
    )
    protected_wrist = set(
        torch.topk(wrist_salience, int(profile["protected_wrist_tokens"])).indices.tolist()
    )
    scene_candidates = [
        index
        for index in torch.argsort(scene_change).tolist()
        if index not in protected_scene
        and float(scene_change[index]) <= float(profile["scene_change_limit"])
    ]
    wrist_candidates = [
        index
        for index in torch.argsort(wrist_change).tolist()
        if index not in protected_wrist
        and float(wrist_change[index]) <= float(profile["wrist_change_limit"])
    ]
    scene_budgets = [int(value) for value in profile["scene_budgets"]]
    wrist_budgets = [int(value) for value in profile["wrist_budgets"]]
    if len(scene_candidates) < scene_offset + scene_budgets[-1] or len(wrist_candidates) < wrist_offset + wrist_budgets[-1]:
        raise B3ProtocolError("B3 profile lacks enough suffix-eligible tokens")
    ordered: list[int] = []
    prior_scene = prior_wrist = 0
    proportions = []
    final_total = scene_budgets[-1] + wrist_budgets[-1]
    for scene_budget, wrist_budget in zip(scene_budgets, wrist_budgets):
        ordered.extend(
            1 + value
            for value in scene_candidates[
                scene_offset + prior_scene : scene_offset + scene_budget
            ]
        )
        ordered.extend(
            257 + value
            for value in wrist_candidates[
                wrist_offset + prior_wrist : wrist_offset + wrist_budget
            ]
        )
        prior_scene, prior_wrist = scene_budget, wrist_budget
        proportions.append((scene_budget + wrist_budget) / final_total)
    if len(ordered) != len(set(ordered)) or len(ordered) != final_total:
        raise B3ProtocolError("B3 ordered profile positions are not unique and complete")
    return torch.as_tensor(ordered, device=scene_change.device, dtype=torch.long), proportions


class SourceTracker:
    """Per-layer/per-visual-token source ownership for B3 contracts."""

    def __init__(self, layers: int = 32, tokens: int = 512, anchor_query: int = 0) -> None:
        self.sources = [[anchor_query] * tokens for _ in range(layers)]
        self.last_query = anchor_query

    def advance(
        self,
        query: int,
        *,
        ordered_positions: Sequence[int],
        profile: Mapping[str, Any],
        pruning_layers: Sequence[int],
    ) -> None:
        if query != self.last_query + 1:
            raise B3ProtocolError("B3 source tracker query order changed")
        budgets = [
            int(scene) + int(wrist)
            for scene, wrist in zip(profile["scene_budgets"], profile["wrist_budgets"])
        ]
        reusable: set[int] = set()
        for layer in range(len(self.sources)):
            if layer in pruning_layers:
                index = tuple(pruning_layers).index(layer)
                reusable = {int(position) - 1 for position in ordered_positions[: budgets[index]]}
            for token in range(len(self.sources[layer])):
                if token not in reusable:
                    self.sources[layer][token] = query
        self.last_query = query
        self.validate(query, profile)

    def validate(self, query: int, profile: Mapping[str, Any]) -> None:
        scene_age = int(profile["scene_max_age"])
        wrist_age = int(profile["wrist_max_age"])
        for layer in self.sources:
            if any(source < 0 or source > query for source in layer):
                raise B3ProtocolError("B3 source ownership is outside the query history")
            if any(query - source > scene_age for source in layer[:256]):
                raise B3ProtocolError("B3 scene source exceeded its maximum age")
            if any(query - source > wrist_age for source in layer[256:]):
                raise B3ProtocolError("B3 wrist source exceeded its maximum age")

    def digest(self) -> str:
        payload = ";".join(
            ",".join(str(source) for source in layer) for layer in self.sources
        ).encode()
        return hashlib.sha256(payload).hexdigest()
