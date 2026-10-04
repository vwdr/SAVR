"""OpenVLA helpers for the outcome-sealed PAIR P4 expert-regret pilot."""

from __future__ import annotations

import hashlib
import time
from typing import Any, Mapping, Sequence

import numpy as np

from savr.brace.b3_openvla import SDPASidecarTap
from savr.pair.features import RouterFeatures, build_group_feature
from savr.pair.p3_openvla import (
    PhysicalSourceTracker,
    PreparedQuery,
    _projected_tile,
    _raw_tile,
    _tile_patch_ids,
)
from savr.pair.p4 import instruction_projection
from savr.pair.types import AtomicGroup, Camera, ONSET_LAYERS, PairValidationError


def action_digest(value: Any) -> str:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (8, 7) or not np.isfinite(array).all():
        raise PairValidationError("P4 action is not finite 8x7")
    return hashlib.sha256(array.tobytes()).hexdigest()


def arbitrary_group_profile(
    groups: Sequence[AtomicGroup], torch_module: Any, *, device: Any
) -> tuple[Any, tuple[float, ...], dict[tuple[Camera, int], int]]:
    """Convert one source-resolved group set into the model's nested schedule."""

    ordered_groups = tuple(sorted(groups, key=lambda g: (g.onset_layer, g.camera.value, g.tile)))
    identities = {(group.camera, group.tile) for group in ordered_groups}
    if len(identities) != len(ordered_groups):
        raise PairValidationError("P4 arbitrary mask assigns one tile more than once")
    for group in ordered_groups:
        group.validate()
    positions = []
    totals = []
    for layer in ONSET_LAYERS:
        for group in ordered_groups:
            if group.onset_layer == layer:
                offset = 1 if group.camera is Camera.PRIMARY else 257
                positions.extend(offset + patch for patch in _tile_patch_ids(group.tile))
        totals.append(len(positions))
    if not positions:
        proportions = (0.0, 0.0, 0.0, 0.0)
    else:
        proportions = tuple(total / len(positions) for total in totals)
    return (
        torch_module.as_tensor(positions, device=device, dtype=torch_module.long),
        proportions,
        {(group.camera, group.tile): group.onset_layer for group in ordered_groups},
    )


def forward_normalized_query(
    *,
    torch_module: Any,
    model: Any,
    action_head: Any,
    prepared: PreparedQuery,
    past_key_values: Any,
    capture_layers: Sequence[int] = (),
) -> dict[str, Any]:
    """Run one query and retain normalized actions only in process memory."""

    torch = torch_module
    masked = prepared.input_embeddings * ~prepared.action_mask.unsqueeze(-1)
    multimodal, multimodal_mask = model._build_multimodal_attention(
        masked, prepared.projected_patches, prepared.attention_mask
    )
    expected_length = 513 + int(prepared.input_embeddings.shape[1])
    if tuple(multimodal.shape) != (1, expected_length, 4096):
        raise PairValidationError("P4 full multimodal sequence shape changed")
    tap = SDPASidecarTap(torch, capture_layers) if capture_layers else None
    torch.cuda.synchronize()
    event_start = torch.cuda.Event(enable_timing=True)
    event_end = torch.cuda.Event(enable_timing=True)
    wall_start = time.perf_counter()
    event_start.record()
    arguments = {
        "input_ids": None,
        "attention_mask": multimodal_mask,
        "position_ids": None,
        "past_key_values": past_key_values,
        "inputs_embeds": multimodal,
        "labels": None,
        "use_cache": True,
        "output_attentions": False,
        "output_hidden_states": True,
        "return_dict": True,
    }
    if tap is None:
        output = model.language_model(**arguments)
    else:
        with tap:
            output = model.language_model(**arguments)
    from savr.openvla.official_semantics import (
        cache_fork_active_positions,
        derive_official_layout,
        select_official_action_hidden,
    )

    layout = derive_official_layout(
        action_mask=prepared.action_mask,
        projected_tokens=int(prepared.projected_patches.shape[1]),
        instruction_token_indices=prepared.instruction_token_indices,
    )
    active_positions = cache_fork_active_positions(output)
    hidden = select_official_action_hidden(
        output.hidden_states[-1], layout, active_positions
    )
    if tuple(hidden.shape) != (1, 56, 4096):
        raise PairValidationError("P4 action hidden-state layout changed")
    normalized = action_head.predict_action(hidden).reshape(8, 7).float().cpu().numpy()
    event_end.record()
    torch.cuda.synchronize()
    if not np.isfinite(normalized).all():
        raise PairValidationError("P4 normalized action is nonfinite")
    return {
        "normalized_action": normalized,
        "action_sha256": action_digest(normalized),
        "cache": output.past_key_values,
        "tap": tap,
        "wall_ms": (time.perf_counter() - wall_start) * 1000,
        "cuda_ms": float(event_start.elapsed_time(event_end)),
        "active_sequence_length": int(output.hidden_states[-1].shape[1]),
        "full_sequence_length": expected_length,
    }


def anchor_instruction_projection(prepared: PreparedQuery, *, seed: int) -> tuple[float, ...]:
    keep = ~prepared.action_mask[0]
    values = prepared.input_embeddings[0, keep].float().mean(dim=0).cpu().numpy()
    return instruction_projection(values, seed=seed)


def group_salience(anchor_salience: Any, group: AtomicGroup) -> float:
    camera_index = 0 if group.camera is Camera.PRIMARY else 1
    values = anchor_salience.reshape(2, 256)[camera_index]
    return float(values[list(_tile_patch_ids(group.tile))].float().mean().item())


def build_router_features(
    *,
    current: PreparedQuery,
    tracker: PhysicalSourceTracker,
    groups: Sequence[AtomicGroup],
    profile_id: str,
    horizon: int,
    query_ordinal: int,
    remaining: int,
    action_history: Mapping[int, Any],
    anchor_salience: Any,
    gripper_transition: bool,
    global_projection: Sequence[float],
    torch_module: Any,
) -> RouterFeatures:
    """Build features strictly before the current branch action is produced."""

    if query_ordinal - 1 not in action_history:
        raise PairValidationError("P4 previous branch action is unavailable")
    rows = []
    for group in groups:
        source_query = tracker.source_query(group.onset_layer, group.camera, group.tile)
        if source_query not in action_history:
            raise PairValidationError("P4 source action history is unavailable")
        source = tracker.source(group.onset_layer, group.camera, group.tile)
        rows.append(
            build_group_feature(
                group=group,
                profile_id=profile_id,
                horizon=horizon,
                source_age=query_ordinal - source_query,
                current_raw_tile=_raw_tile(current, group.camera, group.tile).float().cpu().numpy(),
                source_raw_tile=_raw_tile(source, group.camera, group.tile).float().cpu().numpy(),
                current_projected_tile=_projected_tile(
                    current, group.camera, group.tile, torch_module
                ).float().cpu().numpy(),
                source_projected_tile=_projected_tile(
                    source, group.camera, group.tile, torch_module
                ).float().cpu().numpy(),
                current_proprio=current.normalized_proprio.float().cpu().numpy().reshape(-1),
                source_proprio=source.normalized_proprio.float().cpu().numpy().reshape(-1),
                current_previous_action=np.asarray(action_history[query_ordinal - 1]).reshape(-1),
                source_previous_action=np.asarray(action_history[source_query]).reshape(-1),
                previous_salience=group_salience(anchor_salience, group),
                gripper_transition=gripper_transition,
                remaining=remaining,
                source_mixture_count=tracker.mixture_count(),
            )
        )
    result = RouterFeatures(tuple(rows), tuple(float(value) for value in global_projection))
    result.validate(supported_profiles=(profile_id,))
    return result


def serialize_router_features(features: RouterFeatures) -> dict[str, Any]:
    return {
        "global_continuous": list(features.global_continuous),
        "groups": [
            {
                "camera": item.group.camera.value,
                "tile": item.group.tile,
                "onset_layer": item.group.onset_layer,
                "profile_id": item.profile_id,
                "horizon": item.horizon,
                "continuous": list(item.continuous),
            }
            for item in features.groups
        ],
    }
