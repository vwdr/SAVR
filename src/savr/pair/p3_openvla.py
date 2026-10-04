"""Injected OpenVLA helpers for the outcome-blind PAIR P3 physical gate."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from savr.brace.b3_openvla import SDPASidecarTap
from savr.pair.p3 import P3Profile
from savr.pair.types import Camera, ONSET_LAYERS, PairValidationError


@dataclass
class PreparedQuery:
    input_embeddings: Any
    action_mask: Any
    projected_patches: Any
    attention_mask: Any
    normalized_proprio: Any
    preprocessed_pixels: Any
    instruction_embedding: Any = None
    instruction_token_indices: tuple[int, ...] = ()


def protected_action_record(value: Any, np: Any) -> dict[str, Any]:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (8, 7) or not np.isfinite(array).all():
        raise PairValidationError("P3 action output is not finite 8x7")
    return {
        "shape": [8, 7],
        "finite": True,
        "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
    }


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
    if torch.is_grad_enabled():
        raise PairValidationError("query preparation must run with gradients disabled")
    images = prepare_images([raw_scene, raw_wrist], cfg)
    prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"
    primary = processor(prompt, images[0]).to("cuda:0", dtype=torch.bfloat16)
    wrist = processor(prompt, images[1]).to("cuda:0", dtype=torch.bfloat16)
    pixel_values = torch.cat([primary["pixel_values"], wrist["pixel_values"]], dim=1)
    input_ids = primary["input_ids"]
    original_input_ids = input_ids.clone()
    attention_mask = primary["attention_mask"]
    from savr.cac.c1 import instruction_token_indices

    instruction_indices = instruction_token_indices(
        processor.tokenizer, prompt, instruction, original_input_ids
    )
    if not torch.all(input_ids[:, -1] == 29871):
        input_ids = torch.cat(
            [input_ids, torch.tensor([[29871]], device=input_ids.device, dtype=input_ids.dtype)],
            dim=1,
        )
    labels = input_ids.clone()
    labels[:] = -100
    input_ids, attention_mask = model._prepare_input_for_action_prediction(
        input_ids, attention_mask
    )
    labels = model._prepare_labels_for_action_prediction(labels, input_ids)
    input_embeddings = model.get_input_embeddings()(input_ids)
    instruction_embedding = input_embeddings[0, list(instruction_indices)].mean(dim=0)
    action_mask = model._process_action_masks(labels)
    language_embeddings = input_embeddings[~action_mask].reshape(
        input_embeddings.shape[0], -1, input_embeddings.shape[2]
    )
    projected = model._process_vision_features(pixel_values, language_embeddings, False)
    stats = model.norm_stats[cfg.unnorm_key]["proprio"]
    normalized_np = normalize_proprio(np.asarray(raw_state).copy(), stats)
    normalized = torch.as_tensor(normalized_np, device="cuda:0", dtype=projected.dtype).reshape(
        1, -1
    )
    projected = model._process_proprio_features(projected, normalized, proprio_projector)
    if tuple(projected.shape) != (1, 513, 4096):
        raise PairValidationError("P3 projected visual/proprio layout changed")
    if input_embeddings.ndim != 3 or input_embeddings.shape[:1] != (1,):
        raise PairValidationError("P3 language/action embedding layout changed")
    if tuple(action_mask.shape) != tuple(input_embeddings.shape[:2]):
        raise PairValidationError("P3 action mask is not aligned to input embeddings")
    if int(action_mask.sum()) != 56:
        raise PairValidationError("P3 action-token count changed")
    return PreparedQuery(
        input_embeddings,
        action_mask,
        projected,
        attention_mask,
        normalized,
        pixel_values,
        instruction_embedding,
        instruction_indices,
    )


def runtime_positions(prepared: PreparedQuery, torch_module: Any) -> dict[str, tuple[int, ...]]:
    del torch_module
    from savr.openvla.official_semantics import derive_semantic_runtime_positions

    return derive_semantic_runtime_positions(
        action_mask=prepared.action_mask,
        projected_tokens=int(prepared.projected_patches.shape[1]),
        instruction_token_indices=prepared.instruction_token_indices,
    )


def configure_dense(model: Any) -> None:
    model.language_model.config.proportion_attn_var = None
    model.language_model.config.reusable_patches = None


def configure_profile(
    model: Any, ordered_positions: Any, proportions: Sequence[float], torch_module: Any
) -> None:
    schedule = torch_module.zeros(32, dtype=torch_module.float32, device="cuda:0")
    for layer, value in zip(ONSET_LAYERS, proportions, strict=True):
        schedule[layer] = float(value)
    model.language_model.config.reusable_patches = ordered_positions
    model.language_model.config.proportion_attn_var = schedule


def _tile_patch_ids(tile: int) -> tuple[int, ...]:
    tile_row, tile_column = divmod(tile, 4)
    return tuple(
        (tile_row * 4 + row) * 16 + tile_column * 4 + column
        for row in range(4)
        for column in range(4)
    )


class PhysicalSourceTracker:
    """Exact layer/camera/tile sources plus bounded raw/projected records."""

    def __init__(self, anchor_query: int, prepared: PreparedQuery) -> None:
        self.last_query = anchor_query
        self.sources = {
            (layer, camera, tile): anchor_query
            for layer in range(32)
            for camera in Camera
            for tile in range(16)
        }
        self.records = {anchor_query: prepared}

    def _live_sources(self) -> set[int]:
        return set(self.sources.values())

    def add_record(self, query: int, prepared: PreparedQuery) -> None:
        self.records[query] = prepared
        while len(self.records) > 6:
            evictable = next(
                (source for source in sorted(self.records) if source not in self._live_sources()),
                None,
            )
            if evictable is None:
                raise PairValidationError("P3 source-record eviction would remove a live source")
            del self.records[evictable]

    def source(self, layer: int, camera: Camera, tile: int) -> PreparedQuery:
        query = self.sources[(layer, camera, tile)]
        try:
            return self.records[query]
        except KeyError as error:
            raise PairValidationError("P3 physical source record is unavailable") from error

    def source_query(self, layer: int, camera: Camera, tile: int) -> int:
        return int(self.sources[(layer, camera, tile)])

    def advance(self, query: int, groups: Mapping[tuple[Camera, int], int]) -> None:
        if query != self.last_query + 1:
            raise PairValidationError("P3 physical source queries are not contiguous")
        for layer in range(32):
            for camera in Camera:
                for tile in range(16):
                    onset = groups.get((camera, tile))
                    if onset is None or layer < onset:
                        self.sources[(layer, camera, tile)] = query
        self.last_query = query
        for source in self.sources.values():
            if not 0 <= query - source <= 4:
                raise PairValidationError("P3 physical source age exceeded the frozen cap")

    def digest(self) -> str:
        payload = ";".join(
            f"{layer}:{camera.value}:{tile}:{self.sources[(layer, camera, tile)]}"
            for layer in range(32)
            for camera in Camera
            for tile in range(16)
        ).encode()
        return hashlib.sha256(payload).hexdigest()

    def mixture_count(self) -> int:
        return len(self._live_sources())

    def clone(self) -> "PhysicalSourceTracker":
        """Clone mutable tracking state while sharing immutable prepared tensors."""

        duplicate = object.__new__(PhysicalSourceTracker)
        duplicate.last_query = self.last_query
        duplicate.sources = dict(self.sources)
        duplicate.records = dict(self.records)
        return duplicate


def _raw_tile(prepared: PreparedQuery, camera: Camera, tile: int) -> Any:
    offset = 0 if camera is Camera.PRIMARY else 3
    tile_row, tile_column = divmod(tile, 4)
    row = tile_row * 56
    column = tile_column * 56
    return prepared.preprocessed_pixels[
        :, offset : offset + 3, row : row + 56, column : column + 56
    ]


def _projected_tile(prepared: PreparedQuery, camera: Camera, tile: int, torch: Any) -> Any:
    offset = 0 if camera is Camera.PRIMARY else 256
    indices = (
        torch.as_tensor(
            _tile_patch_ids(tile), device=prepared.projected_patches.device, dtype=torch.long
        )
        + offset
    )
    return prepared.projected_patches[0].index_select(0, indices)


def _tile_index_matrix(torch: Any, device: Any) -> Any:
    grid = torch.arange(256, device=device, dtype=torch.long).reshape(4, 4, 4, 4)
    return grid.permute(0, 2, 1, 3).reshape(16, 16)


def _raw_tiles(prepared: PreparedQuery, camera: Camera) -> Any:
    offset = 0 if camera is Camera.PRIMARY else 3
    pixels = prepared.preprocessed_pixels[:, offset : offset + 3]
    return (
        pixels.reshape(1, 3, 4, 56, 4, 56)
        .permute(0, 2, 4, 1, 3, 5)[0]
        .reshape(16, 3, 56, 56)
    )


def _projected_tiles(prepared: PreparedQuery, camera: Camera, tile_indices: Any) -> Any:
    offset = 0 if camera is Camera.PRIMARY else 256
    positions = (tile_indices + offset).reshape(-1)
    return prepared.projected_patches[0].index_select(0, positions).reshape(16, 16, -1)


def _batched_change_score(current: Any, source: Any, torch: Any) -> Any:
    left = current.float().reshape(*current.shape[:2], -1)
    right = source.float().reshape(*source.shape[:2], -1)
    difference = (left - right).abs()
    span = torch.maximum(left.max(dim=-1).values, right.max(dim=-1).values) - torch.minimum(
        left.min(dim=-1).values, right.min(dim=-1).values
    )
    l1 = difference.mean(dim=-1) / torch.clamp(span, min=1e-8)
    left_norm = torch.linalg.vector_norm(left, dim=-1)
    right_norm = torch.linalg.vector_norm(right, dim=-1)
    norm_product = left_norm * right_norm
    cosine = torch.where(
        norm_product <= 1e-8,
        torch.where((left_norm <= 1e-8) & (right_norm <= 1e-8), 0.0, 1.0),
        torch.clamp(
            (1 - (left * right).sum(dim=-1) / torch.clamp(norm_product, min=1e-8)) / 2,
            0,
            1,
        ),
    )
    return 0.5 * torch.clamp(l1, 0, 1) + 0.5 * cosine


def _vectorized_camera_scores(
    current: PreparedQuery,
    tracker: PhysicalSourceTracker,
    camera: Camera,
    tile_indices: Any,
    torch: Any,
) -> Any:
    source_queries = [
        [tracker.source_query(layer, camera, tile) for tile in range(16)]
        for layer in ONSET_LAYERS
    ]
    unique_sources = sorted({query for row in source_queries for query in row})
    source_slot = {query: index for index, query in enumerate(unique_sources)}
    slots = torch.as_tensor(
        [[source_slot[query] for query in row] for row in source_queries],
        device=current.projected_patches.device,
        dtype=torch.long,
    )
    tiles = torch.arange(16, device=slots.device, dtype=torch.long).expand(4, 16)
    raw_bank = torch.stack([_raw_tiles(tracker.records[query], camera) for query in unique_sources])
    projected_bank = torch.stack(
        [
            _projected_tiles(tracker.records[query], camera, tile_indices)
            for query in unique_sources
        ]
    )
    source_raw = raw_bank[slots, tiles]
    source_projected = projected_bank[slots, tiles]
    current_raw = _raw_tiles(current, camera).unsqueeze(0).expand_as(source_raw)
    current_projected = (
        _projected_tiles(current, camera, tile_indices)
        .unsqueeze(0)
        .expand_as(source_projected)
    )
    return _batched_change_score(current_raw, source_raw, torch) + _batched_change_score(
        current_projected, source_projected, torch
    )


def _change_score(current: Any, source: Any, torch: Any) -> Any:
    left = current.float().reshape(-1)
    right = source.float().reshape(-1)
    difference = (left - right).abs()
    span = torch.maximum(left.max(), right.max()) - torch.minimum(left.min(), right.min())
    l1 = difference.mean() / torch.clamp(span, min=1e-8)
    left_norm = torch.linalg.vector_norm(left)
    right_norm = torch.linalg.vector_norm(right)
    norm_product = left_norm * right_norm
    safe_norm_product = torch.clamp(norm_product, min=1e-8)
    cosine = torch.where(
        norm_product <= 1e-8,
        torch.where((left_norm <= 1e-8) & (right_norm <= 1e-8), 0.0, 1.0),
        torch.clamp((1 - (left * right).sum() / safe_norm_product) / 2, 0, 1),
    )
    return 0.5 * torch.clamp(l1, 0, 1) + 0.5 * cosine


def ordered_tile_profile(
    *,
    current: PreparedQuery,
    tracker: PhysicalSourceTracker,
    profile: P3Profile,
    previous_salience: Any,
    torch_module: Any,
) -> tuple[Any, tuple[float, ...], dict[tuple[Camera, int], int], dict[str, Any]]:
    """Build a tile-aligned nested profile against every group's actual source."""

    torch = torch_module
    salience = previous_salience.reshape(2, 256)
    selected: dict[Camera, list[int]] = {camera: [] for camera in Camera}
    protected: dict[Camera, set[int]] = {}
    protected_counts = {
        Camera.PRIMARY: profile.protected_primary_tiles,
        Camera.WRIST: profile.protected_wrist_tiles,
    }
    for camera_index, camera in enumerate(Camera):
        tile_salience = torch.stack(
            [salience[camera_index, list(_tile_patch_ids(tile))].mean() for tile in range(16)]
        )
        count = protected_counts[camera]
        protected[camera] = (
            set(torch.topk(tile_salience, count).indices.tolist()) if count else set()
        )

    groups: dict[tuple[Camera, int], int] = {}
    ordered_positions: list[int] = []
    total_by_layer = []
    for layer_index, layer in enumerate(ONSET_LAYERS):
        for camera, budgets in (
            (Camera.PRIMARY, profile.primary_budgets),
            (Camera.WRIST, profile.wrist_budgets),
        ):
            target = budgets[layer_index] // 16
            candidate_tiles = []
            candidate_scores = []
            for tile in range(16):
                if tile in protected[camera] or tile in selected[camera]:
                    continue
                source = tracker.source(layer, camera, tile)
                raw = _change_score(
                    _raw_tile(current, camera, tile), _raw_tile(source, camera, tile), torch
                )
                projected = _change_score(
                    _projected_tile(current, camera, tile, torch),
                    _projected_tile(source, camera, tile, torch),
                    torch,
                )
                candidate_tiles.append(tile)
                candidate_scores.append(raw + projected)
            needed = target - len(selected[camera])
            if needed < 0 or len(candidate_tiles) < needed:
                raise PairValidationError("P3 profile lacks enough eligible tile-aligned groups")
            order = (
                torch.argsort(torch.stack(candidate_scores)).tolist() if candidate_scores else []
            )
            additions = [candidate_tiles[index] for index in order[:needed]]
            selected[camera].extend(additions)
            offset = 1 if camera is Camera.PRIMARY else 257
            for tile in additions:
                groups[(camera, tile)] = layer
                ordered_positions.extend(offset + patch for patch in _tile_patch_ids(tile))
        total_by_layer.append(
            profile.primary_budgets[layer_index] + profile.wrist_budgets[layer_index]
        )
    final_total = total_by_layer[-1]
    if len(ordered_positions) != final_total or len(set(ordered_positions)) != final_total:
        raise PairValidationError("P3 tile profile positions are not unique and complete")
    proportions = tuple(value / final_total for value in total_by_layer)
    metadata = {
        "protected_primary_tiles": sorted(protected[Camera.PRIMARY]),
        "protected_wrist_tiles": sorted(protected[Camera.WRIST]),
        "reused_primary_tiles": len(selected[Camera.PRIMARY]),
        "reused_wrist_tiles": len(selected[Camera.WRIST]),
    }
    return (
        torch.as_tensor(
            ordered_positions, device=current.projected_patches.device, dtype=torch.long
        ),
        proportions,
        groups,
        metadata,
    )


def ordered_tile_profile_vectorized(
    *,
    current: PreparedQuery,
    tracker: PhysicalSourceTracker,
    profile: P3Profile,
    previous_salience: Any,
    torch_module: Any,
) -> tuple[Any, tuple[float, ...], dict[tuple[Camera, int], int], dict[str, Any]]:
    """Vectorized implementation required to be exactly equivalent to the P3 reference."""

    torch = torch_module
    tile_indices = _tile_index_matrix(torch, current.projected_patches.device)
    salience = previous_salience.reshape(2, 256)
    selected = {
        camera: torch.zeros(16, device=salience.device, dtype=torch.bool) for camera in Camera
    }
    protected = {}
    scores = {}
    protected_counts = {
        Camera.PRIMARY: profile.protected_primary_tiles,
        Camera.WRIST: profile.protected_wrist_tiles,
    }
    for camera_index, camera in enumerate(Camera):
        tile_salience = salience[camera_index].index_select(0, tile_indices.reshape(-1)).reshape(
            16, 16
        ).mean(dim=-1)
        protected[camera] = torch.zeros(16, device=salience.device, dtype=torch.bool)
        count = protected_counts[camera]
        if count:
            protected[camera][torch.topk(tile_salience, count).indices] = True
        scores[camera] = _vectorized_camera_scores(
            current, tracker, camera, tile_indices, torch
        )

    groups: dict[tuple[Camera, int], int] = {}
    ordered_positions: list[int] = []
    total_by_layer = []
    for layer_index, layer in enumerate(ONSET_LAYERS):
        for camera, budgets in (
            (Camera.PRIMARY, profile.primary_budgets),
            (Camera.WRIST, profile.wrist_budgets),
        ):
            target = budgets[layer_index] // 16
            selected_count = int(selected[camera].sum().item())
            needed = target - selected_count
            eligible = ~(selected[camera] | protected[camera])
            if needed < 0 or int(eligible.sum().item()) < needed:
                raise PairValidationError("P3 profile lacks enough eligible tile-aligned groups")
            ranked = torch.argsort(scores[camera][layer_index].masked_fill(~eligible, float("inf")))
            additions = [int(tile) for tile in ranked[:needed].tolist()]
            if additions:
                selected[camera][torch.as_tensor(additions, device=salience.device)] = True
            offset = 1 if camera is Camera.PRIMARY else 257
            for tile in additions:
                groups[(camera, tile)] = layer
                ordered_positions.extend(offset + patch for patch in _tile_patch_ids(tile))
        total_by_layer.append(
            profile.primary_budgets[layer_index] + profile.wrist_budgets[layer_index]
        )
    final_total = total_by_layer[-1]
    if len(ordered_positions) != final_total or len(set(ordered_positions)) != final_total:
        raise PairValidationError("P3 vectorized tile positions are not unique and complete")
    proportions = tuple(value / final_total for value in total_by_layer)
    metadata = {
        "protected_primary_tiles": torch.nonzero(
            protected[Camera.PRIMARY], as_tuple=False
        ).flatten().tolist(),
        "protected_wrist_tiles": torch.nonzero(
            protected[Camera.WRIST], as_tuple=False
        ).flatten().tolist(),
        "reused_primary_tiles": int(selected[Camera.PRIMARY].sum().item()),
        "reused_wrist_tiles": int(selected[Camera.WRIST].sum().item()),
    }
    return (
        torch.as_tensor(
            ordered_positions, device=current.projected_patches.device, dtype=torch.long
        ),
        proportions,
        groups,
        metadata,
    )


def cache_sample_digest(cache: Any, torch_module: Any) -> str:
    torch = torch_module
    digest = hashlib.sha256()
    for layer, (key, value) in enumerate(zip(cache.key_cache, cache.value_cache, strict=True)):
        positions = sorted(
            {0, 1, int(key.shape[-2]) // 2, int(key.shape[-2]) - 2, int(key.shape[-2]) - 1}
        )
        index = torch.as_tensor(positions, device=key.device, dtype=torch.long)
        for tensor in (key, value):
            sample = tensor.index_select(-2, index).float().cpu().numpy()
            digest.update(layer.to_bytes(2, "little"))
            digest.update(sample.tobytes())
    return digest.hexdigest()


def capture_reused_cache(
    cache: Any, ordered_positions: Any, profile: P3Profile, _torch_module: Any
) -> dict[tuple[int, str], Any]:
    totals = [
        primary + wrist
        for primary, wrist in zip(profile.primary_budgets, profile.wrist_budgets, strict=True)
    ]
    snapshots = {}
    active = 0
    for layer in range(32):
        if layer in ONSET_LAYERS:
            active = totals[ONSET_LAYERS.index(layer)]
        if not active:
            continue
        positions = ordered_positions[:active]
        snapshots[(layer, "key")] = (
            positions.clone(),
            cache.key_cache[layer].index_select(-2, positions).clone(),
        )
        snapshots[(layer, "value")] = (
            positions.clone(),
            cache.value_cache[layer].index_select(-2, positions).clone(),
        )
    return snapshots


def verify_reused_cache(cache: Any, snapshots: Mapping[tuple[int, str], Any]) -> None:
    for (layer, kind), (positions, expected) in snapshots.items():
        tensor = cache.key_cache[layer] if kind == "key" else cache.value_cache[layer]
        if expected.shape[-2] == 0:
            continue
        observed = tensor.index_select(-2, positions)
        if not bool(expected.isfinite().all()) or not bool(observed.isfinite().all()):
            raise PairValidationError("P3 cache provenance contains nonfinite tensors")
        if not bool((expected == observed).all()):
            raise PairValidationError("P3 reused K/V tensor changed at an exact cache position")


def forward_query(
    *,
    torch_module: Any,
    np: Any,
    model: Any,
    action_head: Any,
    cfg: Any,
    prepared: PreparedQuery,
    past_key_values: Any,
    capture_layers: Sequence[int] = (),
    capture_cac: bool = False,
    return_actions: bool = False,
) -> dict[str, Any]:
    torch = torch_module
    masked = prepared.input_embeddings * ~prepared.action_mask.unsqueeze(-1)
    multimodal, multimodal_mask = model._build_multimodal_attention(
        masked, prepared.projected_patches, prepared.attention_mask
    )
    expected_length = 513 + int(prepared.input_embeddings.shape[1])
    if tuple(multimodal.shape) != (1, expected_length, 4096):
        raise PairValidationError("P3 full multimodal sequence shape changed")
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

    last_hidden = output.hidden_states[-1]
    layout = derive_official_layout(
        action_mask=prepared.action_mask,
        projected_tokens=int(prepared.projected_patches.shape[1]),
        instruction_token_indices=prepared.instruction_token_indices,
    )
    active_positions = cache_fork_active_positions(output)
    hidden = select_official_action_hidden(last_hidden, layout, active_positions)
    if tuple(hidden.shape) != (1, 56, 4096):
        raise PairValidationError("P3 action hidden-state layout changed")
    cac_values = {}
    if capture_cac:
        from savr.cac.c1 import action_head_penultimate

        z_cache, base_action = action_head_penultimate(action_head, hidden)
        reference = action_head.predict_action(hidden).reshape(1, 8, 7)
        reproduction_max_abs = float((base_action - reference).abs().max().item())
        if reproduction_max_abs > 1e-6:
            raise PairValidationError("CAC action-head reproduction exceeded 1e-6")
        normalized = base_action[0]
        cac_values = {
            "action_hidden": hidden,
            "z_cache": z_cache,
            "base_action": base_action,
            "head_reproduction_max_abs": reproduction_max_abs,
        }
    else:
        normalized = action_head.predict_action(hidden).reshape(8, 7)
    normalized_cpu = normalized.float().cpu().numpy()
    actions = model._unnormalize_actions(normalized_cpu, cfg.unnorm_key)
    event_end.record()
    torch.cuda.synchronize()
    result = {
        "action_record": protected_action_record(actions, np),
        "cache": output.past_key_values,
        "tap": tap,
        "wall_ms": (time.perf_counter() - wall_start) * 1000,
        "cuda_ms": float(event_start.elapsed_time(event_end)),
        "active_sequence_length": int(last_hidden.shape[1]),
        "full_sequence_length": expected_length,
        **cac_values,
    }
    if return_actions:
        result["actions"] = actions
    return result
