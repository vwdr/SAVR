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
    )


def runtime_positions(prepared: PreparedQuery, torch_module: Any) -> dict[str, tuple[int, ...]]:
    torch = torch_module
    action_input = tuple(
        int(index)
        for index in torch.nonzero(prepared.action_mask[0], as_tuple=False).flatten().tolist()
    )
    action = tuple(513 + index for index in action_input if index > 0)
    nonaction_input = tuple(
        index
        for index in range(1, int(prepared.action_mask.shape[1]) - 1)
        if index not in action_input
    )
    instruction = tuple(513 + index for index in nonaction_input)
    positions = {
        "primary": tuple(range(1, 257)),
        "wrist": tuple(range(257, 513)),
        "proprio": (513,),
        "instruction": instruction,
        "action": action,
    }
    complete = set().union(*(set(value) for value in positions.values()))
    sequence_length = 513 + int(prepared.input_embeddings.shape[1])
    if (
        len(complete) != sequence_length - 2
        or min(complete) != 1
        or max(complete) != sequence_length - 2
        or len(action) != 56
    ):
        raise PairValidationError("P3 runtime sequence map is incomplete or overlapping")
    return positions


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
    last_hidden = output.hidden_states[-1]
    hidden = last_hidden[:, -57:-1, :]
    if tuple(hidden.shape) != (1, 56, 4096):
        raise PairValidationError("P3 action hidden-state layout changed")
    normalized = action_head.predict_action(hidden).reshape(8, 7)
    normalized_cpu = normalized.float().cpu().numpy()
    actions = model._unnormalize_actions(normalized_cpu, cfg.unnorm_key)
    event_end.record()
    torch.cuda.synchronize()
    return {
        "action_record": protected_action_record(actions, np),
        "cache": output.past_key_values,
        "tap": tap,
        "wall_ms": (time.perf_counter() - wall_start) * 1000,
        "cuda_ms": float(event_start.elapsed_time(event_end)),
        "active_sequence_length": int(last_hidden.shape[1]),
        "full_sequence_length": expected_length,
    }
