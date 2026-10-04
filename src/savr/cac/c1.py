"""CAC C1 tensor extraction, adapter, isolation, and bounded-record utilities."""

from __future__ import annotations

import hashlib
import json
import os
import queue
import struct
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ONSET_LAYERS = (2, 6, 9, 11)
NUMERIC_PAYLOAD_BYTES = 1_378_328
ALIGNED_RECORD_BYTES = 1_380_352
SIDECAR_RECORD_BYTES = 192
PROVENANCE_COLUMNS = (
    "reused",
    "fresh",
    "source_age",
    "onset_layer_index",
    "onset_layer_absolute",
    "camera_index",
    "tile_index",
    "tile_row",
    "tile_column",
    "reset_phase",
    "source_query_mod_256",
    "reserved_zero",
)


class CACValidationError(RuntimeError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def semantic_sha256(value: Mapping[str, Any]) -> str:
    body = dict(value)
    body.pop("semantic_sha256", None)
    return hashlib.sha256(canonical_bytes(body)).hexdigest()


def tensor_sha256(tensor: Any) -> str:
    array = tensor.detach().contiguous().cpu().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def action_head_penultimate(action_head: Any, hidden: Any) -> tuple[Any, Any]:
    """Reproduce L1RegressionActionHead while exposing its exact 8x4096 penultimate."""

    if tuple(hidden.shape[1:]) != (56, 4096):
        raise CACValidationError("CAC action hidden-state layout changed")
    model = action_head.model
    value = hidden.reshape(hidden.shape[0], 8, 7 * 4096)
    value = model.layer_norm1(value)
    value = model.fc1(value)
    value = model.relu(value)
    for block in model.mlp_resnet_blocks:
        value = block(value)
    z_cache = model.layer_norm2(value)
    base_action = model.fc2(z_cache)
    if tuple(z_cache.shape) != (1, 8, 4096) or tuple(base_action.shape) != (1, 8, 7):
        raise CACValidationError("CAC action-head penultimate/output layout changed")
    return z_cache, base_action


def instruction_token_indices(tokenizer: Any, prompt: str, instruction: str, input_ids: Any) -> tuple[int, ...]:
    """Locate exactly the instruction substring using fast-tokenizer character offsets."""

    if not getattr(tokenizer, "is_fast", False):
        raise CACValidationError("CAC requires an authenticated fast tokenizer for instruction offsets")
    needle = instruction.lower()
    start = prompt.index(needle)
    end = start + len(needle)
    encoded = tokenizer(prompt, return_offsets_mapping=True, return_tensors="pt")
    expected = encoded["input_ids"][0].tolist()
    observed = input_ids[0].detach().cpu().tolist()
    if expected != observed:
        raise CACValidationError("instruction-offset tokenization differs from processor input IDs")
    offsets = encoded["offset_mapping"][0].tolist()
    indices = tuple(
        index
        for index, (left, right) in enumerate(offsets)
        if right > left and left < end and right > start
    )
    if not indices or min(indices) <= 0:
        raise CACValidationError("instruction token span is empty or includes a special token")
    covered_left = min(offsets[index][0] for index in indices)
    covered_right = max(offsets[index][1] for index in indices)
    if covered_left > start or covered_right < end:
        raise CACValidationError("instruction token span does not cover the full instruction")
    return indices


@dataclass(frozen=True)
class CACFeatureTensors:
    current_tiles: Any
    source_deltas: Any
    z_cache: Any
    base_action: Any
    proprio: Any
    instruction_embedding: Any
    provenance: Any
    source_query_ids: tuple[int, ...]

    def validate(self, torch_module: Any) -> None:
        torch = torch_module
        expected = {
            "current_tiles": ((32, 4096), torch.float16),
            "source_deltas": ((128, 4096), torch.float16),
            "z_cache": ((8, 4096), torch.float16),
            "base_action": ((8, 7), torch.float32),
            "proprio": ((8,), torch.float32),
            "instruction_embedding": ((4096,), torch.float16),
            "provenance": ((128, 12), torch.uint8),
        }
        for name, (shape, dtype) in expected.items():
            value = getattr(self, name)
            if tuple(value.shape) != shape or value.dtype != dtype or not bool(value.isfinite().all()):
                raise CACValidationError(f"CAC feature tensor changed: {name}")
        devices = {
            str(getattr(self, name).device)
            for name in expected
            if name != "provenance"
        }
        if len(devices) != 1 or not next(iter(devices)).startswith("cuda"):
            raise CACValidationError("CAC floating feature tensors are not colocated on one GPU")
        if str(self.provenance.device) not in devices:
            raise CACValidationError("CAC provenance is not colocated with features")
        if len(self.source_query_ids) != 128:
            raise CACValidationError("CAC source-query sidecar length changed")


def _tile_means(prepared: Any, camera_index: int, torch: Any) -> Any:
    offset = 0 if camera_index == 0 else 256
    grid = torch.arange(256, device=prepared.projected_patches.device, dtype=torch.long).reshape(4, 4, 4, 4)
    tile_indices = grid.permute(0, 2, 1, 3).reshape(16, 16) + offset
    return prepared.projected_patches[0].index_select(0, tile_indices.reshape(-1)).reshape(16, 16, 4096).mean(dim=1)


def extract_cac_features(
    *,
    torch_module: Any,
    prepared: Any,
    tracker: Any | None,
    groups: Mapping[tuple[Any, int], int] | None,
    query_ordinal: int,
    z_cache: Any,
    base_action: Any,
) -> CACFeatureTensors:
    """Extract current/source features for the exact physical sources used by this query."""

    torch = torch_module
    current_by_camera = [_tile_means(prepared, camera, torch) for camera in (0, 1)]
    current_tiles = torch.cat(current_by_camera, dim=0).to(torch.float16)
    source_means: dict[tuple[int, int], Any] = {}
    source_deltas = []
    provenance_rows = []
    source_ids = []
    for layer_index, layer in enumerate(ONSET_LAYERS):
        for camera_index, camera_name in enumerate(("primary", "wrist")):
            camera_key = None
            if groups:
                camera_key = next(
                    (key[0] for key in groups if getattr(key[0], "value", str(key[0])) == camera_name),
                    None,
                )
            for tile in range(16):
                onset = groups.get((camera_key, tile)) if groups and camera_key is not None else None
                reused = tracker is not None and onset is not None and layer >= int(onset)
                if reused:
                    source_query = int(tracker.source_query(layer, camera_key, tile))
                    source_prepared = tracker.source(layer, camera_key, tile)
                    source_key = (source_query, camera_index)
                    if source_key not in source_means:
                        source_means[source_key] = _tile_means(source_prepared, camera_index, torch)
                    source_mean = source_means[source_key][tile]
                else:
                    source_query = int(query_ordinal)
                    source_mean = current_by_camera[camera_index][tile]
                age = int(query_ordinal - source_query)
                if not 0 <= age <= 4:
                    raise CACValidationError("CAC physical source age is outside 0-4")
                source_deltas.append((current_by_camera[camera_index][tile] - source_mean).to(torch.float16))
                provenance_rows.append([
                    int(reused), int(not reused), age, layer_index, layer, camera_index,
                    tile, tile // 4, tile % 4, query_ordinal % 5, source_query % 256, 0,
                ])
                source_ids.append(source_query)
    result = CACFeatureTensors(
        current_tiles=current_tiles,
        source_deltas=torch.stack(source_deltas),
        z_cache=z_cache[0].to(torch.float16),
        base_action=base_action[0].float(),
        proprio=prepared.normalized_proprio[0].float(),
        instruction_embedding=prepared.instruction_embedding.to(torch.float16),
        provenance=torch.as_tensor(provenance_rows, device=current_tiles.device, dtype=torch.uint8),
        source_query_ids=tuple(source_ids),
    )
    result.validate(torch)
    return result


class CACCrossAttentionBlock:
    """Factory wrapper so torch is imported only in the pinned runtime."""

    @staticmethod
    def build(torch: Any) -> Any:
        nn = torch.nn

        class Block(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.query_norm = nn.LayerNorm(256)
                self.context_norm = nn.LayerNorm(256)
                self.attention = nn.MultiheadAttention(256, 8, dropout=0.0, batch_first=True)
                self.ffn_norm = nn.LayerNorm(256)
                self.ffn1 = nn.Linear(256, 1024)
                self.activation = nn.GELU()
                self.ffn2 = nn.Linear(1024, 256)

            def forward(self, query: Any, context: Any) -> Any:
                attended, _ = self.attention(
                    self.query_norm(query), self.context_norm(context), self.context_norm(context),
                    need_weights=False,
                )
                query = query + attended
                return query + self.ffn2(self.activation(self.ffn1(self.ffn_norm(query))))

        return Block()


def build_full_adapter(torch: Any) -> Any:
    nn = torch.nn

    class FullAdapter(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.visual_projection = nn.Linear(4096, 256)
            self.z_projection = nn.Linear(4096, 256)
            self.instruction_projection = nn.Linear(4096, 256)
            self.action_projection = nn.Linear(7, 256)
            self.proprio_projection = nn.Linear(8, 256)
            self.action_step = nn.Embedding(8, 256)
            self.type_embedding = nn.Embedding(2, 256)
            self.layer_embedding = nn.Embedding(4, 256)
            self.camera_embedding = nn.Embedding(2, 256)
            self.tile_row_embedding = nn.Embedding(4, 256)
            self.tile_column_embedding = nn.Embedding(4, 256)
            self.age_embedding = nn.Embedding(5, 256)
            self.reset_embedding = nn.Embedding(5, 256)
            self.reuse_embedding = nn.Embedding(2, 256)
            self.global_query_bias = nn.Parameter(torch.zeros(1, 256))
            self.query_fusion = nn.Linear(1280, 256)
            self.blocks = nn.ModuleList([CACCrossAttentionBlock.build(torch) for _ in range(2)])
            self.output = nn.Linear(256, 7)
            nn.init.zeros_(self.output.weight)
            nn.init.zeros_(self.output.bias)

        def forward(self, features: CACFeatureTensors, residual_bound: Any) -> Any:
            device = features.current_tiles.device
            dtype = self.visual_projection.weight.dtype
            current = self.visual_projection(features.current_tiles.to(dtype))
            current_camera = torch.arange(2, device=device).repeat_interleave(16)
            current_tile = torch.arange(16, device=device).repeat(2)
            current = (
                current + self.type_embedding(torch.zeros(32, device=device, dtype=torch.long))
                + self.camera_embedding(current_camera)
                + self.tile_row_embedding(current_tile // 4)
                + self.tile_column_embedding(current_tile % 4)
                + self.layer_embedding(torch.zeros(32, device=device, dtype=torch.long))
                + self.age_embedding(torch.zeros(32, device=device, dtype=torch.long))
                + self.reset_embedding(torch.full((32,), int(features.provenance[0, 9]), device=device, dtype=torch.long))
                + self.reuse_embedding(torch.zeros(32, device=device, dtype=torch.long))
            )
            provenance = features.provenance.long()
            source = (
                self.visual_projection(features.source_deltas.to(dtype))
                + self.type_embedding(torch.ones(128, device=device, dtype=torch.long))
                + self.layer_embedding(provenance[:, 3])
                + self.camera_embedding(provenance[:, 5])
                + self.tile_row_embedding(provenance[:, 7])
                + self.tile_column_embedding(provenance[:, 8])
                + self.age_embedding(provenance[:, 2])
                + self.reset_embedding(provenance[:, 9])
                + self.reuse_embedding(provenance[:, 0])
            )
            context = torch.cat((current, source), dim=0).unsqueeze(0)
            step = torch.arange(8, device=device)
            instruction = self.instruction_projection(features.instruction_embedding.to(dtype)).expand(8, -1)
            proprio = self.proprio_projection(features.proprio.to(dtype)).expand(8, -1)
            query = self.query_fusion(torch.cat((
                self.z_projection(features.z_cache.to(dtype)),
                self.action_projection(features.base_action.to(dtype)),
                proprio,
                instruction,
                self.action_step(step),
            ), dim=-1)) + self.global_query_bias
            query = query.unsqueeze(0)
            for block in self.blocks:
                query = block(query, context)
            raw = self.output(query[0])
            if tuple(residual_bound.shape) != (8, 7):
                raise CACValidationError("CAC residual-bound shape changed")
            return residual_bound.to(dtype) * torch.tanh(raw)

    adapter = FullAdapter()
    count = sum(parameter.numel() for parameter in adapter.parameters())
    if count != 5_070_599:
        raise CACValidationError(f"CAC full-adapter parameter count changed: {count}")
    return adapter


@contextmanager
def transactional_profile(model: Any) -> Iterable[None]:
    config = model.language_model.config
    before = (config.proportion_attn_var, config.reusable_patches)
    try:
        yield
    finally:
        config.proportion_attn_var, config.reusable_patches = before


def cache_digest(cache: Any) -> str:
    import torch

    digest = hashlib.sha256()
    for tensor in (*cache.key_cache, *cache.value_cache):
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(str(tensor.dtype).encode())
        raw = tensor.detach().contiguous().view(torch.uint8).cpu().numpy()
        digest.update(raw.tobytes())
    return digest.hexdigest()


def tracker_digest(tracker: Any) -> str:
    return tracker.digest()


def serialize_numeric_record(
    features: CACFeatureTensors,
    target_delta: Any,
    validity_mask: Any,
) -> bytes:
    arrays = (
        features.current_tiles.detach().cpu().numpy().astype("<f2", copy=False),
        features.source_deltas.detach().cpu().numpy().astype("<f2", copy=False),
        features.z_cache.detach().cpu().numpy().astype("<f2", copy=False),
        features.base_action.detach().cpu().numpy().astype("<f4", copy=False),
        target_delta.detach().cpu().numpy().astype("<f4", copy=False),
        features.proprio.detach().cpu().numpy().astype("<f4", copy=False),
        features.provenance.detach().cpu().numpy().astype("u1", copy=False),
        validity_mask.detach().cpu().numpy().astype("u1", copy=False),
    )
    payload = b"".join(array.tobytes(order="C") for array in arrays)
    if len(payload) != NUMERIC_PAYLOAD_BYTES:
        raise CACValidationError(f"CAC numeric payload changed: {len(payload)}")
    return payload + bytes(ALIGNED_RECORD_BYTES - len(payload))


def serialize_sidecar(metadata: Mapping[str, Any], numeric_sha256: str) -> bytes:
    required = ("contract_id", "trajectory_id", "anchor_query_id", "endpoint_query_id")
    hashes = []
    for key in required:
        value = str(metadata[key])
        if len(value) != 64:
            raise CACValidationError(f"CAC sidecar identity is not SHA-256: {key}")
        hashes.append(bytes.fromhex(value))
    role_ids = {"adapter_fit": 0, "architecture_selection": 1, "checkpoint_validation": 2,
                "development_calibration": 3, "locked_test": 4}
    suite_ids = {"libero_10": 0, "libero_goal": 1, "libero_object": 2, "libero_spatial": 3}
    header = struct.pack(
        "<4sBBBBQ32s",
        b"CAC1",
        role_ids[str(metadata["role"])], suite_ids[str(metadata["suite"])],
        int(metadata["horizon"]), 0, int(metadata["record_index"]), bytes.fromhex(numeric_sha256),
    )
    payload = header + b"".join(hashes)
    if len(payload) > SIDECAR_RECORD_BYTES:
        raise CACValidationError("CAC binary sidecar exceeded its fixed bound")
    return payload + bytes(SIDECAR_RECORD_BYTES - len(payload))


class BoundedRecordWriter:
    """Single-writer bounded queue with append-only, fsynced fixed records."""

    def __init__(self, root: Path, *, queue_size: int = 2) -> None:
        self.root = root
        if root.exists():
            raise CACValidationError("CAC record output already exists")
        root.mkdir(parents=True)
        self.numeric_path = root / "features.bin"
        self.sidecar_path = root / "sidecars.bin"
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=queue_size)
        self._error: BaseException | None = None
        self._count = 0
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, name="cac-record-writer", daemon=False)
        self._thread.start()

    def submit(self, numeric: bytes, sidecar: bytes) -> None:
        self._ready.wait()
        if self._error is not None:
            raise CACValidationError("CAC writer previously failed") from self._error
        if len(numeric) != ALIGNED_RECORD_BYTES or len(sidecar) != SIDECAR_RECORD_BYTES:
            raise CACValidationError("CAC writer received a malformed record")
        self._queue.put((numeric, sidecar))

    def _run(self) -> None:
        numeric_descriptor = None
        sidecar_descriptor = None
        try:
            numeric_descriptor = os.open(
                self.numeric_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
            )
            sidecar_descriptor = os.open(
                self.sidecar_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
            )
            with os.fdopen(numeric_descriptor, "wb") as numeric_stream, os.fdopen(
                sidecar_descriptor, "wb"
            ) as sidecar_stream:
                numeric_descriptor = None
                sidecar_descriptor = None
                self._ready.set()
                while True:
                    item = self._queue.get()
                    try:
                        if item is None:
                            break
                        numeric, sidecar = item
                        numeric_stream.write(numeric)
                        sidecar_stream.write(sidecar)
                        self._count += 1
                    finally:
                        self._queue.task_done()
                numeric_stream.flush(); os.fsync(numeric_stream.fileno())
                sidecar_stream.flush(); os.fsync(sidecar_stream.fileno())
        except BaseException as error:
            self._error = error
            self._ready.set()
        finally:
            if numeric_descriptor is not None:
                os.close(numeric_descriptor)
            if sidecar_descriptor is not None:
                os.close(sidecar_descriptor)

    def close(self) -> dict[str, Any]:
        self._ready.wait()
        if self._error is not None:
            self._thread.join()
            raise CACValidationError("CAC asynchronous writer failed") from self._error
        self._queue.put(None)
        self._thread.join()
        if self._error is not None:
            raise CACValidationError("CAC asynchronous writer failed") from self._error
        return {
            "records": self._count,
            "numeric_bytes": self.numeric_path.stat().st_size,
            "sidecar_bytes": self.sidecar_path.stat().st_size,
            "numeric_sha256": _stream_sha256(self.numeric_path),
            "sidecar_sha256": _stream_sha256(self.sidecar_path),
        }


def _stream_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
