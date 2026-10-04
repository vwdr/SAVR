"""Lossless feature-record interchange, not a qualified GPU feature extractor.

BF16 tensors are stored as their uint16 bit patterns, not cast to float16.
The trusted collection manifest, not a record's own split label, grants access.
"""
import hashlib
import io
import json
import zipfile
import numpy as np

SHAPES = dict(action_features=(8, 4096), current_visual=(512, 4096),
              instruction=(4096,), base_actions=(8, 7), teacher_actions=(8, 7), state=(8,))
BF16 = {"action_features", "current_visual", "instruction"}
MAX_RECORD_BYTES = 5 * 1024 * 1024
META_KEYS = {"schema_version", "sample_id", "trajectory_id", "source_sha256", "frame",
             "split", "split_hash", "student_observation_sha256", "teacher_observation_sha256",
             "compression_budget", "selector_sha256", "backbone_sha256", "extractor_sha256",
             "normalization_statistics_key", "instruction_pooling", "feature_origin"}


def validate_metadata(meta, approved_samples):
    if set(meta) != META_KEYS or meta["schema_version"] != "current-frame-feature-v1":
        raise ValueError("feature metadata schema differs")
    # Explicit sample-level allowlist prevents using arbitrary frames in a training trajectory.
    expected = approved_samples.get(meta["sample_id"])
    if expected is None or expected != meta or meta["split"] != "train":
        raise ValueError("sample is not exactly in the approved training manifest")
    if (type(meta["frame"]) is not int or meta["frame"] < 0
            or type(meta["compression_budget"]) is not int or meta["compression_budget"] not in (256, 384)
            or meta["student_observation_sha256"] != meta["teacher_observation_sha256"]
            or meta["instruction_pooling"] != "fp32_mean_current_instruction_input_embeddings"
            or meta["feature_origin"] != "hard_compacted_current_frame_decoder"):
        raise ValueError("student/teacher alignment or extraction contract differs")
    for key in ("source_sha256", "split_hash", "student_observation_sha256", "teacher_observation_sha256",
                "selector_sha256", "backbone_sha256", "extractor_sha256"):
        if not isinstance(meta[key], str) or len(meta[key]) != 64 or any(c not in "0123456789abcdef" for c in meta[key]):
            raise ValueError("invalid provenance hash")


def validate_arrays(arrays):
    if set(arrays) != set(SHAPES):
        raise ValueError("feature fields differ")
    for name, shape in SHAPES.items():
        value = arrays[name]
        dtype = np.dtype("uint16") if name in BF16 else np.dtype("float32")
        if not isinstance(value, np.ndarray) or value.shape != shape or value.dtype != dtype:
            raise ValueError(f"feature shape/dtype differs: {name}")
        # BF16 exponent all ones signifies either infinity or NaN.
        finite = not np.any((value & 0x7f80) == 0x7f80) if name in BF16 else np.isfinite(value).all()
        if not finite:
            raise ValueError(f"nonfinite feature: {name}")


def encode_record(meta, arrays, approved_samples):
    validate_metadata(meta, approved_samples)
    validate_arrays(arrays)
    body = json.dumps(meta, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    stream = io.BytesIO()
    np.savez(stream, metadata=np.frombuffer(body, dtype=np.uint8), **arrays)
    payload = stream.getvalue()
    if len(payload) > MAX_RECORD_BYTES:
        raise ValueError("record storage budget exceeded")
    return payload, hashlib.sha256(payload).hexdigest()


def decode_record(payload, expected_sha256, approved_samples):
    if len(payload) > MAX_RECORD_BYTES or hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError("record size or hash mismatch")
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        entries = archive.infolist()
        names = {name + ".npy" for name in (*SHAPES, "metadata")}
        if (len(entries) != len(names) or {e.filename for e in entries} != names
                or sum(e.file_size for e in entries) > MAX_RECORD_BYTES):
            raise ValueError("invalid or oversized record archive")
    with np.load(io.BytesIO(payload), allow_pickle=False) as packed:
        raw_meta = packed["metadata"]
        if raw_meta.ndim != 1 or raw_meta.dtype != np.uint8 or raw_meta.size > 16384:
            raise ValueError("invalid metadata payload")
        meta = json.loads(raw_meta.tobytes())
        arrays = {name: packed[name].copy() for name in SHAPES}
    validate_metadata(meta, approved_samples)
    validate_arrays(arrays)
    return meta, arrays


def tensors_to_arrays(features):
    """Called outside backbone forward; does not retain autograd/inference tensor graphs."""
    import torch
    if set(features) != set(SHAPES):
        raise ValueError("tensor field set differs")
    arrays = {}
    for name, value in features.items():
        expected = torch.bfloat16 if name in BF16 else torch.float32
        if value.dtype != expected or tuple(value.shape) != SHAPES[name]:
            raise ValueError("extractor tensor dtype/shape differs")
        value = value.detach().contiguous().cpu()
        arrays[name] = (value.view(torch.int16).numpy().view(np.uint16).copy()
                        if name in BF16 else value.numpy().copy())
    validate_arrays(arrays)
    return arrays


def arrays_to_tensors(arrays):
    import torch
    validate_arrays(arrays)
    return {name: (torch.from_numpy(value.copy().view(np.int16)).view(torch.bfloat16)
                   if name in BF16 else torch.from_numpy(value.copy())) for name, value in arrays.items()}
