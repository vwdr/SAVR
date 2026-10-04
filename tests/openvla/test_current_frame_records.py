import copy
import hashlib
import unittest
import numpy as np
from savr.openvla.current_frame_records import (SHAPES, BF16, encode_record, decode_record,
                                               tensors_to_arrays, arrays_to_tensors)


def fixture():
    meta = dict(schema_version="current-frame-feature-v1", sample_id="synthetic", trajectory_id="train-demo",
        source_sha256="a" * 64, frame=0, split="train", split_hash="b" * 64,
        student_observation_sha256="c" * 64, teacher_observation_sha256="c" * 64,
        compression_budget=384, selector_sha256="d" * 64, backbone_sha256="e" * 64, extractor_sha256="f" * 64,
        normalization_statistics_key="libero_spatial_no_noops",
        instruction_pooling="fp32_mean_current_instruction_input_embeddings",
        feature_origin="hard_compacted_current_frame_decoder")
    arrays = {k: np.zeros(shape, dtype=np.uint16 if k in BF16 else np.float32) for k, shape in SHAPES.items()}
    return meta, arrays, {"synthetic": copy.deepcopy(meta)}


class RecordTests(unittest.TestCase):
    def test_lossless_roundtrip_and_no_input_alias(self):
        meta, arrays, approved = fixture()
        arrays["current_visual"][0, 0] = 0x3f80  # BF16 1.0, not numeric integer 16256.
        payload, sha = encode_record(meta, arrays, approved)
        decoded, values = decode_record(payload, sha, approved)
        self.assertEqual(meta, decoded)
        self.assertLess(len(payload), 5 * 1024 * 1024)
        for k in arrays:
            self.assertEqual(values[k].tobytes(), arrays[k].tobytes())
            self.assertFalse(np.shares_memory(values[k], arrays[k]))

    def test_reject_split_frame_and_teacher_mismatch(self):
        for key, value in (("split", "locked"), ("frame", 1), ("teacher_observation_sha256", "d" * 64),
                           ("compression_budget", 512), ("feature_origin", "synthetic_hidden_states")):
            m, a, approved = fixture(); m[key] = value
            with self.assertRaises(ValueError): encode_record(m, a, approved)

    def test_record_cannot_self_authorize_bad_contract(self):
        for key, value in (("split", "locked"), ("teacher_observation_sha256", "d" * 64), ("frame", -1)):
            m, a, approved = fixture(); m[key] = value; approved["synthetic"] = m.copy()
            with self.assertRaises(ValueError): encode_record(m, a, approved)

    def test_reject_bad_shape_dtype_and_nonfinite(self):
        for key, value in (("action_features", np.zeros((56, 4096), np.uint16)),
                           ("current_visual", np.full((512, 4096), 0x7f80, np.uint16)),
                           ("instruction", np.full((4096,), 0x7fc0, np.uint16)),
                           ("base_actions", np.zeros((8, 7), np.float64)),
                           ("teacher_actions", np.full((8, 7), np.nan, np.float32))):
            m, a, approved = fixture(); a[key] = value
            with self.assertRaises(ValueError): encode_record(m, a, approved)

    def test_reject_tamper_or_unapproved_sample(self):
        m, a, approved = fixture(); payload, sha = encode_record(m, a, approved)
        with self.assertRaises(ValueError): decode_record(payload + b"x", sha, approved)
        with self.assertRaises(ValueError): decode_record(payload, sha, {})

    def test_bfloat16_tensor_roundtrip(self):
        try: import torch
        except ImportError: self.skipTest("torch not installed in local analysis runtime")
        _, arrays, _ = fixture()
        arrays["current_visual"][0, :3] = [0x3f80, 0xbf80, 0x3c80]
        tensors = arrays_to_tensors(arrays)
        self.assertEqual(tensors["current_visual"][0, :3].float().tolist(), [1., -1., .015625])
        restored = tensors_to_arrays(tensors)
        for key in arrays: self.assertEqual(arrays[key].tobytes(), restored[key].tobytes())


if __name__ == "__main__": unittest.main()
