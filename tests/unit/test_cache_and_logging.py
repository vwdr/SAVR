from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from savr.cache import (  # noqa: E402
    CacheCompatibilityError,
    CacheContext,
    ProjectedFeatureCache,
    TensorMetadata,
)
from savr.logging import ImmutableRecordStore, RecordExistsError  # noqa: E402


class FakeTensor:
    def __init__(self, shape=(1, 4, 3), dtype="bfloat16", device="cuda:0"):
        self.shape = shape
        self.dtype = dtype
        self.device = device

    def detach(self):
        return self


class CacheAndLoggingTests(unittest.TestCase):
    def test_cache_validates_context_metadata_and_age(self) -> None:
        context = CacheContext("episode", "task", "checkpoint", "config")
        other = CacheContext("other", "task", "checkpoint", "config")
        cache = ProjectedFeatureCache()
        feature = FakeTensor()
        metadata = cache.store(context, feature)
        self.assertEqual(cache.load(context, metadata), feature)
        cache.mark_reused()
        self.assertEqual(cache.age, 1)
        with self.assertRaises(CacheCompatibilityError):
            cache.load(
                context,
                TensorMetadata(shape=(1, 5, 3), dtype="bfloat16", device="cuda:0"),
            )
        self.assertFalse(cache.available(other))
        cache.invalidate()
        self.assertEqual(cache.age, 0)

    def test_immutable_records_cannot_be_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ImmutableRecordStore(Path(directory))
            path = store.write_query(0, {"query_index": 0, "refresh": True})
            self.assertEqual(json.loads(path.read_text())["query_index"], 0)
            with self.assertRaises(RecordExistsError):
                store.write_query(0, {"query_index": 0, "refresh": False})

            episode = store.write_episode("task_00_state_00", {"success": True})
            self.assertTrue(json.loads(episode.read_text())["success"])
            with self.assertRaises(RecordExistsError):
                store.write_episode("task_00_state_00", {"success": False})

    def test_record_identifier_is_scoped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ImmutableRecordStore(Path(directory))
            with self.assertRaises(ValueError):
                store.write_episode("../outside", {"success": True})


if __name__ == "__main__":
    unittest.main()
