from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from savr.cache import CacheContext  # noqa: E402
from savr.controllers import VisualOnlyRefreshController  # noqa: E402
from savr.integration.openvla_oft import (  # noqa: E402
    OpenVLAProjectedFeatureAdapter,
)


class FakeTensor:
    def __init__(
        self,
        shape,
        *,
        dtype="bfloat16",
        device="cuda:0",
        value=None,
    ):
        self.shape = shape
        self.dtype = dtype
        self.device = device
        self.value = value

    def detach(self):
        return self


class FakeVisionBackbone:
    def get_num_patches(self):
        return 2

    def get_num_images_in_input(self):
        return 2


class FakeModel:
    def __init__(self):
        self.vision_backbone = FakeVisionBackbone()
        self.visual_calls = 0

    def _process_vision_features(
        self,
        pixel_values,
        language_embeddings=None,
        use_film=False,
    ):
        self.visual_calls += 1
        return FakeTensor(
            (pixel_values.shape[0], 4, language_embeddings.shape[-1]),
            dtype=language_embeddings.dtype,
            device=language_embeddings.device,
            value=f"visual-{self.visual_calls}",
        )


IMAGES = {
    "third_person": [[[0, 0, 0]]],
    "wrist": [[[0, 0, 0]]],
}
CONTEXT = CacheContext("episode", "task", "checkpoint", "config")


class AdapterTests(unittest.TestCase):
    def make_adapter(self):
        model = FakeModel()
        controller = VisualOnlyRefreshController(
            image_threshold=1.0,
            max_reuse_horizon=4,
        )
        adapter = OpenVLAProjectedFeatureAdapter(
            model=model,
            controller=controller,
            action_chunk_getter=lambda value: value["actions"],
        )
        adapter.begin_context(CONTEXT)
        return model, adapter

    def make_query(self, model, state, language_dimension=3):
        pixels = FakeTensor((1, 2, 8, 8))
        language = FakeTensor((1, 5, language_dimension))

        def query():
            feature = model._process_vision_features(pixels, language, False)
            return {
                "visual": feature.value,
                "state": list(state),
                "actions": [[0.0, 0.0]],
            }

        return query

    def test_reuse_skips_visual_compute_and_keeps_current_state(self) -> None:
        model, adapter = self.make_adapter()
        original_function = model._process_vision_features.__func__

        first = adapter.run_query(
            query=self.make_query(model, [0.0] * 8),
            images=IMAGES,
            state=[0.0] * 8,
            environment_step=10,
        )
        second = adapter.run_query(
            query=self.make_query(model, [1.0] * 8),
            images=IMAGES,
            state=[1.0] * 8,
            environment_step=18,
        )

        self.assertEqual(model.visual_calls, 1)
        self.assertEqual(first.cache_event, "refresh")
        self.assertEqual(second.cache_event, "reuse")
        self.assertEqual(first.value["visual"], second.value["visual"])
        self.assertEqual(second.value["state"], [1.0] * 8)
        self.assertEqual(model._process_vision_features.__func__, original_function)

    def test_incompatible_cache_forces_refresh(self) -> None:
        model, adapter = self.make_adapter()
        adapter.run_query(
            query=self.make_query(model, [0.0] * 8, language_dimension=3),
            images=IMAGES,
            state=[0.0] * 8,
            environment_step=10,
        )
        second = adapter.run_query(
            query=self.make_query(model, [0.0] * 8, language_dimension=4),
            images=IMAGES,
            state=[0.0] * 8,
            environment_step=18,
        )
        self.assertEqual(model.visual_calls, 2)
        self.assertEqual(second.cache_event, "forced_refresh")
        self.assertIn("cache_incompatible", second.decision.triggers)

    def test_context_change_resets_cache(self) -> None:
        model, adapter = self.make_adapter()
        adapter.run_query(
            query=self.make_query(model, [0.0] * 8),
            images=IMAGES,
            state=[0.0] * 8,
            environment_step=10,
        )
        changed = adapter.begin_context(
            CacheContext("episode-2", "task", "checkpoint", "config")
        )
        result = adapter.run_query(
            query=self.make_query(model, [0.0] * 8),
            images=IMAGES,
            state=[0.0] * 8,
            environment_step=10,
        )
        self.assertTrue(changed)
        self.assertEqual(model.visual_calls, 2)
        self.assertIn("empty_cache", result.decision.triggers)

    def test_every_context_field_change_resets_cache(self) -> None:
        variants = (
            CacheContext("episode-2", "task", "checkpoint", "config"),
            CacheContext("episode", "task-2", "checkpoint", "config"),
            CacheContext("episode", "task", "checkpoint-2", "config"),
            CacheContext("episode", "task", "checkpoint", "config-2"),
        )
        for variant in variants:
            model, adapter = self.make_adapter()
            adapter.run_query(
                query=self.make_query(model, [0.0] * 8),
                images=IMAGES,
                state=[0.0] * 8,
                environment_step=10,
            )
            self.assertTrue(adapter.begin_context(variant))
            self.assertFalse(adapter.cache.available(CONTEXT))

    def test_failure_restores_method_and_invalidates_cache(self) -> None:
        model, adapter = self.make_adapter()
        original_function = model._process_vision_features.__func__
        pixels = FakeTensor((1, 2, 8, 8))
        language = FakeTensor((1, 5, 3))

        def failing_query():
            model._process_vision_features(pixels, language, False)
            raise RuntimeError("downstream failure")

        with self.assertRaisesRegex(RuntimeError, "downstream failure"):
            adapter.run_query(
                query=failing_query,
                images=IMAGES,
                state=[0.0] * 8,
                environment_step=10,
            )
        self.assertEqual(model._process_vision_features.__func__, original_function)
        self.assertFalse(adapter.cache.available(CONTEXT))


if __name__ == "__main__":
    unittest.main()
