from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from savr.cache import CacheContext  # noqa: E402
from savr.controllers import Policy  # noqa: E402
from savr.integration.openvla_oft import OpenVLAProjectedFeatureAdapter  # noqa: E402
from savr.logging import ImmutableRecordStore  # noqa: E402
from savr.savr2 import (  # noqa: E402
    SAVR2Configuration,
    StateAwareVisualRefresh2Controller,
)


IMAGES = {
    "full_image": [[[0, 0, 0]]],
    "wrist_image": [[[0, 0, 0]]],
}
STATE = [0.5] * 8
ACTION = [[0.0] * 7 for _ in range(8)]
CONTEXT = CacheContext("episode", "task", "checkpoint", "savr2-b15")


def configuration(**overrides: object) -> SAVR2Configuration:
    values: dict[str, object] = {
        "configuration_id": "savr2-b15",
        "image_thresholds": {"full_image": 1.0, "wrist_image": 1.0},
        "state_thresholds": {
            "translation": 1.0,
            "orientation": 1.0,
            "gripper": 1.0,
        },
        "action_thresholds": {
            "translation": 1.0,
            "rotation": 1.0,
            "gripper": 1.0,
        },
        "skip_budget": 0.15,
    }
    values.update(overrides)
    return SAVR2Configuration(**values)  # type: ignore[arg-type]


def controller(**overrides: object) -> StateAwareVisualRefresh2Controller:
    value = StateAwareVisualRefresh2Controller(
        configuration=configuration(**overrides),
        state_q01=[0.0] * 8,
        state_q99=[1.0] * 8,
        action_q01=[0.0] * 7,
        action_q99=[1.0] * 7,
    )
    value.reset(CONTEXT)
    return value


def complete_query(
    value: StateAwareVisualRefresh2Controller,
    *,
    images: dict[str, object] = IMAGES,
    state: object = STATE,
    action: object = ACTION,
    cache_age: int = 0,
) -> object:
    decision = value.decide(
        images=images,
        state=state,
        cache_available=value.query_index > 0,
        cache_age=cache_age,
    )
    value.observe(
        decision=decision,
        images=images,
        state=state,
        action_chunk=action,
    )
    return decision


class FakeTensor:
    def __init__(self, shape, *, dtype="bfloat16", device="cuda:0", value=None):
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

    def _process_vision_features(self, pixel_values, language_embeddings=None, use_film=False):
        self.visual_calls += 1
        return FakeTensor(
            (pixel_values.shape[0], 4, language_embeddings.shape[-1]),
            dtype=language_embeddings.dtype,
            device=language_embeddings.device,
            value=f"visual-{self.visual_calls}",
        )


class SAVR2ControllerTests(unittest.TestCase):
    def test_warmup_stability_budget_and_isolated_reuse(self) -> None:
        value = controller()
        decisions = [complete_query(value) for _ in range(7)]
        self.assertTrue(all(decision.refresh for decision in decisions[:6]))
        self.assertIn("skip_budget_prefix_cap", decisions[5].triggers)
        self.assertFalse(decisions[6].refresh)
        self.assertEqual(decisions[6].stable_fresh_before, 4)

        after_reuse = complete_query(value, cache_age=1)
        self.assertTrue(after_reuse.refresh)
        self.assertIn("insufficient_stable_fresh", after_reuse.triggers)
        self.assertIn("maximum_consecutive_reuse", after_reuse.triggers)
        self.assertEqual(value.snapshot().completed_reuses, 1)

    def test_prefix_budget_never_overshoots(self) -> None:
        value = controller()
        reuses = 0
        for _ in range(60):
            decision = complete_query(value, cache_age=0)
            reuses += int(not decision.refresh)
            self.assertLessEqual(reuses / value.query_index, 0.15)

    def test_either_camera_local_change_vetoes_reuse(self) -> None:
        value = controller(
            image_thresholds={"full_image": 0.1, "wrist_image": 0.1}
        )
        for _ in range(6):
            complete_query(value)
        changed = {
            "full_image": IMAGES["full_image"],
            "wrist_image": [[[255, 255, 255]]],
        }
        decision = value.decide(
            images=changed,
            state=STATE,
            cache_available=True,
            cache_age=0,
        )
        self.assertTrue(decision.refresh)
        self.assertIn("image_change.wrist_image", decision.triggers)
        self.assertNotIn("image_change.full_image", decision.triggers)
        self.assertEqual(len(decision.camera_patch_scores["wrist_image"]), 64)

    def test_group_thresholds_and_gripper_transition_veto_independently(self) -> None:
        value = controller(
            state_thresholds={
                "translation": 0.1,
                "orientation": 1.0,
                "gripper": 1.0,
            }
        )
        for _ in range(6):
            complete_query(value)
        changed_state = [1.0, 1.0, 1.0, *STATE[3:]]
        state_decision = value.decide(
            images=IMAGES,
            state=changed_state,
            cache_available=True,
            cache_age=0,
        )
        self.assertIn("state_change.translation", state_decision.triggers)
        self.assertNotIn("state_change.orientation", state_decision.triggers)

        mixed = [[0.0] * 7 for _ in range(8)]
        for index in range(4, 8):
            mixed[index][6] = 1.0
        transition_value = controller()
        complete_query(transition_value, action=ACTION)
        complete_query(transition_value, action=mixed)
        transition_decision = transition_value.decide(
            images=IMAGES,
            state=STATE,
            cache_available=True,
            cache_age=0,
        )
        self.assertTrue(transition_decision.gripper_transition_veto)
        self.assertIn("gripper_transition.mixed_latest", transition_decision.triggers)

    def test_invalid_inputs_and_context_changes_fail_closed(self) -> None:
        value = controller()
        complete_query(value)
        invalid = value.decide(
            images={"full_image": IMAGES["full_image"]},
            state=[float("nan")] * 8,
            cache_available=True,
            cache_age=0,
        )
        self.assertTrue(invalid.refresh)
        self.assertIn("invalid_image", invalid.triggers)
        self.assertIn("invalid_state", invalid.triggers)

        with self.assertRaises(ValueError):
            value.reset(CacheContext("episode-2", "task", "checkpoint", "wrong"))
        value.reset(CacheContext("episode-2", "task", "checkpoint", "savr2-b15"))
        snapshot = value.snapshot()
        self.assertEqual(snapshot.query_index, 0)
        self.assertEqual(snapshot.completed_reuses, 0)
        self.assertFalse(snapshot.has_image_reference)

    def test_only_frozen_budget_candidates_are_accepted(self) -> None:
        for budget in (0.05, 0.10, 0.15):
            self.assertEqual(configuration(skip_budget=budget).skip_budget, budget)
        with self.assertRaises(ValueError):
            configuration(skip_budget=0.20)

    def test_savr3_translation_reversal_veto_is_distinct_and_exact(self) -> None:
        savr3_config = configuration(
            configuration_id="savr3-rv-w375-b15",
            policy=Policy.SAVR3,
            translation_direction_reversal_veto=True,
        )
        value = StateAwareVisualRefresh2Controller(
            configuration=savr3_config,
            state_q01=[0.0] * 8,
            state_q99=[1.0] * 8,
            action_q01=[0.0] * 7,
            action_q99=[1.0] * 7,
        )
        value.reset(
            CacheContext(
                "episode", "task", "checkpoint", "savr3-rv-w375-b15"
            )
        )
        positive = [[0.0] * 7 for _ in range(8)]
        negative = [[0.0] * 7 for _ in range(8)]
        for row in positive:
            row[0] = 0.1
        for row in negative:
            row[0] = -0.1
        for _ in range(5):
            complete_query(value, action=positive)
        complete_query(value, action=negative)
        decision = value.decide(
            images=IMAGES,
            state=STATE,
            cache_available=True,
            cache_age=0,
        )
        self.assertIs(decision.policy, Policy.SAVR3)
        self.assertTrue(decision.translation_direction_reversals[0])
        self.assertFalse(any(decision.translation_direction_reversals[1:]))
        self.assertIn("translation_direction_reversal", decision.triggers)
        self.assertTrue(decision.refresh)

    def test_savr3_zero_mean_is_not_reversal_and_can_reuse(self) -> None:
        savr3_config = configuration(
            configuration_id="savr3-rv-w375-b15",
            policy=Policy.SAVR3,
            translation_direction_reversal_veto=True,
        )
        value = StateAwareVisualRefresh2Controller(
            configuration=savr3_config,
            state_q01=[0.0] * 8,
            state_q99=[1.0] * 8,
            action_q01=[0.0] * 7,
            action_q99=[1.0] * 7,
        )
        value.reset(
            CacheContext(
                "episode", "task", "checkpoint", "savr3-rv-w375-b15"
            )
        )
        for _ in range(7):
            decision = complete_query(value, action=ACTION)
        self.assertFalse(any(decision.translation_direction_reversals))
        self.assertNotIn("translation_direction_reversal", decision.triggers)
        self.assertFalse(decision.refresh)

    def test_savr3_identity_requires_reversal_veto(self) -> None:
        with self.assertRaisesRegex(ValueError, "SAVR3 requires"):
            configuration(policy=Policy.SAVR3)
        with self.assertRaisesRegex(ValueError, "belongs to SAVR3"):
            configuration(translation_direction_reversal_veto=True)

    def test_adapter_reuse_skips_visual_compute_and_records_all_fields(self) -> None:
        model = FakeModel()
        value = controller()
        with tempfile.TemporaryDirectory() as directory:
            store = ImmutableRecordStore(Path(directory))
            adapter = OpenVLAProjectedFeatureAdapter(
                model=model,
                controller=value,
                record_store=store,
                action_chunk_getter=lambda result: result["actions"],
            )
            adapter.begin_context(CONTEXT)

            def query(state_value: float):
                pixels = FakeTensor((1, 2, 8, 8))
                language = FakeTensor((1, 5, 3))

                def run():
                    visual = model._process_vision_features(pixels, language, False)
                    return {
                        "visual": visual.value,
                        "state": [state_value] * 8,
                        "actions": ACTION,
                    }

                return run

            results = []
            for index in range(7):
                results.append(
                    adapter.run_query(
                        query=query(index / 10),
                        images=IMAGES,
                        state=STATE,
                        environment_step=index * 8,
                    )
                )
            self.assertEqual(model.visual_calls, 6)
            self.assertEqual(results[-1].cache_event, "reuse")
            self.assertFalse(results[-1].decision.refresh)
            record = json.loads(
                (store.query_dir / "query_00000006.json").read_text(encoding="utf-8")
            )
            decision = record["decision"]
            for field in (
                "camera_patch_scores",
                "camera_global_scores",
                "state_group_scores",
                "action_group_scores",
                "stable_fresh_before",
                "completed_reuses_before",
                "signals_stable",
            ):
                self.assertIn(field, decision)


if __name__ == "__main__":
    unittest.main()
