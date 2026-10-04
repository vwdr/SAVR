"""CPU contracts for episode scheduling, paired images and query wiring."""
from types import SimpleNamespace as NS
from unittest.mock import patch
import unittest

import numpy as np
import torch

from savr.openvla.attention_diagnostic import execute_episode
from savr.openvla.specprune import EpisodeState
from savr.openvla.specprune_episode import (
    CameraPair, SpecPruneEpisodeQuery, execute_specprune_episode,
    prepare_query_camera_pairs,
)


class EpisodeBridgeTests(unittest.TestCase):
    def fixture(self, horizon=20, done_at=None, fail_at=None):
        trace, queries = [], []

        class Env:
            n = 0

            def reset(self):
                self.n = 0
                trace.append("reset")

            def set_init_state(self, state):
                trace.append(("state", state.tolist()))
                return 0

            def step(self, action):
                self.n += 1
                if self.n == fail_at:
                    raise RuntimeError("simulator fault")
                trace.append(("step", action))
                return self.n, 0, self.n == done_at, {}

        def prepare(obs, size):
            trace.append(("prepare", obs, size))
            return dict(full_image=np.full((224, 224, 3), obs, np.uint8),
                        wrist_image=np.full((224, 224, 3), obs + 100, np.uint8),
                        state=np.zeros(8)), None

        def query(obs, instruction):
            trace.append(("query", int(obs["full_image"][0, 0, 0]), instruction))
            actions = np.zeros((8, 7), np.float32)
            actions[:, 0] = 1
            return actions

        def bridge(obs, instruction, previous, state):
            queries.append((int(obs["full_image"][0, 0, 0]), previous.step,
                            int(previous.scene[0, 0, 0]), int(previous.wrist[0, 0, 0]),
                            state.query, state.precise))
            state.query += 1
            return query(obs, instruction)

        def process(action, family):
            trace.append(("process", action.tolist(), family))
            return action

        cfg = NS(num_open_loop_steps=8, num_steps_wait=10, model_family="openvla",
                 task_suite_name="test")
        evaluation = NS(TASK_MAX_STEPS={"test": horizon}, prepare_observation=prepare,
                        get_libero_dummy_action=lambda _: [0] * 7, process_action=process)
        return NS(trace=trace, queries=queries, env=Env(), cfg=cfg, evaluation=evaluation,
                  query=query, bridge=bridge)

    def run_bridge(self, f, query=None, **kwargs):
        return execute_specprune_episode(f.evaluation, f.cfg, f.env, np.array([3]),
                                        "task", query or f.bridge, 224,
                                        episode_id="same-id", **kwargs)

    def test_disabled_controller_exact_original_trace(self):
        for done in (21, 30, None):
            old, new = self.fixture(done_at=done), self.fixture(done_at=done)
            reference = execute_episode(old.evaluation, old.cfg, old.env, np.array([3]),
                                        "task", old.query, 224)
            actual = self.run_bridge(new, controller=False)
            self.assertEqual(old.trace, new.trace)
            self.assertEqual(reference, {k: actual[k] for k in reference})
            self.assertEqual(actual["replans"], 0)

    def test_history_is_paired_control_steps_not_policy_queries(self):
        f = self.fixture()
        result = self.run_bridge(f, controller=False)
        self.assertEqual(f.queries[:2], [(10, 0, 10, 110, 0, False),
                                         (18, 3, 13, 113, 1, False)])
        self.assertEqual(result["history_frames"], 6)
        self.assertEqual(result["policy_queries"], 3)

    def test_replan_only_after_executing_and_never_skips_early_steps(self):
        f = self.fixture(horizon=16)

        def query(*args):
            actions = f.bridge(*args)
            if args[-1].query == 2:
                actions[3] = 0  # step 11 fast sets coarse; step 12 enters precise.
            return actions

        result = self.run_bridge(f, query)
        self.assertEqual((result["replans"], result["discarded_actions"]), (1, 4))
        self.assertEqual((result["executed_steps"], result["policy_queries"]), (16, 3))
        self.assertEqual(f.env.n, 26)
        self.assertEqual([q[0] for q in f.queries], [10, 18, 22])
        self.assertTrue(f.queries[2][-1])

    def test_done_during_replan_does_not_query_again(self):
        f = self.fixture(done_at=22)

        def query(*args):
            actions = f.bridge(*args)
            if args[-1].query == 2:
                actions[3] = 0
            return actions

        result = self.run_bridge(f, query)
        self.assertTrue(result["success"])
        self.assertEqual((result["executed_steps"], result["policy_queries"]), (12, 2))

    def test_repeat_id_resets_all_state_and_history(self):
        state = EpisodeState()
        for _ in range(2):
            f = self.fixture(horizon=1)
            self.run_bridge(f, state=state)
            self.assertEqual(f.queries[0], (10, 0, 10, 110, 0, False))
            self.assertEqual(state.confidence, {})
            self.assertEqual(state.previous_indices, ())
            self.assertEqual(state.query, 1)
            state.confidence[14] = 9
            state.previous_indices = (7,)
            state.precise = True
            state.action_sum[:] = 10

    def test_query_errors_propagate_without_task_record(self):
        for invalid in (np.zeros((7, 7)), np.full((8, 7), np.nan)):
            f = self.fixture()
            with self.assertRaisesRegex(ValueError, "eight finite"):
                self.run_bridge(f, lambda *args: invalid)
            self.assertEqual(f.env.n, 10)
        f = self.fixture(fail_at=12)
        with self.assertRaisesRegex(RuntimeError, "simulator fault"):
            self.run_bridge(f)

    def test_selector_must_complete_exactly_once(self):
        for increments in (0, 2):
            def query(obs, text, previous, state):
                state.query += increments
                return np.zeros((8, 7))
            with self.assertRaisesRegex(ValueError, "exactly one selector"):
                self.run_bridge(self.fixture(), query)

    def test_frame_copies_are_bounded_and_read_only(self):
        f = self.fixture()
        obs, _ = f.evaluation.prepare_observation(17, 224)
        pair = CameraPair.capture(obs, 7)
        obs["full_image"][:] = 0
        self.assertEqual(int(pair.scene[0, 0, 0]), 17)
        with self.assertRaises(ValueError):
            pair.wrist[0, 0, 0] = 0
        obs["full_image"] = np.zeros((224, 224, 3), np.float32)
        with self.assertRaises(ValueError):
            CameraPair.capture(obs, 7)

    def test_invalid_settings_fail_before_environment_reset(self):
        f = self.fixture()
        f.cfg.num_steps_wait = 9
        with self.assertRaises(ValueError):
            self.run_bridge(f)
        self.assertEqual(f.trace, [])

    def test_preprocessing_once_per_view_and_history_not_mutated(self):
        f = self.fixture()
        obs, _ = f.evaluation.prepare_observation(20, 224)
        old, _ = f.evaluation.prepare_observation(17, 224)
        pair = CameraPair.capture(old, 7)
        calls = []

        def transform(images, cfg):
            calls.append([int(i[0, 0, 0]) for i in images])
            for image in images:
                image[:] += 1
            return images

        current, past = prepare_query_camera_pairs(obs, pair, transform, f.cfg)
        self.assertEqual(calls, [[20, 120], [17, 117]])
        self.assertEqual(int(current[0][0, 0, 0]), 21)
        self.assertEqual(int(past[0][0, 0, 0]), 18)
        self.assertEqual(int(pair.scene[0, 0, 0]), 17)
        self.assertEqual(int(obs["full_image"][0, 0, 0]), 20)


class QueryWiringTests(unittest.TestCase):
    """Mock neural boundaries; numerical parity belongs to real qualification."""
    def exercise(self, enabled, precise=False):
        f = EpisodeBridgeTests().fixture()
        obs, _ = f.evaluation.prepare_observation(20, 224)
        previous_obs, _ = f.evaluation.prepare_observation(17, 224)
        previous = CameraPair.capture(previous_obs, 7)
        state = EpisodeState(); state.reset("x"); state.precise = precise
        preparation_inputs, semantic_inputs, budgets, selectors = [], [], [], []

        def images(values, cfg):
            preparation_inputs.append([int(i[0, 0, 0]) for i in values])
            return [i + 1 for i in values]

        def semantic(**kwargs):
            self.assertFalse(torch.is_grad_enabled())
            current = kwargs["prepare_images"](None, kwargs["cfg"])
            semantic_inputs.extend(int(i[0, 0, 0]) for i in current)
            return NS(action_mask=torch.zeros((1, 2), dtype=torch.bool),
                      input_embeddings=torch.zeros(1, 2, 2), projected_patches=None,
                      attention_mask=None, instruction_token_indices=())

        def low(current, past, **kw):
            budgets.append((int(current[0, 0, 0]), int(past[0, 0, 0]), kw))
            return np.array([257 if kw.get("wrist") else 1])

        def selection(layout, episode, low, **kw):
            selectors.append((episode, low, kw))
            return episode

        def forward(decoder, embeddings, episode):
            episode.query += 1
            return torch.zeros(1, 56, 2), torch.tensor([0, 1, 257, 513])

        model = NS(training=False, language_model=NS(model=object()),
                   _build_multimodal_attention=lambda *args: (torch.zeros(1, 2, 2), torch.ones(1, 2)),
                   _unnormalize_actions=lambda value, key: value.copy())
        head = NS(training=False, predict_action=lambda h: torch.zeros(8, 7))
        f.cfg.unnorm_key = "test"
        query = SpecPruneEpisodeQuery(model=model, head=head, proprio=NS(training=False),
                                     processor=None, cfg=f.cfg,
                                     utils=NS(prepare_images_for_vla=images, normalize_proprio=None),
                                     instruction_indexer=None, enabled=enabled)
        with patch("savr.openvla.official_semantics.prepare_semantic_query", semantic), \
             patch("savr.openvla.official_semantics.derive_official_layout", lambda **kw: object()), \
             patch("savr.openvla.official_semantics.select_official_action_hidden", lambda h, *args: h), \
             patch("savr.openvla.specprune.low_change_indices", low), \
             patch("savr.openvla.specprune.SpecPruneSelection", selection), \
             patch("savr.openvla.specprune.pruned_decoder_forward", forward):
            actions = query(obs, "task", previous, state)
        self.assertEqual(actions.shape, (8, 7))
        self.assertEqual(semantic_inputs, [21, 121])
        self.assertEqual(state.query, 1)
        self.assertEqual(query.last_query["previous_frame"], 7)
        self.assertEqual(selectors[0][2]["enabled"], enabled)
        return preparation_inputs, budgets

    def test_compressed_uses_current_encoder_and_historical_selection(self):
        inputs, budgets = self.exercise(True)
        self.assertEqual(inputs, [[20, 120], [17, 117]])
        self.assertEqual([(c, p) for c, p, kw in budgets], [(21, 18), (121, 118)])
        self.assertEqual([kw["top_k"] for c, p, kw in budgets], [236, 230])

    def test_precise_presets_follow_source_not_names(self):
        _, budgets = self.exercise(True, precise=True)
        self.assertEqual([kw["top_k"] for c, p, kw in budgets], [240, 236])

    def test_dense_omits_only_unneeded_historical_preprocessing(self):
        inputs, budgets = self.exercise(False)
        self.assertEqual(inputs, [[20, 120]])
        self.assertEqual(budgets, [])


if __name__ == "__main__":
    unittest.main()
