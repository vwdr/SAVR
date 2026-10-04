"""CPU tests of actual attention intervention and released episode ordering."""

import ast
from collections import deque
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from savr.openvla.attention_diagnostic import LlamaAttentionDiagnostic, execute_episode

try:
    import torch
    from transformers import LlamaConfig, LlamaModel
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "requires project Torch/Transformers runtime")
class InterventionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        config = LlamaConfig(vocab_size=128, hidden_size=32, intermediate_size=64,
                             num_hidden_layers=2, num_attention_heads=4,
                             num_key_value_heads=4, max_position_embeddings=32)
        config._attn_implementation = "sdpa"
        self.model = LlamaModel(config).eval()
        self.ids = torch.tensor([[1, 5, 9, 13, 17, 21]])
        self.changed = self.ids.clone()
        self.changed[0, -1] = 85

    def forward(self, ids, mask, mode=None):
        with torch.inference_mode():
            if mode is None:
                return self.model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
            with LlamaAttentionDiagnostic(torch, self.model.layers, mode):
                return self.model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state

    def test_native_observer_does_not_change_output(self):
        mask = torch.ones_like(self.ids)
        self.assertTrue(torch.equal(self.forward(self.ids, mask), self.forward(self.ids, mask, "original")))

    def test_causal_control_changes_information_flow(self):
        mask = torch.ones_like(self.ids)
        original = self.forward(self.ids, mask, "original")
        future = self.forward(self.changed, mask, "original")
        causal = self.forward(self.ids, mask, "causal_control")
        causal_future = self.forward(self.changed, mask, "causal_control")
        self.assertGreater(float((original[:, :-1] - future[:, :-1]).abs().max()), 1e-6)
        self.assertTrue(torch.equal(causal[:, :-1], causal_future[:, :-1]))
        self.assertFalse(torch.equal(original, causal))

    def test_padding_is_excluded_in_both_modes(self):
        mask = torch.tensor([[1, 1, 1, 1, 1, 0]])
        for mode in ("original", "causal_control"):
            a = self.forward(self.ids, mask, mode)
            b = self.forward(self.changed, mask, mode)
            self.assertTrue(torch.equal(a[:, :-1], b[:, :-1]))

    def test_restoration_after_control(self):
        mask = torch.ones_like(self.ids)
        original_sdpa = torch.nn.functional.scaled_dot_product_attention
        before = self.forward(self.ids, mask)
        self.forward(self.ids, mask, "causal_control")
        self.assertIs(torch.nn.functional.scaled_dot_product_attention, original_sdpa)
        self.assertTrue(torch.equal(before, self.forward(self.ids, mask)))
        self.assertTrue(all("forward" not in layer.self_attn.__dict__ for layer in self.model.layers))

    def test_restoration_after_exception(self):
        original_sdpa = torch.nn.functional.scaled_dot_product_attention
        with self.assertRaisesRegex(RuntimeError, "intentional"):
            with LlamaAttentionDiagnostic(torch, self.model.layers, "causal_control"):
                raise RuntimeError("intentional")
        self.assertIs(torch.nn.functional.scaled_dot_product_attention, original_sdpa)
        self.assertTrue(all("forward" not in layer.self_attn.__dict__ for layer in self.model.layers))

    def test_non_llama_attention_is_unchanged(self):
        value = torch.randn(1, 2, 6, 8)
        expected = torch.nn.functional.scaled_dot_product_attention(value, value, value)
        with LlamaAttentionDiagnostic(torch, self.model.layers, "causal_control"):
            actual = torch.nn.functional.scaled_dot_product_attention(value, value, value)
            self.model(input_ids=self.ids, attention_mask=torch.ones_like(self.ids), use_cache=False)
        self.assertTrue(torch.equal(expected, actual))

    def test_attention_output_fallback_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "fallback"):
            with LlamaAttentionDiagnostic(torch, self.model.layers, "original"):
                self.model(input_ids=self.ids, output_attentions=True, use_cache=False)


class EpisodeTests(unittest.TestCase):
    def fixture(self, done_at=21, fail_at=None):
        trace = []

        class FakeEnv:
            step_count = 0

            def reset(self):
                trace.append("reset")
                self.step_count = 0

            def set_init_state(self, value):
                trace.append(("state", value.tolist()))
                return 0

            def step(self, action):
                self.step_count += 1
                if self.step_count == fail_at:
                    raise RuntimeError("simulator error")
                trace.append(("step", action))
                return self.step_count, 0, self.step_count == done_at, {}

        def prepare(obs, size):
            trace.append(("prepare", obs, size))
            return obs, None

        def query(obs, text):
            trace.append(("query", obs, text))
            return np.arange(56, dtype=np.float32).reshape(8, 7)

        def process(action, family):
            trace.append(("process", action.tolist(), family))
            return action

        evaluation = SimpleNamespace(TASK_MAX_STEPS={"test": 17},
                                     prepare_observation=prepare, process_action=process,
                                     get_libero_dummy_action=lambda family: [0] * 7)
        cfg = SimpleNamespace(num_open_loop_steps=8, num_steps_wait=10,
                              model_family="openvla", task_suite_name="test", use_film=False)
        return trace, FakeEnv(), evaluation, cfg, query

    def test_queue_and_terminal_accounting(self):
        _, env, evaluation, cfg, query = self.fixture()
        result = execute_episode(evaluation, cfg, env, np.array([3]), "task", query, 224)
        self.assertEqual(result, {"success": True, "executed_steps": 11, "policy_queries": 2})

    def test_horizon_exhaustion_is_a_task_failure(self):
        _, env, evaluation, cfg, query = self.fixture(done_at=999)
        result = execute_episode(evaluation, cfg, env, np.array([3]), "task", query, 224)
        self.assertEqual(result, {"success": False, "executed_steps": 17, "policy_queries": 3})

    def test_technical_error_is_not_a_task_failure(self):
        _, env, evaluation, cfg, query = self.fixture(fail_at=12)
        with self.assertRaisesRegex(RuntimeError, "simulator error"):
            execute_episode(evaluation, cfg, env, np.array([3]), "task", query, 224)

    def test_exact_trace_matches_released_episode(self):
        path = Path("/home/ved/SAVR/third_party/openvla-oft/experiments/robot/libero/run_libero_eval.py")
        if not path.exists():
            self.skipTest("requires pinned original evaluator source on TITAN")
        node = next(n for n in ast.parse(path.read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == "run_episode")
        for done_at in (21, 999):
            own_trace, own_env, evaluation, cfg, query = self.fixture(done_at)
            own = execute_episode(evaluation, cfg, own_env, np.array([3]), "task", query, 224)
            ref_trace, ref_env, ref_eval, ref_cfg, ref_query = self.fixture(done_at)
            namespace = {"GenerateConfig": object, "deque": deque, "NUM_ACTIONS_CHUNK": 8,
                         "TASK_MAX_STEPS": ref_eval.TASK_MAX_STEPS,
                         "get_libero_dummy_action": ref_eval.get_libero_dummy_action,
                         "prepare_observation": ref_eval.prepare_observation,
                         "process_action": ref_eval.process_action,
                         "get_action": lambda cfg, model, obs, task, **kw: ref_query(obs, task),
                         "log_message": lambda *a: self.fail("released evaluator caught an error")}
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
            success, _ = namespace["run_episode"](ref_cfg, ref_env, "task", None, 224,
                                                  initial_state=np.array([3]))
            self.assertEqual(own["success"], success)
            self.assertEqual(own_trace, ref_trace)


if __name__ == "__main__":
    unittest.main()
