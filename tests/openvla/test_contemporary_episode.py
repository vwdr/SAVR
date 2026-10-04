import unittest
import numpy as np
import test_specprune_episode as qualified
from savr.openvla.contemporary_episode import execute_measured_episode
from savr.openvla.specprune import EpisodeState


class MeasuredLoopTests(unittest.TestCase):
    def run_new(self, f, controller=False, state=None, query=None):
        def wrapper(*args):
            raw = (query or f.bridge)(*args)
            return raw, dict(retained_visual_tokens=512, precise=args[-1].precise)
        return execute_measured_episode(f.evaluation, f.cfg, f.env, np.array([3]), "task", wrapper, 224,
            episode_id="same-id", controller=controller, synchronize=lambda: None, state=state)

    def test_same_trace_and_actions_as_qualified_loop(self):
        helper = qualified.EpisodeBridgeTests()
        for controller in (False, True):
            for done in (21, 30, None):
                a, b = helper.fixture(done_at=done), helper.fixture(done_at=done)
                old = helper.run_bridge(a, controller=controller)
                new = self.run_new(b, controller)
                self.assertEqual(a.trace, b.trace)
                self.assertEqual(old, {k: new[k] for k in old})
                self.assertEqual(8 * new["policy_queries"], new["executed_steps"] + new["discarded_actions"] + new["remaining_actions"])
                self.assertLessEqual(sum(q["seconds"] for q in new["queries"]), new["episode_seconds"])

    def test_rounding_before_execution_and_repeated_reset(self):
        state = EpisodeState()
        for _ in range(2):
            f = qualified.EpisodeBridgeTests().fixture(horizon=1)
            def raw(*args):
                f.bridge(*args)
                return np.full((8, 7), .1234567890123, np.float64)
            self.run_new(f, state=state, query=raw)
            self.assertEqual(state.query, 1)
            self.assertEqual(f.trace[-1][1][0], float(np.float32(.1234567890123)))
            state.query = 999

    def test_replanning_accounting(self):
        f = qualified.EpisodeBridgeTests().fixture(horizon=16)
        def query(*args):
            a = f.bridge(*args)
            if args[-1].query == 2: a[3] = 0
            return a
        result = self.run_new(f, True, query=query)
        self.assertEqual((result["replans"], result["discarded_actions"], result["policy_queries"]), (1, 4, 3))
        self.assertEqual(result["remaining_actions"], 4)

    def test_technical_fault_is_not_task_failure(self):
        f = qualified.EpisodeBridgeTests().fixture(fail_at=12)
        with self.assertRaises(RuntimeError): self.run_new(f)


if __name__ == "__main__": unittest.main()
