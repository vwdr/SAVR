from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from savr.timing import SynchronizedQueryTimer  # noqa: E402


class FakeBackend:
    def __init__(self):
        self.calls = []
        self.next_event = 0

    def synchronize(self):
        self.calls.append("synchronize")

    def record_event(self):
        event = self.next_event
        self.next_event += 1
        self.calls.append(("event", event))
        return event

    def elapsed_ms(self, start, end):
        self.calls.append(("elapsed", start, end))
        return float((end - start) * 2)


class TimingTests(unittest.TestCase):
    def test_query_boundaries_are_synchronized_and_components_counted(self) -> None:
        backend = FakeBackend()
        wall_values = iter([10.0, 10.025])
        timer = SynchronizedQueryTimer(backend, wall_clock=lambda: next(wall_values))
        timer.start()
        timer.start_component("vision")
        timer.stop_component("vision")
        result = timer.finish()

        self.assertEqual(backend.calls[0], "synchronize")
        self.assertEqual(backend.calls[5], "synchronize")
        self.assertAlmostEqual(result.wall_ms, 25.0)
        self.assertEqual(result.total_device_ms, 6.0)
        self.assertEqual(result.component_device_ms["vision"], 2.0)
        self.assertEqual(result.component_counts["vision"], 1)

    def test_reuse_can_have_zero_visual_component_calls(self) -> None:
        backend = FakeBackend()
        wall_values = iter([1.0, 1.001])
        timer = SynchronizedQueryTimer(backend, wall_clock=lambda: next(wall_values))
        timer.start()
        result = timer.finish()
        self.assertEqual(result.component_counts, {})
        self.assertEqual(result.component_device_ms, {})

    def test_unbalanced_component_is_rejected(self) -> None:
        backend = FakeBackend()
        timer = SynchronizedQueryTimer(backend)
        timer.start()
        timer.start_component("projector")
        with self.assertRaisesRegex(RuntimeError, "remains open"):
            timer.finish()


if __name__ == "__main__":
    unittest.main()
