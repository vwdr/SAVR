from __future__ import annotations

import importlib.util
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import numpy as np
except ImportError:  # pragma: no cover - exercised by dependency-free CI
    np = None  # type: ignore[assignment]


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "run_phase4_correctness",
    ROOT / "scripts" / "run_phase4_correctness.py",
)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)
JSONSCHEMA_AVAILABLE = importlib.util.find_spec("jsonschema") is not None
NUMPY_AVAILABLE = np is not None


@dataclass
class FakeTiming:
    wall_ms: float = 1.0
    total_device_ms: float = 0.5
    component_device_ms: dict[str, float] | None = None
    component_counts: dict[str, int] | None = None

    def __post_init__(self) -> None:
        self.component_device_ms = self.component_device_ms or {}
        self.component_counts = self.component_counts or {}


class Phase4RunnerTests(unittest.TestCase):
    @unittest.skipUnless(
        NUMPY_AVAILABLE and JSONSCHEMA_AVAILABLE,
        "optional runtime validation dependencies are unavailable",
    )
    def test_all_record_schemas_pass_runtime_preflight(self) -> None:
        assert np is not None
        jsonschema = importlib.import_module("jsonschema")
        validators = importlib.import_module("jsonschema.validators")
        schemas = RUNNER.validate_schemas(
            ROOT,
            jsonschema.Draft202012Validator,
            validators.validate,
        )
        self.assertEqual(
            set(schemas),
            {
                "episode_result.schema.json",
                "query_record.schema.json",
                "run_manifest.schema.json",
            },
        )

    @unittest.skipUnless(NUMPY_AVAILABLE, "numpy is not installed")
    def test_state_b_is_deterministic_finite_in_range_and_distinct(self) -> None:
        assert np is not None
        state_a = np.zeros(8, dtype=np.float64)
        statistics = {
            "q01": [-1.0] * 8,
            "q99": [1.0] * 8,
        }
        first, first_record = RUNNER.make_state_b(state_a, statistics, np)
        second, second_record = RUNNER.make_state_b(state_a, statistics, np)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first_record, second_record)
        self.assertTrue(np.isfinite(first).all())
        self.assertFalse(np.array_equal(first, state_a))
        self.assertGreaterEqual(first[0], -1.0)
        self.assertLessEqual(first[0], 1.0)
        np.testing.assert_array_equal(first[1:], state_a[1:])

    @unittest.skipUnless(NUMPY_AVAILABLE, "numpy is not installed")
    def test_exact_parity_rejects_any_difference(self) -> None:
        assert np is not None
        reference = np.zeros((8, 7), dtype=np.float32)
        self.assertTrue(RUNNER.exact_parity(reference, reference.copy(), np)["array_equal"])
        changed = reference.copy()
        changed[0, 0] = np.nextafter(np.float32(0), np.float32(1))
        with self.assertRaisesRegex(RuntimeError, "Exact action parity failed"):
            RUNNER.exact_parity(reference, changed, np)

    @unittest.skipUnless(NUMPY_AVAILABLE, "numpy is not installed")
    def test_query_record_requires_eight_by_seven_finite_actions(self) -> None:
        assert np is not None
        timing: Any = FakeTiming()
        record = RUNNER.query_record(
            index=1,
            path="test",
            refresh=True,
            cache_event="unmodified",
            actions=np.zeros((8, 7), dtype=np.float32),
            timing=timing,
            decision_wall_ms=0,
            np=np,
            extra={},
        )
        self.assertEqual(record["action_shape"], [8, 7])
        self.assertEqual(
            record["timing"]["component_counts"]["vision_backbone"],
            0,
        )
        with self.assertRaisesRegex(RuntimeError, "Unexpected action shape"):
            RUNNER.query_record(
                index=2,
                path="test",
                refresh=True,
                cache_event="unmodified",
                actions=np.zeros((7,), dtype=np.float32),
                timing=timing,
                decision_wall_ms=0,
                np=np,
                extra={},
            )


if __name__ == "__main__":
    unittest.main()
