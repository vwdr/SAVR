from __future__ import annotations

import importlib.util
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "run_phase5_core_smoke",
    ROOT / "scripts" / "run_phase5_core_smoke.py",
)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


@dataclass
class FakeDecision:
    policy: str
    query_index: int
    refresh: bool
    cache_age_before: int
    triggers: tuple[str, ...]
    image_score: float | None = None
    per_camera_image_scores: dict[str, float] | None = None
    state_score: float | None = None
    action_score: float | None = None
    thresholds: dict[str, float | int] | None = None


class Phase5RunnerTests(unittest.TestCase):
    def test_schedule_is_exact_complete_and_counterbalanced(self) -> None:
        self.assertEqual(len(RUNNER.SCHEDULE), 12)
        self.assertEqual(len(set(RUNNER.SCHEDULE)), 12)
        for state_id in (0, 1, 2):
            policies = {
                policy
                for scheduled_state, policy in RUNNER.SCHEDULE
                if scheduled_state == state_id
            }
            self.assertEqual(policies, {"FR", "PR", "VOR", "SAVR"})

    def test_diagnostic_refresh_trajectories_are_explicit(self) -> None:
        self.assertTrue(
            RUNNER.expected_refresh(
                policy="FR",
                episode_query_index=7,
                cache_age_before=0,
            )
        )
        self.assertEqual(
            [
                RUNNER.expected_refresh(
                    policy="PR",
                    episode_query_index=index,
                    cache_age_before=age,
                )
                for index, age in enumerate((0, 0, 1, 0, 1))
            ],
            [True, False, True, False, True],
        )
        self.assertEqual(
            [
                RUNNER.expected_refresh(
                    policy="VOR",
                    episode_query_index=index,
                    cache_age_before=age,
                )
                for index, age in enumerate((0, 0, 1, 2, 0))
            ],
            [True, False, False, True, False],
        )
        self.assertEqual(
            [
                RUNNER.expected_refresh(
                    policy="SAVR",
                    episode_query_index=index,
                    cache_age_before=age,
                )
                for index, age in enumerate((0, 0, 0, 1, 2))
            ],
            [True, True, False, False, True],
        )

    def test_query_invariants_distinguish_refresh_and_reuse_counts(self) -> None:
        refresh_result = SimpleNamespace(
            decision=FakeDecision(
                policy="VOR",
                query_index=0,
                refresh=True,
                cache_age_before=0,
                triggers=("empty_cache",),
            ),
            cache_event="refresh",
        )
        refresh_timing = SimpleNamespace(
            component_counts={
                "vision_backbone": 1,
                "visual_projector": 1,
                "language_model": 1,
                "action_head": 1,
            }
        )
        RUNNER.assert_query_invariants(
            policy="VOR",
            episode_query_index=0,
            result=refresh_result,
            timing=refresh_timing,
        )

        reuse_result = SimpleNamespace(
            decision=FakeDecision(
                policy="VOR",
                query_index=1,
                refresh=False,
                cache_age_before=0,
                triggers=(),
            ),
            cache_event="reuse",
        )
        reuse_timing = SimpleNamespace(
            component_counts={
                "vision_backbone": 0,
                "visual_projector": 0,
                "language_model": 1,
                "action_head": 1,
            }
        )
        RUNNER.assert_query_invariants(
            policy="VOR",
            episode_query_index=1,
            result=reuse_result,
            timing=reuse_timing,
        )
        reuse_timing.component_counts["vision_backbone"] = 1
        with self.assertRaisesRegex(RuntimeError, "Vision count differs"):
            RUNNER.assert_query_invariants(
                policy="VOR",
                episode_query_index=1,
                result=reuse_result,
                timing=reuse_timing,
            )

    def test_complete_matrix_reconciles_episode_and_query_counts(self) -> None:
        records: list[dict[str, Any]] = []
        for state_id, policy in RUNNER.SCHEDULE:
            records.append(
                {
                    "initial_state_id": state_id,
                    "policy": policy,
                    "status": "completed",
                    "success": policy == "FR",
                    "query_count": 3,
                    "refresh_count": 2,
                    "reuse_count": 1,
                }
            )
        summary = RUNNER.validate_complete_matrix(records, 36)
        self.assertEqual(summary["terminal_episode_count"], 12)
        self.assertEqual(summary["query_record_count"], 36)
        self.assertEqual(summary["policy_successes"]["FR"], 3)

        records[0]["query_count"] = 4
        with self.assertRaisesRegex(RuntimeError, "Query record mismatch"):
            RUNNER.validate_complete_matrix(records, 36)


if __name__ == "__main__":
    unittest.main()
