#!/usr/bin/env python3
"""Validate and summarize the bounded Phase 2B Full Refresh pilot."""

from __future__ import annotations

import json
import statistics
from datetime import datetime
from pathlib import Path


EXPECTED_ROOT = Path("/home/ved/SAVR")
RUN_ID = "phase2b-fr-spatial-pilot-v1"
EXPECTED_PAIRS = {(task_id, state_id) for task_id in range(10) for state_id in range(5)}
SUCCESS_THRESHOLD = 45


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def summary(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("Cannot summarize an empty series")
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p95": percentile(values, 0.95),
        "min": min(values),
        "max": max(values),
    }


def parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value)


def main() -> int:
    root = Path.cwd().resolve()
    if root != EXPECTED_ROOT:
        raise SystemExit(f"Run only from {EXPECTED_ROOT}; current directory is {root}")

    run_dir = root / "results" / RUN_ID
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    episode_paths = sorted((run_dir / "episodes").glob("*.json"))
    episodes = [json.loads(path.read_text(encoding="utf-8")) for path in episode_paths]

    observed_pairs = {
        (int(episode["task_id"]), int(episode["initial_state_id"]))
        for episode in episodes
    }
    if len(episodes) != 50 or observed_pairs != EXPECTED_PAIRS:
        raise SystemExit(
            f"Invalid episode matrix: count={len(episodes)}, "
            f"missing={sorted(EXPECTED_PAIRS - observed_pairs)}, "
            f"extra={sorted(observed_pairs - EXPECTED_PAIRS)}"
        )
    if any(episode["status"] != "COMPLETED" for episode in episodes):
        raise SystemExit("Every episode must have terminal status COMPLETED")
    if not manifest.get("checkpoint_restored"):
        raise SystemExit("Checkpoint restoration was not confirmed")
    if manifest["checkpoint_hashes_before"] != manifest["checkpoint_hashes_after_restore"]:
        raise SystemExit("Checkpoint metadata hashes differ after restoration")

    queries = [query for episode in episodes for query in episode["queries"]]
    warmup_queries = [query for query in queries if query["warmup"]]
    steady_queries = [query for query in queries if not query["warmup"]]
    if len(warmup_queries) != 3:
        raise SystemExit(f"Expected exactly 3 warmup queries, found {len(warmup_queries)}")
    if any(query["action_chunk_length"] != 8 for query in queries):
        raise SystemExit("Every action query must return the frozen chunk length of 8")

    metric_fields = {
        "preprocessing_wall_ms": [
            sample["wall_seconds"] * 1000
            for episode in episodes
            for sample in episode["preprocessing"]
        ],
        "policy_query_wall_ms": [
            query["wall_seconds"] * 1000 for query in steady_queries
        ],
        "total_cuda_ms": [query["total_cuda_ms"] for query in steady_queries],
        "vision_backbone_cuda_ms": [
            query["vision_backbone_cuda_ms"] for query in steady_queries
        ],
        "visual_projector_cuda_ms": [
            query["visual_projector_cuda_ms"] for query in steady_queries
        ],
        "visual_combined_cuda_ms": [
            query["vision_backbone_cuda_ms"] + query["visual_projector_cuda_ms"]
            for query in steady_queries
        ],
        "action_head_cuda_ms": [
            query["action_head_cuda_ms"] for query in steady_queries
        ],
        "downstream_residual_cuda_ms": [
            query["downstream_residual_cuda_ms"] for query in steady_queries
        ],
    }
    timing = {name: summary(values) for name, values in metric_fields.items()}

    per_task = {}
    for task_id in range(10):
        task_episodes = [episode for episode in episodes if episode["task_id"] == task_id]
        successes = sum(bool(episode["success"]) for episode in task_episodes)
        per_task[str(task_id)] = {
            "successes": successes,
            "episodes": len(task_episodes),
            "success_rate": successes / len(task_episodes),
        }

    success_count = sum(bool(episode["success"]) for episode in episodes)
    total_cuda = sum(metric_fields["total_cuda_ms"])
    visual_cuda = sum(metric_fields["visual_combined_cuda_ms"])
    query_wall = sum(metric_fields["policy_query_wall_ms"])
    run_seconds = (
        parse_utc(manifest["completed_at_utc"])
        - parse_utc(manifest["started_at_utc"])
    ).total_seconds()
    threshold_passed = (
        success_count >= SUCCESS_THRESHOLD
        and all(item["successes"] > 0 for item in per_task.values())
    )

    result = {
        "run_id": RUN_ID,
        "analysis_contract": {
            "expected_episodes": 50,
            "expected_tasks": list(range(10)),
            "expected_initial_states": list(range(5)),
            "success_threshold": SUCCESS_THRESHOLD,
            "minimum_successes_per_task": 1,
            "warmup_queries_excluded_from_query_timing": 3,
        },
        "integrity": {
            "manifest_status": manifest["status"],
            "episode_matrix_complete": observed_pairs == EXPECTED_PAIRS,
            "terminal_episode_count": len(episodes),
            "runtime_error_count": manifest["error_episode_count"],
            "checkpoint_restored_byte_for_byte": True,
        },
        "outcomes": {
            "successes": success_count,
            "episodes": len(episodes),
            "success_rate": success_count / len(episodes),
            "threshold_passed": threshold_passed,
            "per_task": per_task,
            "failures": [
                {
                    "task_id": episode["task_id"],
                    "initial_state_id": episode["initial_state_id"],
                    "video_saved": episode["video_saved"],
                    "video_path": episode["video_path"],
                }
                for episode in episodes
                if not episode["success"]
            ],
        },
        "timing": {
            "unit": "milliseconds",
            "total_query_count": len(queries),
            "steady_query_count": len(steady_queries),
            "metrics": timing,
            "visual_share_of_total_cuda": visual_cuda / total_cuda,
            "visual_share_of_policy_query_wall": visual_cuda / query_wall,
            "run_wall_seconds": run_seconds,
            "episode_wall_seconds": summary(
                [episode["episode_seconds"] for episode in episodes]
            ),
        },
        "resources": {
            "physical_gpu_id": manifest["physical_gpu_id"],
            "visible_gpu_name": manifest["visible_gpu_name"],
            "peak_gpu_memory_allocated_bytes": manifest[
                "peak_gpu_memory_allocated_bytes"
            ],
            "peak_gpu_memory_reserved_bytes": manifest["peak_gpu_memory_reserved_bytes"],
            "artifact_bytes": manifest["artifact_bytes"],
            "model_load_seconds": manifest["model_load_seconds"],
        },
        "provenance": {
            "project_commit": manifest["project_commit"],
            "upstream_commit": manifest["frozen_config"]["upstream_commit"],
            "checkpoint_revision": manifest["frozen_config"]["checkpoint_revision"],
        },
    }

    output_path = root / "reports" / "runtime" / "phase2b_aggregate.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if threshold_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
