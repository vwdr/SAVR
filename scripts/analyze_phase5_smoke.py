#!/usr/bin/env python3
"""Reconcile the immutable Phase 5 core-smoke and VLA-Cache audit evidence."""

from __future__ import annotations

import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_ROOT = Path("/home/ved/SAVR")
CORE_RUN_ID = "phase5-core-smoke-v1"
VLA_CACHE_RUN_ID = "phase5-vla-cache-compatibility-v1"
ANALYSIS_RUN_ID = "phase5-analysis-v1"
SCHEDULE = (
    (0, "FR"),
    (1, "PR"),
    (2, "VOR"),
    (0, "SAVR"),
    (1, "FR"),
    (2, "PR"),
    (0, "VOR"),
    (1, "SAVR"),
    (2, "FR"),
    (0, "PR"),
    (1, "VOR"),
    (2, "SAVR"),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_once(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError(f"Analysis result is immutable and already exists: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def validate_episode_matrix(episodes: list[dict[str, Any]]) -> None:
    observed = [
        (int(record["initial_state_id"]), str(record["policy"]))
        for record in episodes
    ]
    if observed != list(SCHEDULE):
        raise RuntimeError(f"Episode order differs: expected={SCHEDULE}, observed={observed}")
    for record in episodes:
        if record["status"] != "completed":
            raise RuntimeError(f"Non-completed episode: {record['episode_id']}")
        if record["query_count"] < 1:
            raise RuntimeError(f"Episode has no queries: {record['episode_id']}")
        if record["refresh_count"] + record["reuse_count"] != record["query_count"]:
            raise RuntimeError(f"Episode count mismatch: {record['episode_id']}")
        if record["skipped_refresh_count"] != record["reuse_count"]:
            raise RuntimeError(f"Skipped count mismatch: {record['episode_id']}")
        expected_rate = record["refresh_count"] / record["query_count"]
        if not math.isclose(record["refresh_rate"], expected_rate):
            raise RuntimeError(f"Refresh-rate mismatch: {record['episode_id']}")
        digest = record["trajectory_sha256"]
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise RuntimeError(f"Invalid trajectory digest: {record['episode_id']}")


def validate_queries(
    queries: list[dict[str, Any]],
    episodes: list[dict[str, Any]],
) -> dict[str, Any]:
    if [record["query_index"] for record in queries] != list(range(len(queries))):
        raise RuntimeError("Global query indices are not contiguous")

    episode_queries: dict[str, list[dict[str, Any]]] = defaultdict(list)
    component_totals: Counter[str] = Counter()
    policy_query_wall_ms: dict[str, list[float]] = defaultdict(list)
    policy_total_cuda_ms: dict[str, list[float]] = defaultdict(list)
    policy_refresh: Counter[str] = Counter()
    policy_reuse: Counter[str] = Counter()
    trigger_totals: dict[str, Counter[str]] = defaultdict(Counter)

    for record in queries:
        if record["status"] != "completed":
            raise RuntimeError(f"Non-completed query: {record['query_index']}")
        if record["action_shape"] != [8, 7]:
            raise RuntimeError(f"Action-shape mismatch: {record['query_index']}")
        episode_queries[record["episode_id"]].append(record)
        decision = record["decision"]
        if decision["query_index"] != record["episode_query_index"]:
            raise RuntimeError(f"Episode query-index mismatch: {record['query_index']}")
        if decision["refresh"] != record["refresh"]:
            raise RuntimeError(f"Decision mismatch: {record['query_index']}")
        if decision["cache_age_before"] != record["cache_age_before"]:
            raise RuntimeError(f"Cache-age-before mismatch: {record['query_index']}")
        expected_event = "refresh" if record["refresh"] else "reuse"
        if record["cache_event"] != expected_event:
            raise RuntimeError(f"Cache-event mismatch: {record['query_index']}")
        if record["refresh"]:
            if record["cache_age_after"] != 0:
                raise RuntimeError(f"Refresh cache age differs: {record['query_index']}")
            policy_refresh[record["policy"]] += 1
        else:
            if record["cache_age_after"] != record["cache_age_before"] + 1:
                raise RuntimeError(f"Reuse cache age differs: {record['query_index']}")
            if record["cache_age_after"] > 2:
                raise RuntimeError(f"Reuse horizon exceeded: {record['query_index']}")
            policy_reuse[record["policy"]] += 1

        counts = record["timing"]["component_counts"]
        visual_expected = 1 if record["refresh"] else 0
        if counts["vision_backbone"] != visual_expected:
            raise RuntimeError(f"Vision count mismatch: {record['query_index']}")
        if counts["visual_projector"] != visual_expected:
            raise RuntimeError(f"Projector count mismatch: {record['query_index']}")
        if counts["language_model"] != 1 or counts["action_head"] != 1:
            raise RuntimeError(f"Downstream count mismatch: {record['query_index']}")
        component_totals.update(counts)

        timing = record["timing"]
        scalar_timings = (
            timing["decision_wall_ms"],
            timing["query_wall_ms"],
            timing["total_cuda_ms"],
            *timing["component_cuda_ms"].values(),
        )
        if any(not math.isfinite(float(value)) or float(value) < 0 for value in scalar_timings):
            raise RuntimeError(f"Invalid timing value: {record['query_index']}")
        policy_query_wall_ms[record["policy"]].append(timing["query_wall_ms"])
        policy_total_cuda_ms[record["policy"]].append(timing["total_cuda_ms"])
        trigger_totals[record["policy"]].update(decision["triggers"])

    for episode in episodes:
        records = episode_queries[episode["episode_id"]]
        if len(records) != episode["query_count"]:
            raise RuntimeError(f"Episode query-file mismatch: {episode['episode_id']}")
        if [record["episode_query_index"] for record in records] != list(
            range(len(records))
        ):
            raise RuntimeError(f"Episode query order differs: {episode['episode_id']}")
        refresh_count = sum(record["refresh"] for record in records)
        if refresh_count != episode["refresh_count"]:
            raise RuntimeError(f"Episode refresh mismatch: {episode['episode_id']}")
        query_triggers = Counter(
            trigger
            for record in records
            for trigger in record["decision"]["triggers"]
        )
        if dict(sorted(query_triggers.items())) != episode["trigger_counts"]:
            raise RuntimeError(f"Episode trigger mismatch: {episode['episode_id']}")

    return {
        "component_invocation_totals": dict(sorted(component_totals.items())),
        "policy_refresh_counts": dict(sorted(policy_refresh.items())),
        "policy_reuse_counts": dict(sorted(policy_reuse.items())),
        "policy_trigger_counts": {
            policy: dict(sorted(counts.items()))
            for policy, counts in sorted(trigger_totals.items())
        },
        "policy_query_wall_ms": {
            policy: {
                "count": len(values),
                "median": statistics.median(values),
                "p95": percentile(values, 95),
            }
            for policy, values in sorted(policy_query_wall_ms.items())
        },
        "policy_total_cuda_ms": {
            policy: {
                "count": len(values),
                "median": statistics.median(values),
                "p95": percentile(values, 95),
            }
            for policy, values in sorted(policy_total_cuda_ms.items())
        },
    }


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    if project_root != EXPECTED_ROOT:
        raise SystemExit(f"Refusing to run outside {EXPECTED_ROOT}: {project_root}")

    core_root = project_root / "results" / CORE_RUN_ID
    external_path = (
        project_root / "results" / VLA_CACHE_RUN_ID / "audit.json"
    )
    output_path = (
        project_root / "results" / ANALYSIS_RUN_ID / "analysis.json"
    )
    manifest = load_json(core_root / "manifest.json")
    summary = load_json(core_root / "run_summary.json")
    episodes = [
        load_json(path)
        for path in sorted((core_root / "episodes").glob("*.json"))
    ]
    queries = [
        load_json(path)
        for path in sorted((core_root / "queries").glob("*.json"))
    ]
    external = load_json(external_path)

    if manifest["status"] != "completed" or summary["status"] != "completed":
        raise RuntimeError("Core Phase 5 run is not completed")
    if manifest["run_id"] != CORE_RUN_ID or summary["run_id"] != CORE_RUN_ID:
        raise RuntimeError("Core run identity differs")
    if not summary["checkpoint_restored"]:
        raise RuntimeError("Checkpoint restoration did not pass")
    if summary["unexpected_new_checkpoint_files"]:
        raise RuntimeError("Unexpected checkpoint files remain")
    if len(episodes) != 12 or len(queries) != summary["query_record_count"]:
        raise RuntimeError("Terminal record counts differ from summary")

    validate_episode_matrix(episodes)
    query_analysis = validate_queries(queries, episodes)

    expected_counts = {
        "FR": {"queries": 31, "refreshes": 31, "reuses": 0, "successes": 3},
        "PR": {"queries": 84, "refreshes": 42, "reuses": 42, "successes": 0},
        "VOR": {"queries": 84, "refreshes": 30, "reuses": 54, "successes": 0},
        "SAVR": {"queries": 84, "refreshes": 30, "reuses": 54, "successes": 0},
    }
    policy_summary = {}
    for policy, expected in expected_counts.items():
        selected = [record for record in episodes if record["policy"] == policy]
        actual = {
            "episodes": len(selected),
            "queries": sum(record["query_count"] for record in selected),
            "refreshes": sum(record["refresh_count"] for record in selected),
            "reuses": sum(record["reuse_count"] for record in selected),
            "successes": sum(bool(record["success"]) for record in selected),
        }
        if actual["episodes"] != 3 or any(
            actual[key] != expected[key]
            for key in ("queries", "refreshes", "reuses", "successes")
        ):
            raise RuntimeError(f"Policy outcome differs for {policy}: {actual}")
        actual["refresh_rate"] = actual["refreshes"] / actual["queries"]
        policy_summary[policy] = actual

    if external["status"] != "TECHNICAL_EXCLUSION":
        raise RuntimeError("VLA-Cache audit lacks technical-exclusion status")
    required_external_findings = (
        "official_previous_frame_aliases_current_frame",
        "official_episode_errors_are_swallowed",
        "isolated_transformers_fork_loaded",
        "official_patch_utility_passed",
    )
    if not all(external["findings"].get(key) for key in required_external_findings):
        raise RuntimeError("VLA-Cache technical evidence is incomplete")
    if external["findings"]["gpu_episode_executed"]:
        raise RuntimeError("VLA-Cache GPU episode should not have executed")

    result = {
        "analysis_run_id": ANALYSIS_RUN_ID,
        "created_at_utc": utc_now(),
        "status": "PASS",
        "core_run_id": CORE_RUN_ID,
        "external_run_id": VLA_CACHE_RUN_ID,
        "claim_boundary": (
            "Diagnostic Phase 5 feasibility only; no calibrated policy, "
            "non-inferiority, latency, or paper-level comparative claim."
        ),
        "core": {
            "episode_count": len(episodes),
            "query_count": len(queries),
            "success_count": sum(bool(record["success"]) for record in episodes),
            "policy_summary": policy_summary,
            "query_analysis": query_analysis,
            "elapsed_seconds": summary["elapsed_seconds"],
            "artifact_bytes": summary["artifact_bytes"],
            "peak_gpu_memory_allocated_bytes": summary[
                "peak_gpu_memory_allocated_bytes"
            ],
            "checkpoint_restored": summary["checkpoint_restored"],
        },
        "vla_cache": {
            "status": external["status"],
            "revisions": external["revisions"],
            "findings": external["findings"],
            "exclusion_reasons": external["exclusion_reasons"],
        },
        "interpretation": [
            "All four project-owned policies completed all three diagnostic trajectories.",
            "Every refresh/reuse, trigger, cache-age, component-count, and immutable-record invariant reconciled.",
            "FR succeeded on all three states; PR, VOR, and SAVR did not succeed under deliberately aggressive uncalibrated diagnostic reuse.",
            "The three-state smoke is insufficient for comparative inference; Phase 6 calibration remains required.",
            "The pinned official VLA-Cache evaluator is technically excluded until its previous-frame and error-reporting semantics receive a reviewed correction.",
        ],
    }
    write_once(output_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
