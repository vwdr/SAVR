#!/usr/bin/env python3
"""Reconcile the frozen Phase 6S-D validation and apply its positive gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


CONFIGURATION_ID = "savr3-rv-w375-b15"
EXPECTED_STATES = frozenset(range(3, 10))
EXPECTED_TASKS = frozenset(range(10))
EXPECTED_EPISODES = 70
MINIMUM_SKIP_RATE = 0.05
MAXIMUM_PREFIX_SKIP = 0.15


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def analyze(run_dir: Path) -> dict[str, Any]:
    manifest = load_json(run_dir / "manifest.json")
    summary = load_json(run_dir / "run_summary.json")
    episodes = [load_json(path) for path in sorted((run_dir / "episodes").glob("*.json"))]
    queries = [load_json(path) for path in sorted((run_dir / "queries").glob("*.json"))]
    expected_pairs = {(task, state) for task in EXPECTED_TASKS for state in EXPECTED_STATES}
    observed_pairs = {
        (int(str(record["task"]).split(":")[-1]), int(record["initial_state_id"]))
        for record in episodes
    }
    query_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in queries:
        query_groups[str(record["episode_id"])].append(record)
    for records in query_groups.values():
        records.sort(key=lambda item: int(item["episode_query_index"]))

    invariant_errors: list[str] = []
    if manifest.get("status") != "completed" or manifest.get("policy") != "SAVR3":
        invariant_errors.append("manifest_status_or_policy")
    if summary.get("status") != "completed" or not summary.get("checkpoint_restored"):
        invariant_errors.append("summary_status_or_checkpoint")
    if summary.get("unexpected_new_checkpoint_files"):
        invariant_errors.append("unexpected_checkpoint_files")
    if len(episodes) != EXPECTED_EPISODES or observed_pairs != expected_pairs:
        invariant_errors.append("episode_matrix")
    if any(record.get("configuration_id") != CONFIGURATION_ID for record in episodes):
        invariant_errors.append("configuration_identity")
    if any(record.get("status") != "completed" for record in episodes):
        invariant_errors.append("terminal_status")
    if sum(int(record.get("query_count", 0)) for record in episodes) != len(queries):
        invariant_errors.append("query_count_reconciliation")

    reuse_count = 0
    refresh_count = 0
    visual_calls = 0
    projector_calls = 0
    for episode in episodes:
        episode_id = str(episode["episode_id"])
        records = query_groups.get(episode_id, [])
        if len(records) != int(episode.get("query_count", -1)):
            invariant_errors.append(f"episode_query_count:{episode_id}")
            continue
        episode_reuses = 0
        previous_reuse = False
        for index, record in enumerate(records):
            decision = record.get("decision", {})
            reuse = not bool(record.get("refresh"))
            counts = record.get("timing", {}).get("component_counts", {})
            expected_visual = 0 if reuse else 1
            if (
                int(counts.get("vision_backbone", -1)) != expected_visual
                or int(counts.get("visual_projector", -1)) != expected_visual
                or int(counts.get("language_model", -1)) != 1
                or int(counts.get("action_head", -1)) != 1
            ):
                invariant_errors.append(f"component_counts:{episode_id}:{index}")
            visual_calls += int(counts.get("vision_backbone", 0))
            projector_calls += int(counts.get("visual_projector", 0))
            if decision.get("policy") != "SAVR3":
                invariant_errors.append(f"decision_policy:{episode_id}:{index}")
            if reuse:
                reuse_count += 1
                episode_reuses += 1
                if index < 5 or previous_reuse:
                    invariant_errors.append(f"temporal_rule:{episode_id}:{index}")
                if any(decision.get("translation_direction_reversals", [])):
                    invariant_errors.append(f"reversal_reuse:{episode_id}:{index}")
            else:
                refresh_count += 1
            if episode_reuses / (index + 1) > MAXIMUM_PREFIX_SKIP + 1e-15:
                invariant_errors.append(f"prefix_budget:{episode_id}:{index}")
            previous_reuse = reuse

    total_queries = len(queries)
    skip_rate = reuse_count / total_queries if total_queries else 0.0
    successes = sum(bool(record.get("success")) for record in episodes)
    task_successes = Counter(
        int(str(record["task"]).split(":")[-1])
        for record in episodes
        if record.get("success")
    )
    gates = {
        "terminal_episodes_70": len(episodes) == EXPECTED_EPISODES,
        "successes_70": successes == EXPECTED_EPISODES,
        "each_task_7_of_7": all(task_successes[task] == 7 for task in EXPECTED_TASKS),
        "skip_rate_at_least_5_percent": skip_rate >= MINIMUM_SKIP_RATE,
        "zero_invariant_errors": not invariant_errors,
        "exact_visual_call_reduction": (
            visual_calls == refresh_count
            and projector_calls == refresh_count
            and total_queries - visual_calls == reuse_count
            and total_queries - projector_calls == reuse_count
        ),
    }
    result: dict[str, Any] = {
        "run_id": manifest.get("run_id"),
        "configuration_id": CONFIGURATION_ID,
        "episodes": len(episodes),
        "successes": successes,
        "task_successes": {str(task): task_successes[task] for task in sorted(EXPECTED_TASKS)},
        "queries": total_queries,
        "refreshes": refresh_count,
        "reuses": reuse_count,
        "skip_rate": skip_rate,
        "vision_backbone_calls": visual_calls,
        "visual_projector_calls": projector_calls,
        "invariant_errors": sorted(set(invariant_errors)),
        "gates": gates,
        "positive_method_result": all(gates.values()),
    }
    semantic = json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False)
    result["analysis_sha256"] = hashlib.sha256(semantic.encode()).hexdigest()
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.run_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["positive_method_result"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
