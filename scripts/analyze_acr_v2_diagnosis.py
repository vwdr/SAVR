#!/usr/bin/env python3
"""Reconcile immutable A4/A5 evidence for the ACR Version 2 redesign."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


FR_RUN = "acr-a4-upstream-fr-object-dev00-09-v01"
A5_ANALYSIS = "acr-a5-stage1-analysis-v01/record/record.json"
CANDIDATE_RUNS = {
    "acr-t25-h2-b30": "acr-a5-sa-acr-object-stage1-acr-t25-h2-b30-v01",
    "acr-t50-h4-b55": "acr-a5-sa-acr-object-stage1-acr-t50-h4-b55-v01",
    "acr-t70-h8-b75": "acr-a5-sa-acr-object-stage1-acr-t70-h8-b75-v01",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def byte_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def semantic_record(payload: dict[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result["semantic_sha256"] = hashlib.sha256(canonical_bytes(payload)).hexdigest()
    return result


def episode_paths(run_root: Path) -> list[Path]:
    return sorted(run_root.rglob("episode/record.json"))


def query_records(episode_path: Path) -> list[dict[str, Any]]:
    attempt_root = episode_path.parent.parent
    return [load_json(path) for path in sorted(attempt_root.glob("query-*/record.json"))]


def first_reuse_summary(queries: list[dict[str, Any]]) -> dict[str, Any] | None:
    reuse = [query for query in queries if not query["decision"]["scene_refresh"]]
    if not reuse:
        return None
    first = reuse[0]
    decision, inputs = first["decision"], first["inputs"]
    return {
        "query_index": first["query_index"],
        "direction_reversal": bool(inputs["direction_reversal"]),
        "scene_threshold_ratio": decision["scene_score"] / decision["scene_threshold"],
        "translation_threshold_ratio": (
            decision["translation_score"] / decision["translation_threshold"]
        ),
        "reuse_query_indices": [query["query_index"] for query in reuse],
        "direction_reversal_reuses": sum(
            bool(query["inputs"]["direction_reversal"]) for query in reuse
        ),
    }


def first_action_divergence(
    candidate_queries: list[dict[str, Any]], fr_queries: list[dict[str, Any]]
) -> dict[str, Any]:
    comparable = min(len(candidate_queries), len(fr_queries))
    first_reuse = next(
        query["query_index"]
        for query in candidate_queries
        if not query["decision"]["scene_refresh"]
    )
    mismatch = None
    for index in range(comparable):
        candidate_hash = candidate_queries[index]["inputs"]["action_sha256"]
        fr_hash = fr_queries[index]["inputs"]["action_sha256"]
        if candidate_hash != fr_hash:
            mismatch = index
            break
    return {
        "comparable_queries": comparable,
        "first_reuse_query_index": first_reuse,
        "first_action_mismatch_query_index": mismatch,
        "pre_reuse_action_hashes_match": mismatch is None or mismatch >= first_reuse,
        "first_reuse_action_hash_differs": mismatch == first_reuse,
    }


def matching_episode(run_root: Path, *, task_id: int, state_id: int) -> tuple[Path, dict[str, Any]]:
    matches: list[tuple[Path, dict[str, Any]]] = []
    for path in episode_paths(run_root):
        episode = load_json(path)
        if episode["task_id"] == task_id and episode["initial_state_id"] == state_id:
            matches.append((path, episode))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one task {task_id}/state {state_id} episode in {run_root}, "
            f"found {len(matches)}"
        )
    return matches[0]


def fr_stage1_summary(fr_root: Path) -> dict[str, Any]:
    episodes: list[dict[str, Any]] = []
    wall: list[float] = []
    visual: list[float] = []
    for path in episode_paths(fr_root):
        episode = load_json(path)
        if episode["task_id"] not in range(10) or episode["initial_state_id"] not in range(3):
            continue
        episodes.append(episode)
        wall.extend(episode["timing"]["steady_query_wall_ms"])
        visual.extend(episode["timing"]["steady_visual_cuda_ms"])
    if len(episodes) != 30 or not wall or len(wall) != len(visual):
        raise RuntimeError("A4 FR Stage 1 population did not reconcile")
    return {
        "episodes": len(episodes),
        "successes": sum(bool(episode["success"]) for episode in episodes),
        "steady_queries": len(wall),
        "steady_query_wall_ms_per_query": sum(wall) / len(wall),
        "steady_visual_cuda_ms_per_query": sum(visual) / len(visual),
    }


def candidate_diagnostics(run_root: Path, analysis_result: dict[str, Any]) -> dict[str, Any]:
    episodes: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    successful_q3_reversal = 0
    successful_first_reversal = 0
    for path in episode_paths(run_root):
        episode = load_json(path)
        queries = query_records(path)
        first = first_reuse_summary(queries)
        row = {
            "task_id": episode["task_id"],
            "initial_state_id": episode["initial_state_id"],
            "success": bool(episode["success"]),
            "queries": len(queries),
            "scene_reuses": episode["counts"]["scene_reuses"],
            "first_reuse": first,
        }
        episodes.append(row)
        if not row["success"]:
            failure_rows.append(row)
        elif first is not None:
            successful_first_reversal += int(first["direction_reversal"])
            successful_q3_reversal += int(first["query_index"] == 3 and first["direction_reversal"])
    if len(episodes) != 30:
        raise RuntimeError(
            f"Candidate {analysis_result['candidate_id']} has {len(episodes)} episodes"
        )
    return {
        "candidate_id": analysis_result["candidate_id"],
        "successes": analysis_result["successes"],
        "scene_reuse_rate": analysis_result["scene_reuse_rate"],
        "steady_query_wall_ms_per_query": analysis_result["steady_query_wall_ms_per_query"],
        "steady_visual_cuda_ms_per_query": analysis_result["steady_visual_cuda_ms_per_query"],
        "failure_episodes": failure_rows,
        "successful_episodes_with_first_reuse_direction_reversal": (successful_first_reversal),
        "successful_episodes_with_first_reuse_at_q3_and_direction_reversal": (
            successful_q3_reversal
        ),
    }


def build_diagnosis(results_root: Path, candidates_path: Path) -> dict[str, Any]:
    analysis_path = results_root / A5_ANALYSIS
    analysis = load_json(analysis_path)
    if analysis["disposition"] != "STOP_NEGATIVE_NO_STAGE1_CANDIDATE":
        raise RuntimeError("Unexpected A5 disposition")
    if analysis["advancing_candidates"]:
        raise RuntimeError("A5 advancing set must be empty")
    by_id = {result["candidate_id"]: result for result in analysis["results"]}
    if set(by_id) != set(CANDIDATE_RUNS):
        raise RuntimeError("A5 candidate set differs from the frozen set")

    fr_root = results_root / FR_RUN
    fr = fr_stage1_summary(fr_root)
    candidates = {
        candidate_id: candidate_diagnostics(results_root / run_id, by_id[candidate_id])
        for candidate_id, run_id in CANDIDATE_RUNS.items()
    }
    conservative = candidates["acr-t25-h2-b30"]
    conservative["query_wall_reduction_vs_fr"] = 1.0 - (
        conservative["steady_query_wall_ms_per_query"] / fr["steady_query_wall_ms_per_query"]
    )
    conservative["visual_cuda_reduction_vs_fr"] = 1.0 - (
        conservative["steady_visual_cuda_ms_per_query"] / fr["steady_visual_cuda_ms_per_query"]
    )

    failure = conservative["failure_episodes"]
    if len(failure) != 1:
        raise RuntimeError("Expected exactly one conservative failure")
    task_id = failure[0]["task_id"]
    state_id = failure[0]["initial_state_id"]
    counterpart_outcomes: dict[str, bool] = {}
    for candidate_id, run_id in CANDIDATE_RUNS.items():
        _, episode = matching_episode(results_root / run_id, task_id=task_id, state_id=state_id)
        counterpart_outcomes[candidate_id] = bool(episode["success"])
    fr_path, fr_episode = matching_episode(fr_root, task_id=task_id, state_id=state_id)
    counterpart_outcomes["upstream-fr"] = bool(fr_episode["success"])
    conservative_path, _ = matching_episode(
        results_root / CANDIDATE_RUNS["acr-t25-h2-b30"],
        task_id=task_id,
        state_id=state_id,
    )

    middle_failures = {
        (row["task_id"], row["initial_state_id"])
        for row in candidates["acr-t50-h4-b55"]["failure_episodes"]
    }
    aggressive_failures = {
        (row["task_id"], row["initial_state_id"])
        for row in candidates["acr-t70-h8-b75"]["failure_episodes"]
    }
    payload = {
        "schema_version": "acr.v2-diagnosis.v1",
        "phase": "ACR-V2-DIAGNOSIS",
        "source_hashes": {
            "a5_analysis_sha256": byte_sha256(analysis_path),
            "a5_analysis_semantic_sha256": analysis["semantic_sha256"],
            "candidates_sha256": byte_sha256(candidates_path),
        },
        "fr_stage1": fr,
        "candidates": candidates,
        "conservative_failure": {
            "task_id": task_id,
            "initial_state_id": state_id,
            "counterpart_outcomes": counterpart_outcomes,
            "action_divergence": first_action_divergence(
                query_records(conservative_path), query_records(fr_path)
            ),
        },
        "aggressive_failure_overlap": {
            "middle_failure_count": len(middle_failures),
            "aggressive_failure_count": len(aggressive_failures),
            "shared_failure_count": len(middle_failures & aggressive_failures),
            "shared_failures": [
                list(item) for item in sorted(middle_failures & aggressive_failures)
            ],
        },
        "interpretation_guards": [
            "A5 outcomes are exploratory inputs to Version 2 design, not confirmation evidence.",
            "The conservative failure trigger pattern also occurs in successful episodes.",
            "The conservative failure is not monotonic because both more aggressive candidates succeed on the same task/state.",
            "Version 1 reduces visual CUDA time but increases synchronized query wall time.",
            "No single logged signal is identified as a causal failure mechanism.",
        ],
    }
    return semantic_record(payload)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    diagnosis = build_diagnosis(arguments.results_root, arguments.candidates)
    encoded = canonical_bytes(diagnosis) + b"\n"
    if arguments.output is None:
        print(encoded.decode("utf-8"), end="")
    else:
        if arguments.output.exists():
            raise FileExistsError(f"Refusing to overwrite {arguments.output}")
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_bytes(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
