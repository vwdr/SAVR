#!/usr/bin/env python3
"""Mechanically adjudicate frozen ACR A5 Stage 1 and Stage 2 records."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_ROOT = Path("/home/ved/SAVR")
CANDIDATE_IDS = (
    "acr-t25-h2-b30",
    "acr-t50-h4-b55",
    "acr-t70-h8-b75",
)
A4_RUN_ID = "acr-a4-upstream-fr-object-dev00-09-v01"
STAGE1_ANALYSIS_ROOT = "acr-a5-stage1-analysis-v01"
FINAL_ANALYSIS_ROOT = "acr-a5-final-analysis-v01"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def value_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_id_for(stage: str, candidate_id: str) -> str:
    if stage not in {"stage1", "stage2"} or candidate_id not in CANDIDATE_IDS:
        raise ValueError("Unsupported A5 stage/candidate")
    return f"acr-a5-sa-acr-object-{stage}-{candidate_id}-v01"


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def validate_semantic_record(record: dict[str, Any]) -> None:
    semantic = dict(record)
    claimed = semantic.pop("semantic_sha256", None)
    if claimed != value_sha256(semantic):
        raise RuntimeError("Immutable A5 analysis semantic hash changed")


def point_from_episode_arrays(episodes: list[dict[str, Any]], key: str) -> float:
    values = [
        float(value) for episode in episodes for value in (episode.get("timing", {}).get(key) or [])
    ]
    if not values:
        raise ValueError(f"No steady timing values for {key}")
    return sum(values) / len(values)


def reduction(acr_point: float, fr_point: float) -> float:
    if acr_point < 0 or fr_point <= 0:
        raise ValueError("Timing points must be non-negative with positive FR")
    return 1.0 - acr_point / fr_point


def summarize_run(
    project_root: Path,
    *,
    stage: str,
    candidate_id: str,
) -> dict[str, Any]:
    sys.path.insert(0, str(project_root / "src"))
    from savr.acr.records import reconcile_episode_counts, validate_record

    run_id = run_id_for(stage, candidate_id)
    run_root = project_root / "results" / run_id
    if not run_root.is_dir():
        raise FileNotFoundError(f"Missing A5 run: {run_id}")
    expected_states = tuple(range(3)) if stage == "stage1" else tuple(range(3, 10))
    expected_episodes = 30 if stage == "stage1" else 70
    query_schema = load_json(project_root / "schemas/acr_query.schema.json")
    episode_schema = load_json(project_root / "schemas/acr_episode.schema.json")
    run_schema = load_json(project_root / "schemas/acr_run.schema.json")
    completion = load_json(run_root / "completion/record.json")
    summary = load_json(run_root / "summary/record.json")
    validate_record(completion, run_schema)
    if completion.get("status") != "completed" or summary.get("status") != "completed":
        raise RuntimeError(f"A5 run did not complete cleanly: {run_id}")
    episodes = [load_json(path) for path in sorted(run_root.rglob("episode/record.json"))]
    queries = [load_json(path) for path in sorted(run_root.rglob("query-*/record.json"))]
    if len(episodes) != expected_episodes:
        raise RuntimeError(
            f"Expected {expected_episodes} episodes for {run_id}, found {len(episodes)}"
        )
    expected_pairings = {(task, state, 0) for task in range(10) for state in expected_states}
    observed_pairings = {
        (int(item["task_id"]), int(item["initial_state_id"]), int(item["seed"]))
        for item in episodes
    }
    if observed_pairings != expected_pairings:
        raise RuntimeError(f"A5 population mismatch for {run_id}")
    queries_by_attempt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    records_hash_input: list[dict[str, Any]] = []
    for query in queries:
        validate_record(query, query_schema)
        if query["run_id"] != run_id or query["policy"] != "sa-acr":
            raise RuntimeError("A5 query identity changed")
        if query["provenance"]["configuration_sha256"] != completion["configuration_sha256"]:
            raise RuntimeError("A5 query configuration hash changed")
        work = query["camera_work"]
        refresh = bool(query["decision"]["scene_refresh"])
        expected_scene = 1 if refresh else 0
        if (
            work["scene_siglip_calls"],
            work["scene_dinov2_calls"],
            work["scene_projector_calls"],
        ) != (expected_scene, expected_scene, expected_scene):
            raise RuntimeError("A5 scene work differs from its decision")
        if (
            work["wrist_siglip_calls"],
            work["wrist_dinov2_calls"],
            work["wrist_projector_calls"],
            work["downstream_calls"],
        ) != (1, 1, 1, 1):
            raise RuntimeError("A5 fresh-wrist/downstream invariant failed")
        timing = query["timing"]
        if refresh and timing["scene_visual_cuda_ms"] is None:
            raise RuntimeError("A5 scene refresh lacks timing")
        if not refresh and float(timing["scene_visual_cuda_ms"]) != 0.0:
            raise RuntimeError("A5 scene reuse performed measured scene CUDA work")
        queries_by_attempt[query["attempt_id"]].append(query)
        records_hash_input.append(query)
    aggregate = {
        name: 0
        for name in (
            "queries",
            "scene_refreshes",
            "scene_reuses",
            "wrist_refreshes",
            "scene_siglip_calls",
            "scene_dinov2_calls",
            "scene_projector_calls",
            "wrist_siglip_calls",
            "wrist_dinov2_calls",
            "wrist_projector_calls",
            "downstream_calls",
        )
    }
    per_task_success = {str(task): 0 for task in range(10)}
    successes = technical_failures = 0
    configuration_hashes: set[str] = set()
    for episode in episodes:
        validate_record(episode, episode_schema)
        records_hash_input.append(episode)
        if episode["run_id"] != run_id or episode["policy"] != "sa-acr":
            raise RuntimeError("A5 episode identity changed")
        configuration_hashes.add(str(episode["configuration_sha256"]))
        if episode["status"] != "completed":
            technical_failures += 1
        else:
            successes += int(bool(episode["success"]))
            per_task_success[str(episode["task_id"])] += int(bool(episode["success"]))
        attempt_queries = queries_by_attempt.get(episode["attempt_id"], [])
        attempt_queries.sort(key=lambda item: int(item["query_index"]))
        if len(attempt_queries) != int(episode["counts"]["queries"]):
            raise RuntimeError("A5 episode/query record counts differ")
        if [int(item["query_index"]) for item in attempt_queries] != list(
            range(len(attempt_queries))
        ):
            raise RuntimeError("A5 query indices are not contiguous")
        if episode["records_sha256"] != value_sha256({"queries": attempt_queries}):
            raise RuntimeError("A5 episode record hash changed")
        if len(episode["timing"]["inclusive_query_wall_ms"] or []) != len(attempt_queries):
            raise RuntimeError("A5 inclusive query timing count changed")
        if len(episode["timing"]["inclusive_visual_cuda_ms"] or []) != len(attempt_queries):
            raise RuntimeError("A5 inclusive visual timing count changed")
        reconcile_episode_counts(episode["counts"])
        for name in aggregate:
            aggregate[name] += int(episode["counts"][name])
    if len(configuration_hashes) != 1:
        raise RuntimeError("A5 run contains multiple configuration hashes")
    reconcile_episode_counts(aggregate)
    if aggregate != summary.get("counts"):
        raise RuntimeError("A5 aggregate counts differ from runner summary")
    if len(queries) != aggregate["queries"]:
        raise RuntimeError("A5 query total does not reconcile")
    ordered_episode_ids = [item["episode_id"] for item in episodes]
    if [item.removesuffix("/episode") for item in ordered_episode_ids] != completion[
        "planned_attempts"
    ]:
        raise RuntimeError("A5 planned attempts differ from terminal records")
    if completion["records_sha256"] != value_sha256(ordered_episode_ids):
        raise RuntimeError("A5 completion record hash changed")
    if (
        summary.get("attempts_started") != expected_episodes
        or summary.get("terminal_episodes") != expected_episodes
        or summary.get("successes") != successes
        or summary.get("per_task_successes") != per_task_success
        or summary.get("configuration_sha256") != next(iter(configuration_hashes))
    ):
        raise RuntimeError("A5 runner summary differs from immutable records")
    scene_reuse_rate = aggregate["scene_reuses"] / aggregate["queries"]
    return {
        "run_id": run_id,
        "stage": stage,
        "candidate_id": candidate_id,
        "terminal_episodes": sum(item["status"] == "completed" for item in episodes),
        "successes": successes,
        "per_task_successes": per_task_success,
        "technical_failures": technical_failures,
        "counts": aggregate,
        "scene_reuse_rate": scene_reuse_rate,
        "steady_visual_cuda_ms_per_query": point_from_episode_arrays(
            episodes, "steady_visual_cuda_ms"
        ),
        "steady_query_wall_ms_per_query": point_from_episode_arrays(
            episodes, "steady_query_wall_ms"
        ),
        "configuration_sha256": next(iter(configuration_hashes)),
        "records_sha256": value_sha256(records_hash_input),
        "episodes": episodes,
    }


def stage1_pass(summary: dict[str, Any]) -> tuple[bool, list[str]]:
    failures: list[str] = []
    if summary["terminal_episodes"] != 30:
        failures.append("terminal-episodes")
    if summary["successes"] != 30:
        failures.append("successes")
    if any(value != 3 for value in summary["per_task_successes"].values()):
        failures.append("per-task-success")
    if summary["scene_reuse_rate"] < 0.15:
        failures.append("scene-reuse")
    if summary["technical_failures"] != 0:
        failures.append("technical-failures")
    return not failures, failures


def development_eligibility(
    candidate: dict[str, Any],
    fr: dict[str, Any],
) -> tuple[bool, list[str]]:
    failures: list[str] = []
    if candidate["technical_failures"] != 0 or candidate["terminal_episodes"] != 100:
        failures.append("integrity")
    if candidate["successes"] < fr["successes"] - 2:
        failures.append("success")
    for task in range(10):
        if candidate["per_task_successes"][str(task)] < fr["per_task_successes"][str(task)] - 1:
            failures.append(f"task-{task}-success")
    if candidate["scene_reuse_rate"] < 0.40:
        failures.append("scene-reuse")
    if candidate["visual_cuda_reduction"] < 0.10:
        failures.append("visual-cuda")
    return not failures, failures


def select_candidate(eligible: list[dict[str, Any]], horizons: dict[str, int]) -> str | None:
    if not eligible:
        return None
    best_success = max(int(item["success_difference_vs_fr"]) for item in eligible)
    pool = [item for item in eligible if int(item["success_difference_vs_fr"]) >= best_success - 1]
    pool.sort(
        key=lambda item: (
            -float(item["query_latency_reduction"]),
            -float(item["visual_cuda_reduction"]),
            horizons[str(item["candidate_id"])],
            str(item["candidate_id"]),
        )
    )
    return str(pool[0]["candidate_id"])


def write_analysis(project_root: Path, root_name: str, record: dict[str, Any]) -> Path:
    sys.path.insert(0, str(project_root / "src"))
    from savr.acr.records import ImmutableRecordStore

    root = project_root / "results" / root_name
    if root.exists():
        raise FileExistsError(f"Immutable A5 analysis already exists: {root}")
    record["semantic_sha256"] = value_sha256(record)
    return ImmutableRecordStore(root).write_once("record", record)


def analyze_stage1(project_root: Path) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    advancing: list[str] = []
    for candidate_id in CANDIDATE_IDS:
        summary = summarize_run(project_root, stage="stage1", candidate_id=candidate_id)
        passed, failures = stage1_pass(summary)
        summary.pop("episodes")
        summary["passed"] = passed
        summary["failed_gates"] = failures
        results.append(summary)
        if passed:
            advancing.append(candidate_id)
    record = {
        "schema_version": "acr.a5-stage1-analysis.v1",
        "phase": "A5",
        "stage": "stage1",
        "candidate_source_sha256": file_sha256(project_root / "configs/acr/candidates.json"),
        "a5_configuration_sha256": file_sha256(project_root / "configs/acr/development_a5.json"),
        "analyzer_sha256": file_sha256(Path(__file__)),
        "results": results,
        "advancing_candidates": advancing,
        "disposition": "ADVANCE_TO_STAGE2" if advancing else "STOP_NEGATIVE_NO_STAGE1_CANDIDATE",
        "recorded_at_utc": utc_now(),
    }
    write_analysis(project_root, STAGE1_ANALYSIS_ROOT, record)
    return record


def combine_summaries(stage1: dict[str, Any], stage2: dict[str, Any]) -> dict[str, Any]:
    counts = {
        key: int(stage1["counts"][key]) + int(stage2["counts"][key]) for key in stage1["counts"]
    }
    episodes = [*stage1["episodes"], *stage2["episodes"]]
    per_task = {
        str(task): int(stage1["per_task_successes"][str(task)])
        + int(stage2["per_task_successes"][str(task)])
        for task in range(10)
    }
    return {
        "candidate_id": stage1["candidate_id"],
        "terminal_episodes": stage1["terminal_episodes"] + stage2["terminal_episodes"],
        "successes": stage1["successes"] + stage2["successes"],
        "per_task_successes": per_task,
        "technical_failures": stage1["technical_failures"] + stage2["technical_failures"],
        "counts": counts,
        "scene_reuse_rate": counts["scene_reuses"] / counts["queries"],
        "steady_visual_cuda_ms_per_query": point_from_episode_arrays(
            episodes, "steady_visual_cuda_ms"
        ),
        "steady_query_wall_ms_per_query": point_from_episode_arrays(
            episodes, "steady_query_wall_ms"
        ),
        "stage1_run_id": stage1["run_id"],
        "stage2_run_id": stage2["run_id"],
        "configuration_sha256": stage1["configuration_sha256"],
        "records_sha256": value_sha256([stage1["records_sha256"], stage2["records_sha256"]]),
    }


def fr_summary(project_root: Path) -> dict[str, Any]:
    root = project_root / "results" / A4_RUN_ID
    episodes = [load_json(path) for path in sorted(root.rglob("episode/record.json"))]
    if len(episodes) != 100 or any(item["status"] != "completed" for item in episodes):
        raise RuntimeError("A4 FR source does not contain 100 completed episodes")
    per_task = {str(task): 0 for task in range(10)}
    for episode in episodes:
        per_task[str(episode["task_id"])] += int(bool(episode["success"]))
    return {
        "run_id": A4_RUN_ID,
        "successes": sum(int(bool(item["success"])) for item in episodes),
        "per_task_successes": per_task,
        "steady_visual_cuda_ms_per_query": point_from_episode_arrays(
            episodes, "steady_visual_cuda_ms"
        ),
        "steady_query_wall_ms_per_query": point_from_episode_arrays(
            episodes, "steady_query_wall_ms"
        ),
    }


def analyze_stage2(project_root: Path) -> dict[str, Any]:
    stage1_record = load_json(
        project_root / "results" / STAGE1_ANALYSIS_ROOT / "record/record.json"
    )
    validate_semantic_record(stage1_record)
    if (
        stage1_record.get("analyzer_sha256") != file_sha256(Path(__file__))
        or stage1_record.get("a5_configuration_sha256")
        != file_sha256(project_root / "configs/acr/development_a5.json")
        or stage1_record.get("candidate_source_sha256")
        != file_sha256(project_root / "configs/acr/candidates.json")
    ):
        raise RuntimeError("Stage 1 analysis provenance differs from frozen A5 code")
    advancing = list(stage1_record.get("advancing_candidates", []))
    if not advancing:
        raise RuntimeError("Stage 2 is ineligible because no candidate advanced")
    candidate_payload = load_json(project_root / "configs/acr/candidates.json")
    horizons = {
        item["configuration_id"]: int(item["horizon"]) for item in candidate_payload["candidates"]
    }
    fr = fr_summary(project_root)
    results: list[dict[str, Any]] = []
    eligible: list[dict[str, Any]] = []
    for candidate_id in advancing:
        first = summarize_run(project_root, stage="stage1", candidate_id=candidate_id)
        second = summarize_run(project_root, stage="stage2", candidate_id=candidate_id)
        combined = combine_summaries(first, second)
        combined["success_difference_vs_fr"] = combined["successes"] - fr["successes"]
        combined["visual_cuda_reduction"] = reduction(
            combined["steady_visual_cuda_ms_per_query"], fr["steady_visual_cuda_ms_per_query"]
        )
        combined["query_latency_reduction"] = reduction(
            combined["steady_query_wall_ms_per_query"], fr["steady_query_wall_ms_per_query"]
        )
        passed, failures = development_eligibility(combined, fr)
        combined["eligible"] = passed
        combined["failed_gates"] = failures
        results.append(combined)
        if passed:
            eligible.append(combined)
    selected = select_candidate(eligible, horizons)
    selected_payload = next(
        (item for item in candidate_payload["candidates"] if item["configuration_id"] == selected),
        None,
    )
    record = {
        "schema_version": "acr.a5-final-analysis.v1",
        "phase": "A5",
        "stage": "stage2",
        "a5_configuration_sha256": file_sha256(project_root / "configs/acr/development_a5.json"),
        "analyzer_sha256": file_sha256(Path(__file__)),
        "stage1_analysis_semantic_sha256": stage1_record["semantic_sha256"],
        "fr": fr,
        "results": results,
        "selected_candidate": selected,
        "selected_candidate_payload": selected_payload,
        "selected_candidate_semantic_sha256": value_sha256(selected_payload)
        if selected_payload
        else None,
        "disposition": "PASS_PRIMARY_FROZEN"
        if selected
        else "STOP_NEGATIVE_NO_DEVELOPMENT_CANDIDATE",
        "recorded_at_utc": utc_now(),
    }
    write_analysis(project_root, FINAL_ANALYSIS_ROOT, record)
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("stage1", "stage2"), required=True)
    arguments = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    if project_root != EXPECTED_ROOT:
        raise SystemExit(f"Refusing to run outside {EXPECTED_ROOT}: {project_root}")
    record = (
        analyze_stage1(project_root)
        if arguments.stage == "stage1"
        else analyze_stage2(project_root)
    )
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
