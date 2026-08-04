#!/usr/bin/env python3
"""Mechanically reconcile and gate the complete frozen ACR V3-D matrix."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_ROOT = Path("/home/ved/SAVR")
PRIMARY_RUN_ID = "acr-v3d-paired-object-dev03-09-v01"
RECOVERY_1_RUN_ID = "acr-v3d-paired-object-dev03-09-recovery-v01"
RUN_ID = "acr-v3d-paired-object-dev03-09-recovery-02-v01"
ANALYSIS_ID = "acr-v3d-analysis-v01"
FR_RUN_ID = "acr-a4-upstream-fr-object-dev00-09-v01"
TASK_IDS = tuple(range(10))
STATE_IDS = tuple(range(3, 10))
POLICIES = ("batched-fr", "sa-bdp-acr-t25-h2-b30-v01")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def value_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_semantic(record: dict[str, Any]) -> None:
    semantic = dict(record)
    claimed = semantic.pop("semantic_sha256", None)
    if claimed != value_sha256(semantic):
        raise RuntimeError(
            f"Semantic hash mismatch: {record.get('query_id') or record.get('episode_id')}"
        )


def mean_flat(episodes: list[dict[str, Any]], key: str) -> tuple[float, int]:
    values = [float(value) for episode in episodes for value in episode["timing"][key]]
    if not values:
        raise RuntimeError(f"No retained measurements for {key}")
    return sum(values) / len(values), len(values)


def evaluate_gate(metrics: dict[str, Any]) -> tuple[bool, list[str]]:
    failures: list[str] = []
    bfr, v3 = metrics["policies"][POLICIES[0]], metrics["policies"][POLICIES[1]]
    if bfr["terminal_episodes"] != 70 or v3["terminal_episodes"] != 70:
        failures.append("terminal-episodes")
    if bfr["technical_failures"] != 0 or v3["technical_failures"] != 0:
        failures.append("technical-failures")
    if v3["successes"] < bfr["successes"] - 2:
        failures.append("aggregate-success")
    for task in TASK_IDS:
        if v3["per_task_successes"][str(task)] < bfr["per_task_successes"][str(task)] - 1:
            failures.append(f"task-{task}-success")
    if v3["scene_reuse_rate"] < 0.2:
        failures.append("scene-reuse")
    if metrics["comparisons"]["v3_visual_cuda_reduction_vs_bfr"] < 0.1:
        failures.append("visual-cuda")
    if metrics["comparisons"]["v3_query_wall_ratio_vs_sequential_fr"] > 0.98:
        failures.append("wall-vs-sequential-fr")
    if metrics["comparisons"]["v3_query_wall_ratio_vs_bfr"] > 1.0:
        failures.append("wall-vs-bfr")
    if metrics.get("all_invariants_pass") is not True:
        failures.append("invariants")
    return not failures, failures


def summarize_policy(
    policy: str,
    episodes: list[dict[str, Any]],
    queries: list[dict[str, Any]],
    *,
    configuration_sha256: str,
) -> dict[str, Any]:
    expected_pairs = {(task, state, 0) for task in TASK_IDS for state in STATE_IDS}
    observed_pairs = {
        (int(item["task_id"]), int(item["initial_state_id"]), int(item["seed"]))
        for item in episodes
    }
    if len(episodes) != 70 or observed_pairs != expected_pairs:
        raise RuntimeError(f"V3-D population mismatch for {policy}")
    by_attempt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for query in queries:
        verify_semantic(query)
        if (
            query.get("schema_version") != "acr.v3d-query.v1"
            or query.get("run_id") != RUN_ID
            or query.get("policy") != policy
        ):
            raise RuntimeError("V3-D query identity changed")
        if query["provenance"]["configuration_sha256"] != configuration_sha256:
            raise RuntimeError("V3-D query configuration hash changed")
        work = query["camera_work"]
        refresh = bool(query["decision"]["scene_refresh"])
        if (
            work["physical_siglip_calls"],
            work["physical_dinov2_calls"],
            work["physical_projector_calls"],
            work["logical_wrist_backbone_calls"],
            work["logical_wrist_projector_calls"],
            work["downstream_calls"],
        ) != (1, 1, 1, 1, 1, 1):
            raise RuntimeError("V3-D physical/fresh-work invariant failed")
        expected_scene = int(refresh)
        if (work["logical_scene_backbone_calls"], work["logical_scene_projector_calls"]) != (
            expected_scene,
            expected_scene,
        ):
            raise RuntimeError("V3-D scene work differs from its decision")
        if policy == POLICIES[0] and not refresh:
            raise RuntimeError("BFR contains a scene reuse")
        if (
            float(query["timing"]["total_visual_cuda_ms"]) < 0
            or float(query["timing"]["query_wall_ms"]) <= 0
        ):
            raise RuntimeError("V3-D timing is invalid")
        by_attempt[query["attempt_id"]].append(query)

    successes = technical_failures = 0
    per_task = {str(task): 0 for task in TASK_IDS}
    counts = {
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
    for episode in episodes:
        verify_semantic(episode)
        if (
            episode.get("schema_version") != "acr.v3d-episode.v1"
            or episode.get("run_id") != RUN_ID
            or episode.get("policy") != policy
        ):
            raise RuntimeError("V3-D episode identity changed")
        if episode["configuration_sha256"] != configuration_sha256:
            raise RuntimeError("V3-D episode configuration hash changed")
        if episode["status"] != "completed":
            technical_failures += 1
            continue
        successes += int(bool(episode["success"]))
        per_task[str(episode["task_id"])] += int(bool(episode["success"]))
        attempt_queries = sorted(
            by_attempt.get(episode["attempt_id"], []), key=lambda item: int(item["query_index"])
        )
        if [int(item["query_index"]) for item in attempt_queries] != list(
            range(len(attempt_queries))
        ):
            raise RuntimeError("V3-D query indices are not contiguous")
        if len(attempt_queries) != int(episode["counts"]["queries"]):
            raise RuntimeError("V3-D episode/query counts differ")
        if episode["records_sha256"] != value_sha256({"queries": attempt_queries}):
            raise RuntimeError("V3-D episode query hash changed")
        steady_wall = [
            item["timing"]["query_wall_ms"] for item in attempt_queries if item["steady_state"]
        ]
        steady_visual = [
            item["timing"]["total_visual_cuda_ms"]
            for item in attempt_queries
            if item["steady_state"]
        ]
        if (
            episode["timing"]["steady_query_wall_ms"] != steady_wall
            or episode["timing"]["steady_visual_cuda_ms"] != steady_visual
        ):
            raise RuntimeError("V3-D steady timing arrays differ from query truth")
        for name in counts:
            counts[name] += int(episode["counts"][name])

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from savr.acr.records import reconcile_episode_counts

    reconcile_episode_counts(counts)
    if len(queries) != counts["queries"]:
        raise RuntimeError("V3-D aggregate query total differs")
    wall_point, wall_queries = mean_flat(episodes, "steady_query_wall_ms")
    visual_point, visual_queries = mean_flat(episodes, "steady_visual_cuda_ms")
    if wall_queries != visual_queries:
        raise RuntimeError("V3-D steady timing populations differ")
    return {
        "policy": policy,
        "terminal_episodes": len(episodes),
        "successes": successes,
        "per_task_successes": per_task,
        "technical_failures": technical_failures,
        "counts": counts,
        "scene_reuse_rate": counts["scene_reuses"] / counts["queries"],
        "steady_query_wall_ms_per_query": wall_point,
        "steady_visual_cuda_ms_per_query": visual_point,
        "steady_queries": wall_queries,
        "records_sha256": value_sha256({"episodes": episodes, "queries": queries}),
    }


def sequential_fr_point(project_root: Path) -> dict[str, Any]:
    run_root = project_root / "results" / FR_RUN_ID
    episodes = [load_json(path) for path in sorted(run_root.rglob("episode/record.json"))]
    selected = [item for item in episodes if int(item["initial_state_id"]) in STATE_IDS]
    expected = {(task, state, 0) for task in TASK_IDS for state in STATE_IDS}
    observed = {
        (int(item["task_id"]), int(item["initial_state_id"]), int(item["seed"]))
        for item in selected
    }
    if (
        len(selected) != 70
        or observed != expected
        or any(item["status"] != "completed" for item in selected)
    ):
        raise RuntimeError("Immutable A4 sequential-FR reference population is incomplete")
    point, count = mean_flat(selected, "steady_query_wall_ms")
    return {
        "run_id": FR_RUN_ID,
        "states": list(STATE_IDS),
        "terminal_episodes": len(selected),
        "steady_queries": count,
        "steady_query_wall_ms_per_query": point,
        "records_sha256": value_sha256(selected),
    }


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    if project_root != EXPECTED_ROOT:
        raise SystemExit(f"Refusing to analyze outside {EXPECTED_ROOT}: {project_root}")
    run_root = project_root / "results" / RUN_ID
    output_root = project_root / "results" / ANALYSIS_ID
    if output_root.exists():
        raise SystemExit(f"Immutable V3-D analysis already exists: {output_root}")
    completion = load_json(run_root / "completion/record.json")
    summary = load_json(run_root / "summary/record.json")
    manifest = load_json(run_root / "manifest/record.json")
    if completion.get("status") != "completed" or summary.get("status") != "completed":
        raise RuntimeError("V3-D outcomes remain blind until the complete terminal matrix exists")
    if summary.get("terminal_records") != 140 or summary.get("attempts_started") != 140:
        raise RuntimeError("V3-D run did not complete exactly 140 attempts")
    if (
        summary.get("cumulative_attempts_started") != 143
        or manifest.get("recovery") is not True
        or manifest.get("recovery_index") != 2
    ):
        raise RuntimeError("V3-D recovery 2 did not preserve exactly three prior starts")
    if summary.get("restoration_error") is not None or summary.get(
        "checkpoint_before"
    ) != summary.get("checkpoint_after"):
        raise RuntimeError("V3-D restoration invariant failed")
    if (
        summary.get("upstream_revision_after") != manifest["revisions"]["openvla_oft"]
        or summary.get("libero_revision_after") != manifest["revisions"]["libero"]
    ):
        raise RuntimeError("V3-D source restoration invariant failed")
    episodes = [load_json(path) for path in sorted(run_root.rglob("episode/record.json"))]
    queries = [load_json(path) for path in sorted(run_root.rglob("query-*/record.json"))]
    if len(episodes) != 140:
        raise RuntimeError("V3-D does not have 140 terminal records")
    observed_order = sorted(
        episodes, key=lambda item: (int(item["pair_index"]), int(item["pair_position"]))
    )
    expected_episode_ids = [f"{item}/episode" for item in manifest["planned_attempts"]]
    if [item["episode_id"] for item in observed_order] != expected_episode_ids:
        raise RuntimeError("V3-D paired counterbalance differs from the frozen manifest")
    if completion["terminal_record_ids_sha256"] != value_sha256(expected_episode_ids):
        raise RuntimeError("V3-D completion identity hash changed")
    configuration_sha256 = str(manifest["configuration_sha256"])
    policy_results = {}
    for policy in POLICIES:
        policy_results[policy] = summarize_policy(
            policy,
            [item for item in episodes if item["policy"] == policy],
            [item for item in queries if item["policy"] == policy],
            configuration_sha256=configuration_sha256,
        )
        if policy_results[policy]["counts"] != summary["work_counts_per_policy"][policy]:
            raise RuntimeError("V3-D runner/analyzer work totals differ")
    sequential = sequential_fr_point(project_root)
    bfr, v3 = policy_results[POLICIES[0]], policy_results[POLICIES[1]]
    comparisons = {
        "v3_success_difference_vs_bfr": v3["successes"] - bfr["successes"],
        "v3_visual_cuda_reduction_vs_bfr": 1.0
        - v3["steady_visual_cuda_ms_per_query"] / bfr["steady_visual_cuda_ms_per_query"],
        "v3_query_wall_ratio_vs_sequential_fr": v3["steady_query_wall_ms_per_query"]
        / sequential["steady_query_wall_ms_per_query"],
        "v3_query_wall_ratio_vs_bfr": v3["steady_query_wall_ms_per_query"]
        / bfr["steady_query_wall_ms_per_query"],
    }
    metrics = {
        "policies": policy_results,
        "sequential_fr_reference": sequential,
        "comparisons": comparisons,
        "all_invariants_pass": True,
    }
    passed, failures = evaluate_gate(metrics)
    record = {
        "schema_version": "acr.v3d-analysis.v1",
        "phase": "V3-D",
        "run_id": RUN_ID,
        "configuration_sha256": configuration_sha256,
        "preserved_prior_run_ids": [PRIMARY_RUN_ID, RECOVERY_1_RUN_ID],
        "recovery_configuration_sha256": hashlib.sha256(
            (project_root / "configs/acr/v3_d_recovery_2.json").read_bytes()
        ).hexdigest(),
        "run_records_sha256": value_sha256(
            {"episodes": episodes, "queries": queries, "completion": completion}
        ),
        **metrics,
        "passed": passed,
        "failed_gates": failures,
        "disposition": "PASS_POSITIVE_STOP_BEFORE_V3_E" if passed else "STOP_NEGATIVE_BEFORE_V3_E",
        "analyzer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "recorded_at_utc": utc_now(),
    }
    record["semantic_sha256"] = value_sha256(record)
    sys.path.insert(0, str(project_root / "src"))
    from savr.acr.records import ImmutableRecordStore

    ImmutableRecordStore(output_root).write_once("record", record)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
