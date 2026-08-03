#!/usr/bin/env python3
"""Reconcile ACR A4 FR evidence and derive the frozen candidates twice."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from savr.acr.candidates import FRTraceQuery, derive_candidates_bytes
from savr.acr.records import (
    ImmutableRecordStore,
    canonical_json_bytes,
    decode_float_sequence,
    reconcile_episode_counts,
    reconcile_run,
    semantic_sha256,
    validate_record,
)


EXPECTED_ROOT = Path("/home/ved/SAVR")
RUN_ID = "acr-a4-upstream-fr-object-dev00-09-v01"
EXPECTED_PAIRINGS = {(task, state) for task in range(10) for state in range(10)}
MINIMUM_SUCCESSES = 90
MINIMUM_PER_TASK = 8


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def read_records(root: Path, suffix: str) -> list[dict[str, Any]]:
    return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(root.rglob(suffix))]


def feasibility(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    pairings = {(int(item["task_id"]), int(item["initial_state_id"])) for item in episodes}
    per_task = Counter(
        int(item["task_id"])
        for item in episodes
        if item["status"] == "completed" and item["success"] is True
    )
    technical = sum(item["status"] != "completed" for item in episodes)
    successes = sum(item["success"] is True for item in episodes)
    passed = (
        len(episodes) == 100
        and pairings == EXPECTED_PAIRINGS
        and technical == 0
        and successes >= MINIMUM_SUCCESSES
        and all(per_task[task] >= MINIMUM_PER_TASK for task in range(10))
    )
    return {
        "passed": passed,
        "terminal_episodes": len(episodes),
        "successes": successes,
        "technical_failures": technical,
        "per_task_successes": {str(task): per_task[task] for task in range(10)},
        "missing_pairings": sorted([list(item) for item in EXPECTED_PAIRINGS - pairings]),
        "extra_pairings": sorted([list(item) for item in pairings - EXPECTED_PAIRINGS]),
    }


def load_trace(run_dir: Path) -> tuple[FRTraceQuery, ...]:
    traces = read_records(run_dir, "trace/record.json")
    result: list[FRTraceQuery] = []
    seen: set[tuple[str, int]] = set()
    for record in traces:
        if record.get("schema_version") != "acr.fr-trace-query.v1":
            raise RuntimeError("Unexpected A4 trace schema")
        semantic = dict(record)
        claimed = semantic.pop("semantic_sha256", None)
        if claimed != semantic_sha256(semantic):
            raise RuntimeError("A4 trace semantic hash mismatch")
        action = decode_float_sequence(record["action_chunk"])
        if record.get("action_shape") != [8, 7] or len(action) != 56:
            raise RuntimeError("A4 trace action shape changed")
        if record.get("upstream_component_invocations") != {
            "vision_backbone": 1,
            "visual_projector": 1,
            "language_model": 1,
            "action_head": 1,
        }:
            raise RuntimeError("A4 upstream component invocation count changed")
        scene = decode_float_sequence(record["scene_representation"])
        position = tuple(float(value) for value in record["normalized_eef_position"])
        if len(scene) not in {1024, 3072} or len(position) != 3:
            raise RuntimeError("A4 trace signal dimensions changed")
        key = (str(record["episode_id"]), int(record["query_index"]))
        if key in seen:
            raise RuntimeError("Duplicate A4 trace query")
        seen.add(key)
        result.append(
            FRTraceQuery(
                episode_id=key[0],
                query_index=key[1],
                scene_representation=scene,
                normalized_eef_position=(position[0], position[1], position[2]),
                action_chunk=action,
            )
        )
    return tuple(sorted(result, key=lambda item: (item.episode_id, item.query_index)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=EXPECTED_ROOT / "results" / RUN_ID)
    arguments = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    if project_root != EXPECTED_ROOT:
        raise SystemExit(f"Refusing to run outside {EXPECTED_ROOT}: {project_root}")
    run_dir = arguments.run_dir.resolve()
    if run_dir != EXPECTED_ROOT / "results" / RUN_ID:
        raise SystemExit("A4 analyzer accepts only the frozen result directory")
    output_root = run_dir / "analysis"
    if output_root.exists():
        raise SystemExit(f"Immutable A4 analysis already exists: {output_root}")

    query_schema = json.loads((project_root / "schemas/acr_query.schema.json").read_text())
    episode_schema = json.loads((project_root / "schemas/acr_episode.schema.json").read_text())
    run_schema = json.loads((project_root / "schemas/acr_run.schema.json").read_text())
    queries = read_records(run_dir, "query-*/record.json")
    episodes = read_records(run_dir, "episode/record.json")
    completions = read_records(run_dir, "completion/record.json")
    if len(completions) != 1 or completions[0]["status"] != "completed":
        raise RuntimeError("A4 run does not have one completed manifest")
    validate_record(completions[0], run_schema)
    for record in queries:
        validate_record(record, query_schema)
    for record in episodes:
        validate_record(record, episode_schema)
        reconcile_episode_counts(record["counts"])
    reconcile_run(scheduled_attempts=100, terminal_episodes=len(episodes), failures=0)
    gate = feasibility(episodes)
    trace_first = load_trace(run_dir)
    trace_second = load_trace(run_dir)
    if len(trace_first) != len(queries) or trace_first != trace_second:
        raise RuntimeError("A4 query/trace reconciliation failed")

    store = ImmutableRecordStore(output_root)
    candidates_sha256 = None
    candidates = None
    disposition = "STOPPED_FR_INELIGIBLE"
    if gate["passed"]:
        first = derive_candidates_bytes(trace_first)
        second = derive_candidates_bytes(trace_second)
        if first != second:
            raise RuntimeError("A4 candidate derivations are not byte-identical")
        first_payload = json.loads(first)
        if len(first_payload.get("candidates", [])) != 3:
            raise RuntimeError("A4 did not derive exactly three candidates")
        store.write_once("candidates-pass-1", first_payload)
        store.write_once("candidates-pass-2", json.loads(second))
        candidates_sha256 = sha256_bytes(first)
        candidates = first_payload
        disposition = "PASS_CANDIDATES_FROZEN"

    revision = subprocess.check_output(
        ["git", "-C", str(project_root), "rev-parse", "HEAD"], text=True
    ).strip()
    record = {
        "schema_version": "acr.a4-analysis.v1",
        "run_id": RUN_ID,
        "disposition": disposition,
        "feasibility": gate,
        "query_records": len(queries),
        "trace_records": len(trace_first),
        "candidate_derivations_byte_identical": candidates is not None,
        "candidates_sha256": candidates_sha256,
        "candidate_payload": candidates,
        "run_records_sha256": sha256_bytes(
            canonical_json_bytes(
                {"queries": queries, "episodes": episodes, "completion": completions[0]}
            )
        ),
        "analyzer_revision": revision,
        "recorded_at_utc": utc_now(),
    }
    record["semantic_sha256"] = semantic_sha256(record)
    store.write_once("record", record)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0 if gate["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
