#!/usr/bin/env python3
"""Run frozen outcome-blind IR-SA-ACR V5-B CPU screening."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import subprocess
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


RUN_ID = "acr-v5b-output-blind-screening-v01"
CONFIG_PATH = Path("configs/acr/v5_b_output_blind_preflight.json")
DEFAULT_OUTPUT = Path("reports/runtime/acr_v5_b.json")
TRACE_KEYS = {
    "schema_version",
    "run_id",
    "attempt_id",
    "query_id",
    "episode_id",
    "query_index",
    "scene_representation",
    "normalized_eef_position",
    "action_chunk",
    "action_shape",
    "action_sha256",
    "gripper_transition_veto",
    "translation_direction_reversals",
    "upstream_component_invocations",
    "semantic_sha256",
}
FORBIDDEN_KEY_FRAGMENTS = ("success", "failure", "reward", "timing", "latency")


@dataclass(frozen=True)
class ReplayQuery:
    episode_id: str
    query_index: int
    scene_representation: tuple[float, ...]
    normalized_eef_position: tuple[float, float, float]
    action_chunk: tuple[float, ...]


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def value_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def semantic_sha256(record: dict[str, Any]) -> str:
    semantic = dict(record)
    semantic.pop("semantic_sha256", None)
    return value_sha256(semantic)


def percentile(values: list[float], probability: float) -> float:
    if not values or not 0 <= probability <= 1:
        raise ValueError("Percentile input is invalid")
    ordered = sorted(values)
    rank = (len(ordered) - 1) * probability
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    fraction = rank - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def trace_manifest(root: Path, pattern: str) -> dict[str, Any]:
    paths = sorted(root.rglob(pattern))
    digest = hashlib.sha256()
    total = 0
    for path in paths:
        data = path.read_bytes()
        relative = path.relative_to(root).as_posix().encode()
        total += len(data)
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return {
        "trace_records": len(paths),
        "trace_artifact_bytes": total,
        "ordered_path_content_sha256": digest.hexdigest(),
        "paths": paths,
    }


def verify_trace_schema(record: dict[str, Any]) -> None:
    if set(record) != TRACE_KEYS:
        extra = sorted(set(record) - TRACE_KEYS)
        missing = sorted(TRACE_KEYS - set(record))
        raise RuntimeError(f"Trace schema changed; extra={extra}, missing={missing}")
    lowered = {key.lower() for key in record}
    if any(fragment in key for key in lowered for fragment in FORBIDDEN_KEY_FRAGMENTS):
        raise RuntimeError("Outcome or timing field entered the sealed V5-B trace")
    if record["schema_version"] != "acr.fr-trace-query.v1":
        raise RuntimeError("Trace schema version changed")
    if record.get("semantic_sha256") != semantic_sha256(record):
        raise RuntimeError(f"Trace semantic hash mismatch: {record.get('query_id')}")


def load_traces(
    project_root: Path, config: dict[str, Any]
) -> tuple[dict[str, tuple[ReplayQuery, ...]], dict[str, Any]]:
    from savr.acr.records import decode_float_sequence

    frozen = config["input"]
    trace_root = project_root / "results" / str(frozen["run_id"])
    manifest = trace_manifest(trace_root, str(frozen["record_glob"]))
    for key in (
        "trace_records",
        "trace_artifact_bytes",
        "ordered_path_content_sha256",
    ):
        if manifest[key] != frozen[key]:
            raise RuntimeError(f"Frozen trace manifest mismatch: {key}")

    grouped: dict[str, list[ReplayQuery]] = defaultdict(list)
    for path in manifest.pop("paths"):
        record = json.loads(path.read_text(encoding="utf-8"))
        verify_trace_schema(record)
        if record["run_id"] != frozen["run_id"]:
            raise RuntimeError("Trace run identity changed")
        scene = decode_float_sequence(record["scene_representation"])
        action = decode_float_sequence(record["action_chunk"])
        position = tuple(float(value) for value in record["normalized_eef_position"])
        if len(scene) not in {1024, 3072} or len(action) != 56 or len(position) != 3:
            raise RuntimeError("Trace dimensions changed")
        episode_id = str(record["episode_id"])
        grouped[episode_id].append(
            ReplayQuery(
                episode_id=episode_id,
                query_index=int(record["query_index"]),
                scene_representation=scene,
                normalized_eef_position=(position[0], position[1], position[2]),
                action_chunk=action,
            )
        )
    if len(grouped) != frozen["episodes"]:
        raise RuntimeError("Frozen episode count changed")
    result: dict[str, tuple[ReplayQuery, ...]] = {}
    for episode_id, queries in grouped.items():
        ordered = tuple(sorted(queries, key=lambda item: item.query_index))
        if [item.query_index for item in ordered] != list(range(len(ordered))):
            raise RuntimeError(f"Non-contiguous query trace: {episode_id}")
        result[episode_id] = ordered
    return dict(sorted(result.items())), manifest


def context(configuration: Any, episode_id: str) -> Any:
    from savr.acr.types import ACRContext

    return ACRContext(
        episode_id=episode_id,
        attempt_id=f"v5b-{episode_id}",
        task_id="outcome-blind-development",
        instruction_sha256="0" * 64,
        checkpoint_id="a4-full-refresh-trace",
        upstream_revision="frozen-a4",
        configuration_id=configuration.configuration_id,
        controller_version=configuration.controller_version,
        preprocessing_id="acr-scene-32-v1",
        action_head_id="frozen-a4-action",
        dtype="encoded-float",
        device="cpu",
        patch_count=1,
    )


def replay_candidate(
    episodes: dict[str, tuple[ReplayQuery, ...]], candidate: dict[str, Any]
) -> dict[str, Any]:
    from savr.acr.isolated_controller import IsolatedACRController
    from savr.acr.types import ACRConfiguration, ACRPolicy

    configuration = ACRConfiguration(
        configuration_id=str(candidate["candidate_id"]),
        policy=ACRPolicy.SA_ACR,
        scene_threshold=float(candidate["scene_threshold"]),
        translation_threshold=float(candidate["translation_threshold"]),
        horizon=1,
        hard_reuse_cap=float(candidate["hard_reuse_cap"]),
        controller_version="acr-isolated-controller-v1",
    )
    episode_summaries: list[dict[str, int | float]] = []
    reason_counts: Counter[str] = Counter()
    maximum_streak = 0
    maximum_prefix = 0.0
    prefix_cap_violations = 0
    gripper_reuses = 0
    mismatch_count = 0
    post_reuse_refreshes = 0

    for queries in episodes.values():
        controller = IsolatedACRController(configuration)
        controller.reset(context(configuration, queries[0].episode_id))
        cache_available = False
        cache_age = 0
        reuses = 0
        current_streak = 0
        episode_maximum_prefix = 0.0
        for completed, query in enumerate(queries, start=1):
            decision = controller.decide(
                scene_representation=query.scene_representation,
                normalized_eef_position=query.normalized_eef_position,
                cache_available=cache_available,
                cache_age=cache_age,
            )
            reason_counts.update(decision.reasons)
            mismatch_count += int("isolation-state-mismatch" in decision.reasons)
            post_reuse_refreshes += int("post-reuse-refresh" in decision.reasons)
            if decision.refresh:
                cache_available = True
                cache_age = 0
                current_streak = 0
            else:
                reuses += 1
                cache_age += 1
                current_streak += 1
                maximum_streak = max(maximum_streak, current_streak)
                gripper_reuses += int(bool(decision.gripper_transition_veto))
            controller.observe(
                decision=decision,
                scene_representation=query.scene_representation,
                normalized_eef_position=query.normalized_eef_position,
                action_chunk=query.action_chunk,
            )
            prefix = reuses / completed
            episode_maximum_prefix = max(episode_maximum_prefix, prefix)
            maximum_prefix = max(maximum_prefix, prefix)
            if prefix > float(candidate["hard_reuse_cap"]) + 1e-12:
                prefix_cap_violations += 1
        snapshot = controller.snapshot()
        if snapshot.completed_queries != len(queries) or snapshot.completed_reuses != reuses:
            raise RuntimeError("Controller snapshot disagrees with replay accounting")
        episode_summaries.append(
            {
                "queries": len(queries),
                "reuses": reuses,
                "maximum_prefix_reuse": episode_maximum_prefix,
            }
        )

    queries_total = sum(int(item["queries"]) for item in episode_summaries)
    reuses_total = sum(int(item["reuses"]) for item in episode_summaries)
    return {
        "candidate_id": candidate["candidate_id"],
        "level": candidate["level"],
        "scene_threshold": candidate["scene_threshold"],
        "translation_threshold": candidate["translation_threshold"],
        "hard_reuse_cap": candidate["hard_reuse_cap"],
        "episodes": len(episode_summaries),
        "queries": queries_total,
        "refreshes": queries_total - reuses_total,
        "reuses": reuses_total,
        "reuse_rate": reuses_total / queries_total,
        "maximum_reuse_streak": maximum_streak,
        "maximum_prefix_reuse": maximum_prefix,
        "prefix_cap_violations": prefix_cap_violations,
        "gripper_transition_reuses": gripper_reuses,
        "isolation_state_mismatches": mismatch_count,
        "invariant_failures": 0,
        "post_reuse_refreshes": post_reuse_refreshes,
        "reason_counts": dict(sorted(reason_counts.items())),
        "episode_summaries": episode_summaries,
    }


def bootstrap_reuse(
    episodes: list[dict[str, int | float]], *, seed: int, resamples: int
) -> dict[str, float]:
    generator = random.Random(seed)
    values: list[float] = []
    for _ in range(resamples):
        selected = [episodes[generator.randrange(len(episodes))] for _ in episodes]
        values.append(
            sum(int(item["reuses"]) for item in selected)
            / sum(int(item["queries"]) for item in selected)
        )
    return {
        "lower_95": percentile(values, 0.025),
        "median": percentile(values, 0.5),
        "upper_95": percentile(values, 0.975),
    }


def finalize_candidate(raw: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    bootstrap = config["bootstrap"]
    interval = bootstrap_reuse(
        raw["episode_summaries"],
        seed=int(bootstrap["seed"]),
        resamples=int(bootstrap["resamples"]),
    )
    scene_fraction = float(config["accounting"]["scene_fraction_of_logical_visual_components"])
    visual_point = raw["reuse_rate"] * scene_fraction
    visual_interval = {key: value * scene_fraction for key, value in interval.items()}
    gates = config["gates"]
    checks = {
        "population": raw["episodes"] == config["input"]["episodes"]
        and raw["queries"] == config["input"]["trace_records"],
        "maximum_streak": raw["maximum_reuse_streak"] == gates["maximum_reuse_streak"],
        "prefix_cap": raw["prefix_cap_violations"] <= gates["maximum_prefix_cap_violations"],
        "gripper_transition": raw["gripper_transition_reuses"]
        <= gates["gripper_transition_reuses_max"],
        "isolation_state": raw["isolation_state_mismatches"]
        <= gates["isolation_state_mismatches_max"],
        "invariants": raw["invariant_failures"] <= gates["invariant_failures_max"],
        "post_reuse_refresh": raw["post_reuse_refreshes"] >= gates["post_reuse_refreshes_min"],
        "reuse_point": raw["reuse_rate"] >= gates["reuse_point_min"],
        "reuse_lower_95": interval["lower_95"] >= gates["reuse_lower_95_min"],
        "logical_visual_point": visual_point >= gates["logical_visual_reduction_point_min"],
        "logical_visual_lower_95": visual_interval["lower_95"]
        >= gates["logical_visual_reduction_lower_95_min"],
    }
    result = {key: value for key, value in raw.items() if key != "episode_summaries"}
    result.update(
        {
            "reuse_rate_interval": interval,
            "logical_visual_reduction_point": visual_point,
            "logical_visual_reduction_interval": visual_interval,
            "gates": checks,
            "eligible": all(checks.values()),
        }
    )
    return result


def analyze(project_root: Path, config: dict[str, Any]) -> dict[str, Any]:
    episodes, manifest = load_traces(project_root, config)
    first = [
        finalize_candidate(replay_candidate(episodes, candidate), config)
        for candidate in config["candidates"]
    ]
    second = [
        finalize_candidate(replay_candidate(episodes, candidate), config)
        for candidate in config["candidates"]
    ]
    if canonical_bytes(first) != canonical_bytes(second):
        raise RuntimeError("Repeated V5-B replay is not byte-identical")
    eligible = [item for item in first if item["eligible"]]
    eligible.sort(
        key=lambda item: (
            float(item["level"]),
            float(item["hard_reuse_cap"]),
            int(item["isolation_state_mismatches"]),
            float(item["reuse_rate"]),
        )
    )
    selected = eligible[0]["candidate_id"] if eligible else None
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    result = {
        "schema_version": "acr.v5b-result.v1",
        "run_id": RUN_ID,
        "phase": "V5-B",
        "status": "COMPLETE",
        "configuration_sha256": file_sha256(project_root / CONFIG_PATH),
        "savr_revision": revision,
        "input_manifest": manifest,
        "candidate_count": len(first),
        "replay_repetitions": 2,
        "replay_byte_identical": True,
        "candidates": first,
        "eligible_candidate_ids": [item["candidate_id"] for item in eligible],
        "selected_candidate_id": selected,
        "disposition": "ADVANCE_TO_V5_C_PROTOCOL" if selected else "STOP_BEFORE_V5_C",
        "resources": {
            "gpu_count": 0,
            "model_queries": 0,
            "simulator_episodes": 0,
            "simulator_resets": 0,
            "downloads": 0,
            "new_task_outcomes": 0,
        },
        "protected": config["protected"],
        "claim_boundary": "outcome-blind_offline_telemetry_only",
    }
    result["semantic_sha256"] = semantic_sha256(result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    started = time.monotonic()
    config = json.loads((args.root / CONFIG_PATH).read_text(encoding="utf-8"))
    result = analyze(args.root, config)
    elapsed = time.monotonic() - started
    if elapsed > float(config["resource_caps"]["cpu_wall_seconds"]):
        raise RuntimeError("V5-B CPU wall-time cap exceeded")
    output = args.output if args.output.is_absolute() else args.root / args.output
    payload = canonical_bytes(result) + b"\n"
    if len(payload) > int(config["resource_caps"]["artifact_bytes"]):
        raise RuntimeError("V5-B artifact cap exceeded")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(payload)
    print(json.dumps({"output": str(output), "semantic_sha256": result["semantic_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
