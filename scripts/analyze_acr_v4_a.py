#!/usr/bin/env python3
"""Run the frozen CPU-only ACR V4-A diagnosis and candidate screening."""

from __future__ import annotations

import hashlib
import json
import math
import random
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from statistics import fmean
from typing import Any


EXPECTED_ROOT = Path("/home/ved/SAVR")
V3_RUN_ID = "acr-v3d-paired-object-dev03-09-recovery-02-v01"
A4_RUN_ID = "acr-a4-upstream-fr-object-dev00-09-v01"
A5_RUN_IDS = (
    "acr-a5-sa-acr-object-stage1-acr-t25-h2-b30-v01",
    "acr-a5-sa-acr-object-stage1-acr-t50-h4-b55-v01",
    "acr-a5-sa-acr-object-stage1-acr-t70-h8-b75-v01",
)
OUTPUT_ID = "acr-v4a-diagnosis-v01"
BFR_POLICY = "batched-fr"
V3_POLICY = "sa-bdp-acr-t25-h2-b30-v01"


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


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_records(root: Path, pattern: str) -> list[dict[str, Any]]:
    return [load_json(path) for path in sorted(root.rglob(pattern))]


def verify_semantic(record: dict[str, Any]) -> None:
    semantic = dict(record)
    claimed = semantic.pop("semantic_sha256", None)
    if claimed != value_sha256(semantic):
        raise RuntimeError(f"Semantic hash mismatch: {record.get('query_id')}")


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


def distribution(values: list[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("Distribution cannot be empty")
    return {
        "count": len(values),
        "mean": fmean(values),
        "minimum": min(values),
        "p05": percentile(values, 0.05),
        "p25": percentile(values, 0.25),
        "median": percentile(values, 0.5),
        "p75": percentile(values, 0.75),
        "p95": percentile(values, 0.95),
        "maximum": max(values),
    }


def candidate_id(alpha: float, transition_policy: str) -> str:
    level = int(round(alpha * 100))
    suffix = "g" if transition_policy == "gripper_only" else "gr"
    return f"v4-a{level:03d}-h2-b40-{suffix}"


def context(configuration: Any, episode_id: str) -> Any:
    from savr.acr.types import ACRContext

    return ACRContext(
        episode_id=episode_id,
        attempt_id=f"offline-{episode_id}",
        task_id="offline",
        instruction_sha256="0" * 64,
        checkpoint_id="offline",
        upstream_revision="offline",
        configuration_id=configuration.configuration_id,
        controller_version=configuration.controller_version,
        preprocessing_id="acr-scene-32-v1",
        action_head_id="offline",
        dtype="offline",
        device="cpu",
        patch_count=1,
    )


def load_a4_trace(project_root: Path) -> dict[str, tuple[ReplayQuery, ...]]:
    from savr.acr.records import decode_float_sequence

    root = project_root / "results" / A4_RUN_ID
    records = read_records(root, "trace/record.json")
    if len(records) != 1773:
        raise RuntimeError(f"A4 trace count changed: {len(records)}")
    grouped: dict[str, list[ReplayQuery]] = defaultdict(list)
    for record in records:
        verify_semantic(record)
        if record.get("schema_version") != "acr.fr-trace-query.v1":
            raise RuntimeError("A4 trace schema changed")
        scene = decode_float_sequence(record["scene_representation"])
        action = decode_float_sequence(record["action_chunk"])
        position = tuple(float(value) for value in record["normalized_eef_position"])
        if len(scene) not in {1024, 3072} or len(action) != 56 or len(position) != 3:
            raise RuntimeError("A4 trace dimensions changed")
        grouped[str(record["episode_id"])].append(
            ReplayQuery(
                episode_id=str(record["episode_id"]),
                query_index=int(record["query_index"]),
                scene_representation=scene,
                normalized_eef_position=(position[0], position[1], position[2]),
                action_chunk=action,
            )
        )
    if len(grouped) != 100:
        raise RuntimeError("A4 episode population changed")
    result: dict[str, tuple[ReplayQuery, ...]] = {}
    for episode_id, values in grouped.items():
        ordered = tuple(sorted(values, key=lambda item: item.query_index))
        if [item.query_index for item in ordered] != list(range(len(ordered))):
            raise RuntimeError(f"A4 trace is not contiguous: {episode_id}")
        result[episode_id] = ordered
    return dict(sorted(result.items()))


def action_rms(first: tuple[float, ...], second: tuple[float, ...]) -> float:
    if len(first) != len(second) or not first:
        raise ValueError("Action chunks must have equal nonzero width")
    return math.sqrt(sum((left - right) ** 2 for left, right in zip(first, second)) / len(first))


def replay_candidate(
    episodes: dict[str, tuple[ReplayQuery, ...]],
    *,
    alpha: float,
    transition_policy: str,
    family: dict[str, Any],
    high_action_threshold: float,
) -> dict[str, Any]:
    from savr.acr.controller import ACRController
    from savr.acr.types import ACRConfiguration, ACRPolicy

    scene = float(family["scene_threshold_low"]) + alpha * (
        float(family["scene_threshold_high"]) - float(family["scene_threshold_low"])
    )
    translation = float(family["translation_threshold_low"]) + alpha * (
        float(family["translation_threshold_high"]) - float(family["translation_threshold_low"])
    )
    identifier = candidate_id(alpha, transition_policy)
    configuration = ACRConfiguration(
        configuration_id=identifier,
        policy=ACRPolicy.SA_ACR,
        scene_threshold=scene,
        translation_threshold=translation,
        horizon=int(family["horizon"]),
        hard_reuse_cap=float(family["hard_reuse_cap"]),
    )
    episode_results: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    for episode_id, queries in episodes.items():
        controller = ACRController(configuration)
        controller.reset(context(configuration, episode_id))
        cache_available = False
        cache_age = 0
        reuses = current_streak = maximum_streak = 0
        gripper_reuses = reversal_reuses = high_action_reuses = early_reuses = 0
        history: list[tuple[float, ...]] = []
        for query in queries:
            decision = controller.decide(
                scene_representation=query.scene_representation,
                normalized_eef_position=query.normalized_eef_position,
                cache_available=cache_available,
                cache_age=cache_age,
            )
            reversal = any(decision.translation_direction_reversals)
            if (
                transition_policy == "gripper_or_translation_direction_reversal"
                and reversal
                and not decision.refresh
            ):
                decision = replace(
                    decision,
                    refresh=True,
                    reasons=(*decision.reasons, "analysis-direction-reversal"),
                )
            reason_counts.update(decision.reasons)
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
                reversal_reuses += int(reversal)
                early_reuses += int(query.query_index < 4)
                if len(history) >= 2:
                    high_action_reuses += int(
                        action_rms(history[-1], history[-2]) >= high_action_threshold
                    )
            controller.observe(
                decision=decision,
                scene_representation=query.scene_representation,
                normalized_eef_position=query.normalized_eef_position,
                action_chunk=query.action_chunk,
            )
            history.append(query.action_chunk)
        episode_results.append(
            {
                "episode_id": episode_id,
                "queries": len(queries),
                "reuses": reuses,
                "maximum_reuse_streak": maximum_streak,
                "gripper_transition_reuses": gripper_reuses,
                "direction_reversal_reuses": reversal_reuses,
                "high_action_change_reuses": high_action_reuses,
                "early_query_reuses": early_reuses,
            }
        )
    queries_total = sum(item["queries"] for item in episode_results)
    reuses_total = sum(item["reuses"] for item in episode_results)
    return {
        "candidate_id": identifier,
        "threshold_interpolation": alpha,
        "transition_policy": transition_policy,
        "scene_threshold": scene,
        "translation_threshold": translation,
        "queries": queries_total,
        "reuses": reuses_total,
        "reuse_rate": reuses_total / queries_total,
        "maximum_reuse_streak": max(item["maximum_reuse_streak"] for item in episode_results),
        "gripper_transition_reuses": sum(
            item["gripper_transition_reuses"] for item in episode_results
        ),
        "direction_reversal_reuses": sum(
            item["direction_reversal_reuses"] for item in episode_results
        ),
        "high_action_change_reuses": sum(
            item["high_action_change_reuses"] for item in episode_results
        ),
        "early_query_reuses": sum(item["early_query_reuses"] for item in episode_results),
        "reason_counts": dict(sorted(reason_counts.items())),
        "episodes": episode_results,
    }


def bootstrap_ratio(
    episodes: list[dict[str, Any]], *, seed: int, resamples: int
) -> tuple[list[float], dict[str, float]]:
    generator = random.Random(seed)
    values: list[float] = []
    for _ in range(resamples):
        selected = [episodes[generator.randrange(len(episodes))] for _ in episodes]
        queries = sum(int(item["queries"]) for item in selected)
        reuses = sum(int(item["reuses"]) for item in selected)
        values.append(reuses / queries)
    return values, {
        "lower_95": percentile(values, 0.025),
        "median": percentile(values, 0.5),
        "upper_95": percentile(values, 0.975),
    }


def load_v3_evidence(project_root: Path) -> dict[str, Any]:
    root = project_root / "results" / V3_RUN_ID
    queries = read_records(root, "query-*/record.json")
    episodes = read_records(root, "episode/record.json")
    completion = load_json(root / "completion/record.json")
    summary = load_json(root / "summary/record.json")
    if len(episodes) != 140 or len(queries) != 2589:
        raise RuntimeError("V3-D record counts changed")
    if completion.get("status") != "completed" or summary.get("terminal_records") != 140:
        raise RuntimeError("V3-D completion changed")
    for record in [*queries, *episodes, summary]:
        verify_semantic(record)
    episode_by_attempt = {str(item["attempt_id"]): item for item in episodes}
    grouped: dict[tuple[int, int], dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for query in queries:
        pair = (int(query["task_id"]), int(query["initial_state_id"]))
        grouped[pair][str(query["policy"])].append(query)
    expected_pairs = {(task, state) for task in range(10) for state in range(3, 10)}
    if set(grouped) != expected_pairs:
        raise RuntimeError("V3-D paired population changed")
    for policies in grouped.values():
        if set(policies) != {BFR_POLICY, V3_POLICY}:
            raise RuntimeError("V3-D paired policies changed")

    def values(policy: str, key: str, refresh: bool | None = None) -> list[float]:
        result = []
        for query in queries:
            if query["policy"] != policy or not query["steady_state"]:
                continue
            if refresh is not None and bool(query["decision"]["scene_refresh"]) != refresh:
                continue
            result.append(float(query["timing"][key]))
        return result

    timing = {
        "bfr_wall_ms": distribution(values(BFR_POLICY, "query_wall_ms")),
        "bfr_cuda_ms": distribution(values(BFR_POLICY, "query_cuda_ms")),
        "bfr_visual_cuda_ms": distribution(values(BFR_POLICY, "total_visual_cuda_ms")),
        "v3_refresh_wall_ms": distribution(values(V3_POLICY, "query_wall_ms", True)),
        "v3_refresh_cuda_ms": distribution(values(V3_POLICY, "query_cuda_ms", True)),
        "v3_refresh_visual_cuda_ms": distribution(values(V3_POLICY, "total_visual_cuda_ms", True)),
        "v3_reuse_wall_ms": distribution(values(V3_POLICY, "query_wall_ms", False)),
        "v3_reuse_cuda_ms": distribution(values(V3_POLICY, "query_cuda_ms", False)),
        "v3_reuse_visual_cuda_ms": distribution(values(V3_POLICY, "total_visual_cuda_ms", False)),
    }
    v3_queries = [item for item in queries if item["policy"] == V3_POLICY]
    reason_counts: Counter[str] = Counter(
        reason for item in v3_queries for reason in item["decision"]["reasons"]
    )
    cache_ages = Counter(
        str(item["decision"]["cache_age_before"])
        for item in v3_queries
        if item["decision"]["cache_age_before"] is not None
    )
    task_stats: dict[str, dict[str, int]] = {}
    for task in range(10):
        selected = [item for item in v3_queries if int(item["task_id"]) == task]
        task_stats[str(task)] = {
            "queries": len(selected),
            "reuses": sum(not item["decision"]["scene_refresh"] for item in selected),
            "successful_episode_queries": sum(
                episode_by_attempt[item["attempt_id"]]["success"] is True for item in selected
            ),
            "failed_episode_queries": sum(
                episode_by_attempt[item["attempt_id"]]["success"] is False for item in selected
            ),
        }

    action_comparison = {
        "paired_query_indices": 0,
        "action_hash_matches": 0,
        "v3_reuse_indices": 0,
        "v3_reuse_action_hash_matches": 0,
        "post_reuse_indices": 0,
        "post_reuse_action_hash_matches": 0,
    }
    for policies in grouped.values():
        bfr = {int(item["query_index"]): item for item in policies[BFR_POLICY]}
        v3 = {int(item["query_index"]): item for item in policies[V3_POLICY]}
        for index in sorted(set(bfr) & set(v3)):
            equal = bfr[index]["inputs"]["action_sha256"] == v3[index]["inputs"]["action_sha256"]
            action_comparison["paired_query_indices"] += 1
            action_comparison["action_hash_matches"] += int(equal)
            if not v3[index]["decision"]["scene_refresh"]:
                action_comparison["v3_reuse_indices"] += 1
                action_comparison["v3_reuse_action_hash_matches"] += int(equal)
            if index > 0 and index - 1 in v3 and not v3[index - 1]["decision"]["scene_refresh"]:
                action_comparison["post_reuse_indices"] += 1
                action_comparison["post_reuse_action_hash_matches"] += int(equal)

    clusters: dict[tuple[int, int], dict[str, Any]] = {}
    for pair, policies in grouped.items():
        clusters[pair] = {
            "bfr_wall": [
                float(item["timing"]["query_wall_ms"])
                for item in policies[BFR_POLICY]
                if item["steady_state"]
            ],
            "bfr_visual": [
                float(item["timing"]["total_visual_cuda_ms"])
                for item in policies[BFR_POLICY]
                if item["steady_state"]
            ],
            "refresh_wall": [
                float(item["timing"]["query_wall_ms"])
                for item in policies[V3_POLICY]
                if item["steady_state"] and item["decision"]["scene_refresh"]
            ],
            "refresh_visual": [
                float(item["timing"]["total_visual_cuda_ms"])
                for item in policies[V3_POLICY]
                if item["steady_state"] and item["decision"]["scene_refresh"]
            ],
            "reuse_wall": [
                float(item["timing"]["query_wall_ms"])
                for item in policies[V3_POLICY]
                if item["steady_state"] and not item["decision"]["scene_refresh"]
            ],
            "reuse_visual": [
                float(item["timing"]["total_visual_cuda_ms"])
                for item in policies[V3_POLICY]
                if item["steady_state"] and not item["decision"]["scene_refresh"]
            ],
        }
    return {
        "queries": queries,
        "episodes": episodes,
        "clusters": clusters,
        "timing": timing,
        "refresh_reason_counts": dict(sorted(reason_counts.items())),
        "cache_age_counts": dict(sorted(cache_ages.items())),
        "task_stats": task_stats,
        "action_hash_comparison": action_comparison,
        "source_records_sha256": value_sha256(
            {"queries": queries, "episodes": episodes, "completion": completion}
        ),
    }


def bootstrap_efficiency(
    v3_clusters: dict[tuple[int, int], dict[str, Any]],
    replay_episodes: list[dict[str, Any]],
    *,
    seed: int,
    resamples: int,
) -> dict[str, Any]:
    generator = random.Random(seed)
    pairs = sorted(v3_clusters)
    predicted_visual: list[float] = []
    refresh_ratios: list[float] = []
    current_wall_ratios: list[float] = []
    for _ in range(resamples):
        selected_pairs = [pairs[generator.randrange(len(pairs))] for _ in pairs]
        selected_replay = [
            replay_episodes[generator.randrange(len(replay_episodes))] for _ in replay_episodes
        ]
        bfr_wall = [value for pair in selected_pairs for value in v3_clusters[pair]["bfr_wall"]]
        bfr_visual = [value for pair in selected_pairs for value in v3_clusters[pair]["bfr_visual"]]
        refresh_wall = [
            value for pair in selected_pairs for value in v3_clusters[pair]["refresh_wall"]
        ]
        refresh_visual = [
            value for pair in selected_pairs for value in v3_clusters[pair]["refresh_visual"]
        ]
        reuse_wall = [value for pair in selected_pairs for value in v3_clusters[pair]["reuse_wall"]]
        reuse_visual = [
            value for pair in selected_pairs for value in v3_clusters[pair]["reuse_visual"]
        ]
        reuse_rate = sum(item["reuses"] for item in selected_replay) / sum(
            item["queries"] for item in selected_replay
        )
        bfr_wall_mean = fmean(bfr_wall)
        refresh_ratio = fmean(refresh_wall) / bfr_wall_mean
        current_ratio = (
            (1.0 - reuse_rate) * fmean(refresh_wall) + reuse_rate * fmean(reuse_wall)
        ) / bfr_wall_mean
        visual = 1.0 - (
            (1.0 - reuse_rate) * fmean(refresh_visual) + reuse_rate * fmean(reuse_visual)
        ) / fmean(bfr_visual)
        predicted_visual.append(visual)
        refresh_ratios.append(refresh_ratio)
        current_wall_ratios.append(current_ratio)
    return {
        "predicted_visual_reduction": {
            "lower_95": percentile(predicted_visual, 0.025),
            "median": percentile(predicted_visual, 0.5),
            "upper_95": percentile(predicted_visual, 0.975),
        },
        "refresh_wall_ratio": {
            "lower_95": percentile(refresh_ratios, 0.025),
            "median": percentile(refresh_ratios, 0.5),
            "upper_95": percentile(refresh_ratios, 0.975),
        },
        "current_executor_weighted_wall_ratio": {
            "lower_95": percentile(current_wall_ratios, 0.025),
            "median": percentile(current_wall_ratios, 0.5),
            "upper_95": percentile(current_wall_ratios, 0.975),
        },
    }


def a5_risk_summary(project_root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for run_id in A5_RUN_IDS:
        root = project_root / "results" / run_id
        queries = read_records(root, "query-*/record.json")
        episodes = read_records(root, "episode/record.json")
        if len(episodes) != 30:
            raise RuntimeError(f"A5 population changed: {run_id}")
        episode_by_attempt = {str(item["attempt_id"]): item for item in episodes}
        for record in [*queries, *episodes]:
            verify_semantic(record)
        reuses = [item for item in queries if not item["decision"]["scene_refresh"]]
        by_attempt: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in queries:
            by_attempt[str(item["attempt_id"])].append(item)
        maximum_streak = 0
        first_reuse_indices: list[int] = []
        for values in by_attempt.values():
            streak = 0
            reuse_indices = []
            for item in sorted(values, key=lambda entry: int(entry["query_index"])):
                if item["decision"]["scene_refresh"]:
                    streak = 0
                else:
                    streak += 1
                    maximum_streak = max(maximum_streak, streak)
                    reuse_indices.append(int(item["query_index"]))
            if reuse_indices:
                first_reuse_indices.append(min(reuse_indices))
        result[run_id] = {
            "terminal_episodes": len(episodes),
            "successes": sum(item["success"] is True for item in episodes),
            "queries": len(queries),
            "reuses": len(reuses),
            "reuse_rate": len(reuses) / len(queries),
            "maximum_reuse_streak": maximum_streak,
            "median_first_reuse_query": percentile(
                [float(item) for item in first_reuse_indices], 0.5
            ),
            "direction_reversal_reuses": sum(
                bool(item["inputs"].get("direction_reversal")) for item in reuses
            ),
            "gripper_transition_reuses": sum(
                bool(item["decision"].get("gripper_transition_veto")) for item in reuses
            ),
            "reuses_in_successful_episodes": sum(
                episode_by_attempt[item["attempt_id"]]["success"] is True for item in reuses
            ),
            "reuses_in_failed_episodes": sum(
                episode_by_attempt[item["attempt_id"]]["success"] is False for item in reuses
            ),
            "records_sha256": value_sha256({"queries": queries, "episodes": episodes}),
        }
    return result


def source_audit(project_root: Path) -> dict[str, Any]:
    project_path = project_root / "src/savr/acr/batched_dual_path.py"
    model_path = project_root / "third_party/openvla-oft/prismatic/extern/hf/modeling_prismatic.py"
    robot_path = project_root / "third_party/openvla-oft/experiments/robot/openvla_utils.py"
    sources = {
        "project_adapter": project_path.read_text(encoding="utf-8"),
        "pinned_model": model_path.read_text(encoding="utf-8"),
        "pinned_robot_utils": robot_path.read_text(encoding="utf-8"),
    }
    checks = {
        "reuse_has_fixed_one_wrist_camera_shape": "_, wrist_pixels = self.tensor_ops.split(pixel_values, (6, 6), dim=1)"
        in sources["project_adapter"],
        "reuse_uses_owned_scene_cache": '"scene.cache-load"' in sources["project_adapter"],
        "regression_action_shape_is_fixed": "NUM_ACTIONS_CHUNK, ACTION_DIM"
        in sources["pinned_model"],
        "direct_upstream_query_contains_cpu_transfer": ".cpu().detach().numpy()"
        in sources["pinned_model"],
        "direct_upstream_query_allocates_proprio_tensor": "torch.Tensor(proprio).to("
        in sources["pinned_model"],
        "prompt_depends_on_task_text": "task_label.lower()" in sources["pinned_robot_utils"],
    }
    if not all(checks.values()):
        raise RuntimeError(f"Pinned executor source audit changed: {checks}")
    return {
        "checks": checks,
        "direct_upstream_complete_query_graph_safe": False,
        "direct_graph_blockers": [
            "CPU NumPy transfer synchronizes after the action head",
            "proprioception tensor is allocated inside predict_action",
            "prompt token length can vary by task text",
            "input addresses are not persistent static buffers",
        ],
        "testable_project_owned_boundary": (
            "static-buffer GPU inference core from prepared fixed-shape inputs through "
            "the regression action head, bucketed by prompt shape, with CPU preprocessing, "
            "controller, validation, unnormalization, and eager fallback outside capture"
        ),
        "testable_boundary_supported_by_source": True,
        "source_sha256": {
            "project_adapter": file_sha256(project_path),
            "pinned_model": file_sha256(model_path),
            "pinned_robot_utils": file_sha256(robot_path),
        },
        "claim_boundary": (
            "Source feasibility only; performance, capture safety, memory fit, and numerical "
            "equivalence remain unmeasured until V4-C."
        ),
    }


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    if project_root != EXPECTED_ROOT:
        raise SystemExit(f"Refusing to analyze outside {EXPECTED_ROOT}: {project_root}")
    sys.path.insert(0, str(project_root / "src"))
    preflight_path = project_root / "configs/acr/v4_a_diagnosis_preflight.json"
    preflight = load_json(preflight_path)
    if preflight.get("status") != "FROZEN_BEFORE_V4_A_CANDIDATE_OUTPUTS":
        raise RuntimeError("V4-A preflight is not frozen")
    output_root = project_root / "results" / OUTPUT_ID
    if output_root.exists():
        raise SystemExit(f"Immutable V4-A diagnosis already exists: {output_root}")

    v3 = load_v3_evidence(project_root)
    a4 = load_a4_trace(project_root)
    all_action_deltas = [
        action_rms(queries[index - 1].action_chunk, queries[index - 2].action_chunk)
        for queries in a4.values()
        for index in range(2, len(queries))
    ]
    high_action_threshold = percentile(all_action_deltas, 0.9)
    family = preflight["controller_family"]
    replay_first = [
        replay_candidate(
            a4,
            alpha=float(alpha),
            transition_policy=str(policy),
            family=family,
            high_action_threshold=high_action_threshold,
        )
        for alpha in family["interpolation_fractions"]
        for policy in family["transition_policies"]
    ]
    replay_second = [
        replay_candidate(
            a4,
            alpha=float(alpha),
            transition_policy=str(policy),
            family=family,
            high_action_threshold=high_action_threshold,
        )
        for alpha in family["interpolation_fractions"]
        for policy in family["transition_policies"]
    ]
    if canonical_bytes(replay_first) != canonical_bytes(replay_second):
        raise RuntimeError("V4-A candidate replay is not byte-identical")

    timing = v3["timing"]
    bfr_wall = float(timing["bfr_wall_ms"]["mean"])
    bfr_visual = float(timing["bfr_visual_cuda_ms"]["mean"])
    refresh_wall = float(timing["v3_refresh_wall_ms"]["mean"])
    refresh_visual = float(timing["v3_refresh_visual_cuda_ms"]["mean"])
    reuse_wall = float(timing["v3_reuse_wall_ms"]["mean"])
    reuse_visual = float(timing["v3_reuse_visual_cuda_ms"]["mean"])
    resamples = int(preflight["bootstrap"]["resamples"])
    seed = int(preflight["bootstrap"]["seed"])
    candidates: list[dict[str, Any]] = []
    for index, replay in enumerate(replay_first):
        bootstrap_values, reuse_interval = bootstrap_ratio(
            replay["episodes"], seed=seed + index, resamples=resamples
        )
        del bootstrap_values
        efficiency = bootstrap_efficiency(
            v3["clusters"],
            replay["episodes"],
            seed=seed + 100 + index,
            resamples=resamples,
        )
        reuse_rate = float(replay["reuse_rate"])
        visual_point = (
            1.0 - ((1.0 - reuse_rate) * refresh_visual + reuse_rate * reuse_visual) / bfr_visual
        )
        current_wall_point = (
            (1.0 - reuse_rate) * refresh_wall + reuse_rate * reuse_wall
        ) / bfr_wall
        conservative_reuse = float(reuse_interval["lower_95"])
        conservative_refresh_ratio = float(efficiency["refresh_wall_ratio"]["upper_95"])
        required_reuse_ratio = (
            0.98 - (1.0 - conservative_reuse) * conservative_refresh_ratio
        ) / conservative_reuse
        gates = {
            "reuse_point": reuse_rate
            >= float(preflight["controller_gates"]["replay_reuse_point_min"]),
            "reuse_lower_bound": conservative_reuse
            >= float(preflight["controller_gates"]["replay_reuse_lower_bound_min"]),
            "maximum_streak": replay["maximum_reuse_streak"]
            <= int(preflight["controller_gates"]["maximum_reuse_streak"]),
            "gripper_transition": replay["gripper_transition_reuses"]
            <= int(preflight["controller_gates"]["gripper_transition_reuses_max"]),
            "visual_point": visual_point
            >= float(preflight["controller_gates"]["predicted_visual_cuda_reduction_point_min"]),
            "executor_feasibility": required_reuse_ratio
            >= float(
                preflight["complete_method_gates"]["required_reuse_wall_ratio_feasibility_floor"]
            ),
        }
        candidates.append(
            {
                **{key: value for key, value in replay.items() if key != "episodes"},
                "reuse_rate_interval": reuse_interval,
                "predicted_visual_reduction_point": visual_point,
                "predicted_visual_reduction_interval": efficiency["predicted_visual_reduction"],
                "current_executor_weighted_wall_ratio_point": current_wall_point,
                "current_executor_weighted_wall_ratio_interval": efficiency[
                    "current_executor_weighted_wall_ratio"
                ],
                "conservative_refresh_wall_ratio": conservative_refresh_ratio,
                "required_reuse_wall_ratio_vs_bfr": required_reuse_ratio,
                "gates": gates,
                "eligible": all(gates.values()),
                "episode_replay_sha256": value_sha256(replay["episodes"]),
            }
        )

    eligible = [item for item in candidates if item["eligible"]]
    eligible.sort(
        key=lambda item: (
            item["transition_policy"] != "gripper_or_translation_direction_reversal",
            item["direction_reversal_reuses"] + item["high_action_change_reuses"],
            item["threshold_interpolation"],
            item["reuse_rate"],
            item["candidate_id"],
        )
    )
    audit = source_audit(project_root)
    selected = eligible[0] if eligible and audit["testable_boundary_supported_by_source"] else None
    disposition = "PASS_FREEZE_ONE_V4_DESIGN" if selected else "STOP_BEFORE_V4_B"
    record = {
        "schema_version": "acr.v4a-diagnosis.v1",
        "phase": "V4-A",
        "run_id": OUTPUT_ID,
        "repository_revision": __import__("subprocess")
        .check_output(["git", "-C", str(project_root), "rev-parse", "HEAD"], text=True)
        .strip(),
        "preflight_sha256": file_sha256(preflight_path),
        "inputs": {
            "v3_d_source_records_sha256": v3["source_records_sha256"],
            "a4_trace_sha256": value_sha256(
                [asdict(item) for queries in a4.values() for item in queries]
            ),
            "a5_run_ids": list(A5_RUN_IDS),
        },
        "v3_d_reconciliation": {
            "terminal_episodes": len(v3["episodes"]),
            "queries": len(v3["queries"]),
            "timing": v3["timing"],
            "refresh_reason_counts": v3["refresh_reason_counts"],
            "cache_age_counts": v3["cache_age_counts"],
            "task_stats": v3["task_stats"],
            "action_hash_comparison": v3["action_hash_comparison"],
            "all_records_semantically_valid": True,
            "outliers_deleted": 0,
        },
        "a5_risk_summary": a5_risk_summary(project_root),
        "high_action_change_definition": {
            "metric": "RMS difference between the two prior 8x7 action chunks",
            "global_a4_p90_threshold": high_action_threshold,
        },
        "candidate_replays_byte_identical": True,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "executor_source_audit": audit,
        "selected_candidate_id": selected["candidate_id"] if selected else None,
        "selected_executor": audit["testable_project_owned_boundary"] if selected else None,
        "disposition": disposition,
        "scientific_claim_boundary": (
            "Offline replay and source feasibility only; no model latency, closed-loop success, "
            "or paper-level positive result is established."
        ),
        "resources": {
            "gpu_count": 0,
            "model_queries": 0,
            "simulator_episodes": 0,
            "downloads": 0,
            "protected_outcomes_opened": 0,
        },
    }
    record["semantic_sha256"] = value_sha256(record)
    from savr.acr.records import ImmutableRecordStore

    ImmutableRecordStore(output_root).write_once("record", record)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0 if selected else 2


if __name__ == "__main__":
    raise SystemExit(main())
