#!/usr/bin/env python3
"""Reconcile Phase 6 failures and extract redesign-relevant diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


FR_RUN_ID = "phase6-fr-signals-v1"
GRID_RUN_ID = "phase6-savr-grid-v1"
THRESHOLD_RUN_ID = "phase6-savr-thresholds-v1"
EXPECTED_FR_EPISODES = 100
EXPECTED_GRID_EPISODES = 900
EXPECTED_CONFIGURATIONS = 9


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def percentile(values: Iterable[float], quantile: float) -> float | None:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("Quantile must lie in [0, 1]")
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def pairing_key(record: dict[str, Any]) -> tuple[int, int, int]:
    return (
        int(str(record["task"]).split(":")[-1]),
        int(record["initial_state_id"]),
        int(record["seed"]),
    )


def episode_pairing_from_id(episode_id: str) -> tuple[int, int]:
    parts = episode_id.rsplit("_", 4)
    if len(parts) < 5 or parts[-4] != "task" or parts[-2] != "state":
        raise RuntimeError(f"Unexpected episode identifier: {episode_id}")
    return int(parts[-3]), int(parts[-1])


def load_complete_run(
    run_dir: Path,
    *,
    expected_episodes: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    manifest_path = run_dir / "manifest.json"
    summary_path = run_dir / "run_summary.json"
    manifest = load_json(manifest_path)
    summary = load_json(summary_path)
    if manifest["status"] != "completed" or summary["status"] != "completed":
        raise RuntimeError(f"Run is not complete: {run_dir.name}")
    episodes = [
        load_json(path) for path in sorted((run_dir / "episodes").glob("*.json"))
    ]
    queries = [
        load_json(path) for path in sorted((run_dir / "queries").glob("*.json"))
    ]
    if len(episodes) != expected_episodes:
        raise RuntimeError(
            f"{run_dir.name} has {len(episodes)} episodes, expected {expected_episodes}"
        )
    if any(record["status"] != "completed" for record in episodes):
        raise RuntimeError(f"{run_dir.name} includes a non-complete terminal episode")
    if sum(int(record["query_count"]) for record in episodes) != len(queries):
        raise RuntimeError(f"{run_dir.name} query records do not reconcile")
    return manifest, episodes, queries


def group_queries(
    records: Iterable[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["episode_id"])].append(record)
    for episode_id, queries in grouped.items():
        queries.sort(key=lambda item: int(item["episode_query_index"]))
        expected = list(range(len(queries)))
        actual = [int(item["episode_query_index"]) for item in queries]
        if actual != expected:
            raise RuntimeError(f"Non-contiguous query indices for {episode_id}")
    return dict(grouped)


def ratio(value: Any, threshold: Any) -> float | None:
    if value is None or threshold is None:
        return None
    threshold_value = float(threshold)
    if threshold_value == 0.0:
        return math.inf if float(value) > 0.0 else 0.0
    return float(value) / threshold_value


def distribution(values: Iterable[float]) -> dict[str, float | int | None]:
    materialized = [float(value) for value in values]
    return {
        "count": len(materialized),
        "minimum": min(materialized) if materialized else None,
        "median": percentile(materialized, 0.5),
        "p95": percentile(materialized, 0.95),
        "maximum": max(materialized) if materialized else None,
    }


def task_successes(episodes: Iterable[dict[str, Any]]) -> dict[str, dict[str, int]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in episodes:
        grouped[pairing_key(record)[0]].append(record)
    return {
        str(task): {
            "successes": sum(bool(record["success"]) for record in records),
            "episodes": len(records),
        }
        for task, records in sorted(grouped.items())
    }


def maximum_reuse_streak(queries: list[dict[str, Any]]) -> int:
    longest = 0
    current = 0
    for record in queries:
        if not bool(record["refresh"]):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def position_bin(query_index: int, query_count: int) -> str:
    if query_count <= 1:
        return "only"
    fraction = query_index / (query_count - 1)
    if fraction < 1.0 / 3.0:
        return "early"
    if fraction < 2.0 / 3.0:
        return "middle"
    return "late"


def build_episode_diagnostic(
    episode: dict[str, Any],
    queries: list[dict[str, Any]],
    fr_queries: list[dict[str, Any]],
) -> dict[str, Any]:
    reuse_queries = [record for record in queries if not bool(record["refresh"])]
    if len(reuse_queries) != int(episode["reuse_count"]):
        raise RuntimeError(f"Reuse count differs for {episode['episode_id']}")
    first_reuse = reuse_queries[0] if reuse_queries else None

    comparable = min(len(queries), len(fr_queries))
    first_action_mismatch = next(
        (
            index
            for index in range(comparable)
            if queries[index]["actions_sha256"] != fr_queries[index]["actions_sha256"]
        ),
        None,
    )
    prefix_matches = first_action_mismatch if first_action_mismatch is not None else comparable
    first_reuse_index = (
        int(first_reuse["episode_query_index"]) if first_reuse is not None else None
    )
    if first_reuse_index is None:
        action_divergence_class = "no_reuse"
    elif first_action_mismatch is None:
        action_divergence_class = "no_mismatch_in_comparable_prefix"
    elif first_action_mismatch < first_reuse_index:
        action_divergence_class = "mismatch_before_first_reuse"
    elif first_action_mismatch == first_reuse_index:
        action_divergence_class = "mismatch_at_first_reuse"
    else:
        action_divergence_class = "mismatch_after_first_reuse"

    hidden_camera_exceedances = 0
    consecutive_reuses = 0
    camera_exceedance_counts: Counter[str] = Counter()
    reuse_position_counts: Counter[str] = Counter()
    image_ratios: list[float] = []
    state_ratios: list[float] = []
    action_ratios: list[float] = []
    for record in reuse_queries:
        query_index = int(record["episode_query_index"])
        if query_index > 0 and not bool(queries[query_index - 1]["refresh"]):
            consecutive_reuses += 1
        decision = record["decision"]
        thresholds = decision["thresholds"]
        image_ratio = ratio(decision["image_score"], thresholds.get("image"))
        state_ratio = ratio(decision["state_score"], thresholds.get("state"))
        action_ratio = ratio(decision["action_score"], thresholds.get("action"))
        if image_ratio is not None:
            image_ratios.append(image_ratio)
        if state_ratio is not None:
            state_ratios.append(state_ratio)
        if action_ratio is not None:
            action_ratios.append(action_ratio)

        image_threshold = float(thresholds["image"])
        exceeding = [
            name
            for name, score in decision["per_camera_image_scores"].items()
            if float(score) > image_threshold
        ]
        if exceeding:
            hidden_camera_exceedances += 1
            camera_exceedance_counts.update(exceeding)
        reuse_position_counts[position_bin(
            int(record["episode_query_index"]), len(queries)
        )] += 1

    return {
        "configuration_id": episode["configuration_id"],
        "episode_id": episode["episode_id"],
        "task_id": pairing_key(episode)[0],
        "initial_state_id": int(episode["initial_state_id"]),
        "success": bool(episode["success"]),
        "query_count": len(queries),
        "fr_query_count": len(fr_queries),
        "reuse_count": len(reuse_queries),
        "refresh_rate": float(episode["refresh_rate"]),
        "first_reuse_query_index": first_reuse_index,
        "first_reuse_position": (
            position_bin(first_reuse_index, len(queries))
            if first_reuse_index is not None
            else None
        ),
        "maximum_reuse_streak": maximum_reuse_streak(queries),
        "first_action_mismatch_query_index": first_action_mismatch,
        "matching_action_prefix_queries": prefix_matches,
        "action_divergence_class": action_divergence_class,
        "first_mismatch_equals_first_reuse": (
            first_action_mismatch == first_reuse_index
            if first_reuse_index is not None and first_action_mismatch is not None
            else None
        ),
        "hidden_camera_exceedance_reuses": hidden_camera_exceedances,
        "consecutive_reuse_queries": consecutive_reuses,
        "camera_exceedance_counts": dict(sorted(camera_exceedance_counts.items())),
        "reuse_position_counts": dict(sorted(reuse_position_counts.items())),
        "reuse_signal_ratio_maxima": {
            "image": max(image_ratios) if image_ratios else None,
            "state": max(state_ratios) if state_ratios else None,
            "action": max(action_ratios) if action_ratios else None,
        },
    }


def summarize_episode_diagnostics(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    successes = [record for record in records if record["success"]]
    failures = [record for record in records if not record["success"]]
    reuse_total = sum(int(record["reuse_count"]) for record in records)
    hidden_total = sum(
        int(record["hidden_camera_exceedance_reuses"]) for record in records
    )
    mismatch_comparable = [
        record for record in records
        if record["first_mismatch_equals_first_reuse"] is not None
    ]
    position_counts: Counter[str] = Counter()
    camera_counts: Counter[str] = Counter()
    for record in records:
        position_counts.update(record["reuse_position_counts"])
        camera_counts.update(record["camera_exceedance_counts"])

    def cohort_summary(cohort: list[dict[str, Any]]) -> dict[str, Any]:
        cohort_reuses = sum(int(record["reuse_count"]) for record in cohort)
        cohort_hidden = sum(
            int(record["hidden_camera_exceedance_reuses"]) for record in cohort
        )
        cohort_consecutive = sum(
            int(record["consecutive_reuse_queries"]) for record in cohort
        )
        cohort_positions: Counter[str] = Counter()
        cohort_cameras: Counter[str] = Counter()
        for record in cohort:
            cohort_positions.update(record["reuse_position_counts"])
            cohort_cameras.update(record["camera_exceedance_counts"])
        return {
            "episodes": len(cohort),
            "reuse_queries": cohort_reuses,
            "hidden_camera_exceedance_reuses": cohort_hidden,
            "hidden_camera_exceedance_fraction": (
                cohort_hidden / cohort_reuses if cohort_reuses else None
            ),
            "consecutive_reuse_queries": cohort_consecutive,
            "consecutive_reuse_fraction": (
                cohort_consecutive / cohort_reuses if cohort_reuses else None
            ),
            "camera_exceedance_counts": dict(sorted(cohort_cameras.items())),
            "reuse_position_counts": dict(sorted(cohort_positions.items())),
            "reuse_count": distribution(
                int(record["reuse_count"]) for record in cohort
            ),
            "first_reuse_query_index": distribution(
                int(record["first_reuse_query_index"])
                for record in cohort
                if record["first_reuse_query_index"] is not None
            ),
            "maximum_reuse_streak": distribution(
                int(record["maximum_reuse_streak"]) for record in cohort
            ),
            "query_count": distribution(int(record["query_count"]) for record in cohort),
            "reuse_signal_ratio_maxima": {
                signal: {
                    **distribution(
                        float(record["reuse_signal_ratio_maxima"][signal])
                        for record in cohort
                        if record["reuse_signal_ratio_maxima"][signal] is not None
                    ),
                    "episode_fraction_at_or_above_0_9": (
                        sum(
                            float(record["reuse_signal_ratio_maxima"][signal]) >= 0.9
                            for record in cohort
                            if record["reuse_signal_ratio_maxima"][signal] is not None
                        )
                        / sum(
                            record["reuse_signal_ratio_maxima"][signal] is not None
                            for record in cohort
                        )
                        if any(
                            record["reuse_signal_ratio_maxima"][signal] is not None
                            for record in cohort
                        )
                        else None
                    ),
                }
                for signal in ("image", "state", "action")
            },
        }

    return {
        "episodes": len(records),
        "successes": len(successes),
        "failures": len(failures),
        "reuse_queries": reuse_total,
        "reuse_position_counts": dict(sorted(position_counts.items())),
        "hidden_camera_exceedance_reuses": hidden_total,
        "hidden_camera_exceedance_fraction": (
            hidden_total / reuse_total if reuse_total else None
        ),
        "camera_exceedance_counts": dict(sorted(camera_counts.items())),
        "first_mismatch_equals_first_reuse": {
            "episodes_compared": len(mismatch_comparable),
            "episodes_equal": sum(
                bool(record["first_mismatch_equals_first_reuse"])
                for record in mismatch_comparable
            ),
        },
        "action_divergence_classes": dict(sorted(Counter(
            str(record["action_divergence_class"]) for record in records
        ).items())),
        "successful_episodes": cohort_summary(successes),
        "failed_episodes": cohort_summary(failures),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    project_root = arguments.project_root.resolve()

    fr_dir = project_root / "results" / FR_RUN_ID
    grid_dir = project_root / "results" / GRID_RUN_ID
    threshold_path = (
        project_root
        / "results"
        / THRESHOLD_RUN_ID
        / "threshold_derivation.json"
    )
    fr_manifest, fr_episodes, fr_queries = load_complete_run(
        fr_dir,
        expected_episodes=EXPECTED_FR_EPISODES,
    )
    grid_manifest, grid_episodes, grid_queries = load_complete_run(
        grid_dir,
        expected_episodes=EXPECTED_GRID_EPISODES,
    )
    threshold_artifact = load_json(threshold_path)

    fr_episode_by_pairing = {pairing_key(record): record for record in fr_episodes}
    if len(fr_episode_by_pairing) != EXPECTED_FR_EPISODES:
        raise RuntimeError("FR pairing keys are not unique")
    fr_queries_by_episode = group_queries(fr_queries)
    grid_queries_by_episode = group_queries(grid_queries)

    episodes_by_configuration: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for episode in grid_episodes:
        episodes_by_configuration[str(episode["configuration_id"])].append(episode)
    if len(episodes_by_configuration) != EXPECTED_CONFIGURATIONS:
        raise RuntimeError("Phase 6 grid does not contain nine configurations")

    offline_by_configuration = {
        str(record["configuration_id"]): record
        for record in threshold_artifact["candidates"]
    }
    configuration_summaries = []
    episode_diagnostics: list[dict[str, Any]] = []
    for configuration_id, episodes in sorted(episodes_by_configuration.items()):
        diagnostics = []
        for episode in sorted(episodes, key=pairing_key):
            task_id, state_id, seed = pairing_key(episode)
            fr_episode = fr_episode_by_pairing[(task_id, state_id, seed)]
            fr_episode_id = str(fr_episode["episode_id"])
            record = build_episode_diagnostic(
                episode,
                grid_queries_by_episode[str(episode["episode_id"])],
                fr_queries_by_episode[fr_episode_id],
            )
            diagnostics.append(record)
            episode_diagnostics.append(record)

        query_count = sum(int(record["query_count"]) for record in episodes)
        reuse_count = sum(int(record["reuse_count"]) for record in episodes)
        online_skip_rate = reuse_count / query_count
        offline_skip_rate = float(
            offline_by_configuration[configuration_id]["simulated_skip_rate"]
        )
        configuration_summaries.append(
            {
                "configuration_id": configuration_id,
                "successes": sum(bool(record["success"]) for record in episodes),
                "episodes": len(episodes),
                "query_count": query_count,
                "reuse_count": reuse_count,
                "online_skip_rate": online_skip_rate,
                "offline_skip_rate": offline_skip_rate,
                "online_minus_offline_skip_rate": online_skip_rate - offline_skip_rate,
                "task_successes": task_successes(episodes),
                "diagnostics": summarize_episode_diagnostics(diagnostics),
            }
        )

    best_configuration = max(
        configuration_summaries,
        key=lambda record: (int(record["successes"]), -float(record["online_skip_rate"])),
    )
    best_id = str(best_configuration["configuration_id"])
    best_episode_diagnostics = [
        record
        for record in episode_diagnostics
        if record["configuration_id"] == best_id
    ]

    first_reuse_buckets: dict[str, list[bool]] = defaultdict(list)
    for record in best_episode_diagnostics:
        index = record["first_reuse_query_index"]
        if index is None:
            bucket = "none"
        elif index <= 2:
            bucket = "query_2_or_earlier"
        elif index <= 4:
            bucket = "query_3_to_4"
        else:
            bucket = "query_5_or_later"
        first_reuse_buckets[bucket].append(bool(record["success"]))

    artifact = {
        "analysis_id": "phase6r-a-diagnosis-v1",
        "claim_boundary": (
            "Forensic analysis of previously observed Phase 6 calibration outcomes; "
            "not a final-holdout result and not proof of causality."
        ),
        "source_integrity": {
            "fr_manifest_sha256": sha256(fr_dir / "manifest.json"),
            "fr_summary_sha256": sha256(fr_dir / "run_summary.json"),
            "grid_manifest_sha256": sha256(grid_dir / "manifest.json"),
            "grid_summary_sha256": sha256(grid_dir / "run_summary.json"),
            "threshold_derivation_sha256": sha256(threshold_path),
            "fr_trace_input_sha256": threshold_artifact[
                "source_input_combined_sha256"
            ],
            "grid_git_revision": grid_manifest["savr_git_revision"],
        },
        "fr": {
            "episodes": len(fr_episodes),
            "successes": sum(bool(record["success"]) for record in fr_episodes),
            "query_count": len(fr_queries),
            "task_successes": task_successes(fr_episodes),
        },
        "configurations": configuration_summaries,
        "best_observed_configuration": best_id,
        "best_configuration_first_reuse_buckets": {
            bucket: {
                "episodes": len(outcomes),
                "successes": sum(outcomes),
                "success_rate": sum(outcomes) / len(outcomes),
            }
            for bucket, outcomes in sorted(first_reuse_buckets.items())
        },
        "episode_diagnostics": episode_diagnostics,
        "limitations": [
            (
                "SAVR query records contain aggregate signal scores but not raw "
                "online images, states, or actions."
            ),
            "No rollout video or task-phase annotation was collected in Phase 6.",
            (
                "After the first action divergence, same-index FR/SAVR action hashes "
                "are not paired observations."
            ),
            (
                "Associations between reuse patterns and success do not by "
                "themselves establish a causal failure mechanism."
            ),
        ],
    }

    rendered = json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if arguments.output is None:
        print(rendered, end="")
    else:
        output = arguments.output.resolve()
        if project_root not in output.parents:
            raise SystemExit("Output must remain inside the SAVR project")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
