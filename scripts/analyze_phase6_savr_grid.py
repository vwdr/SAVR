#!/usr/bin/env python3
"""Select primary SAVR and derive first matched-budget baseline settings."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_ROOT = Path("/home/ved/SAVR")
FR_RUN_ID = "phase6-fr-signals-v1"
GRID_RUN_ID = "phase6-savr-grid-v1"
OUTPUT_ID = "phase6-savr-selection-v1"
SUCCESS_MARGIN = 0.02
BUDGET_TOLERANCE = 0.02


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("Cannot summarize an empty distribution")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def pairing_key(record: dict[str, Any]) -> tuple[int, int, int]:
    return (
        int(record["task"].split(":")[-1]),
        int(record["initial_state_id"]),
        int(record["seed"]),
    )


def load_terminal_episodes(run_dir: Path) -> list[dict[str, Any]]:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
    if manifest["status"] != "completed" or summary["status"] != "completed":
        raise RuntimeError(f"Run is not complete: {run_dir.name}")
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((run_dir / "episodes").glob("*.json"))
    ]


def load_fr_traces(project_root: Path) -> tuple[list[Any], Any, list[int]]:
    from savr.calibration import SignalBounds, query_from_record

    fr_dir = project_root / "results" / FR_RUN_ID
    manifest = json.loads((fr_dir / "manifest.json").read_text(encoding="utf-8"))
    episodes = load_terminal_episodes(fr_dir)
    if len(episodes) != 100 or any(record["status"] != "completed" for record in episodes):
        raise RuntimeError("FR calibration matrix is not 100 complete episodes")
    queries_by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in sorted((fr_dir / "queries").glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        queries_by_episode[record["episode_id"]].append(record)
    traces = []
    lengths = []
    for episode in sorted(episodes, key=pairing_key):
        queries = sorted(
            queries_by_episode[episode["episode_id"]],
            key=lambda item: int(item["episode_query_index"]),
        )
        if len(queries) != int(episode["query_count"]):
            raise RuntimeError("FR trace count differs from episode count")
        traces.append(tuple(query_from_record(record) for record in queries))
        lengths.append(len(queries))
    statistics = manifest["normalization_statistics"]
    bounds = SignalBounds(
        state_q01=tuple(statistics["state_q01"]),
        state_q99=tuple(statistics["state_q99"]),
        action_q01=tuple(statistics["action_q01"]),
        action_q99=tuple(statistics["action_q99"]),
    )
    return traces, bounds, lengths


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    if project_root != EXPECTED_ROOT:
        raise SystemExit(f"Refusing to run outside {EXPECTED_ROOT}: {project_root}")
    sys.path.insert(0, str(project_root / "src"))

    from savr.analysis.statistics import (
        paired_binary_counts,
        planning_power_result,
        power_sensitivity,
    )
    from savr.calibration import derive_vor_candidates, select_period

    fr_dir = project_root / "results" / FR_RUN_ID
    grid_dir = project_root / "results" / GRID_RUN_ID
    fr_episodes = load_terminal_episodes(fr_dir)
    grid_episodes = load_terminal_episodes(grid_dir)
    if len(fr_episodes) != 100 or len(grid_episodes) != 900:
        raise RuntimeError("FR or SAVR grid episode count differs from protocol")
    fr_outcomes = {pairing_key(record): bool(record["success"]) for record in fr_episodes}
    if len(fr_outcomes) != 100:
        raise RuntimeError("FR pairings are not unique")

    grid_manifest = json.loads(
        (grid_dir / "manifest.json").read_text(encoding="utf-8")
    )
    settings = {
        item["configuration_id"]: item
        for item in grid_manifest["configuration"]["settings"]
    }
    if len(settings) != 9:
        raise RuntimeError("SAVR grid does not contain nine frozen settings")

    queries_by_setting: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in sorted((grid_dir / "queries").glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        queries_by_setting[record["configuration_id"]].append(record)
    episodes_by_setting: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in grid_episodes:
        episodes_by_setting[record["configuration_id"]].append(record)

    candidate_summaries = []
    for identifier, setting in sorted(settings.items()):
        episodes = episodes_by_setting[identifier]
        outcomes = {pairing_key(record): bool(record["success"]) for record in episodes}
        if len(episodes) != 100 or len(outcomes) != 100:
            raise RuntimeError(f"{identifier} lacks 100 unique pairings")
        counts = paired_binary_counts(fr_outcomes, outcomes)
        failures = sum(record["status"] != "completed" for record in episodes)
        queries = sum(int(record["query_count"]) for record in episodes)
        refreshes = sum(int(record["refresh_count"]) for record in episodes)
        reuses = sum(int(record["reuse_count"]) for record in episodes)
        if refreshes + reuses != queries:
            raise RuntimeError(f"{identifier} refresh counts do not reconcile")
        task_zero = []
        for task_id in range(10):
            fr_task_success = sum(
                success for (task, _state, _seed), success in fr_outcomes.items()
                if task == task_id
            )
            candidate_task_success = sum(
                success for (task, _state, _seed), success in outcomes.items()
                if task == task_id
            )
            if fr_task_success >= 1 and candidate_task_success == 0:
                task_zero.append(task_id)
        latencies = [
            float(record["timing"]["query_wall_ms"])
            for record in queries_by_setting[identifier]
        ]
        if len(latencies) != queries:
            raise RuntimeError(f"{identifier} query latency count differs")
        eligible = (
            failures == 0
            and counts.success_difference >= -SUCCESS_MARGIN
            and not task_zero
        )
        candidate_summaries.append(
            {
                "configuration_id": identifier,
                "setting": setting,
                "terminal_episodes": len(episodes),
                "runtime_error_episodes": failures,
                "successes": sum(outcomes.values()),
                "paired_counts": {
                    "both_success": counts.both_success,
                    "fr_only_success": counts.fr_only_success,
                    "candidate_only_success": counts.candidate_only_success,
                    "both_failure": counts.both_failure,
                },
                "paired_success_difference": counts.success_difference,
                "discordance_rate": counts.discordance_rate,
                "queries": queries,
                "refreshes": refreshes,
                "reuses": reuses,
                "refresh_rate": refreshes / queries,
                "skip_rate": reuses / queries,
                "query_wall_ms_median": percentile(latencies, 0.5),
                "query_wall_ms_p95": percentile(latencies, 0.95),
                "zero_success_tasks_with_fr_success": task_zero,
                "eligible": eligible,
            }
        )

    eligible = [item for item in candidate_summaries if item["eligible"]]
    if not eligible:
        output_dir = project_root / "results" / OUTPUT_ID
        output_dir.mkdir(parents=True, exist_ok=True)
        artifact = {
            "run_id": OUTPUT_ID,
            "status": "no_eligible_savr_candidate",
            "created_at_utc": utc_now(),
            "success_margin": SUCCESS_MARGIN,
            "candidates": candidate_summaries,
            "claim_boundary": "Negative calibration result; no final inference.",
        }
        atomic_json(output_dir / "selection.json", artifact)
        print(json.dumps(artifact, indent=2, sort_keys=True))
        return 2

    selected = min(
        eligible,
        key=lambda item: (
            item["refresh_rate"],
            -item["paired_success_difference"],
            item["query_wall_ms_median"],
            int(item["setting"]["max_reuse_horizon"]),
            item["configuration_id"],
        ),
    )
    traces, bounds, query_lengths = load_fr_traces(project_root)
    target_refresh_rate = float(selected["refresh_rate"])
    horizon = int(selected["setting"]["max_reuse_horizon"])
    vor_ranked = derive_vor_candidates(
        traces,
        bounds=bounds,
        target_refresh_rate=target_refresh_rate,
        max_reuse_horizon=horizon,
    )
    first_vor = vor_ranked[0]
    period, expected_pr_refresh_rate = select_period(
        query_lengths,
        target_refresh_rate=target_refresh_rate,
    )

    matched_config = {
        "artifact_cap_bytes": 536870912,
        "protocol": "PHASE6_CALIBRATION_PROTOCOL.md",
        "run_id": "phase6-matched-baselines-v1",
        "settings": [
            {
                "configuration_id": "vor-attempt-1",
                "policy": "VOR",
                "image_threshold": first_vor.image_threshold,
                "max_reuse_horizon": horizon,
                "quantile": first_vor.quantile,
                "target_refresh_rate": target_refresh_rate,
                "simulated_refresh_rate": (
                    first_vor.simulated_refreshes
                    / (first_vor.simulated_refreshes + first_vor.simulated_reuses)
                ),
            },
            {
                "configuration_id": f"pr-k{period}",
                "policy": "PR",
                "period": period,
                "target_refresh_rate": target_refresh_rate,
                "expected_refresh_rate": expected_pr_refresh_rate,
            },
        ],
        "wall_cap_seconds": 25200,
    }

    selected_counts = paired_binary_counts(
        fr_outcomes,
        {
            pairing_key(record): bool(record["success"])
            for record in episodes_by_setting[selected["configuration_id"]]
        },
    )
    input_paths = [
        fr_dir / "manifest.json",
        fr_dir / "run_summary.json",
        grid_dir / "manifest.json",
        grid_dir / "run_summary.json",
    ]
    artifact = {
        "run_id": OUTPUT_ID,
        "status": "completed",
        "created_at_utc": utc_now(),
        "success_margin": SUCCESS_MARGIN,
        "budget_tolerance": BUDGET_TOLERANCE,
        "fr_successes": sum(fr_outcomes.values()),
        "candidate_selection_order": [
            "lowest_refresh_rate",
            "highest_paired_success_difference",
            "lowest_median_query_wall_time",
            "smallest_horizon",
            "configuration_id",
        ],
        "candidates": candidate_summaries,
        "selected_savr": selected,
        "provisional_power": planning_power_result(selected_counts),
        "power_sensitivity": power_sensitivity(),
        "vor_ranked_unique_thresholds": [
            {
                "quantile": item.quantile,
                "image_threshold": item.image_threshold,
                "simulated_refresh_rate": (
                    item.simulated_refreshes
                    / (item.simulated_refreshes + item.simulated_reuses)
                ),
            }
            for item in list(
                {
                    item.image_threshold: item for item in vor_ranked
                }.values()
            )
        ],
        "selected_pr": {
            "period": period,
            "expected_refresh_rate": expected_pr_refresh_rate,
        },
        "generated_matched_baseline_config": matched_config,
        "input_hashes": {
            str(path.relative_to(project_root)): sha256(path) for path in input_paths
        },
        "claim_boundary": "Calibration selection only; no final inference.",
    }
    output_dir = project_root / "results" / OUTPUT_ID
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(output_dir / "selection.json", artifact)
    atomic_json(output_dir / "phase6_matched_baselines.generated.json", matched_config)
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
