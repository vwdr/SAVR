#!/usr/bin/env python3
"""Reconcile matched baselines, request bounded VOR retries, and freeze Phase 6."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_ROOT = Path("/home/ved/SAVR")
FR_RUN_ID = "phase6-fr-signals-v1"
SELECTION_RUN_ID = "phase6-savr-selection-v1"
BASELINE_RUN_ID = "phase6-matched-baselines-v1"
OUTPUT_ID = "phase6-final-calibration-v1"
BUDGET_TOLERANCE = 0.02


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def pairing_key(record: dict[str, Any]) -> tuple[int, int, int]:
    return (
        int(record["task"].split(":")[-1]),
        int(record["initial_state_id"]),
        int(record["seed"]),
    )


def load_complete_run(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((path / "run_summary.json").read_text(encoding="utf-8"))
    if manifest["status"] != "completed" or summary["status"] != "completed":
        raise RuntimeError(f"Run is not complete: {path.name}")
    episodes = [
        json.loads(item.read_text(encoding="utf-8"))
        for item in sorted((path / "episodes").glob("*.json"))
    ]
    return manifest, episodes


def summarize_setting(
    records: list[dict[str, Any]],
    fr_outcomes: dict[tuple[int, int, int], bool],
    paired_binary_counts: Any,
) -> dict[str, Any]:
    outcomes = {pairing_key(record): bool(record["success"]) for record in records}
    if len(records) != 100 or len(outcomes) != 100 or set(outcomes) != set(fr_outcomes):
        raise RuntimeError("Baseline setting does not contain 100 unique FR pairings")
    counts = paired_binary_counts(fr_outcomes, outcomes)
    queries = sum(int(record["query_count"]) for record in records)
    refreshes = sum(int(record["refresh_count"]) for record in records)
    reuses = sum(int(record["reuse_count"]) for record in records)
    if refreshes + reuses != queries:
        raise RuntimeError("Baseline refresh counts do not reconcile")
    return {
        "configuration_id": records[0]["configuration_id"],
        "policy": records[0]["policy"],
        "terminal_episodes": len(records),
        "runtime_error_episodes": sum(record["status"] != "completed" for record in records),
        "successes": sum(outcomes.values()),
        "paired_counts": {
            "both_success": counts.both_success,
            "fr_only_success": counts.fr_only_success,
            "candidate_only_success": counts.candidate_only_success,
            "both_failure": counts.both_failure,
        },
        "paired_success_difference": counts.success_difference,
        "queries": queries,
        "refreshes": refreshes,
        "reuses": reuses,
        "refresh_rate": refreshes / queries,
        "skip_rate": reuses / queries,
    }


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    if project_root != EXPECTED_ROOT:
        raise SystemExit(f"Refusing to run outside {EXPECTED_ROOT}: {project_root}")
    sys.path.insert(0, str(project_root / "src"))

    from savr.analysis.statistics import paired_binary_counts

    selection_path = (
        project_root / "results" / SELECTION_RUN_ID / "selection.json"
    )
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection["status"] != "completed":
        raise RuntimeError("SAVR selection is not complete")
    selected_savr = selection["selected_savr"]
    target = float(selected_savr["refresh_rate"])

    _, fr_records = load_complete_run(project_root / "results" / FR_RUN_ID)
    fr_outcomes = {pairing_key(record): bool(record["success"]) for record in fr_records}
    if len(fr_outcomes) != 100:
        raise RuntimeError("FR outcomes do not contain 100 unique pairings")

    baseline_manifest, baseline_records = load_complete_run(
        project_root / "results" / BASELINE_RUN_ID
    )
    records_by_setting: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in baseline_records:
        records_by_setting[record["configuration_id"]].append(record)
    if len(records_by_setting) != 2:
        raise RuntimeError("Initial matched-baseline run must contain VOR and PR")

    summaries = {
        identifier: summarize_setting(records, fr_outcomes, paired_binary_counts)
        for identifier, records in records_by_setting.items()
    }
    vor_attempts = [
        {
            "attempt": 1,
            "run_id": BASELINE_RUN_ID,
            "setting": next(
                setting
                for setting in baseline_manifest["configuration"]["settings"]
                if setting["policy"] == "VOR"
            ),
            "summary": next(
                summary for summary in summaries.values()
                if summary["policy"] == "VOR"
            ),
        }
    ]
    pr_summary = next(
        summary for summary in summaries.values() if summary["policy"] == "PR"
    )
    pr_setting = next(
        setting
        for setting in baseline_manifest["configuration"]["settings"]
        if setting["policy"] == "PR"
    )

    for attempt in (2, 3):
        run_id = f"phase6-vor-attempt-{attempt}-v1"
        run_dir = project_root / "results" / run_id
        if not (run_dir / "manifest.json").exists():
            break
        manifest, records = load_complete_run(run_dir)
        if len(manifest["configuration"]["settings"]) != 1:
            raise RuntimeError(f"{run_id} must contain exactly one VOR setting")
        vor_attempts.append(
            {
                "attempt": attempt,
                "run_id": run_id,
                "setting": manifest["configuration"]["settings"][0],
                "summary": summarize_setting(
                    records,
                    fr_outcomes,
                    paired_binary_counts,
                ),
            }
        )

    matched = [
        attempt
        for attempt in vor_attempts
        if abs(float(attempt["summary"]["refresh_rate"]) - target)
        <= BUDGET_TOLERANCE
    ]
    if matched:
        frozen_vor = matched[0]
        vor_budget_status = "matched-budget"
    elif len(vor_attempts) < 3:
        latest = vor_attempts[-1]
        observed = float(latest["summary"]["refresh_rate"])
        latest_simulated = float(latest["setting"]["simulated_refresh_rate"])
        need_lower = observed > target
        used_thresholds = {
            float(attempt["setting"]["image_threshold"]) for attempt in vor_attempts
        }
        ranked = selection["vor_ranked_unique_thresholds"]
        directional = [
            item
            for item in ranked
            if float(item["image_threshold"]) not in used_thresholds
            and (
                float(item["simulated_refresh_rate"]) < latest_simulated
                if need_lower
                else float(item["simulated_refresh_rate"]) > latest_simulated
            )
        ]
        if not directional:
            raise RuntimeError("No unused VOR threshold remains in the needed direction")
        selected = directional[0]
        attempt_number = len(vor_attempts) + 1
        retry_config = {
            "artifact_cap_bytes": 268435456,
            "protocol": "PHASE6_CALIBRATION_PROTOCOL.md",
            "run_id": f"phase6-vor-attempt-{attempt_number}-v1",
            "settings": [
                {
                    "configuration_id": f"vor-attempt-{attempt_number}",
                    "policy": "VOR",
                    "image_threshold": selected["image_threshold"],
                    "max_reuse_horizon": int(
                        selected_savr["setting"]["max_reuse_horizon"]
                    ),
                    "quantile": selected["quantile"],
                    "target_refresh_rate": target,
                    "simulated_refresh_rate": selected["simulated_refresh_rate"],
                }
            ],
            "wall_cap_seconds": 10800,
        }
        output_dir = project_root / "results" / OUTPUT_ID
        output_dir.mkdir(parents=True, exist_ok=True)
        artifact = {
            "run_id": OUTPUT_ID,
            "status": "needs_vor_retry",
            "created_at_utc": utc_now(),
            "target_refresh_rate": target,
            "budget_tolerance": BUDGET_TOLERANCE,
            "vor_attempts": vor_attempts,
            "generated_retry_config": retry_config,
        }
        atomic_json(output_dir / "baseline_analysis.json", artifact)
        atomic_json(
            output_dir / f"phase6_vor_attempt_{attempt_number}.generated.json",
            retry_config,
        )
        print(json.dumps(artifact, indent=2, sort_keys=True))
        return 3
    else:
        frozen_vor = min(
            vor_attempts,
            key=lambda attempt: abs(
                float(attempt["summary"]["refresh_rate"]) - target
            ),
        )
        vor_budget_status = "nearest-budget"

    pr_budget_gap = abs(float(pr_summary["refresh_rate"]) - target)
    pr_budget_status = (
        "matched-budget"
        if pr_budget_gap <= BUDGET_TOLERANCE
        else "nearest-budget"
    )
    frozen = {
        "FR": {
            "configuration_id": "fr",
            "policy": "FR",
            "refresh_rate": 1.0,
        },
        "SAVR": {
            **selected_savr["setting"],
            "observed_refresh_rate": selected_savr["refresh_rate"],
            "observed_successes": selected_savr["successes"],
        },
        "VOR": {
            **frozen_vor["setting"],
            "observed_refresh_rate": frozen_vor["summary"]["refresh_rate"],
            "observed_successes": frozen_vor["summary"]["successes"],
            "budget_status": vor_budget_status,
        },
        "PR": {
            **pr_setting,
            "observed_refresh_rate": pr_summary["refresh_rate"],
            "observed_successes": pr_summary["successes"],
            "budget_status": pr_budget_status,
        },
    }
    artifact = {
        "run_id": OUTPUT_ID,
        "status": "completed",
        "created_at_utc": utc_now(),
        "target_refresh_rate": target,
        "budget_tolerance": BUDGET_TOLERANCE,
        "vor_attempts": vor_attempts,
        "frozen_vor_attempt": frozen_vor["attempt"],
        "vor_budget_status": vor_budget_status,
        "pr_summary": pr_summary,
        "pr_budget_status": pr_budget_status,
        "provisional_power": selection["provisional_power"],
        "power_sensitivity": selection["power_sensitivity"],
        "frozen_primary_configurations": frozen,
        "claim_boundary": "Phase 6 calibration only; no final inference.",
    }
    output_dir = project_root / "results" / OUTPUT_ID
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(output_dir / "baseline_analysis.json", artifact)
    atomic_json(output_dir / "frozen_primary_configurations.json", frozen)
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
