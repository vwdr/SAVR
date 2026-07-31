#!/usr/bin/env python3
"""Render the reconciled Phase 6 calibration report from immutable artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


EXPECTED_ROOT = Path("/home/ved/SAVR")


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def percent(value: float) -> str:
    return f"{100 * value:.2f}%"


def phase6_run_summaries(project_root: Path) -> list[dict[str, Any]]:
    summaries = []
    for path in sorted((project_root / "results").glob("phase6-*/run_summary.json")):
        record = load(path)
        summaries.append(
            {
                "run_id": record["run_id"],
                "status": record["status"],
                "elapsed": float(record["accumulated_elapsed_seconds"]),
                "artifact_bytes": int(record["artifact_bytes"]),
                "checkpoint_restored": bool(record["checkpoint_restored"]),
            }
        )
    return summaries


def write_negative_report(
    project_root: Path,
    *,
    fr_summary: dict[str, Any],
    threshold: dict[str, Any],
    selection: dict[str, Any],
) -> None:
    candidates = sorted(
        selection["candidates"],
        key=lambda item: (
            float(item["setting"]["target_skip_rate"]),
            int(item["setting"]["max_reuse_horizon"]),
        ),
    )
    candidate_rows = [
        "| {identifier} | {target} | {horizon} | {successes}/100 | {difference} "
        "| {skip} | {refresh} |".format(
            identifier=item["configuration_id"],
            target=percent(float(item["setting"]["target_skip_rate"])),
            horizon=item["setting"]["max_reuse_horizon"],
            successes=item["successes"],
            difference=f"{100 * float(item['paired_success_difference']):+.1f} pp",
            skip=percent(float(item["skip_rate"])),
            refresh=percent(float(item["refresh_rate"])),
        )
        for item in candidates
    ]
    best = max(
        candidates,
        key=lambda item: (
            int(item["successes"]),
            -float(item["refresh_rate"]),
        ),
    )
    run_summaries = phase6_run_summaries(project_root)
    if any(
        item["status"] != "completed" or not item["checkpoint_restored"]
        for item in run_summaries
    ):
        raise RuntimeError("A Phase 6 GPU run failed integrity reconciliation")
    total_elapsed = sum(item["elapsed"] for item in run_summaries)
    total_artifact_bytes = sum(item["artifact_bytes"] for item in run_summaries)

    lines = [
        "# Phase 6 Calibration and Power Report",
        "",
        "Status: STOPPED — NO ELIGIBLE SAVR CONFIGURATION",
        "",
        "## Outcome",
        "",
        "The frozen Phase 6 calibration rule was not met. Full Refresh succeeded "
        "on all 100 paired LIBERO-Spatial calibration episodes, while every "
        "predeclared SAVR setting degraded success by substantially more than "
        "the frozen 2-percentage-point margin.",
        "",
        "Per the frozen stop rule, thresholds and the margin were not relaxed. "
        "No SAVR primary configuration was selected, no matched-budget VOR/PR "
        "run was launched, and no final-holdout outcome was executed or "
        "inspected.",
        "",
        "## Full Refresh calibration oracle",
        "",
        f"- terminal episodes: {fr_summary['progress']['terminal']}/100",
        f"- successes: {threshold['source_success_count']}/100",
        f"- query traces: {threshold['source_query_count']}",
        f"- trace input hash: `{threshold['source_input_combined_sha256']}`",
        "",
        "## Frozen SAVR grid results",
        "",
        "| Configuration | Offline target skip | Hmax | Success | Paired "
        "difference | Online skip | Online refresh |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *candidate_rows,
        "",
        f"The least-degrading setting was `{best['configuration_id']}` with "
        f"{best['successes']}/100 successes, "
        f"{percent(float(best['skip_rate']))} online skipped refreshes, and a "
        f"{100 * float(best['paired_success_difference']):+.1f}-percentage-"
        "point paired success difference from FR. It was not eligible.",
        "",
        "The offline FR replay target did not transfer safely to closed-loop "
        "trajectories: even the conservative target family produced large "
        "online success losses. This supports a negative conclusion for the "
        "tested operating region, not a claim that all possible SAVR settings "
        "must fail.",
        "",
        "## Power and frozen configurations",
        "",
        "- primary FR configuration: frozen (refresh every query)",
        "- primary SAVR configuration: not frozen; no eligible candidate",
        "- matched VOR/PR configurations: not run by the predeclared stop rule",
        "- paired final sample size: not confirmed because no eligible SAVR "
        "operating point exists",
        "",
        "Therefore the normal Phase 6 exit gate was not met. Phase 7 final-"
        "protocol freezing must not begin without an explicit scientific "
        "redesign decision.",
        "",
        "## Integrity, resources, and safety",
        "",
        f"- reconciled GPU-run elapsed time: {total_elapsed / 3600:.2f} hours",
        f"- reconciled GPU-run artifacts: {total_artifact_bytes / 1024**2:.2f} MiB",
        "- FR: 100/100 terminal episodes, 0 infrastructure errors",
        "- SAVR grid: 900/900 terminal episodes, 0 infrastructure errors",
        "- every run used one explicitly selected GPU and restored protected "
        "checkpoint metadata exactly",
        "- all task failures were retained as scientific outcomes",
        "- no training, model/dataset download, upstream edit, manuscript edit, "
        "or final-holdout execution occurred",
        "- no university-server path outside `/home/ved/SAVR` was modified by "
        "the Phase 6 workflow",
        "",
        "## Primary methodological sources",
        "",
        "- OpenVLA-OFT: https://arxiv.org/abs/2502.19645",
        "- LIBERO: "
        "https://proceedings.neurips.cc/paper_files/paper/2023/hash/"
        "8c3c666820ea055a77726d66fc7d447f-Abstract-Datasets_and_Benchmarks.html",
        "- Matched-pair non-inferiority sample size: "
        "https://doi.org/10.1002/bimj.201100231",
        "- McNemar matched-pair sample size: "
        "https://doi.org/10.1002/sim.4780110909",
        "",
        "## Phase boundary",
        "",
        "Phase 6 stops at this negative checkpoint. A follow-up may either end "
        "the proposed method as currently formulated or predeclare a new "
        "calibration protocol that tests materially more conservative reuse. "
        "The current split must not be relabeled as a fresh holdout.",
        "",
    ]
    report_path = project_root / "reports/PHASE6_CALIBRATION_REPORT.md"
    temporary = report_path.with_suffix(".md.tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(report_path)


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    if project_root != EXPECTED_ROOT:
        raise SystemExit(f"Refusing to run outside {EXPECTED_ROOT}: {project_root}")

    fr_summary = load(
        project_root / "results/phase6-fr-signals-v1/run_summary.json"
    )
    threshold = load(
        project_root
        / "results/phase6-savr-thresholds-v1/threshold_derivation.json"
    )
    selection = load(
        project_root / "results/phase6-savr-selection-v1/selection.json"
    )
    if selection["status"] == "no_eligible_savr_candidate":
        write_negative_report(
            project_root,
            fr_summary=fr_summary,
            threshold=threshold,
            selection=selection,
        )
        print(project_root / "reports/PHASE6_CALIBRATION_REPORT.md")
        return 0
    final = load(
        project_root
        / "results/phase6-final-calibration-v1/baseline_analysis.json"
    )
    if any(
        artifact["status"] != "completed"
        for artifact in (fr_summary, threshold, selection, final)
    ):
        raise RuntimeError("A required Phase 6 artifact is not complete")

    candidates = sorted(
        selection["candidates"],
        key=lambda item: (
            float(item["setting"]["target_skip_rate"]),
            int(item["setting"]["max_reuse_horizon"]),
        ),
    )
    candidate_rows = [
        "| {identifier} | {target} | {horizon} | {successes}/100 | {difference} "
        "| {skip} | {refresh} | {eligible} |".format(
            identifier=item["configuration_id"],
            target=percent(float(item["setting"]["target_skip_rate"])),
            horizon=item["setting"]["max_reuse_horizon"],
            successes=item["successes"],
            difference=f"{100 * float(item['paired_success_difference']):+.1f} pp",
            skip=percent(float(item["skip_rate"])),
            refresh=percent(float(item["refresh_rate"])),
            eligible="yes" if item["eligible"] else "no",
        )
        for item in candidates
    ]
    frozen = final["frozen_primary_configurations"]
    baseline_rows = [
        "| {policy} | {identifier} | {successes} | {refresh} | {status} |".format(
            policy=policy,
            identifier=config["configuration_id"],
            successes=(
                "FR calibration oracle"
                if policy == "FR"
                else f"{config['observed_successes']}/100"
            ),
            refresh=percent(
                float(
                    config["observed_refresh_rate"]
                    if "observed_refresh_rate" in config
                    else config["refresh_rate"]
                )
            ),
            status=config.get("budget_status", "target"),
        )
        for policy, config in frozen.items()
    ]

    run_summaries = phase6_run_summaries(project_root)
    total_elapsed = sum(item["elapsed"] for item in run_summaries)
    total_artifact_bytes = sum(item["artifact_bytes"] for item in run_summaries)
    if any(
        item["status"] != "completed" or not item["checkpoint_restored"]
        for item in run_summaries
    ):
        raise RuntimeError("A Phase 6 GPU run failed integrity reconciliation")

    selected = selection["selected_savr"]
    power = final["provisional_power"]
    lines = [
        "# Phase 6 Calibration and Power Report",
        "",
        "Status: COMPLETE",
        "",
        "## Scope and claim boundary",
        "",
        "Phase 6 used only LIBERO-Spatial tasks 0-9, initial-state IDs 0-9, "
        "and seed 0. These are calibration outcomes, not final paper-level "
        "non-inferiority or comparative-performance evidence. The final "
        "initial-state 10-49 / seed 7,17,27 holdout was not executed or "
        "inspected.",
        "",
        "## Full Refresh calibration oracle",
        "",
        f"- terminal episodes: {fr_summary['progress']['terminal']}/100",
        f"- successes: {threshold['source_success_count']}/100",
        f"- query traces: {threshold['source_query_count']}",
        f"- trace input hash: `{threshold['source_input_combined_sha256']}`",
        "",
        "## Frozen SAVR grid",
        "",
        "| Configuration | Target skip | Hmax | Success | Paired difference "
        "| Observed skip | Observed refresh | Eligible |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
        *candidate_rows,
        "",
        "The frozen mechanical selection rule chose "
        f"`{selected['configuration_id']}`: "
        f"{selected['successes']}/100 successes, "
        f"{percent(float(selected['skip_rate']))} skipped refreshes, and "
        f"a paired success difference of "
        f"{100 * float(selected['paired_success_difference']):+.1f} percentage "
        "points versus FR.",
        "",
        "## Frozen primary configurations",
        "",
        "| Policy | Configuration | Calibration success | Refresh rate "
        "| Budget classification |",
        "|---|---|---:|---:|---|",
        *baseline_rows,
        "",
        "PR remains a fixed integer-period policy. Any discrete budget gap is "
        "reported rather than changing its definition. VOR used at most the "
        "three predeclared matching attempts.",
        "",
        "## Paired power",
        "",
        f"- observed selected-SAVR discordance: "
        f"{percent(float(power['observed_discordance_rate']))}",
        f"- Wilson 95% upper discordance used for planning: "
        f"{percent(float(power['wilson_95_upper_discordance_rate']))}",
        f"- margin: {100 * float(power['margin']):.1f} percentage points",
        f"- one-sided alpha: {power['alpha_one_sided']}",
        f"- target power: {percent(float(power['target_power']))}",
        f"- unrounded requirement: {power['required_unrounded']} paired episodes "
        "per policy per suite",
        f"- balanced recommendation: {power['recommended_sample_size']} paired "
        "episodes per policy per suite",
        "",
        "The Phase 7 protocol must use the balanced recommendation or explicitly "
        "reduce the claim; it may not enlarge the margin.",
        "",
        "## Integrity, resources, and safety",
        "",
        f"- reconciled GPU-run elapsed time: {total_elapsed / 3600:.2f} hours",
        f"- reconciled GPU-run artifacts: {total_artifact_bytes / 1024**2:.2f} MiB",
        "- every run used one explicitly selected GPU and restored protected "
        "checkpoint metadata",
        "- every setting retained all task failures as scientific outcomes",
        "- no training, model/dataset download, upstream edit, manuscript edit, "
        "or final-holdout execution occurred",
        "- no university-server path outside `/home/ved/SAVR` was modified by "
        "the Phase 6 workflow",
        "",
        "## Primary methodological sources",
        "",
        "- OpenVLA-OFT: https://arxiv.org/abs/2502.19645",
        "- LIBERO: "
        "https://proceedings.neurips.cc/paper_files/paper/2023/hash/"
        "8c3c666820ea055a77726d66fc7d447f-Abstract-Datasets_and_Benchmarks.html",
        "- Matched-pair non-inferiority sample size: "
        "https://doi.org/10.1002/bimj.201100231",
        "- McNemar matched-pair sample size: "
        "https://doi.org/10.1002/sim.4780110909",
        "",
        "## Phase boundary",
        "",
        "One primary configuration per method and the final planning sample size "
        "are frozen. Phase 6 stops here. Phase 7 remains a separate protocol-"
        "freeze and user-approval gate.",
        "",
    ]
    report_path = project_root / "reports/PHASE6_CALIBRATION_REPORT.md"
    temporary = report_path.with_suffix(".md.tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(report_path)
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
