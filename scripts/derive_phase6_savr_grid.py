#!/usr/bin/env python3
"""Derive the frozen nine-setting SAVR grid from Phase 6 FR traces."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_ROOT = Path("/home/ved/SAVR")
FR_RUN_ID = "phase6-fr-signals-v1"
OUTPUT_ID = "phase6-savr-thresholds-v1"
TARGETS = (0.25, 0.50, 0.75)
HORIZONS = (2, 4, 8)
QUANTILE_STEP = 1000


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
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    if project_root != EXPECTED_ROOT:
        raise SystemExit(f"Refusing to run outside {EXPECTED_ROOT}: {project_root}")
    sys.path.insert(0, str(project_root / "src"))

    from savr.calibration import (
        SignalBounds,
        derive_savr_grid,
        query_from_record,
        signal_distributions,
    )

    fr_dir = project_root / "results" / FR_RUN_ID
    manifest_path = fr_dir / "manifest.json"
    summary_path = fr_dir / "run_summary.json"
    if not manifest_path.is_file() or not summary_path.is_file():
        raise RuntimeError("Phase 6 FR manifest or summary is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if manifest["status"] != "completed" or summary["status"] != "completed":
        raise RuntimeError("Phase 6 FR collection is not complete")
    if manifest["configuration"]["initial_state_ids"] != list(range(10)):
        raise RuntimeError("FR calibration state IDs differ from the frozen split")
    if manifest["configuration"]["task_ids"] != list(range(10)):
        raise RuntimeError("FR calibration task IDs differ from the frozen split")
    if manifest["configuration"]["seed"] != 0:
        raise RuntimeError("FR calibration seed differs")

    episode_paths = sorted((fr_dir / "episodes").glob("*.json"))
    query_paths = sorted((fr_dir / "queries").glob("*.json"))
    if len(episode_paths) != 100:
        raise RuntimeError(f"Expected 100 FR episodes, found {len(episode_paths)}")
    episode_records = [
        json.loads(path.read_text(encoding="utf-8")) for path in episode_paths
    ]
    if any(
        record["status"] != "completed"
        or record["policy"] != "FR"
        or record["configuration_id"] != "fr"
        for record in episode_records
    ):
        raise RuntimeError("FR episode matrix contains a non-complete/non-FR record")
    expected_pairs = {
        (task_id, state_id)
        for task_id in range(10)
        for state_id in range(10)
    }
    observed_pairs = {
        (int(record["task"].split(":")[-1]), int(record["initial_state_id"]))
        for record in episode_records
    }
    if observed_pairs != expected_pairs:
        raise RuntimeError("FR episode pairings differ from the frozen matrix")

    queries_by_episode: dict[str, list[dict[str, Any]]] = {
        record["episode_id"]: [] for record in episode_records
    }
    for path in query_paths:
        record = json.loads(path.read_text(encoding="utf-8"))
        episode_id = record.get("episode_id")
        if episode_id not in queries_by_episode:
            raise RuntimeError(f"Query names an unknown FR episode: {episode_id}")
        if (
            record["status"] != "completed"
            or record["policy"] != "FR"
            or not record["refresh"]
        ):
            raise RuntimeError("FR query record violates Full Refresh invariants")
        queries_by_episode[episode_id].append(record)

    expected_query_count = sum(int(record["query_count"]) for record in episode_records)
    if len(query_paths) != expected_query_count:
        raise RuntimeError(
            f"FR query count differs: files={len(query_paths)}, "
            f"episodes={expected_query_count}"
        )

    episodes = []
    for episode_record in sorted(
        episode_records,
        key=lambda item: (
            int(item["task"].split(":")[-1]),
            int(item["initial_state_id"]),
        ),
    ):
        records = sorted(
            queries_by_episode[episode_record["episode_id"]],
            key=lambda item: int(item["episode_query_index"]),
        )
        if [record["episode_query_index"] for record in records] != list(
            range(len(records))
        ):
            raise RuntimeError("FR episode query indices are not contiguous")
        episodes.append(tuple(query_from_record(record) for record in records))

    statistics = manifest.get("normalization_statistics")
    if not isinstance(statistics, dict):
        raise RuntimeError("FR manifest lacks normalization statistics")
    bounds = SignalBounds(
        state_q01=tuple(statistics["state_q01"]),
        state_q99=tuple(statistics["state_q99"]),
        action_q01=tuple(statistics["action_q01"]),
        action_q99=tuple(statistics["action_q99"]),
    )
    distributions = signal_distributions(episodes, bounds)
    grid = derive_savr_grid(
        episodes,
        bounds=bounds,
        target_skip_rates=TARGETS,
        max_reuse_horizons=HORIZONS,
        quantile_step=QUANTILE_STEP,
    )

    settings = []
    candidate_records = []
    for target in TARGETS:
        for horizon in HORIZONS:
            candidate = grid[(target, horizon)]
            target_label = int(round(target * 100))
            identifier = f"savr-s{target_label:02d}-h{horizon}"
            settings.append(
                {
                    "configuration_id": identifier,
                    "policy": "SAVR",
                    "target_skip_rate": target,
                    "quantile": candidate.quantile,
                    "image_threshold": candidate.image_threshold,
                    "state_threshold": candidate.state_threshold,
                    "action_threshold": candidate.action_threshold,
                    "max_reuse_horizon": horizon,
                }
            )
            candidate_records.append(
                {
                    "configuration_id": identifier,
                    "target_skip_rate": target,
                    **asdict(candidate),
                    "simulated_skip_rate": candidate.simulated_skip_rate,
                }
            )

    input_paths = [manifest_path, summary_path, *episode_paths, *query_paths]
    input_hashes = {str(path.relative_to(project_root)): sha256(path) for path in input_paths}
    combined = hashlib.sha256()
    for path, digest in sorted(input_hashes.items()):
        combined.update(path.encode("utf-8"))
        combined.update(digest.encode("ascii"))

    generated_config = {
        "artifact_cap_bytes": 1073741824,
        "protocol": "PHASE6_CALIBRATION_PROTOCOL.md",
        "run_id": "phase6-savr-grid-v1",
        "settings": settings,
        "wall_cap_seconds": 115200
    }
    output_dir = project_root / "results" / OUTPUT_ID
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact = {
        "run_id": OUTPUT_ID,
        "status": "completed",
        "created_at_utc": utc_now(),
        "source_run_id": FR_RUN_ID,
        "source_savr_git_revision": manifest["savr_git_revision"],
        "source_episode_count": len(episode_paths),
        "source_query_count": len(query_paths),
        "source_success_count": sum(bool(record["success"]) for record in episode_records),
        "source_input_combined_sha256": combined.hexdigest(),
        "source_input_hashes": input_hashes,
        "quantile_grid": {
            "minimum": 0.0,
            "maximum": 1.0,
            "step": 1 / QUANTILE_STEP,
            "count": QUANTILE_STEP + 1,
        },
        "signal_distributions": {
            name: {
                "count": len(values),
                "minimum": min(values),
                "median": percentile(list(values), 0.5),
                "p95": percentile(list(values), 0.95),
                "maximum": max(values),
            }
            for name, values in distributions.items()
        },
        "candidates": candidate_records,
        "generated_config": generated_config,
        "claim_boundary": "Calibration threshold derivation only; no final holdout.",
    }
    atomic_json(output_dir / "threshold_derivation.json", artifact)
    atomic_json(output_dir / "phase6_savr_grid.generated.json", generated_config)
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
