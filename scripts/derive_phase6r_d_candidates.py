#!/usr/bin/env python3
"""Derive the frozen Phase 6R-D SAVR 2.0 candidates from Phase 6 FR traces."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from savr.signals import (  # noqa: E402
    action_transition,
    grouped_action_change,
    grouped_state_change,
    patch_image_change,
)


EXPECTED_EPISODES = 100
EXPECTED_TASK_STATES = {(task, state) for task in range(10) for state in range(10)}
QUANTILES = tuple(index / 1000 for index in range(1001))
SAFETY_MARGIN = 0.90
BUDGETS = (0.05, 0.10, 0.15)


@dataclass(frozen=True)
class QuerySignals:
    visual_lag1: dict[str, float] | None
    visual_lag2: dict[str, float] | None
    state: dict[str, float] | None
    action: dict[str, float] | None
    gripper_veto: bool


@dataclass(frozen=True)
class EpisodeTrace:
    episode_id: str
    task_id: int
    state_id: int
    signals: tuple[QuerySignals, ...]


def canonical_sha256(records: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(
            json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        )
        digest.update(b"\n")
    return digest.hexdigest()


def linear_quantile(values: list[float], quantile: float) -> float:
    if not values or not 0 <= quantile <= 1:
        raise ValueError("Quantile input is invalid")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _images(trace: dict[str, Any]) -> dict[str, Any]:
    images: dict[str, Any] = {}
    for name in ("full_image", "wrist_image"):
        shape = tuple(int(value) for value in trace["image_shapes"][name])
        if shape != (32, 32, 3):
            raise RuntimeError(f"Unexpected trace image shape for {name}: {shape}")
        flattened = [float(value) for value in trace["images"][name]]
        if len(flattened) != 32 * 32 * 3:
            raise RuntimeError(f"Unexpected trace image width for {name}")
        images[name] = [
            [flattened[(row * 32 + column) * 3 : (row * 32 + column + 1) * 3]
            for column in range(32)]
            for row in range(32)
        ]
    return images


def _reference(trace: dict[str, Any]) -> dict[str, tuple[float, ...]]:
    return {
        name: tuple(float(value) for value in trace["images"][name])
        for name in ("full_image", "wrist_image")
    }


def _pairing(episode_id: str) -> tuple[int, int]:
    parts = episode_id.rsplit("_", 4)
    if len(parts) != 5 or parts[-4] != "task" or parts[-2] != "state":
        raise RuntimeError(f"Unexpected FR episode identifier: {episode_id}")
    return int(parts[-3]), int(parts[-1])


def load_traces(
    run_dir: Path,
    *,
    state_q01: list[float],
    state_q99: list[float],
    action_q01: list[float],
    action_q99: list[float],
) -> tuple[list[EpisodeTrace], str, dict[str, list[float]]]:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
    if manifest["status"] != "completed" or summary["status"] != "completed":
        raise RuntimeError("Phase 6 FR source run is not complete")
    records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((run_dir / "queries").glob("query_*.json"))
    ]
    if not records or any(
        record.get("configuration_id") != "fr"
        or record.get("status") != "completed"
        or "calibration_trace" not in record
        for record in records
    ):
        raise RuntimeError("Phase 6 FR query records are incomplete or incompatible")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["episode_id"])].append(record)
    if len(grouped) != EXPECTED_EPISODES:
        raise RuntimeError(f"Expected {EXPECTED_EPISODES} FR episodes, found {len(grouped)}")

    distributions: dict[str, list[float]] = defaultdict(list)
    episodes: list[EpisodeTrace] = []
    pairings: set[tuple[int, int]] = set()
    for episode_id, episode_records in sorted(grouped.items()):
        episode_records.sort(key=lambda record: int(record["episode_query_index"]))
        if [int(record["episode_query_index"]) for record in episode_records] != list(
            range(len(episode_records))
        ):
            raise RuntimeError(f"Non-contiguous FR queries for {episode_id}")
        task_id, state_id = _pairing(episode_id)
        pairings.add((task_id, state_id))
        traces = [record["calibration_trace"] for record in episode_records]
        images = [_images(trace) for trace in traces]
        references = [_reference(trace) for trace in traces]
        states = [trace["state"] for trace in traces]
        actions = [trace["actions"] for trace in traces]
        signals: list[QuerySignals] = []
        for index in range(len(traces)):
            visual_lag1 = visual_lag2 = state_scores = action_scores = None
            gripper_veto = False
            if index >= 1:
                lag1 = patch_image_change(images[index], references[index - 1])
                visual_lag1 = {
                    name: result.top_k_mean for name, result in lag1.per_camera.items()
                }
                state_scores = grouped_state_change(
                    states[index], states[index - 1], state_q01, state_q99
                ).scores
                for name, value in visual_lag1.items():
                    distributions[f"image.{name}"].append(value)
                for name, value in state_scores.items():
                    distributions[f"state.{name}"].append(value)
            if index >= 2:
                lag2 = patch_image_change(images[index], references[index - 2])
                visual_lag2 = {
                    name: result.top_k_mean for name, result in lag2.per_camera.items()
                }
                action_scores = grouped_action_change(
                    actions[index - 1], actions[index - 2], action_q01, action_q99
                ).scores
                gripper_veto = action_transition(
                    actions[index - 1], actions[index - 2]
                ).gripper_veto
                for name, value in action_scores.items():
                    distributions[f"action.{name}"].append(value)
            signals.append(
                QuerySignals(
                    visual_lag1=visual_lag1,
                    visual_lag2=visual_lag2,
                    state=state_scores,
                    action=action_scores,
                    gripper_veto=gripper_veto,
                )
            )
        episodes.append(
            EpisodeTrace(
                episode_id=episode_id,
                task_id=task_id,
                state_id=state_id,
                signals=tuple(signals),
            )
        )
    if pairings != EXPECTED_TASK_STATES:
        raise RuntimeError("Phase 6 FR task/state pairings differ from the frozen set")
    return episodes, canonical_sha256(records), dict(distributions)


def thresholds_at(
    distributions: dict[str, list[float]], quantile: float
) -> dict[str, dict[str, float]]:
    result = {"image": {}, "state": {}, "action": {}}
    for family, names in {
        "image": ("full_image", "wrist_image"),
        "state": ("translation", "orientation", "gripper"),
        "action": ("translation", "rotation", "gripper"),
    }.items():
        for name in names:
            result[family][name] = SAFETY_MARGIN * linear_quantile(
                distributions[f"{family}.{name}"], quantile
            )
    return result


def replay(
    episodes: list[EpisodeTrace],
    thresholds: dict[str, dict[str, float]],
    budget: float,
) -> dict[str, Any]:
    total_queries = total_reuses = 0
    episode_reuses: dict[str, int] = {}
    first_reuse_queries: list[int] = []
    for episode in episodes:
        stable_fresh = completed_reuses = 0
        previous_reuse = False
        reuses = 0
        for index, signals in enumerate(episode.signals):
            visual = signals.visual_lag2 if previous_reuse else signals.visual_lag1
            valid = visual is not None and signals.state is not None and signals.action is not None
            stable = bool(valid and not signals.gripper_veto)
            if stable:
                assert visual is not None and signals.state is not None and signals.action is not None
                stable = all(
                    visual[name] <= thresholds["image"][name]
                    for name in ("full_image", "wrist_image")
                ) and all(
                    signals.state[name] <= thresholds["state"][name]
                    for name in ("translation", "orientation", "gripper")
                ) and all(
                    signals.action[name] <= thresholds["action"][name]
                    for name in ("translation", "rotation", "gripper")
                )
            budget_allows = (completed_reuses + 1) / (index + 1) <= budget
            reuse = bool(
                stable
                and index >= 5
                and stable_fresh >= 2
                and not previous_reuse
                and budget_allows
            )
            if reuse:
                if reuses == 0:
                    first_reuse_queries.append(index)
                reuses += 1
                completed_reuses += 1
                stable_fresh = 0
            else:
                stable_fresh = stable_fresh + 1 if stable else 0
            previous_reuse = reuse
        episode_reuses[episode.episode_id] = reuses
        total_reuses += reuses
        total_queries += len(episode.signals)
    return {
        "queries": total_queries,
        "reuses": total_reuses,
        "skip_rate": total_reuses / total_queries,
        "episodes_with_reuse": sum(value > 0 for value in episode_reuses.values()),
        "maximum_episode_reuses": max(episode_reuses.values()),
        "earliest_first_reuse_query": (
            min(first_reuse_queries) if first_reuse_queries else None
        ),
    }


def derive(
    episodes: list[EpisodeTrace], distributions: dict[str, list[float]]
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    threshold_cache = {quantile: thresholds_at(distributions, quantile) for quantile in QUANTILES}
    for budget in BUDGETS:
        eligible: list[tuple[float, float, dict[str, dict[str, float]], dict[str, Any]]] = []
        for quantile in QUANTILES:
            thresholds = threshold_cache[quantile]
            result = replay(episodes, thresholds, budget)
            if result["skip_rate"] <= budget + 1e-15:
                eligible.append((result["skip_rate"], quantile, thresholds, result))
        if not eligible:
            raise RuntimeError(f"No offline candidate respects budget {budget}")
        eligible.sort(key=lambda item: (-item[0], item[1]))
        skip_rate, quantile, thresholds, result = eligible[0]
        identifier = f"savr2-b{int(round(budget * 100)):02d}"
        candidates.append(
            {
                "configuration_id": identifier,
                "policy": "SAVR2",
                "skip_budget": budget,
                "threshold_quantile": quantile,
                "safety_margin": SAFETY_MARGIN,
                "image_thresholds": thresholds["image"],
                "state_thresholds": thresholds["state"],
                "action_thresholds": thresholds["action"],
                "offline_replay": {**result, "skip_rate": skip_rate},
            }
        )
    return candidates


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    manifest = json.loads((arguments.run_dir / "manifest.json").read_text(encoding="utf-8"))
    stats = manifest["normalization_statistics"]
    episodes, trace_sha256, distributions = load_traces(
        arguments.run_dir,
        state_q01=stats["state_q01"],
        state_q99=stats["state_q99"],
        action_q01=stats["action_q01"],
        action_q99=stats["action_q99"],
    )
    candidates = derive(episodes, distributions)
    output = {
        "protocol": "PHASE6R_PROTOCOL_V1.md",
        "protocol_version": "1.0",
        "run_id": "phase6r-d-stage1-v1",
        "stage": 1,
        "suite": "libero_spatial",
        "task_ids": list(range(10)),
        "initial_state_ids": [0, 1, 2],
        "seed": 0,
        "settings": candidates,
        "save_calibration_traces": True,
        "wall_cap_seconds": 16 * 60 * 60,
        "artifact_cap_bytes": 1024**3,
        "source": {
            "run_id": arguments.run_dir.name,
            "trace_input_sha256": trace_sha256,
            "episode_count": len(episodes),
            "query_count": sum(len(episode.signals) for episode in episodes),
            "quantile_grid_count": len(QUANTILES),
            "quantile_step": 0.001,
            "safety_margin": SAFETY_MARGIN,
        },
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
