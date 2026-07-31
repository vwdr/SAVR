#!/usr/bin/env python3
"""Reproduce the Phase 6S-A failure-localization analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


FAILED_EPISODES = (
    "savr2-b10_task_08_state_02",
    "savr2-b15_task_01_state_00",
    "savr2-b15_task_04_state_02",
    "savr2-b15_task_09_state_01",
)
WRIST_CAPS = (0.300, 0.325, 0.350, 0.375, 0.400)


def load_records(directory: Path) -> list[dict[str, Any]]:
    return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(directory.glob("*.json"))]


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    episodes = {record["episode_id"]: record for record in load_records(args.run_dir / "episodes")}
    queries: dict[str, list[dict[str, Any]]] = {}
    for record in load_records(args.run_dir / "queries"):
        queries.setdefault(record["episode_id"], []).append(record)
    for records in queries.values():
        records.sort(key=lambda record: int(record["episode_query_index"]))
    if tuple(sorted(episode_id for episode_id, record in episodes.items() if not record["success"])) != tuple(sorted(FAILED_EPISODES)):
        raise RuntimeError("Stage 1 failed-episode identities changed")

    failures = []
    for episode_id in FAILED_EPISODES:
        episode = episodes[episode_id]
        task = int(str(episode["task"]).split(":")[-1])
        state = int(episode["initial_state_id"])
        reference_id = f"savr2-b05_task_{task:02d}_state_{state:02d}"
        current = queries[episode_id]
        reference = queries[reference_id]
        first_mismatch = next(
            (index for index in range(min(len(current), len(reference))) if current[index]["actions_sha256"] != reference[index]["actions_sha256"]),
            None,
        )
        reuses = []
        for record in current:
            if record["refresh"]:
                continue
            decision = record["decision"]
            reuses.append(
                {
                    "query_index": record["episode_query_index"],
                    "environment_step": record["environment_step"],
                    "action_mismatch_vs_b05": (
                        int(record["episode_query_index"]) < len(reference)
                        and record["actions_sha256"]
                        != reference[int(record["episode_query_index"])]["actions_sha256"]
                    ),
                    "translation_reversal": any(decision["translation_direction_reversals"]),
                    "wrist_score": decision["per_camera_image_scores"]["wrist_image"],
                    "full_image_score": decision["per_camera_image_scores"]["full_image"],
                }
            )
        failures.append(
            {
                "episode_id": episode_id,
                "reference_episode_id": reference_id,
                "first_action_mismatch": first_mismatch,
                "reuses": reuses,
            }
        )

    b15_reuses = []
    for episode_id, episode in episodes.items():
        if episode["configuration_id"] != "savr2-b15":
            continue
        for record in queries[episode_id]:
            if record["refresh"]:
                continue
            decision = record["decision"]
            b15_reuses.append(
                {
                    "episode_id": episode_id,
                    "episode_success": bool(episode["success"]),
                    "query_index": int(record["episode_query_index"]),
                    "translation_reversal": any(decision["translation_direction_reversals"]),
                    "wrist_score": float(decision["per_camera_image_scores"]["wrist_image"]),
                }
            )
    search = []
    for cap in WRIST_CAPS:
        retained = [
            record
            for record in b15_reuses
            if not record["translation_reversal"] and record["wrist_score"] <= cap
        ]
        search.append(
            {
                "wrist_cap": cap,
                "retained_reuses": len(retained),
                "retrospective_skip_rate": len(retained) / 473,
                "retained_episodes": len({record["episode_id"] for record in retained}),
                "retained_failed_path_reuses": sum(not record["episode_success"] for record in retained),
            }
        )

    output = {
        "source_run": args.run_dir.name,
        "source_summary_sha256": hashlib.sha256((args.run_dir / "run_summary.json").read_bytes()).hexdigest(),
        "failed_episodes": failures,
        "b10_reuse_count": sum(
            not record["refresh"]
            for episode_id, records in queries.items()
            if episodes[episode_id]["configuration_id"] == "savr2-b10"
            for record in records
        ),
        "b15_reuse_count": len(b15_reuses),
        "wrist_cap_search": search,
        "selected_design": {
            "base": "savr2-b15",
            "translation_reversal_veto": True,
            "wrist_threshold": 0.375,
            "selection_rule": "smallest 0.025-grid cap retaining at least 5% retrospective reuse while filtering every first unsafe failed-trajectory reuse",
        },
    }
    output["analysis_sha256"] = sha256_json(output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
