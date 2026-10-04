from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from savr.pair.p3 import P3Profile, block_schedule, semantic_sha256


ROOT = Path(__file__).resolve().parents[2]


def write_json(path: Path, value: dict) -> None:
    value["semantic_sha256"] = semantic_sha256(value)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def action_record() -> dict:
    return {"shape": [8, 7], "finite": True, "sha256": "a" * 64}


def test_p3_analyzer_accepts_complete_outcome_blind_passing_population(tmp_path):
    config = json.loads((ROOT / "configs/pair/p3_physical_v1.json").read_text())
    inputs = json.loads((ROOT / config["input_manifest"]).read_text())
    profiles = {
        value["profile_id"]: P3Profile.from_mapping(value) for value in config["profiles"]
    }
    block_rows = []
    for block in block_schedule(config, len(inputs["inputs"])):
        profile = profiles[block.profile_id]
        reduction = profile.primary_budgets[-1] + profile.wrist_budgets[-1]

        def arm(cache: bool) -> dict:
            queries = []
            for ordinal in range(block.horizon + 1):
                queries.append(
                    {
                        "action_record": action_record(),
                        "total_wall_ms": 60.0 if cache else 100.0,
                        "total_cuda_ms": 50.0 if cache else 90.0,
                        "decoder_wall_ms": 50.0 if cache else 90.0,
                        "decoder_cuda_ms": 45.0 if cache else 85.0,
                        "prepare_wall_ms": 5.0,
                        "gate_wall_ms": 0.05 if cache and ordinal else 0.0,
                        "sidecar_wall_ms": 0.05 if cache and not ordinal else 0.0,
                        "provenance_wall_ms": 0.01,
                        "active_sequence_length": (
                            600 - reduction if cache and ordinal else 600
                        ),
                        "full_sequence_length": 600,
                        "source_digest": "b" * 64 if cache else None,
                        "source_mixture_count": 2 if cache and ordinal else 1,
                        "mask_metadata": {} if cache and ordinal else None,
                        "underlying_sdpa_calls": 32 if cache and not ordinal else 0,
                        "original_step": ordinal * 8,
                        "observation_sha256": "c" * 64,
                    }
                )
            result = {
                "cycle_wall_ms": sum(query["total_wall_ms"] for query in queries),
                "cycle_cuda_ms": sum(query["total_cuda_ms"] for query in queries),
                "queries": queries,
                "service_count": block.horizon if cache else 0,
                "fallback_count": 0,
            }
            if cache:
                result.update(
                    final_source_digest="d" * 64,
                    maximum_source_mixture_count=2,
                )
            return result

        row = {
            "schema_version": "pair-p3-block-v1",
            "run_id": config["run_id"],
            "block_id": block.block_id,
            "profile_id": block.profile_id,
            "horizon": block.horizon,
            "repetition": block.repetition,
            "arm_order": block.arm_order,
            "suite": "technical",
            "task_id": "technical",
            "trajectory_id": "technical",
            "start_query_ordinal": 0,
            "arms": {"dense": arm(False), "cache": arm(True)},
        }
        row["semantic_sha256"] = semantic_sha256(row)
        block_rows.append(row)
    (tmp_path / "blocks.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in block_rows),
        encoding="utf-8",
    )
    worker = {
        "schema_version": "pair-p3-worker-v1",
        "run_id": config["run_id"],
        "status": "completed",
        "queries_used": 688,
        "queries_planned": 688,
        "blocks_completed": 96,
        "blocks_expected": 96,
        "wall_seconds": 10.0,
        "sidecar_control": {
            "cache_sample_match": True,
            "underlying_sdpa_calls": 32,
            "incremental_upper_ms": 0.05,
            "off_action_record": action_record(),
            "on_action_record": action_record(),
        },
        "all_fresh_control": {
            "cache_sample_match": True,
            "dense_action_record": action_record(),
            "all_fresh_action_record": action_record(),
        },
        "overhead_mock": {"router_p99_ms": 0.05, "reset_p99_ms": 0.05},
        "memory": {
            "peak_reserved_mib": 20000,
            "peak_aggregate_mib": 20000,
            "strict_limit_mib": 23552,
        },
        "protection": {
            "action_values_persisted": False,
            "action_comparisons_performed": False,
            "expert_actions_accessed": False,
            "terminal_outcomes_accessed": False,
            "simulator_used": False,
            "downloads": 0,
        },
    }
    write_json(tmp_path / "worker_summary.json", worker)
    process = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/analyze_pair_p3.py"),
            "--config",
            str(ROOT / "configs/pair/p3_physical_v1.json"),
            "--run-root",
            str(tmp_path),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    analysis = json.loads((tmp_path / "analysis.json").read_text())
    assert analysis["status"] == "passed"
    assert analysis["gates"]["at_least_one_headroom_point"] is True
    assert analysis["advance"] == {
        "next_phase": "P4",
        "authorized": False,
        "stop_before_next_phase": True,
    }
