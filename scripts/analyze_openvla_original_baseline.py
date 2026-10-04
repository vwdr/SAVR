#!/usr/bin/env python3
"""Reconcile the complete diagnostic before reporting any task outcomes."""

import argparse
import hashlib
import json
import math
from pathlib import Path


def check(condition, message):
    if not condition:
        raise ValueError(message)


def reconcile(root, config_path):
    check(not (root / "technical_stop.json").exists(), "technical stop prohibits analysis")
    summary = json.loads((root / "worker_summary.json").read_text())
    check(summary.get("complete") is True, "worker is not complete")
    launch = json.loads((root / "launch.json").read_text())
    config = json.loads(config_path.read_text())
    check(launch["config"] == config, "launch/config disagreement")
    check(launch["config_sha256"] == hashlib.sha256(config_path.read_bytes()).hexdigest(),
          "configuration hash mismatch")
    for name, expected in summary["artifact_sha256"].items():
        check(name in {"reference_check.json", "episodes.json", "loaded_runtime.json"},
              "unexpected artifact reference")
        check(hashlib.sha256((root / name).read_bytes()).hexdigest() == expected,
              f"artifact hash mismatch: {name}")
    check(set(summary["artifact_sha256"]) == {"reference_check.json", "episodes.json", "loaded_runtime.json"},
          "missing authenticated artifact")
    offline = json.loads((root / "reference_check.json").read_text())
    terminal = json.loads((root / "episodes.json").read_text())
    check(offline["complete"] is True and terminal["complete"] is True, "incomplete stage")
    check(offline["queries"] == 32 and len(offline["records"]) == 8, "offline counts differ")
    check(len({r["observation_id"] for r in offline["records"]}) == 8, "duplicate observation")
    for record in offline["records"]:
        for field in ("parity_max_abs", "restoration_max_abs"):
            check(bool(record[field]) and all(math.isfinite(v) and 0 <= v <= 1e-6
                                             for v in record[field].values()), "reference check failed")
        for name, mode in (("original_masks", "original"), ("causal_masks", "causal_control")):
            masks = record[name]
            check([m["layer"] for m in masks] == list(range(32)), "attention layers differ")
            check(all(m["effective_mode"] == mode and m["reference_is_causal"] is False
                      for m in masks), "attention mode differs")
    records = terminal["records"]
    check(len(records) == summary["episode_count"] == 40, "episode count differs")
    expected_order = [(r["condition_id"], mode) for r in config["episode_conditions"] for mode in r["arm_order"]]
    check([(r["condition_id"], r["mode"]) for r in records] == expected_order, "paired schedule differs")
    check(sum(r["policy_queries"] for r in records) + 32 == summary["model_queries"] <= 1692,
          "query counts do not reconcile")
    check(summary["peak_aggregate_gpu_memory_mib"] < 23552, "memory ceiling exceeded")
    check(summary["elapsed_seconds"] <= 7200, "wall ceiling exceeded")
    check(summary["checkpoint_unchanged"] is True and summary["training_performed"] is False,
          "protected boundary violated")
    step_limits = dict(zip(config["suites"], [220, 280, 300, 520]))
    conditions = {r["condition_id"]: r for r in config["episode_conditions"]}
    for record in records:
        for key, value in conditions[record["condition_id"]].items():
            check(record[key] == value, "episode identity differs")
        check(type(record["success"]) is bool, "success is not Boolean")
        check(1 <= record["executed_steps"] <= step_limits[record["suite"]], "invalid episode length")
        check(record["policy_queries"] == math.ceil(record["executed_steps"] / 8), "action queue count differs")
    modes = ["original"]
    successes = {mode: sum(r["success"] for r in records if r["mode"] == mode) for mode in modes}
    check(successes == summary["successes"], "success aggregation differs")
    pairs = []
    for condition in config["episode_conditions"]:
        rows = {r["mode"]: r for r in records if r["condition_id"] == condition["condition_id"]}
        pairs.append({"condition_id": condition["condition_id"], "suite": condition["suite"],
                      "task_id": condition["task_id"],
                      **{mode: rows[mode]["success"] for mode in modes}})
    check(config["schema_version"] == "openvla-original-baseline-40task-v1", "invalid baseline schema")
    check(len(conditions) == 40 and all(r["arm_order"] == ["original"] and r["seed"] == 7
                                       and r["initial_state_id"] == 0 for r in config["episode_conditions"]),
          "frozen original-only population differs")
    check(summary["authenticated_files_unchanged"] is True, "authenticated files changed")
    for suite in config["suites"]:
        rows = [r for r in records if r["suite"] == suite]
        check(len(rows) == 10 and len({r["task_id"] for r in rows}) == 10, "suite inventory differs")
    for record in records:
        state_hash = record.get("initial_state_sha256", "")
        check(len(state_hash) == 64 and all(c in "0123456789abcdef" for c in state_hash),
              "invalid initial-state hash")
    runtime = json.loads((root / "loaded_runtime.json").read_text())
    check(runtime["transformers_version"] == "4.40.1" and
          runtime["attention_classes"] == ["LlamaSdpaAttention"] * 32,
          "runtime contract differs")
    check(runtime["checkpoint_metadata_mutation_disabled"] is True and
          runtime["tensorflow_gpu_disabled"] is True, "runtime protections differ")
    return {
        "complete": True, "reconciled": True, "conditions": 40,
        "successes": successes, "episodes": 40,
        "per_suite": {suite: {"successes": sum(r["original"] for r in pairs if r["suite"] == suite),
                              "episodes": 10} for suite in config["suites"]},
        "conditions_outcomes": pairs,
        "failures": [{"condition_id": r["condition_id"], "suite": r["suite"],
                      "task_id": r["task_id"], "executed_steps": r["executed_steps"]}
                     for r in records if not r["success"]],
        "model_queries": summary["model_queries"],
        "peak_aggregate_gpu_memory_mib": summary["peak_aggregate_gpu_memory_mib"],
        "elapsed_seconds": summary["elapsed_seconds"],
        "positive_method_result": False,
        "interpretation_scope": "Forty previously consumed development conditions, one state per task, one seed. Task coverage, not a held-out benchmark or cache-method test.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(reconcile(args.root, args.config), indent=2, sort_keys=True, allow_nan=False))

