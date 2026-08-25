#!/usr/bin/env python3
"""Reconcile every frozen BRACE-B3 physical gate without task outcomes."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ROOT = Path("/home/ved/SAVR")
RUN = ROOT / "results/brace-b3-physical-v01"
CONFIG = ROOT / "configs/brace/b3_physical_v1.json"


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def write_once(path: Path, value: dict[str, Any]) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(canonical_bytes(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def action_close(left: dict[str, Any], right: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    a = np.asarray(left["values"], dtype=np.float32)
    b = np.asarray(right["values"], dtype=np.float32)
    finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
    close = finite and bool(
        np.allclose(
            a,
            b,
            rtol=float(config["parity"]["unnormalized_action_rtol"]),
            atol=float(config["parity"]["unnormalized_action_atol"]),
        )
    )
    gripper = bool(np.array_equal(a[:, -1] > 0, b[:, -1] > 0))
    return {
        "passed": close and gripper,
        "finite": finite,
        "gripper_exact": gripper,
        "maximum_absolute_difference": float(np.max(np.abs(a - b))),
    }


def main() -> int:
    if ROOT != EXPECTED_ROOT or Path.cwd().resolve() != EXPECTED_ROOT:
        raise SystemExit(f"B3 analysis is restricted to {EXPECTED_ROOT}")
    sys.path.insert(0, str(ROOT / "src"))
    from savr.brace.b3 import profile_speed_gate, summarize_timings, validate_config

    if (RUN / "analysis.json").exists():
        raise SystemExit("Immutable B3 analysis already exists")
    if (RUN / "technical_stop.json").exists():
        raise SystemExit("B3 ended technically; scientific analysis is prohibited")
    run = json.loads((RUN / "run_summary.json").read_text())
    config = json.loads(CONFIG.read_text())
    validate_config(config)
    run_hash = dict(run)
    recorded_run_hash = run_hash.pop("semantic_sha256")
    if hashlib.sha256(canonical_bytes(run_hash)).hexdigest() != recorded_run_hash:
        raise RuntimeError("B3 run-summary semantic hash mismatch")
    workers = {
        method: json.loads((RUN / "workers" / f"{method}.json").read_text())
        for method in ("core_fr", "cache_suite", "vla_adp", "vla_pruner")
    }
    identities = all(
        worker["status"] == "completed"
        and worker["configuration_semantic_sha256"] == config["semantic_sha256"]
        and worker["source_revision"] == run["source_revision"]
        for worker in workers.values()
    )
    core = workers["core_fr"]
    cache = workers["cache_suite"]
    p0_parity = {
        label: action_close(core["action_references"][label], cache["p0_action_references"][label], config)
        for label in sorted(core["action_references"])
    }
    p0_matches_fr = len(p0_parity) == 3 and all(value["passed"] for value in p0_parity.values())
    p0_values = [float(record["wall_ms"]) for record in cache["p0_timings"]]
    p0_median = summarize_timings(p0_values)["p50"]
    profiles = {}
    passing_profiles = []
    all_timed_parity = True
    for profile_id, evidence in cache["profiles"].items():
        accelerated = [item for cycle in evidence["cycles"] for item in cycle["accelerated"]]
        cycles = evidence["cycles"]
        action_parity = all(item["action_parity"]["passed"] for item in accelerated)
        all_timed_parity = all_timed_parity and action_parity
        speed = profile_speed_gate(
            p0_accelerated_ms=p0_values,
            accelerated_ms=[float(item["wall_ms"]) for item in accelerated],
            p0_cycle_ms=[p0_median * (int(cycle["horizon"]) + 1) for cycle in cycles],
            contract_cycle_ms=[float(cycle["cycle_wall_ms"]) for cycle in cycles],
            minimum_accelerated=float(config["gates"]["minimum_accelerated_query_reduction"]),
            minimum_cycle=float(config["gates"]["minimum_amortized_cycle_reduction"]),
        )
        per_horizon = {}
        for horizon in config["measurement"]["horizons"]:
            selected = [cycle for cycle in cycles if int(cycle["horizon"]) == int(horizon)]
            selected_accelerated = [item for cycle in selected for item in cycle["accelerated"]]
            per_horizon[str(horizon)] = {
                "accelerated_wall": summarize_timings([item["wall_ms"] for item in selected_accelerated]),
                "cycle_wall": summarize_timings([cycle["cycle_wall_ms"] for cycle in selected]),
                "reference_cycle_ms": p0_median * (int(horizon) + 1),
            }
        passed = speed["passed"] and action_parity
        if passed:
            passing_profiles.append(profile_id)
        profiles[profile_id] = {
            "passed": passed,
            "action_parity": action_parity,
            "speed_gate": speed,
            "accelerated_wall": summarize_timings([item["wall_ms"] for item in accelerated]),
            "cycle_wall": summarize_timings([cycle["cycle_wall_ms"] for cycle in cycles]),
            "per_horizon": per_horizon,
        }
    memory_limit = int(config["gates"]["peak_reserved_bytes_strictly_below"])
    worker_peak = max(int(worker["peak_reserved_bytes"]) for worker in workers.values())
    aggregate_peak = int(run["peak_aggregate_gpu_memory_used_mib"]) * 1024 * 1024
    memory_pass = worker_peak < memory_limit and aggregate_peak < memory_limit
    corrected_cache = bool(cache["corrected_vla_cache"]["all_action_parity"])
    provenance = bool(cache["cache_provenance_reset_invariants"])
    warm_parity = all(record["passed"] for record in cache["warm_profile_parity"])
    sidecar = bool(cache["sidecar_parity"]["passed"])
    comparator_dispositions = {
        "corrected_vla_cache": "valid_real_timing" if corrected_cache else "failed_parity",
        "vla_adp": workers["vla_adp"]["timing_validity"],
        "vla_pruner": workers["vla_pruner"]["timing_validity"],
        "specprune_vla": config["comparators"]["specprune_vla"],
        "gated_vla_cache": config["comparators"]["gated_vla_cache"],
    }
    comparator_pass = (
        corrected_cache
        and workers["vla_adp"]["timing_validity"]
        == "component_timing_only_episode_coupled_dynamic_controller_excluded"
        and workers["vla_pruner"]["timing_validity"] == "official_temporal_semantic_action_timing"
    )
    p0_summary = summarize_timings(p0_values)
    p4 = {
        "control_window_ms": float(config["p4"]["measured_window_seconds"]) * 1000,
        "dense_completion_wall": p0_summary,
        "p50_spills_past_window": p0_summary["p50"] > float(config["p4"]["measured_window_seconds"]) * 1000,
        "p95_spills_past_window": p0_summary["p95"] > float(config["p4"]["measured_window_seconds"]) * 1000,
        "artificial_sleep": False,
        "measured_disposition": True,
    }
    gates = {
        "identity_and_query_accounting": identities and int(run["queries"]) == 388,
        "p0_matches_optimized_fr": p0_matches_fr,
        "dense_sidecar_parity": sidecar,
        "all_profile_action_parity": warm_parity and all_timed_parity,
        "cache_provenance_and_reset_invariants": provenance,
        "corrected_vla_cache_parity": corrected_cache,
        "at_least_one_clean_profile_speed_gate": bool(passing_profiles),
        "peak_memory_strictly_below_23_gib": memory_pass,
        "mandatory_comparator_dispositions": comparator_pass,
        "p4_measured_disposition": p4["measured_disposition"],
    }
    accepted = all(gates.values())
    analysis = {
        "schema_version": "brace.b3-analysis.v1",
        "run_id": config["run_id"],
        "status": "accepted" if accepted else "stopped_negative",
        "configuration_semantic_sha256": config["semantic_sha256"],
        "source_revision": run["source_revision"],
        "queries": int(run["queries"]),
        "gates": gates,
        "p0_action_parity": p0_parity,
        "p0_wall": p0_summary,
        "profiles": profiles,
        "passing_profiles": passing_profiles,
        "memory": {
            "worker_peak_reserved_bytes": worker_peak,
            "aggregate_peak_bytes": aggregate_peak,
            "strict_limit_bytes": memory_limit,
        },
        "comparator_dispositions": comparator_dispositions,
        "comparator_wall": {
            method: summarize_timings([record["wall_ms"] for record in workers[method]["timings"]])
            for method in ("vla_adp", "vla_pruner")
        },
        "corrected_vla_cache": cache["corrected_vla_cache"],
        "sidecar_parity": cache["sidecar_parity"],
        "p4": p4,
        "interpretation_boundary": "No simulator outcomes or task-success fields were accessed.",
        "b4_authorized": False,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    analysis["semantic_sha256"] = hashlib.sha256(canonical_bytes(analysis)).hexdigest()
    if not math.isfinite(float(p0_summary["p50"])):
        raise RuntimeError("B3 analysis produced nonfinite timing")
    write_once(RUN / "analysis.json", analysis)
    print(json.dumps({"status": analysis["status"], "passing_profiles": passing_profiles, "gates": gates}))
    return 0 if accepted else 3


if __name__ == "__main__":
    raise SystemExit(main())
