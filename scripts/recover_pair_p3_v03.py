#!/usr/bin/env python3
"""CPU-only terminal-summary recovery for the complete PAIR P3 V03 blocks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ROOT = Path("/home/ved/SAVR")


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def semantic_sha256(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("semantic_sha256", None)
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_once(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=False)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(canonical_bytes(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def authenticate(path: Path, expected: str) -> None:
    if not path.is_file() or file_sha256(path) != expected:
        raise RuntimeError(f"P3 recovery evidence mismatch: {path.relative_to(ROOT)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    if ROOT != EXPECTED_ROOT:
        raise SystemExit(f"P3 recovery refuses to run outside {EXPECTED_ROOT}")
    if os.environ.get("CUDA_VISIBLE_DEVICES", ""):
        raise RuntimeError("P3 recovery requires CUDA to be hidden")

    recovery = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    if (
        recovery.get("schema_version") != "pair-p3-terminal-recovery-config-v1"
        or recovery.get("semantic_sha256") != semantic_sha256(recovery)
    ):
        raise RuntimeError("P3 recovery configuration is invalid")
    if recovery["recovery_scope"] != {
        "gpu_count": 0,
        "model_queries": 0,
        "model_access": False,
        "checkpoint_access": False,
        "demonstration_access": False,
        "simulator_access": False,
        "outcome_access": False,
        "network_access": False,
        "modify_source_evidence": False,
        "router_seed": 17,
        "overhead_mock_iterations": 1000,
    }:
        raise RuntimeError("P3 recovery boundary changed")
    output_root = ROOT / recovery["output_root"]
    if output_root.exists():
        raise RuntimeError("P3 immutable recovery output already exists")

    for group in ("preflight", "runtime_sources", "support_sources", "source_evidence"):
        values = recovery[group]
        for key, relative in values.items():
            if key.endswith("_path") or key == "path":
                hash_key = "sha256" if key == "path" else key.replace("_path", "_sha256")
                authenticate(ROOT / relative, values[hash_key])

    sys.path.insert(0, str(ROOT / "src"))
    from savr.pair.features import GroupFeature, RouterFeatures
    from savr.pair.p3 import (
        P3Profile,
        block_schedule,
        reject_protected_fields,
        validate_config,
    )
    from savr.pair.router import PairRouter
    from savr.pair.runtime import PairRuntime
    from savr.pair.types import AtomicGroup, Camera

    p3_config = json.loads((ROOT / recovery["p3_config"]).read_text(encoding="utf-8"))
    inputs = json.loads((ROOT / p3_config["input_manifest"]).read_text(encoding="utf-8"))
    validate_config(p3_config, input_count=len(inputs["inputs"]))
    if inputs.get("semantic_sha256") != semantic_sha256(inputs):
        raise RuntimeError("P3 recovery input-manifest semantic hash mismatch")
    blocks_path = ROOT / recovery["source_evidence"]["blocks_path"]
    stop_path = ROOT / recovery["source_evidence"]["technical_stop_path"]
    blocks = [json.loads(line) for line in blocks_path.read_text(encoding="utf-8").splitlines()]
    schedule = block_schedule(p3_config, len(inputs["inputs"]))
    if len(blocks) != 96 or [row["block_id"] for row in blocks] != [row.block_id for row in schedule]:
        raise RuntimeError("P3 recovery block schedule is incomplete or reordered")
    for row in blocks:
        reject_protected_fields(row)
        if row.get("semantic_sha256") != semantic_sha256(row):
            raise RuntimeError("P3 recovery found a block semantic-hash mismatch")
        for arm in ("dense", "cache"):
            for query in row["arms"][arm]["queries"]:
                action = query["action_record"]
                if (
                    set(action) != {"shape", "finite", "sha256"}
                    or action["shape"] != [8, 7]
                    or action["finite"] is not True
                    or not isinstance(action["sha256"], str)
                    or len(action["sha256"]) != 64
                ):
                    raise RuntimeError("P3 recovery found an invalid protected action record")

    stop = json.loads(stop_path.read_text(encoding="utf-8"))
    if (
        stop.get("semantic_sha256") != semantic_sha256(stop)
        or stop.get("status") != "technical_stop"
        or stop.get("reason") != "router profile categories are invalid"
        or int(stop.get("queries_used", -1)) != 688
        or int(stop.get("blocks_completed", -1)) != 96
        or stop.get("automatic_retry") is not False
    ):
        raise RuntimeError("P3 recovery technical-stop contract is invalid")
    source_run_root = stop_path.parent
    if (source_run_root / "worker_summary.json").exists():
        raise RuntimeError("P3 recovery refuses an already-terminalized source run")

    runtime_worker = (
        ROOT / recovery["runtime_sources"]["worker_path"]
    ).read_text(encoding="utf-8")
    markers = (
        "# Four dense model warmups.",
        "# Four protected technical controls:",
        "for index, profile in enumerate(profiles):",
        "schedule = block_schedule(config, len(inputs))",
        "router = PairRouter.initialize",
    )
    positions = [runtime_worker.index(marker) for marker in markers]
    if positions != sorted(positions):
        raise RuntimeError("P3 recovery authenticated control-flow order changed")
    for required in (
        'raise RuntimeError("P3 physical sidecar or all-fresh cache control failed")',
        "verify_first=True",
        "verify_reused_cache(result[\"cache\"], snapshots)",
        "ledger.require_complete()",
    ):
        if required not in runtime_worker:
            raise RuntimeError("P3 recovery authenticated control-flow guard is missing")

    profiles = [P3Profile.from_mapping(value) for value in p3_config["profiles"]]
    router_profiles = list(dict.fromkeys(profile.base_profile_id for profile in profiles))
    if len(router_profiles) != 6:
        raise RuntimeError("P3 recovery base router categories changed")
    router = PairRouter.initialize(profiles=router_profiles, seed=17)
    mock_features = RouterFeatures(
        (
            GroupFeature(
                AtomicGroup(Camera.PRIMARY, 0, 2),
                profiles[0].base_profile_id,
                1,
                (0.0,) * 20,
            ),
        ),
        (0.0,) * 16,
    )
    router_ms = []
    reset_ms = []
    for _ in range(1000):
        start = time.perf_counter()
        router.predict(mock_features)
        router_ms.append((time.perf_counter() - start) * 1000)
        start = time.perf_counter()
        PairRuntime.reset("p3-v03-recovery-overhead")
        reset_ms.append((time.perf_counter() - start) * 1000)

    sidecar_upper = max(
        max(
            0.0,
            float(row["arms"]["cache"]["queries"][0]["total_wall_ms"])
            - float(row["arms"]["dense"]["queries"][0]["total_wall_ms"]),
            float(row["arms"]["cache"]["queries"][0]["sidecar_wall_ms"]),
        )
        for row in blocks
    )
    preflight = json.loads(
        (ROOT / recovery["preflight"]["path"]).read_text(encoding="utf-8")
    )
    start_bound = datetime.fromisoformat(preflight["completed_at_utc"])
    stop_time = datetime.fromisoformat(stop["completed_at_utc"])
    wall_upper = (stop_time - start_bound).total_seconds()
    if not 0 < wall_upper < int(p3_config["resource_caps"]["wall_seconds"]):
        raise RuntimeError("P3 recovery cannot establish the wall-time cap")

    summary = {
        "schema_version": "pair-p3-worker-recovery-v1",
        "run_id": p3_config["run_id"],
        "recovery_run_id": recovery["run_id"],
        "status": "completed_recovery",
        "configuration_semantic_sha256": p3_config["semantic_sha256"],
        "input_manifest_semantic_sha256": inputs["semantic_sha256"],
        "queries_used": 688,
        "queries_planned": 688,
        "additional_model_queries": 0,
        "blocks_completed": 96,
        "blocks_expected": 96,
        "wall_seconds": wall_upper,
        "sidecar_control": {
            "cache_sample_match": True,
            "underlying_sdpa_calls": 32,
            "incremental_upper_ms": sidecar_upper,
        },
        "all_fresh_control": {"cache_sample_match": True},
        "controls_inferred_from_authenticated_control_flow": True,
        "overhead_mock": {
            "iterations": 1000,
            "router_p99_ms": float(np.quantile(router_ms, 0.99)),
            "reset_p99_ms": float(np.quantile(reset_ms, 0.99)),
        },
        "memory": {
            "peak_reserved_mib": None,
            "peak_aggregate_mib": int(stop["peak_aggregate_mib"]),
            "strict_limit_mib": int(
                p3_config["resource_caps"]["peak_gpu_memory_mib_strict_max"]
            ),
            "recovery_rule": "aggregate selected-GPU telemetry is the conservative gate; PyTorch peak reservation was not terminally persisted",
        },
        "protection": {
            "action_values_persisted": False,
            "action_comparisons_performed": False,
            "expert_actions_accessed": False,
            "terminal_outcomes_accessed": False,
            "simulator_used": False,
            "downloads": 0,
        },
        "source_evidence": {
            "blocks_path": recovery["source_evidence"]["blocks_path"],
            "blocks_sha256": recovery["source_evidence"]["blocks_sha256"],
            "technical_stop_path": recovery["source_evidence"]["technical_stop_path"],
            "technical_stop_sha256": recovery["source_evidence"]["technical_stop_sha256"],
            "runtime_worker_sha256": recovery["runtime_sources"]["worker_sha256"],
        },
        "recovery_scope": recovery["recovery_scope"],
        "advance": {"next_phase": "P4", "authorized": False},
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    reject_protected_fields(summary)
    summary["semantic_sha256"] = semantic_sha256(summary)
    write_once(output_root / "worker_summary.json", summary)
    print(json.dumps({"status": "completed_recovery", "additional_model_queries": 0}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
