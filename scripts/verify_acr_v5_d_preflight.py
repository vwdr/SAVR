#!/usr/bin/env python3
"""Dependency-free deterministic verification of the pre-GPU V5-D implementation."""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify() -> dict[str, Any]:
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from savr.acr.v5_d_runtime import (
        BackendKind,
        BackendWaterfall,
        FrozenQueryLedger,
        TechnicalReason,
        V5DProtocolViolation,
        frozen_query_schedule,
        load_v5_d_freeze,
    )
    from savr.acr.v5_d_torch_backend import build_openvla_core_functions

    config = load_v5_d_freeze(ROOT)
    schedule = frozen_query_schedule(config)
    ledger = FrozenQueryLedger(config)
    for identity in schedule:
        ledger.consume(identity.label)
    ledger.require_complete()

    waterfall = BackendWaterfall(config)
    waterfall.begin(BackendKind.TORCH_COMPILE, process_token="compile-process")
    waterfall.record_preparation_launch("compile-wrist")
    raw_permitted = waterfall.technical_failure(
        TechnicalReason.FULL_GRAPH_CAPTURE_ERROR, "pre-output technical failure"
    )
    waterfall.begin(BackendKind.RAW_CUDAGRAPH, process_token="raw-process")
    waterfall.begin_correctness()
    waterfall.record_correctness()
    post_output_permitted = waterfall.technical_failure(
        TechnicalReason.FULL_GRAPH_CAPTURE_ERROR, "post-output failure"
    )
    raw_after_output_rejected = False
    try:
        waterfall.begin(BackendKind.RAW_CUDAGRAPH, process_token="third-process")
    except V5DProtocolViolation:
        raw_after_output_rejected = True

    core_source = inspect.getsource(build_openvla_core_functions)
    runner_source = (ROOT / "scripts/run_acr_v5_d.py").read_text(encoding="utf-8")
    selector_source = (ROOT / "scripts/select_acr_v5_d_gpu.py").read_text(encoding="utf-8")
    launch_source = (ROOT / "scripts/launch_acr_v5_d.sh").read_text(encoding="utf-8")
    files = (
        "src/savr/acr/v5_d_runtime.py",
        "src/savr/acr/v5_d_torch_backend.py",
        "scripts/select_acr_v5_d_gpu.py",
        "scripts/run_acr_v5_d.py",
        "scripts/launch_acr_v5_d.sh",
        "scripts/analyze_acr_v5_d.py",
        "scripts/verify_acr_v5_d.py",
        "scripts/finalize_acr_v5_d.py",
    )
    checks = {
        "freeze_semantic_valid": True,
        "query_count_111": len(schedule) == 111,
        "correctness_count_7": sum(item.kind == "correctness" for item in schedule) == 7,
        "warmup_count_8": sum(item.kind == "warmup" for item in schedule) == 8,
        "timed_count_96": sum(item.kind == "timed" for item in schedule) == 96,
        "raw_pre_output_permitted": raw_permitted,
        "raw_post_output_prohibited": not post_output_permitted and raw_after_output_rejected,
        "cores_have_no_host_transfer": all(
            token not in core_source
            for token in (".cpu(", ".numpy(", "synchronize", "hashlib", "open(")
        ),
        "runner_has_zero_simulator": all(
            token in runner_source
            for token in (
                "num_trials_per_task=0",
                '"simulator_episodes": 0',
                '"simulator_resets": 0',
                '"new_task_outcomes": 0',
            )
        ),
        "selector_aggregate_only": "--query-gpu=" in selector_source
        and "--query-compute-apps" not in selector_source
        and "process_identities_inspected" in selector_source,
        "launch_compile_first": launch_source.index("--backend torch-compile")
        < launch_source.index("--backend raw-cudagraph"),
        "launch_raw_only_exit_20": "if [[ ${status} -eq 20 ]]" in launch_source,
        "local_cache_roots": all(
            name in launch_source
            for name in (
                "HF_HOME",
                "HF_HUB_CACHE",
                "TORCH_HOME",
                "TORCHINDUCTOR_CACHE_DIR",
                "TRITON_CACHE_DIR",
            )
        ),
        "implementation_files_present": all((ROOT / name).is_file() for name in files),
    }
    record = {
        "schema_version": "acr.v5d-preflight-verification.v1",
        "status": "pass" if all(checks.values()) else "fail",
        "configuration_semantic_sha256": config["semantic_sha256"],
        "backend_version": "acr-v5d-static-backend-v1",
        "query_labels_sha256": hashlib.sha256(
            canonical_bytes([item.label for item in schedule])
        ).hexdigest(),
        "checks": checks,
        "source_hashes": {name: file_sha256(ROOT / name) for name in files},
        "resources_used": {
            "gpu_count": 0,
            "model_queries": 0,
            "simulator_episodes": 0,
            "simulator_resets": 0,
            "downloads": 0,
            "new_task_outcomes": 0,
        },
        "advance_only_to": "EXPLICIT_USER_COORDINATION_BEFORE_GPU_SELECTION",
    }
    record["semantic_sha256"] = hashlib.sha256(canonical_bytes(record)).hexdigest()
    return record


def main() -> int:
    record = verify()
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0 if record["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
