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
        resolve_v5_d_recovery,
    )
    from savr.acr.v5_d_torch_backend import build_openvla_core_functions

    config = load_v5_d_freeze(ROOT)
    base = json.loads(
        (ROOT / "configs/acr/v5_d_gpu_feasibility_freeze.json").read_text(encoding="utf-8")
    )
    recovery_v02 = json.loads(
        (ROOT / "configs/acr/v5_d_gpu_feasibility_recovery_v02.json").read_text(
            encoding="utf-8"
        )
    )
    resolved_v02 = resolve_v5_d_recovery(base, recovery_v02)
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
    recovery_source = (ROOT / "src/savr/acr/v5_d_recovery.py").read_text(encoding="utf-8")
    scientific_sections = (
        "selected_method",
        "pinned_stack",
        "environment_hashes",
        "checkpoint_hashes",
        "upstream_source_hashes",
        "identities",
        "backend_waterfall",
        "inputs",
        "tensor_contract",
        "correctness",
        "timing",
        "analysis",
        "gates",
        "gpu_selection",
        "memory",
        "resource_caps",
        "recovery",
    )
    files = (
        "src/savr/acr/v5_d_runtime.py",
        "src/savr/acr/v5_d_torch_backend.py",
        "src/savr/acr/v5_d_recovery.py",
        "scripts/select_acr_v5_d_gpu.py",
        "scripts/prepare_acr_v5_d_libero_config.py",
        "scripts/verify_acr_v5_d_v03_import.py",
        "scripts/run_acr_v5_d.py",
        "scripts/launch_acr_v5_d.sh",
        "scripts/analyze_acr_v5_d.py",
        "scripts/verify_acr_v5_d.py",
        "scripts/finalize_acr_v5_d.py",
    )
    checks = {
        "freeze_semantic_valid": True,
        "v03_run_identity": config["run_id"] == "acr-v5d-real-tensor-feasibility-v03",
        "v01_evidence_linked": config["recovery_v02"]["v01_technical_stop_semantic_sha256"]
        == "edf5872fa818f5806601f52143cb17cec7dd4974e03cc4e2ed43c3d042fb4412",
        "v02_evidence_linked": config["recovery_v03"][
            "v02_technical_stop_semantic_sha256"
        ]
        == "0a30bd847bf2e1549c376200e559a23c670b33c0b01215926c90a15704487661",
        "scientific_contract_unchanged": all(
            config[key] == base[key] for key in scientific_sections
        ),
        "scientific_contract_equals_v02": all(
            config[key] == resolved_v02[key] for key in scientific_sections
        ),
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
        "libero_config_precedes_runner": launch_source.index(
            "scripts/prepare_acr_v5_d_libero_config.py"
        )
        < launch_source.index("scripts/run_acr_v5_d.py"),
        "libero_config_is_canonical_and_gpu_free": all(
            token in recovery_source
            for token in (
                "canonical_libero_bytes",
                "create_libero_config_once",
                "validate_libero_config",
                "os.O_EXCL",
            )
        )
        and all(
            token not in recovery_source
            for token in ("nvidia-smi", "import torch", "env.step", "env.reset")
        ),
        "pre_model_failures_are_recorded": all(
            token in runner_source
            for token in (
                "record_pre_model_stop",
                "build_pre_model_stop_record",
                "return 4",
            )
        ),
        "exact_checkpoint_restoration_integrated": all(
            token in recovery_source
            for token in (
                "capture_checkpoint_baseline",
                "restore_checkpoint_exact",
                "_LOADER_BACKUP_PATTERN",
                "nonprotected_signatures",
                "backup content changed",
                "idempotent_ready",
            )
        )
        and all(
            token in runner_source
            for token in (
                "capture_checkpoint_baseline",
                "restore_checkpoint_exact",
                "selected_gpu_compute_capability",
            )
        ),
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
        "schema_version": "acr.v5d-preflight-verification.v3",
        "status": "pass" if all(checks.values()) else "fail",
        "configuration_semantic_sha256": config["semantic_sha256"],
        "backend_version": "acr-v5d-static-backend-v03-recovery",
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
        "advance_only_to": "EXPLICIT_USER_COORDINATION_BEFORE_V03_GPU_SELECTION",
    }
    record["semantic_sha256"] = hashlib.sha256(canonical_bytes(record)).hexdigest()
    return record


def main() -> int:
    record = verify()
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0 if record["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
