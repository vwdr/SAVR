#!/usr/bin/env python3
"""Dependency-free, CUDA-free verification of the frozen V5-D v05 recovery."""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify() -> dict[str, Any]:
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from savr.acr.v5_d_runtime import FrozenQueryLedger, frozen_query_schedule, load_v5_d_freeze
    from savr.acr.v5_d_v04_runtime import load_v04
    from savr.acr.v5_d_v05_runtime import load_v05
    from savr.acr.v5_d_v05_transition import V05TransitionSampler

    v03 = load_v5_d_freeze(ROOT)
    v04 = load_v04(ROOT)
    v05 = load_v05(ROOT)
    schedule = frozen_query_schedule(v03)
    ledger = FrozenQueryLedger(v03)
    for identity in schedule:
        ledger.consume(identity.label)
    ledger.require_complete()
    unchanged_sections = (
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
        "resource_caps",
        "recovery",
        "recovery_v02",
        "recovery_v03",
        "recovery_v04",
        "raw_cuda_graph",
        "memory",
    )
    transition_source = inspect.getsource(V05TransitionSampler)
    run_source = (ROOT / "scripts/run_acr_v5_d_v05.py").read_text(encoding="utf-8")
    launch_source = (ROOT / "scripts/launch_acr_v5_d_v05.sh").read_text(encoding="utf-8")
    files = (
        "configs/acr/v5_d_transition_recovery_v05.json",
        "docs/ACR_V5_D_V05_TRANSITION_RECOVERY_PROTOCOL.md",
        "reports/runtime/acr_v5_d_v04_technical_stop.json",
        "src/savr/acr/v5_d_v05_runtime.py",
        "src/savr/acr/v5_d_v05_transition.py",
        "src/savr/acr/v5_d_v05_adapter.py",
        "scripts/select_acr_v5_d_v05_gpu.py",
        "scripts/prepare_acr_v5_d_v05_libero_config.py",
        "scripts/run_acr_v5_d_v05.py",
        "scripts/launch_acr_v5_d_v05.sh",
        "scripts/finalize_acr_v5_d_v05.py",
        "scripts/verify_acr_v5_d_v05_preflight.py",
        "scripts/verify_acr_v5_d_v05_import.py",
        "tests/acr/test_v5_d_v05_recovery.py",
    )
    transition = v05["transition_revalidation"]
    checks = {
        "v05_identity": v05["run_id"] == "acr-v5d-real-tensor-feasibility-v05",
        "v04_stop_linked": v05["recovery_v05"]["v04_technical_stop_semantic_sha256"]
        == "a3515180022df7938b50956851a2ca05b698819da38b387ddc23b54e59769811",
        "scientific_contract_unchanged": all(v05[key] == v04[key] for key in unchanged_sections),
        "query_budget_unchanged": len(schedule) == 111
        and sum(item.kind == "correctness" for item in schedule) == 7
        and sum(item.kind == "warmup" for item in schedule) == 8
        and sum(item.kind == "timed" for item in schedule) == 96,
        "transition_window_frozen": transition["initial_discard_seconds"] == 2
        and transition["sample_count"] == 3
        and transition["sample_interval_seconds"] == 5
        and transition["maximum_memory_used_mib_each_sample"] == 512
        and transition["maximum_utilization_percent_each_sample"] == 5
        and transition["additional_sampling_windows"] == 0
        and transition["automatic_retries"] == 0,
        "transition_fail_closed": all(
            token in transition_source
            for token in (
                "self._sleep",
                "all(all(item.values()) for item in checks)",
                "V05 transition eligibility window failed",
                "self._write_once(record)",
            )
        ),
        "raw_requires_permit_before_sampler": run_source.index("raw-transition-permit")
        < run_source.index("V05TransitionSampler("),
        "launch_compile_first": launch_source.index("--backend torch-compile")
        < launch_source.index("--backend raw-cudagraph"),
        "launch_is_v05": "scripts/run_acr_v5_d_v05.py" in launch_source
        and "acr-v5d-real-tensor-feasibility-v05" in launch_source,
        "shared_pool_unchanged": v05["raw_cuda_graph"] == v04["raw_cuda_graph"],
        "memory_cap_unchanged": v05["memory"] == v04["memory"]
        and v05["memory"]["peak_reserved_bytes_max"] == 23 * 1024**3,
        "immutable_run_paths_unused": not (
            ROOT / "results/acr-v5d-real-tensor-feasibility-v05"
        ).exists()
        and not (ROOT / "results/acr-v5d-analysis-v05").exists()
        and not (ROOT / "results/acr-v5d-verification-v05").exists(),
        "implementation_files_present": all((ROOT / name).is_file() for name in files),
    }
    record = {
        "schema_version": "acr.v5d-preflight-verification.v5",
        "status": "pass" if all(checks.values()) else "fail",
        "configuration_semantic_sha256": v05["semantic_sha256"],
        "v04_configuration_semantic_sha256": v04["semantic_sha256"],
        "query_labels_sha256": hashlib.sha256(
            canonical_bytes([item.label for item in schedule])
        ).hexdigest(),
        "checks": checks,
        "source_hashes": {name: file_sha256(ROOT / name) for name in files},
        "resources_used": {
            "gpu_count": 0,
            "gpu_inspections": 0,
            "cuda_initialized": False,
            "model_queries": 0,
            "simulator_episodes": 0,
            "simulator_resets": 0,
            "downloads": 0,
            "new_task_outcomes": 0,
        },
        "advance_only_to": "EXPLICIT_USER_COORDINATION_BEFORE_V05_GPU_SELECTION",
    }
    record["semantic_sha256"] = hashlib.sha256(canonical_bytes(record)).hexdigest()
    return record


def main() -> int:
    record = verify()
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0 if record["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
