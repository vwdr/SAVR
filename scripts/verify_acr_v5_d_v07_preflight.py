#!/usr/bin/env python3
"""CUDA-free verification of the frozen V5-D v07 allocator recovery."""

from __future__ import annotations

import hashlib
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
    from savr.acr.v5_d_v06_runtime import load_v06
    from savr.acr.v5_d_v07_runtime import ALLOCATOR, V06_MEASURED_MEMORY, load_v07

    v03 = load_v5_d_freeze(ROOT)
    v06 = load_v06(ROOT)
    v07 = load_v07(ROOT)
    schedule = frozen_query_schedule(v03)
    ledger = FrozenQueryLedger(v03)
    for identity in schedule:
        ledger.consume(identity.label)
    ledger.require_complete()
    changed = {
        "schema_version",
        "status",
        "authorized_at",
        "authorized_scope",
        "protocol",
        "run_id",
        "allocator",
        "recovery_v07",
        "current_authorization",
        "advance_only_to",
        "semantic_sha256",
    }
    launcher = (ROOT / "scripts/launch_acr_v5_d_v07.sh").read_text(encoding="utf-8")
    runner = (ROOT / "scripts/run_acr_v5_d_v07.py").read_text(encoding="utf-8")
    files = (
        "configs/acr/v5_d_expandable_segments_recovery_v07.json",
        "docs/ACR_V5_D_V07_EXPANDABLE_SEGMENTS_PROTOCOL.md",
        "src/savr/acr/v5_d_v07_runtime.py",
        "src/savr/acr/v5_d_v07_adapter.py",
        "scripts/select_acr_v5_d_v07_gpu.py",
        "scripts/prepare_acr_v5_d_v07_libero_config.py",
        "scripts/run_acr_v5_d_v07.py",
        "scripts/launch_acr_v5_d_v07.sh",
        "scripts/finalize_acr_v5_d_v07.py",
        "scripts/verify_acr_v5_d_v07_preflight.py",
        "tests/acr/test_v5_d_v07_recovery.py",
    )
    checks = {
        "v07_identity": v07["run_id"] == "acr-v5d-real-tensor-feasibility-v07",
        "v06_stop_linked": v07["recovery_v07"]["v06_technical_stop_semantic_sha256"]
        == "0588f628a118a2f467215c2337bc23452f3b8e98d0b5865c37be0d2892a18edb",
        "only_identity_allocator_authorization_changed": {
            key: value for key, value in v07.items() if key not in changed
        }
        == {key: value for key, value in v06.items() if key not in changed},
        "allocator_exact": v07["allocator"] == ALLOCATOR
        and ALLOCATOR["exact_value"] == "expandable_segments:True"
        and ALLOCATOR["other_allocator_options"] == 0
        and ALLOCATOR["automatic_retries"] == 0,
        "v06_memory_rationale_exact": v07["recovery_v07"]["v06_measured_memory"]
        == V06_MEASURED_MEMORY,
        "query_budget_unchanged": len(schedule) == 111
        and sum(item.kind == "correctness" for item in schedule) == 7
        and sum(item.kind == "warmup" for item in schedule) == 8
        and sum(item.kind == "timed" for item in schedule) == 96,
        "raw_only_environment_scope": launcher.index("--backend torch-compile")
        < launcher.index("PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
        < launcher.index("--backend raw-cudagraph")
        and "Refusing inherited allocator configuration" in launcher,
        "runtime_enforces_scope": "lacks the exact expandable-segments setting" in runner
        and "compiler process must not receive an allocator override" in runner,
        "no_other_memory_change": "empty_cache" not in launcher
        and "max_split_size_mb" not in launcher
        and "cudaMallocAsync" not in launcher,
        "memory_cap_unchanged": v07["memory"] == v06["memory"]
        and v07["memory"]["peak_reserved_bytes_max"] == 23 * 1024**3,
        "v06_backend_unchanged": v07["raw_cuda_graph"] == v06["raw_cuda_graph"]
        and v07["pre_capture_warmup"] == v06["pre_capture_warmup"],
        "immutable_paths_unused": not (
            ROOT / "results/acr-v5d-real-tensor-feasibility-v07"
        ).exists()
        and not (ROOT / "results/acr-v5d-analysis-v07").exists()
        and not (ROOT / "results/acr-v5d-verification-v07").exists(),
        "implementation_files_present": all((ROOT / name).is_file() for name in files),
    }
    record = {
        "schema_version": "acr.v5d-preflight-verification.v7",
        "status": "pass" if all(checks.values()) else "fail",
        "configuration_semantic_sha256": v07["semantic_sha256"],
        "v06_configuration_semantic_sha256": v06["semantic_sha256"],
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
            "downloads": 0,
            "new_task_outcomes": 0,
        },
        "advance_only_to": "ONE_FROZEN_V07_GPU_ATTEMPT",
    }
    record["semantic_sha256"] = hashlib.sha256(canonical_bytes(record)).hexdigest()
    return record


def main() -> int:
    record = verify()
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0 if record["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
