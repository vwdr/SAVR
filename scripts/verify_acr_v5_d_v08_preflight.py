#!/usr/bin/env python3
"""CUDA-free verification of the frozen V5-D V08 inference recovery."""

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
    from savr.acr.v5_d_v07_runtime import load_v07
    from savr.acr.v5_d_v08_runtime import INFERENCE_SEMANTICS, V07_MEASURED_MEMORY, load_v08

    v03 = load_v5_d_freeze(ROOT)
    v07 = load_v07(ROOT)
    v08 = load_v08(ROOT)
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
        "inference_semantics",
        "recovery_v08",
        "current_authorization",
        "advance_only_to",
        "semantic_sha256",
    }
    files = (
        "configs/acr/v5_d_inference_semantics_recovery_v08.json",
        "docs/ACR_V5_D_V08_INFERENCE_SEMANTICS_PROTOCOL.md",
        "src/savr/acr/v5_d_v08_runtime.py",
        "src/savr/acr/v5_d_v08_inference.py",
        "src/savr/acr/v5_d_v08_adapter.py",
        "scripts/select_acr_v5_d_v08_gpu.py",
        "scripts/prepare_acr_v5_d_v08_libero_config.py",
        "scripts/run_acr_v5_d_v08.py",
        "scripts/launch_acr_v5_d_v08.sh",
        "scripts/finalize_acr_v5_d_v08.py",
        "scripts/verify_acr_v5_d_v08_preflight.py",
        "tests/acr/test_v5_d_v08_recovery.py",
    )
    runner = (ROOT / "scripts/run_acr_v5_d_v08.py").read_text(encoding="utf-8")
    launcher = (ROOT / "scripts/launch_acr_v5_d_v08.sh").read_text(encoding="utf-8")
    checks = {
        "v08_identity": v08["run_id"] == "acr-v5d-real-tensor-feasibility-v08",
        "v07_stop_linked": v08["recovery_v08"]["v07_technical_stop_semantic_sha256"]
        == "17c6c68ed075f6848768d81eb158ae1d522b2b670df37d0c1db3ab54439bc8c1",
        "only_identity_inference_authorization_changed": {
            key: value for key, value in v08.items() if key not in changed
        }
        == {key: value for key, value in v07.items() if key not in changed},
        "inference_contract_exact": v08["inference_semantics"] == INFERENCE_SEMANTICS,
        "v07_memory_rationale_exact": v08["recovery_v08"]["v07_measured_memory"]
        == V07_MEASURED_MEMORY,
        "query_budget_unchanged": len(schedule) == 111
        and sum(item.kind == "correctness" for item in schedule) == 7
        and sum(item.kind == "warmup" for item in schedule) == 8
        and sum(item.kind == "timed" for item in schedule) == 96,
        "v07_allocator_preserved": v08["allocator"] == v07["allocator"]
        and launcher.index("--backend torch-compile")
        < launcher.index("PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
        < launcher.index("--backend raw-cudagraph"),
        "whole_attempt_guard": "runner._run_attempt = guarded_attempt" in runner
        and "lifecycle.close()" in runner
        and "inference-lifecycle" in runner,
        "no_output_path_change": INFERENCE_SEMANTICS["direct-decoder-pruning"] is False
        and INFERENCE_SEMANTICS["kv-cache-change"] is False
        and INFERENCE_SEMANTICS["hidden-state-output-change"] is False,
        "memory_and_backend_unchanged": v08["memory"] == v07["memory"]
        and v08["raw_cuda_graph"] == v07["raw_cuda_graph"]
        and v08["pre_capture_warmup"] == v07["pre_capture_warmup"],
        "pre_gpu_stop": v08["current_authorization"]["gpu_selection"] is False
        and v08["current_authorization"]["model_queries_max"] == 0,
        "immutable_paths_unused": not (
            ROOT / "results/acr-v5d-real-tensor-feasibility-v08"
        ).exists(),
        "implementation_files_present": all((ROOT / name).is_file() for name in files),
    }
    record = {
        "schema_version": "acr.v5d-preflight-verification.v8",
        "status": "pass" if all(checks.values()) else "fail",
        "configuration_semantic_sha256": v08["semantic_sha256"],
        "v07_configuration_semantic_sha256": v07["semantic_sha256"],
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
        "advance_only_to": "V08_PRE_GPU_CHECKPOINT",
    }
    record["semantic_sha256"] = hashlib.sha256(canonical_bytes(record)).hexdigest()
    return record


def main() -> int:
    record = verify()
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0 if record["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
