#!/usr/bin/env python3
"""CUDA-free verification of the frozen V5-D V09 default-allocator recovery."""

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
    from savr.acr.v5_d_v08_runtime import load_v08
    from savr.acr.v5_d_v09_runtime import DEFAULT_ALLOCATOR, V08_MEASURED_EVIDENCE, load_v09

    v03 = load_v5_d_freeze(ROOT)
    v08 = load_v08(ROOT)
    v09 = load_v09(ROOT)
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
        "recovery_v09",
        "current_authorization",
        "advance_only_to",
        "semantic_sha256",
    }
    files = (
        "configs/acr/v5_d_default_allocator_recovery_v09.json",
        "docs/ACR_V5_D_V09_DEFAULT_ALLOCATOR_PROTOCOL.md",
        "src/savr/acr/v5_d_v09_runtime.py",
        "src/savr/acr/v5_d_v09_adapter.py",
        "scripts/select_acr_v5_d_v09_gpu.py",
        "scripts/prepare_acr_v5_d_v09_libero_config.py",
        "scripts/run_acr_v5_d_v09.py",
        "scripts/launch_acr_v5_d_v09.sh",
        "scripts/finalize_acr_v5_d_v09.py",
        "scripts/verify_acr_v5_d_v09_preflight.py",
        "tests/acr/test_v5_d_v09_recovery.py",
    )
    runner = (ROOT / "scripts/run_acr_v5_d_v09.py").read_text(encoding="utf-8")
    launcher = (ROOT / "scripts/launch_acr_v5_d_v09.sh").read_text(encoding="utf-8")
    adapter = (ROOT / "src/savr/acr/v5_d_v09_adapter.py").read_text(encoding="utf-8")
    checks = {
        "v09_identity": v09["run_id"] == "acr-v5d-real-tensor-feasibility-v09",
        "v08_stop_linked": v09["recovery_v09"]["v08_technical_stop_semantic_sha256"]
        == "3572abf107ad1b0ef10557e27c66b3d5ad1d967a5f82b633c572bef907d16d98",
        "only_identity_allocator_authorization_changed": {
            key: value for key, value in v09.items() if key not in changed
        }
        == {key: value for key, value in v08.items() if key not in changed},
        "default_allocator_exact": v09["allocator"] == DEFAULT_ALLOCATOR
        and DEFAULT_ALLOCATOR["exact_value"] is None
        and DEFAULT_ALLOCATOR["must_be_unset_before_torch_import"] is True
        and DEFAULT_ALLOCATOR["experimental"] is False,
        "v08_evidence_exact": v09["recovery_v09"]["v08_measured_evidence"]
        == V08_MEASURED_EVIDENCE,
        "query_budget_unchanged": len(schedule) == 111
        and sum(item.kind == "correctness" for item in schedule) == 7
        and sum(item.kind == "warmup" for item in schedule) == 8
        and sum(item.kind == "timed" for item in schedule) == 96,
        "allocator_override_absent": "expandable_segments:True" not in launcher
        and "PYTORCH_CUDA_ALLOC_CONF=" not in launcher
        and "Refusing inherited allocator configuration" in launcher,
        "runtime_enforces_default": (
            'os.environ.get("PYTORCH_CUDA_ALLOC_CONF") is not None' in runner
            and "must use the default allocator environment" in runner
            and 'observed_backend != "native"' in adapter
            and '"observed_backend": observed_backend' in adapter
        ),
        "compiler_then_raw_unchanged": launcher.index("--backend torch-compile")
        < launcher.index("--backend raw-cudagraph"),
        "no_other_system_change": "empty_cache" not in launcher
        and "max_split_size_mb" not in launcher
        and "cudaMallocAsync" not in launcher,
        "v08_backend_inference_memory_unchanged": v09["raw_cuda_graph"]
        == v08["raw_cuda_graph"]
        and v09["pre_capture_warmup"] == v08["pre_capture_warmup"]
        and v09["inference_semantics"] == v08["inference_semantics"]
        and v09["memory"] == v08["memory"],
        "one_gpu_attempt_authorized": v09["current_authorization"]["gpu_selection"] is True
        and v09["current_authorization"]["model_queries_max"] == 111
        and v09["current_authorization"]["automatic_retry"] is False
        and v09["current_authorization"]["simulator_use"] is False
        and v09["current_authorization"]["protected_outcome_access"] is False
        and v09["current_authorization"]["manuscript_changes"] is False,
        "immutable_paths_unused": not (
            ROOT / "results/acr-v5d-real-tensor-feasibility-v09"
        ).exists(),
        "implementation_files_present": all((ROOT / name).is_file() for name in files),
    }
    record = {
        "schema_version": "acr.v5d-preflight-verification.v9",
        "status": "pass" if all(checks.values()) else "fail",
        "configuration_semantic_sha256": v09["semantic_sha256"],
        "v08_configuration_semantic_sha256": v08["semantic_sha256"],
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
        "advance_only_to": "ONE_FROZEN_V09_GPU_ATTEMPT",
    }
    record["semantic_sha256"] = hashlib.sha256(canonical_bytes(record)).hexdigest()
    return record


def main() -> int:
    record = verify()
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0 if record["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
