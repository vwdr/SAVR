#!/usr/bin/env python3
"""Dependency-free, CUDA-free verification of the frozen V5-D v06 recovery."""

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
    from savr.acr.v5_d_v05_runtime import load_v05
    from savr.acr.v5_d_v06_runtime import PRE_CAPTURE_WARMUP, V05_MEASURED_MEMORY, load_v06
    from savr.acr.v5_d_v06_torch_backend import V06PreCaptureWarmupRawCudaGraphCorePair

    v03 = load_v5_d_freeze(ROOT)
    v05 = load_v05(ROOT)
    v06 = load_v06(ROOT)
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
        "memory",
        "transition_revalidation",
    )
    backend_source = inspect.getsource(V06PreCaptureWarmupRawCudaGraphCorePair)
    run_source = (ROOT / "scripts/run_acr_v5_d_v06.py").read_text(encoding="utf-8")
    launch_source = (ROOT / "scripts/launch_acr_v5_d_v06.sh").read_text(encoding="utf-8")
    v05_hashes = {
        "configs/acr/v5_d_transition_recovery_v05.json": "73e892321424e07c53c2dd5e81679a5c0ee6f6137ef2f5fc18c4f1d919663f5d",
        "docs/ACR_V5_D_V05_TRANSITION_RECOVERY_PROTOCOL.md": "3c507ae72fa4c4686533b7aaa222375591c417b7024dce8d4c5a517dd38c76bb",
        "reports/runtime/acr_v5_d_v05_technical_stop.json": "9d8deb9dc2d7ea061cf8d1a6a2a145369cddef40574c3b04a787d8cddc9b82db",
        "src/savr/acr/v5_d_v05_runtime.py": "3818f8bcd8ed29e72fd6bee7e287175a9d78af6ad621a649de77104dbec5084f",
        "src/savr/acr/v5_d_v05_transition.py": "fce291321d95c93750049aee5fd369f52491d615086290648d6c587189aaa84c",
        "src/savr/acr/v5_d_v05_adapter.py": "d574a25fe77c98f3091f36ce180734d73bfb2536b2c28ce2ab79c83ef513bece",
        "src/savr/acr/v5_d_v04_torch_backend.py": "747cb972d8c797c96cc69684d26b7ef4d55dc9e073ba5c63412a78770a87c0f7",
        "src/savr/acr/v5_d_v04_adapter.py": "3c6236f9ab05f09948044c2b54d157d30f75c72339ebfba41955ca526501d8a5",
        "scripts/run_acr_v5_d_v05.py": "18b942f280237cf617ac89b4779c89b4ea85690f9e2491ff46e6f395bddc9687",
        "tests/acr/test_v5_d_v05_recovery.py": "a20858a2e974a0c8f7271d1435a66d238409be39cde4e7b7e164c104e9fb34c9",
    }
    files = (
        "configs/acr/v5_d_precapture_warmup_recovery_v06.json",
        "docs/ACR_V5_D_V06_PRECAPTURE_WARMUP_PROTOCOL.md",
        "src/savr/acr/v5_d_v06_runtime.py",
        "src/savr/acr/v5_d_v06_torch_backend.py",
        "src/savr/acr/v5_d_v06_adapter.py",
        "scripts/select_acr_v5_d_v06_gpu.py",
        "scripts/prepare_acr_v5_d_v06_libero_config.py",
        "scripts/run_acr_v5_d_v06.py",
        "scripts/launch_acr_v5_d_v06.sh",
        "scripts/finalize_acr_v5_d_v06.py",
        "scripts/verify_acr_v5_d_v06_preflight.py",
        "scripts/verify_acr_v5_d_v06_import.py",
        "tests/acr/test_v5_d_v06_recovery.py",
        "tests/acr/test_v5_d_v06_torch_backend.py",
    )
    raw_delta = {
        key: value
        for key, value in v06["raw_cuda_graph"].items()
        if v05["raw_cuda_graph"].get(key) != value
    }
    checks = {
        "v06_identity": v06["run_id"] == "acr-v5d-real-tensor-feasibility-v06",
        "v05_stop_linked": v06["recovery_v06"]["v05_technical_stop_semantic_sha256"]
        == "cb6d9120fc2e6ee69aaa83d677598d21741be8eaf5a3456bc21461d30eb3cc3f",
        "scientific_contract_unchanged": all(v06[key] == v05[key] for key in unchanged_sections),
        "query_budget_unchanged": len(schedule) == 111
        and sum(item.kind == "correctness" for item in schedule) == 7
        and sum(item.kind == "warmup" for item in schedule) == 8
        and sum(item.kind == "timed" for item in schedule) == 96,
        "only_raw_lifecycle_changed": raw_delta
        == {
            "warmup_order": ["wrist", "downstream"],
            "all_warmups_before_any_capture": True,
            "inter_capture_warmups": 0,
            "empty_cache_calls": 0,
        },
        "warmup_lifecycle_frozen": v06["pre_capture_warmup"] == PRE_CAPTURE_WARMUP
        and PRE_CAPTURE_WARMUP["iterations_per_core"] == 3
        and PRE_CAPTURE_WARMUP["order"] == ["wrist", "downstream"]
        and PRE_CAPTURE_WARMUP["capture_order"] == ["wrist", "downstream"],
        "v05_memory_rationale_frozen": v06["recovery_v06"]["v05_measured_memory"]
        == V05_MEASURED_MEMORY,
        "no_empty_cache_or_allocator_change": ".empty_cache(" not in backend_source
        and "PYTORCH_CUDA_ALLOC_CONF" not in launch_source
        and "max_split_size_mb" not in launch_source,
        "same_stream_and_shared_pool": all(
            token in backend_source
            for token in (
                "self._capture_stream = self.torch.cuda.Stream()",
                'self._warmup("wrist", wrist_call)',
                'self._warmup("downstream", downstream_call)',
                'self._capture_without_warmup("wrist", wrist_call)',
                "pool=shared_pool",
            )
        ),
        "raw_requires_permit_before_sampler": run_source.index("raw-transition-permit")
        < run_source.index("V05TransitionSampler("),
        "launch_compile_first": launch_source.index("--backend torch-compile")
        < launch_source.index("--backend raw-cudagraph"),
        "launch_is_v06": "scripts/run_acr_v5_d_v06.py" in launch_source
        and "acr-v5d-real-tensor-feasibility-v06" in launch_source,
        "transition_rule_unchanged": v06["transition_revalidation"]
        == v05["transition_revalidation"],
        "memory_cap_unchanged": v06["memory"] == v05["memory"]
        and v06["memory"]["peak_reserved_bytes_max"] == 23 * 1024**3,
        "v01_v05_critical_files_unchanged": all(
            file_sha256(ROOT / name) == expected for name, expected in v05_hashes.items()
        ),
        "immutable_run_paths_unused": not (
            ROOT / "results/acr-v5d-real-tensor-feasibility-v06"
        ).exists()
        and not (ROOT / "results/acr-v5d-analysis-v06").exists()
        and not (ROOT / "results/acr-v5d-verification-v06").exists(),
        "implementation_files_present": all((ROOT / name).is_file() for name in files),
    }
    record = {
        "schema_version": "acr.v5d-preflight-verification.v6",
        "status": "pass" if all(checks.values()) else "fail",
        "configuration_semantic_sha256": v06["semantic_sha256"],
        "v05_configuration_semantic_sha256": v05["semantic_sha256"],
        "query_labels_sha256": hashlib.sha256(
            canonical_bytes([item.label for item in schedule])
        ).hexdigest(),
        "checks": checks,
        "source_hashes": {name: file_sha256(ROOT / name) for name in files},
        "protected_v05_hashes": v05_hashes,
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
        "advance_only_to": "EXPLICIT_USER_COORDINATION_BEFORE_V06_GPU_SELECTION",
    }
    record["semantic_sha256"] = hashlib.sha256(canonical_bytes(record)).hexdigest()
    return record


def main() -> int:
    record = verify()
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0 if record["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
