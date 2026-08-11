#!/usr/bin/env python3
"""Dependency-free, CUDA-free verification of the frozen V5-D v04 protocol."""

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
    from savr.acr.v5_d_runtime import (
        FrozenQueryLedger,
        frozen_query_schedule,
        load_v5_d_freeze,
    )
    from savr.acr.v5_d_v04_runtime import load_v04
    from savr.acr.v5_d_v04_torch_backend import V04SharedPoolRawCudaGraphCorePair

    v03 = load_v5_d_freeze(ROOT)
    v04 = load_v04(ROOT)
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
    )
    backend_source = inspect.getsource(V04SharedPoolRawCudaGraphCorePair)
    adapter_source = (ROOT / "src/savr/acr/v5_d_v04_adapter.py").read_text(encoding="utf-8")
    runner_source = (ROOT / "scripts/run_acr_v5_d_v04.py").read_text(encoding="utf-8")
    launch_source = (ROOT / "scripts/launch_acr_v5_d_v04.sh").read_text(encoding="utf-8")
    files = (
        "configs/acr/v5_d_titan_memory_recovery_v04.json",
        "docs/ACR_V5_D_V04_TITAN_MEMORY_REMEDIATION_PROTOCOL.md",
        "src/savr/acr/v5_d_runtime.py",
        "src/savr/acr/v5_d_torch_backend.py",
        "src/savr/acr/v5_d_v04_runtime.py",
        "src/savr/acr/v5_d_v04_torch_backend.py",
        "src/savr/acr/v5_d_v04_adapter.py",
        "scripts/select_acr_v5_d_v04_gpu.py",
        "scripts/prepare_acr_v5_d_v04_libero_config.py",
        "scripts/run_acr_v5_d_v04.py",
        "scripts/launch_acr_v5_d_v04.sh",
        "scripts/finalize_acr_v5_d_v04.py",
        "scripts/verify_acr_v5_d_v04_preflight.py",
        "scripts/verify_acr_v5_d_v04_import.py",
    )
    checks = {
        "v04_identity": v04["run_id"] == "acr-v5d-real-tensor-feasibility-v04",
        "v03_stop_linked": v04["recovery_v04"]["v03_technical_stop_semantic_sha256"]
        == "1016569f642b21266e8f0b75b5906716200055f5d37385c5501b6711f9a6bd54",
        "scientific_contract_unchanged": all(v04[key] == v03[key] for key in unchanged_sections),
        "query_budget_unchanged": len(schedule) == 111
        and sum(item.kind == "correctness" for item in schedule) == 7
        and sum(item.kind == "warmup" for item in schedule) == 8
        and sum(item.kind == "timed" for item in schedule) == 96,
        "shared_pool_contract": v04["raw_cuda_graph"]["shared_private_pool"] is True
        and v04["raw_cuda_graph"]["capture_order"] == ["wrist", "downstream"]
        and v04["raw_cuda_graph"]["replay_order"] == ["wrist", "downstream"]
        and v04["raw_cuda_graph"]["concurrent_replay"] is False,
        "memory_cap_unchanged": v04["memory"]["peak_reserved_gib_max"] == 23
        and v04["memory"]["peak_reserved_bytes_max"] == 23 * 1024**3
        and v04["memory"]["raise_cap"] is False,
        "backend_fail_closed": all(
            token in backend_source
            for token in (
                "pool_method()",
                "V04 shared-pool wrist replayed out of order",
                "V04 shared-pool downstream replayed out of order",
                "V04 shared-pool replay stream changed",
                "_invalidated",
            )
        ),
        "runner_activates_only_frozen_pool": "install_v04_adapters" in runner_source
        and "V04SharedPoolRawCudaGraphCorePair" in adapter_source
        and "raw_graph_memory_trace" in adapter_source,
        "launch_compile_first": launch_source.index("--backend torch-compile")
        < launch_source.index("--backend raw-cudagraph"),
        "launch_is_v04": "scripts/run_acr_v5_d_v04.py" in launch_source
        and "acr-v5d-real-tensor-feasibility-v04" in launch_source,
        "immutable_run_paths_unused": not (
            ROOT / "results/acr-v5d-real-tensor-feasibility-v04"
        ).exists()
        and not (ROOT / "results/acr-v5d-analysis-v04").exists()
        and not (ROOT / "results/acr-v5d-verification-v04").exists(),
        "implementation_files_present": all((ROOT / name).is_file() for name in files),
    }
    record = {
        "schema_version": "acr.v5d-preflight-verification.v4",
        "status": "pass" if all(checks.values()) else "fail",
        "configuration_semantic_sha256": v04["semantic_sha256"],
        "v03_configuration_semantic_sha256": v03["semantic_sha256"],
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
        "advance_only_to": "EXPLICIT_USER_COORDINATION_BEFORE_V04_GPU_SELECTION",
    }
    record["semantic_sha256"] = hashlib.sha256(canonical_bytes(record)).hexdigest()
    return record


def main() -> int:
    record = verify()
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0 if record["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
