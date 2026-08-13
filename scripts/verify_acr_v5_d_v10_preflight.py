#!/usr/bin/env python3
"""CUDA-hidden verification of the frozen V10 downstream-only graph recovery."""

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
    from savr.acr.v5_d_v09_runtime import load_v09
    from savr.acr.v5_d_v10_runtime import (
        HYBRID_ARCHITECTURE,
        LIVE_QUERY,
        PRE_CAPTURE_WARMUP,
        PRIOR_TIMING_RATIONALE,
        load_v10,
    )

    v03 = load_v5_d_freeze(ROOT)
    v09 = load_v09(ROOT)
    v10 = load_v10(ROOT)
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
        "raw_cuda_graph",
        "pre_capture_warmup",
        "live_query",
        "recovery_v10",
        "current_authorization",
        "advance_only_to",
        "semantic_sha256",
    }
    files = (
        "configs/acr/v5_d_downstream_only_graph_recovery_v10.json",
        "docs/ACR_V5_D_V10_DOWNSTREAM_ONLY_GRAPH_PROTOCOL.md",
        "src/savr/acr/v5_d_v10_runtime.py",
        "src/savr/acr/v5_d_v10_torch_backend.py",
        "src/savr/acr/v5_d_v10_adapter.py",
        "src/savr/acr/v5_d_v10_verification.py",
        "scripts/select_acr_v5_d_v10_gpu.py",
        "scripts/prepare_acr_v5_d_v10_libero_config.py",
        "scripts/run_acr_v5_d_v10.py",
        "scripts/launch_acr_v5_d_v10.sh",
        "scripts/finalize_acr_v5_d_v10.py",
        "scripts/verify_acr_v5_d_v10_preflight.py",
        "tests/acr/test_v5_d_v10_recovery.py",
        "tests/acr/test_v5_d_v10_torch_backend.py",
        "tests/acr/test_v5_d_v10_preflight_verifier.py",
    )
    backend = (ROOT / "src/savr/acr/v5_d_v10_torch_backend.py").read_text(encoding="utf-8")
    adapter = (ROOT / "src/savr/acr/v5_d_v10_adapter.py").read_text(encoding="utf-8")
    verifier = (ROOT / "src/savr/acr/v5_d_v10_verification.py").read_text(encoding="utf-8")
    runner = (ROOT / "scripts/run_acr_v5_d_v10.py").read_text(encoding="utf-8")
    selector = (ROOT / "scripts/select_acr_v5_d_v10_gpu.py").read_text(encoding="utf-8")
    launcher = (ROOT / "scripts/launch_acr_v5_d_v10.sh").read_text(encoding="utf-8")
    checks = {
        "v10_identity": v10["run_id"] == "acr-v5d-real-tensor-feasibility-v10",
        "v09_stop_and_raw_linked": v10["recovery_v10"]["v09_technical_stop_semantic_sha256"]
        == "2113acaad46550b26da8bbfcfe25de4e78312e55e7047974d5b555dd88316209"
        and v10["recovery_v10"]["v09_raw_attempt_semantic_sha256"]
        == "e2d5058732ed93fc9a4c10b327279af14de5e8c62dcda003f3bb48d9ba01214a",
        "only_identity_architecture_authorization_changed": {
            key: value for key, value in v10.items() if key not in changed
        }
        == {key: value for key, value in v09.items() if key not in changed},
        "hybrid_architecture_exact": v10["raw_cuda_graph"] == HYBRID_ARCHITECTURE
        and HYBRID_ARCHITECTURE["graph_object_count"] == 1
        and HYBRID_ARCHITECTURE["wrist_capture_count"] == 0
        and HYBRID_ARCHITECTURE["shared_private_pool"] is False,
        "warmup_and_live_query_exact": v10["pre_capture_warmup"] == PRE_CAPTURE_WARMUP
        and v10["live_query"] == LIVE_QUERY,
        "prior_timing_rationale_exact": v10["recovery_v10"]["prior_timing_rationale"]
        == PRIOR_TIMING_RATIONALE,
        "scientific_runtime_unchanged": v10["allocator"] == v09["allocator"]
        and v10["inference_semantics"] == v09["inference_semantics"]
        and v10["correctness"] == v09["correctness"]
        and v10["timing"] == v09["timing"]
        and v10["analysis"] == v09["analysis"]
        and v10["gates"] == v09["gates"]
        and v10["memory"] == v09["memory"]
        and v10["resource_caps"] == v09["resource_caps"],
        "query_budget_unchanged": len(schedule) == 111
        and sum(item.kind == "correctness" for item in schedule) == 7
        and sum(item.kind == "warmup" for item in schedule) == 8
        and sum(item.kind == "timed" for item in schedule) == 96,
        "backend_is_one_downstream_graph": "V10DownstreamOnlyCudaGraphCorePair" in backend
        and "self.eager.wrist(pixels, output)" in backend
        and "self._downstream_graph.replay()" in backend
        and 'capture_error_mode="global"' in backend
        and ".pool(" not in backend
        and "empty_cache(" not in backend,
        "hybrid_provenance_is_published_and_verified": (
            'augmented["hybrid_architecture"]' in adapter
            and 'augmented["preparation_labels"]' in adapter
            and "V10 published preparation labels changed" in verifier
            and "missing V10 hybrid architecture provenance" in verifier
        ),
        "launch_waterfall_and_default_allocator": launcher.index("--backend torch-compile")
        < launcher.index("--backend raw-cudagraph")
        and "expandable_segments:True" not in launcher
        and "PYTORCH_CUDA_ALLOC_CONF=" not in launcher
        and "Refusing inherited allocator configuration" in launcher,
        "execution_requires_exact_authorization": "V10 GPU execution is not authorized" in runner
        and "V10 GPU execution lacks the exact query authorization" in runner
        and "V10 GPU inspection or selection is not authorized" in selector,
        "single_gpu_attempt_authorized": v10["current_authorization"]
        == {
            "protocol_documentation": True,
            "pre_gpu_implementation": True,
            "gpu_inspection_or_selection": True,
            "model_queries_max": 111,
            "automatic_retry": False,
            "simulator_use": False,
            "protected_outcome_access": False,
            "manuscript_changes": False,
        },
        "immutable_paths_unused": not (
            ROOT / "results/acr-v5d-real-tensor-feasibility-v10"
        ).exists()
        and not (ROOT / "results/acr-v5d-analysis-v10").exists()
        and not (ROOT / "results/acr-v5d-verification-v10").exists(),
        "implementation_files_present": all((ROOT / name).is_file() for name in files),
    }
    record = {
        "schema_version": "acr.v5d-preflight-verification.v10",
        "status": "pass" if all(checks.values()) else "fail",
        "configuration_semantic_sha256": v10["semantic_sha256"],
        "v09_configuration_semantic_sha256": v09["semantic_sha256"],
        "query_labels_sha256": hashlib.sha256(
            canonical_bytes([item.label for item in schedule])
        ).hexdigest(),
        "checks": checks,
        "source_hashes": {name: file_sha256(ROOT / name) for name in files},
        "resources_used": {
            "gpu_count": 0,
            "gpu_inspections": 0,
            "cuda_visible_devices": "",
            "cuda_initialized": False,
            "model_queries": 0,
            "simulator_episodes": 0,
            "downloads": 0,
            "new_task_outcomes": 0,
        },
        "advance_only_to": "ONE_FROZEN_V10_GPU_ATTEMPT",
    }
    record["semantic_sha256"] = hashlib.sha256(canonical_bytes(record)).hexdigest()
    return record


def main() -> int:
    record = verify()
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0 if record["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
