"""Isolated V5-D v07 allocator recovery; V06 remains immutable."""

from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from savr.acr.v5_d_runtime import semantic_sha256
from savr.acr.v5_d_v06_runtime import load_v06


V07_RUN_ID = "acr-v5d-real-tensor-feasibility-v07"
V07_RESOLVED_SCHEMA = "acr.v5d-gpu-feasibility-resolved.v7"
V07_RECOVERY_SCHEMA = "acr.v5d-expandable-segments-recovery.v7"
V07_RECOVERY_RELATIVE = Path("configs/acr/v5_d_expandable_segments_recovery_v07.json")
V06_CONFIG_SHA256 = "7d0976512b15c6d14486f9e83e5b14513ab7fc919bbf9b55b75c9536b90b92e6"
V06_STOP_SHA256 = "0588f628a118a2f467215c2337bc23452f3b8e98d0b5865c37be0d2892a18edb"
ALLOCATOR = {
    "applies_to_backend": "raw-cudagraph",
    "environment_variable": "PYTORCH_CUDA_ALLOC_CONF",
    "exact_value": "expandable_segments:True",
    "set_before_torch_import": True,
    "backend": "native",
    "experimental": True,
    "compiler_process_unchanged": True,
    "empty_cache_calls": 0,
    "other_allocator_options": 0,
    "automatic_retries": 0,
}
V06_MEASURED_MEMORY = {
    "raw_peak_allocated_bytes": 24259268096,
    "raw_peak_reserved_bytes": 24941428736,
    "reported_reserved_unallocated_mib": 648.56,
    "cap_exceeded_bytes": 245366784,
    "failed_allocation_bytes": 23068672,
    "completed_capture_count": 0,
}


def validate_v07_recovery(recovery: Mapping[str, Any], v06: Mapping[str, Any]) -> None:
    if recovery.get("schema_version") != V07_RECOVERY_SCHEMA:
        raise ValueError("V5-D v07 recovery schema changed")
    if recovery.get("semantic_sha256") != semantic_sha256(recovery):
        raise ValueError("V5-D v07 recovery semantic hash mismatch")
    if (
        v06.get("semantic_sha256") != V06_CONFIG_SHA256
        or recovery.get("base_v06_configuration_semantic_sha256") != V06_CONFIG_SHA256
    ):
        raise ValueError("V5-D v07 base V06 identity changed")
    if (
        recovery.get("v06_run_id") != "acr-v5d-real-tensor-feasibility-v06"
        or recovery.get("run_id") != V07_RUN_ID
        or recovery.get("v06_technical_stop_semantic_sha256") != V06_STOP_SHA256
    ):
        raise ValueError("V5-D v07 provenance changed")
    if recovery.get("permitted_changes") != [
        "raw-process-pytorch-cuda-alloc-conf-expandable-segments-true",
        "allocator-configuration-provenance",
    ]:
        raise ValueError("V5-D v07 recovery scope changed")
    if recovery.get("allocator") != ALLOCATOR:
        raise ValueError("V5-D v07 allocator contract changed")
    if recovery.get("v06_measured_memory") != V06_MEASURED_MEMORY:
        raise ValueError("V5-D v07 memory rationale changed")


def resolve_v07(v06: Mapping[str, Any], recovery: Mapping[str, Any]) -> dict[str, Any]:
    validate_v07_recovery(recovery, v06)
    resolved = deepcopy(dict(v06))
    resolved.update(
        {
            "schema_version": V07_RESOLVED_SCHEMA,
            "status": recovery["status"],
            "authorized_at": recovery["authorized_at"],
            "authorized_scope": recovery["authorized_scope"],
            "protocol": recovery["protocol"],
            "run_id": recovery["run_id"],
            "allocator": deepcopy(recovery["allocator"]),
            "recovery_v07": {
                "base_v06_configuration_semantic_sha256": recovery[
                    "base_v06_configuration_semantic_sha256"
                ],
                "v06_run_id": recovery["v06_run_id"],
                "v06_technical_stop_semantic_sha256": recovery[
                    "v06_technical_stop_semantic_sha256"
                ],
                "permitted_changes": recovery["permitted_changes"],
                "v06_measured_memory": deepcopy(recovery["v06_measured_memory"]),
            },
            "current_authorization": deepcopy(recovery["current_authorization"]),
            "advance_only_to": recovery["advance_only_to"],
            "semantic_sha256": recovery["resolved_configuration_semantic_sha256"],
        }
    )
    validate_v07_resolved(resolved)
    return resolved


def validate_v07_resolved(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != V07_RESOLVED_SCHEMA:
        raise ValueError("V5-D v07 resolved schema changed")
    if config.get("run_id") != V07_RUN_ID:
        raise ValueError("V5-D v07 resolved run identity changed")
    if config.get("semantic_sha256") != semantic_sha256(config):
        raise ValueError("V5-D v07 resolved semantic hash mismatch")
    if config.get("recovery_v07", {}).get("v06_technical_stop_semantic_sha256") != (
        V06_STOP_SHA256
    ):
        raise ValueError("V5-D v07 resolved provenance changed")
    if config.get("allocator") != ALLOCATOR:
        raise ValueError("V5-D v07 resolved allocator changed")
    if config.get("memory", {}).get("peak_reserved_bytes_max") != 23 * 1024**3:
        raise ValueError("V5-D v07 memory cap changed")


def load_v07(project_root: Path) -> dict[str, Any]:
    v06 = load_v06(project_root)
    recovery = json.loads((project_root / V07_RECOVERY_RELATIVE).read_text(encoding="utf-8"))
    return resolve_v07(v06, recovery)
