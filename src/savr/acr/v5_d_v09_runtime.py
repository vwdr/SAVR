"""Isolated V5-D V09 default-allocator recovery; V08 remains immutable."""

from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from savr.acr.v5_d_runtime import semantic_sha256
from savr.acr.v5_d_v08_runtime import INFERENCE_SEMANTICS, load_v08


V09_RUN_ID = "acr-v5d-real-tensor-feasibility-v09"
V09_RESOLVED_SCHEMA = "acr.v5d-gpu-feasibility-resolved.v9"
V09_RECOVERY_SCHEMA = "acr.v5d-default-allocator-recovery.v9"
V09_RECOVERY_RELATIVE = Path("configs/acr/v5_d_default_allocator_recovery_v09.json")
V08_CONFIG_SHA256 = "70db1b6b4b2259d326a6eb45de52c12b5372157f62d739e47e0ec27b5230ce21"
V08_STOP_SHA256 = "3572abf107ad1b0ef10557e27c66b3d5ad1d967a5f82b633c572bef907d16d98"
DEFAULT_ALLOCATOR = {
    "applies_to_backend": "raw-cudagraph",
    "environment_variable": "PYTORCH_CUDA_ALLOC_CONF",
    "exact_value": None,
    "must_be_unset_before_torch_import": True,
    "backend": "native-default",
    "experimental": False,
    "compiler_process_unchanged": True,
    "empty_cache_calls": 0,
    "other_allocator_options": 0,
    "automatic_retries": 0,
}
V08_MEASURED_EVIDENCE = {
    "raw_peak_allocated_bytes": 16062644736,
    "raw_peak_reserved_bytes": 16403922944,
    "peak_reserved_cap_margin_bytes": 8292139008,
    "completed_pre_capture_warmup_order": ["wrist", "downstream"],
    "completed_capture_order": ["wrist"],
    "downstream_capture_failed": True,
    "model_queries": 0,
}


def validate_v09_recovery(recovery: Mapping[str, Any], v08: Mapping[str, Any]) -> None:
    if recovery.get("schema_version") != V09_RECOVERY_SCHEMA:
        raise ValueError("V5-D V09 recovery schema changed")
    if recovery.get("semantic_sha256") != semantic_sha256(recovery):
        raise ValueError("V5-D V09 recovery semantic hash mismatch")
    if (
        v08.get("semantic_sha256") != V08_CONFIG_SHA256
        or recovery.get("base_v08_configuration_semantic_sha256") != V08_CONFIG_SHA256
    ):
        raise ValueError("V5-D V09 base V08 identity changed")
    if (
        recovery.get("v08_run_id") != "acr-v5d-real-tensor-feasibility-v08"
        or recovery.get("run_id") != V09_RUN_ID
        or recovery.get("v08_technical_stop_semantic_sha256") != V08_STOP_SHA256
    ):
        raise ValueError("V5-D V09 provenance changed")
    if recovery.get("permitted_changes") != [
        "raw-process-remove-expandable-segments-override",
        "default-native-allocator-provenance",
    ]:
        raise ValueError("V5-D V09 recovery scope changed")
    if recovery.get("allocator") != DEFAULT_ALLOCATOR:
        raise ValueError("V5-D V09 allocator contract changed")
    if recovery.get("v08_measured_evidence") != V08_MEASURED_EVIDENCE:
        raise ValueError("V5-D V09 measured rationale changed")


def resolve_v09(v08: Mapping[str, Any], recovery: Mapping[str, Any]) -> dict[str, Any]:
    validate_v09_recovery(recovery, v08)
    resolved = deepcopy(dict(v08))
    resolved.update(
        {
            "schema_version": V09_RESOLVED_SCHEMA,
            "status": recovery["status"],
            "authorized_at": recovery["authorized_at"],
            "authorized_scope": recovery["authorized_scope"],
            "protocol": recovery["protocol"],
            "run_id": recovery["run_id"],
            "allocator": deepcopy(recovery["allocator"]),
            "recovery_v09": {
                "base_v08_configuration_semantic_sha256": recovery[
                    "base_v08_configuration_semantic_sha256"
                ],
                "v08_run_id": recovery["v08_run_id"],
                "v08_technical_stop_semantic_sha256": recovery[
                    "v08_technical_stop_semantic_sha256"
                ],
                "permitted_changes": recovery["permitted_changes"],
                "v08_measured_evidence": deepcopy(recovery["v08_measured_evidence"]),
            },
            "current_authorization": deepcopy(recovery["current_authorization"]),
            "advance_only_to": recovery["advance_only_to"],
            "semantic_sha256": recovery["resolved_configuration_semantic_sha256"],
        }
    )
    validate_v09_resolved(resolved)
    return resolved


def validate_v09_resolved(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != V09_RESOLVED_SCHEMA:
        raise ValueError("V5-D V09 resolved schema changed")
    if config.get("run_id") != V09_RUN_ID:
        raise ValueError("V5-D V09 resolved run identity changed")
    if config.get("semantic_sha256") != semantic_sha256(config):
        raise ValueError("V5-D V09 resolved semantic hash mismatch")
    if config.get("recovery_v09", {}).get("v08_technical_stop_semantic_sha256") != (
        V08_STOP_SHA256
    ):
        raise ValueError("V5-D V09 resolved provenance changed")
    if config.get("allocator") != DEFAULT_ALLOCATOR:
        raise ValueError("V5-D V09 resolved allocator changed")
    if config.get("inference_semantics") != INFERENCE_SEMANTICS:
        raise ValueError("V5-D V09 inference semantics changed")
    if config.get("memory", {}).get("peak_reserved_bytes_max") != 23 * 1024**3:
        raise ValueError("V5-D V09 memory cap changed")


def load_v09(project_root: Path) -> dict[str, Any]:
    v08 = load_v08(project_root)
    recovery = json.loads((project_root / V09_RECOVERY_RELATIVE).read_text(encoding="utf-8"))
    return resolve_v09(v08, recovery)
