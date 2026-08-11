"""Isolated V5-D v05 transition recovery; V04 remains immutable."""

from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from savr.acr.v5_d_runtime import semantic_sha256
from savr.acr.v5_d_v04_runtime import load_v04


V05_RUN_ID = "acr-v5d-real-tensor-feasibility-v05"
V05_RESOLVED_SCHEMA = "acr.v5d-gpu-feasibility-resolved.v5"
V05_RECOVERY_SCHEMA = "acr.v5d-transition-recovery.v5"
V05_RECOVERY_RELATIVE = Path("configs/acr/v5_d_transition_recovery_v05.json")
V04_CONFIG_SHA256 = "f5bd1e2c1622acbade05f9b67d8e7ee6e53dee935d9d315ecd7614f5a62f22e3"
V04_STOP_SHA256 = "a3515180022df7938b50956851a2ca05b698819da38b387ddc23b54e59769811"


TRANSITION_REVALIDATION = {
    "applies_to_backend": "raw-cudagraph",
    "requires_authorized_compiler_permit": True,
    "before_torch_import_cuda_initialization_or_model_load": True,
    "telemetry_scope": "aggregate-selected-gpu-only",
    "initial_discard_seconds": 2,
    "sample_count": 3,
    "sample_interval_seconds": 5,
    "require_every_sample": True,
    "maximum_memory_used_mib_each_sample": 512,
    "maximum_utilization_percent_each_sample": 5,
    "require_same_physical_index_and_uuid": True,
    "process_identity_inspection": False,
    "allocation_inspection": False,
    "additional_sampling_windows": 0,
    "automatic_retries": 0,
    "gpu_switch_on_failure": False,
}


def validate_v05_recovery(recovery: Mapping[str, Any], v04: Mapping[str, Any]) -> None:
    if recovery.get("schema_version") != V05_RECOVERY_SCHEMA:
        raise ValueError("V5-D v05 recovery schema changed")
    if recovery.get("semantic_sha256") != semantic_sha256(recovery):
        raise ValueError("V5-D v05 recovery semantic hash mismatch")
    if (
        v04.get("semantic_sha256") != V04_CONFIG_SHA256
        or recovery.get("base_v04_configuration_semantic_sha256") != V04_CONFIG_SHA256
    ):
        raise ValueError("V5-D v05 base V04 identity changed")
    if (
        recovery.get("v04_run_id") != "acr-v5d-real-tensor-feasibility-v04"
        or recovery.get("run_id") != V05_RUN_ID
        or recovery.get("v04_technical_stop_semantic_sha256") != V04_STOP_SHA256
    ):
        raise ValueError("V5-D v05 provenance changed")
    if recovery.get("permitted_changes") != [
        "fresh-raw-process-transition-revalidation"
    ]:
        raise ValueError("V5-D v05 recovery scope changed")
    if recovery.get("transition_revalidation") != TRANSITION_REVALIDATION:
        raise ValueError("V5-D v05 transition rule changed")
    if recovery.get("current_authorization") != {
        "research": True,
        "protocol_documentation": True,
        "pre_gpu_implementation": True,
        "cuda_hidden_verification": True,
        "gpu_inspection": False,
        "gpu_selection": False,
        "model_loading": False,
        "model_queries": False,
        "cuda_compile_capture_or_timing": False,
        "simulator_use": False,
        "protected_outcome_access": False,
        "manuscript_changes": False,
    }:
        raise ValueError("V5-D v05 authorization boundary changed")


def resolve_v05(v04: Mapping[str, Any], recovery: Mapping[str, Any]) -> dict[str, Any]:
    validate_v05_recovery(recovery, v04)
    resolved = deepcopy(dict(v04))
    resolved.update(
        {
            "schema_version": V05_RESOLVED_SCHEMA,
            "status": recovery["status"],
            "authorized_at": recovery["authorized_at"],
            "authorized_scope": recovery["authorized_scope"],
            "protocol": recovery["protocol"],
            "run_id": recovery["run_id"],
            "transition_revalidation": deepcopy(recovery["transition_revalidation"]),
            "recovery_v05": {
                "base_v04_configuration_semantic_sha256": recovery[
                    "base_v04_configuration_semantic_sha256"
                ],
                "v04_run_id": recovery["v04_run_id"],
                "v04_technical_stop_semantic_sha256": recovery[
                    "v04_technical_stop_semantic_sha256"
                ],
                "permitted_changes": recovery["permitted_changes"],
            },
            "current_authorization": recovery["current_authorization"],
            "advance_only_to": recovery["advance_only_to"],
            "semantic_sha256": recovery["resolved_configuration_semantic_sha256"],
        }
    )
    validate_v05_resolved(resolved)
    return resolved


def validate_v05_resolved(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != V05_RESOLVED_SCHEMA:
        raise ValueError("V5-D v05 resolved schema changed")
    if config.get("run_id") != V05_RUN_ID:
        raise ValueError("V5-D v05 resolved run identity changed")
    if config.get("semantic_sha256") != semantic_sha256(config):
        raise ValueError("V5-D v05 resolved semantic hash mismatch")
    if config.get("recovery_v05", {}).get("v04_technical_stop_semantic_sha256") != V04_STOP_SHA256:
        raise ValueError("V5-D v05 resolved provenance changed")
    if config.get("transition_revalidation") != TRANSITION_REVALIDATION:
        raise ValueError("V5-D v05 resolved transition changed")
    if config.get("raw_cuda_graph", {}).get("shared_private_pool") is not True:
        raise ValueError("V5-D v05 shared-pool contract changed")
    if config.get("memory", {}).get("peak_reserved_bytes_max") != 23 * 1024**3:
        raise ValueError("V5-D v05 memory cap changed")


def load_v05(project_root: Path) -> dict[str, Any]:
    v04 = load_v04(project_root)
    recovery = json.loads((project_root / V05_RECOVERY_RELATIVE).read_text(encoding="utf-8"))
    return resolve_v05(v04, recovery)
