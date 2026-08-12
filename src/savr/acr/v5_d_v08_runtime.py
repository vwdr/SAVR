"""Isolated V5-D V08 inference-semantics recovery; V07 remains immutable."""

from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from savr.acr.v5_d_runtime import semantic_sha256
from savr.acr.v5_d_v07_runtime import load_v07


V08_RUN_ID = "acr-v5d-real-tensor-feasibility-v08"
V08_RESOLVED_SCHEMA = "acr.v5d-gpu-feasibility-resolved.v8"
V08_RECOVERY_SCHEMA = "acr.v5d-inference-semantics-recovery.v8"
V08_RECOVERY_RELATIVE = Path("configs/acr/v5_d_inference_semantics_recovery_v08.json")
V07_CONFIG_SHA256 = "fa9a5785eb2cb885bd98f05355db61251b840b4c6e5ce19760cff088880b88d5"
V07_STOP_SHA256 = "17c6c68ed075f6848768d81eb158ae1d522b2b670df37d0c1db3ab54439bc8c1"
INFERENCE_SEMANTICS = {
    "applies_to_backend": "raw-cudagraph",
    "mode": "torch.inference_mode",
    "entry_after_transition_revalidation": True,
    "entry_before_model_initialization": True,
    "covers_input-preparation": True,
    "covers_static-buffer-population": True,
    "covers_eager-warmup": True,
    "covers_graph-capture": True,
    "covers-correctness-and-timing": True,
    "require_grad_enabled_false": True,
    "require_inference_mode_enabled_true": True,
    "restore_prior_thread_state_on_exit": True,
    "compiler_process_unchanged": True,
    "direct-decoder-pruning": False,
    "kv-cache-change": False,
    "hidden-state-output-change": False,
    "automatic_retries": 0,
}
V07_MEASURED_MEMORY = {
    "raw_peak_allocated_bytes": 24712435200,
    "raw_peak_reserved_bytes": 24937234432,
    "peak_reserved_cap_exceeded_bytes": 241172480,
    "reported_reserved_unallocated_mib": 203.64,
    "failed_allocation_bytes": 23068672,
    "completed_capture_count": 0,
}


def validate_v08_recovery(recovery: Mapping[str, Any], v07: Mapping[str, Any]) -> None:
    if recovery.get("schema_version") != V08_RECOVERY_SCHEMA:
        raise ValueError("V5-D V08 recovery schema changed")
    if recovery.get("semantic_sha256") != semantic_sha256(recovery):
        raise ValueError("V5-D V08 recovery semantic hash mismatch")
    if (
        v07.get("semantic_sha256") != V07_CONFIG_SHA256
        or recovery.get("base_v07_configuration_semantic_sha256") != V07_CONFIG_SHA256
    ):
        raise ValueError("V5-D V08 base V07 identity changed")
    if (
        recovery.get("v07_run_id") != "acr-v5d-real-tensor-feasibility-v07"
        or recovery.get("run_id") != V08_RUN_ID
        or recovery.get("v07_technical_stop_semantic_sha256") != V07_STOP_SHA256
    ):
        raise ValueError("V5-D V08 provenance changed")
    if recovery.get("permitted_changes") != [
        "raw-process-whole-attempt-torch-inference-mode",
        "inference-lifecycle-provenance",
    ]:
        raise ValueError("V5-D V08 recovery scope changed")
    if recovery.get("inference_semantics") != INFERENCE_SEMANTICS:
        raise ValueError("V5-D V08 inference contract changed")
    if recovery.get("v07_measured_memory") != V07_MEASURED_MEMORY:
        raise ValueError("V5-D V08 memory rationale changed")


def resolve_v08(v07: Mapping[str, Any], recovery: Mapping[str, Any]) -> dict[str, Any]:
    validate_v08_recovery(recovery, v07)
    resolved = deepcopy(dict(v07))
    resolved.update(
        {
            "schema_version": V08_RESOLVED_SCHEMA,
            "status": recovery["status"],
            "authorized_at": recovery["authorized_at"],
            "authorized_scope": recovery["authorized_scope"],
            "protocol": recovery["protocol"],
            "run_id": recovery["run_id"],
            "inference_semantics": deepcopy(recovery["inference_semantics"]),
            "recovery_v08": {
                "base_v07_configuration_semantic_sha256": recovery[
                    "base_v07_configuration_semantic_sha256"
                ],
                "v07_run_id": recovery["v07_run_id"],
                "v07_technical_stop_semantic_sha256": recovery[
                    "v07_technical_stop_semantic_sha256"
                ],
                "permitted_changes": recovery["permitted_changes"],
                "v07_measured_memory": deepcopy(recovery["v07_measured_memory"]),
            },
            "current_authorization": deepcopy(recovery["current_authorization"]),
            "advance_only_to": recovery["advance_only_to"],
            "semantic_sha256": recovery["resolved_configuration_semantic_sha256"],
        }
    )
    validate_v08_resolved(resolved)
    return resolved


def validate_v08_resolved(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != V08_RESOLVED_SCHEMA:
        raise ValueError("V5-D V08 resolved schema changed")
    if config.get("run_id") != V08_RUN_ID:
        raise ValueError("V5-D V08 resolved run identity changed")
    if config.get("semantic_sha256") != semantic_sha256(config):
        raise ValueError("V5-D V08 resolved semantic hash mismatch")
    if config.get("recovery_v08", {}).get("v07_technical_stop_semantic_sha256") != V07_STOP_SHA256:
        raise ValueError("V5-D V08 resolved provenance changed")
    if config.get("inference_semantics") != INFERENCE_SEMANTICS:
        raise ValueError("V5-D V08 resolved inference contract changed")
    if config.get("allocator", {}).get("exact_value") != "expandable_segments:True":
        raise ValueError("V5-D V08 did not preserve V07 allocator")
    if config.get("memory", {}).get("peak_reserved_bytes_max") != 23 * 1024**3:
        raise ValueError("V5-D V08 memory cap changed")


def load_v08(project_root: Path) -> dict[str, Any]:
    v07 = load_v07(project_root)
    recovery = json.loads((project_root / V08_RECOVERY_RELATIVE).read_text(encoding="utf-8"))
    return resolve_v08(v07, recovery)
