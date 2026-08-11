"""Isolated V5-D v06 pre-capture warm-up recovery; V05 remains immutable."""

from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from savr.acr.v5_d_runtime import semantic_sha256
from savr.acr.v5_d_v05_runtime import load_v05


V06_RUN_ID = "acr-v5d-real-tensor-feasibility-v06"
V06_RESOLVED_SCHEMA = "acr.v5d-gpu-feasibility-resolved.v6"
V06_RECOVERY_SCHEMA = "acr.v5d-precapture-warmup-recovery.v6"
V06_RECOVERY_RELATIVE = Path("configs/acr/v5_d_precapture_warmup_recovery_v06.json")
V05_CONFIG_SHA256 = "b34c1d70bbc7163419597148906c22daa82cea3b497405aeeb82afcb4802b2cf"
V05_STOP_SHA256 = "cb6d9120fc2e6ee69aaa83d677598d21741be8eaf5a3456bc21461d30eb3cc3f"


PRE_CAPTURE_WARMUP = {
    "stream": "same-explicit-stream-used-for-both-captures",
    "order": ["wrist", "downstream"],
    "iterations_per_core": 3,
    "all_warmups_before_any_capture": True,
    "inter_capture_warmups": 0,
    "capture_order": ["wrist", "downstream"],
    "shared_private_pool": True,
    "replay_order": ["wrist", "downstream"],
    "concurrent_replay": False,
    "empty_cache_calls": 0,
    "allocator_environment_change": False,
    "record_stage_memory": True,
}


V05_MEASURED_MEMORY = {
    "wrist_after_warmup_allocated_bytes": 17429342720,
    "wrist_after_warmup_reserved_bytes": 17983078400,
    "wrist_after_capture_allocated_bytes": 18077560320,
    "wrist_after_capture_reserved_bytes": 18417188864,
    "wrist_capture_allocated_increment_bytes": 648217600,
    "wrist_capture_reserved_increment_bytes": 434110464,
    "raw_peak_allocated_bytes": 24226396160,
    "raw_peak_reserved_bytes": 24939331584,
    "cap_exceeded_bytes": 243269632,
    "failed_allocation_bytes": 14680064,
}


def validate_v06_recovery(recovery: Mapping[str, Any], v05: Mapping[str, Any]) -> None:
    if recovery.get("schema_version") != V06_RECOVERY_SCHEMA:
        raise ValueError("V5-D v06 recovery schema changed")
    if recovery.get("semantic_sha256") != semantic_sha256(recovery):
        raise ValueError("V5-D v06 recovery semantic hash mismatch")
    if (
        v05.get("semantic_sha256") != V05_CONFIG_SHA256
        or recovery.get("base_v05_configuration_semantic_sha256") != V05_CONFIG_SHA256
    ):
        raise ValueError("V5-D v06 base V05 identity changed")
    if (
        recovery.get("v05_run_id") != "acr-v5d-real-tensor-feasibility-v05"
        or recovery.get("run_id") != V06_RUN_ID
        or recovery.get("v05_technical_stop_semantic_sha256") != V05_STOP_SHA256
    ):
        raise ValueError("V5-D v06 provenance changed")
    if recovery.get("permitted_changes") != [
        "all-core-warmups-before-any-graph-capture",
        "no-inter-capture-eager-warmup",
        "precapture-stage-memory-evidence",
    ]:
        raise ValueError("V5-D v06 recovery scope changed")
    if recovery.get("pre_capture_warmup") != PRE_CAPTURE_WARMUP:
        raise ValueError("V5-D v06 warm-up lifecycle changed")
    if recovery.get("v05_measured_memory") != V05_MEASURED_MEMORY:
        raise ValueError("V5-D v06 measured rationale changed")
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
        raise ValueError("V5-D v06 authorization boundary changed")


def resolve_v06(v05: Mapping[str, Any], recovery: Mapping[str, Any]) -> dict[str, Any]:
    validate_v06_recovery(recovery, v05)
    resolved = deepcopy(dict(v05))
    raw_graph = deepcopy(dict(v05["raw_cuda_graph"]))
    raw_graph.update(
        {
            "warmup_order": ["wrist", "downstream"],
            "all_warmups_before_any_capture": True,
            "inter_capture_warmups": 0,
            "empty_cache_calls": 0,
        }
    )
    resolved.update(
        {
            "schema_version": V06_RESOLVED_SCHEMA,
            "status": recovery["status"],
            "authorized_at": recovery["authorized_at"],
            "authorized_scope": recovery["authorized_scope"],
            "protocol": recovery["protocol"],
            "run_id": recovery["run_id"],
            "raw_cuda_graph": raw_graph,
            "pre_capture_warmup": deepcopy(recovery["pre_capture_warmup"]),
            "recovery_v06": {
                "base_v05_configuration_semantic_sha256": recovery[
                    "base_v05_configuration_semantic_sha256"
                ],
                "v05_run_id": recovery["v05_run_id"],
                "v05_technical_stop_semantic_sha256": recovery[
                    "v05_technical_stop_semantic_sha256"
                ],
                "permitted_changes": recovery["permitted_changes"],
                "v05_measured_memory": deepcopy(recovery["v05_measured_memory"]),
            },
            "current_authorization": recovery["current_authorization"],
            "advance_only_to": recovery["advance_only_to"],
            "semantic_sha256": recovery["resolved_configuration_semantic_sha256"],
        }
    )
    validate_v06_resolved(resolved)
    return resolved


def validate_v06_resolved(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != V06_RESOLVED_SCHEMA:
        raise ValueError("V5-D v06 resolved schema changed")
    if config.get("run_id") != V06_RUN_ID:
        raise ValueError("V5-D v06 resolved run identity changed")
    if config.get("semantic_sha256") != semantic_sha256(config):
        raise ValueError("V5-D v06 resolved semantic hash mismatch")
    if config.get("recovery_v06", {}).get("v05_technical_stop_semantic_sha256") != V05_STOP_SHA256:
        raise ValueError("V5-D v06 resolved provenance changed")
    if config.get("pre_capture_warmup") != PRE_CAPTURE_WARMUP:
        raise ValueError("V5-D v06 resolved warm-up lifecycle changed")
    if config.get("raw_cuda_graph", {}).get("shared_private_pool") is not True:
        raise ValueError("V5-D v06 shared-pool contract changed")
    if config.get("memory", {}).get("peak_reserved_bytes_max") != 23 * 1024**3:
        raise ValueError("V5-D v06 memory cap changed")


def load_v06(project_root: Path) -> dict[str, Any]:
    v05 = load_v05(project_root)
    recovery = json.loads((project_root / V06_RECOVERY_RELATIVE).read_text(encoding="utf-8"))
    return resolve_v06(v05, recovery)
