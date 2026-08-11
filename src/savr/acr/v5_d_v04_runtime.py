"""Isolated V5-D v04 overlay; V03 runtime and evidence remain byte-identical."""

from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from savr.acr.v5_d_runtime import load_v5_d_freeze, semantic_sha256


V04_RUN_ID = "acr-v5d-real-tensor-feasibility-v04"
V04_RESOLVED_SCHEMA = "acr.v5d-gpu-feasibility-resolved.v4"
V04_RECOVERY_SCHEMA = "acr.v5d-titan-memory-recovery.v4"
V04_RECOVERY_RELATIVE = Path("configs/acr/v5_d_titan_memory_recovery_v04.json")
V03_CONFIG_SHA256 = "a9447cd385b4229e54cf85ba8fc7e06e4b4d283b9ac5c655e0c5201fb5d3f297"
V03_STOP_SHA256 = "1016569f642b21266e8f0b75b5906716200055f5d37385c5501b6711f9a6bd54"


def validate_v04_recovery(recovery: Mapping[str, Any], v03: Mapping[str, Any]) -> None:
    if recovery.get("schema_version") != V04_RECOVERY_SCHEMA:
        raise ValueError("V5-D v04 recovery schema changed")
    if recovery.get("semantic_sha256") != semantic_sha256(recovery):
        raise ValueError("V5-D v04 recovery semantic hash mismatch")
    if (
        v03.get("semantic_sha256") != V03_CONFIG_SHA256
        or recovery.get("base_resolved_configuration_semantic_sha256") != V03_CONFIG_SHA256
    ):
        raise ValueError("V5-D v04 base identity changed")
    if (
        recovery.get("v03_run_id") != "acr-v5d-real-tensor-feasibility-v03"
        or recovery.get("run_id") != V04_RUN_ID
        or recovery.get("v03_technical_stop_semantic_sha256") != V03_STOP_SHA256
    ):
        raise ValueError("V5-D v04 run provenance changed")
    if recovery.get("permitted_changes") != [
        "raw-graphs-share-one-private-pool",
        "shared-capture-stream",
        "fail-closed-replay-order-and-stream-enforcement",
        "capture-stage-memory-evidence",
    ]:
        raise ValueError("V5-D v04 recovery scope changed")
    if recovery.get("raw_cuda_graph") != {
        "graph_count": 2,
        "capture_order": ["wrist", "downstream"],
        "replay_order": ["wrist", "downstream"],
        "shared_private_pool": True,
        "same_capture_stream": True,
        "same_replay_stream_within_and_across_queries": True,
        "concurrent_replay": False,
        "static_buffers_retained": True,
        "capture_warmups_per_core": 3,
        "capture_calls_per_core": 1,
        "allocator_environment_change": False,
    }:
        raise ValueError("V5-D v04 raw CUDA graph contract changed")
    if recovery.get("memory") != {
        "v03_peak_allocated_bytes": 24184212992,
        "v03_peak_reserved_bytes": 24937234432,
        "peak_reserved_bytes_max": 23 * 1024**3,
        "minimum_reserved_reduction_bytes": 241172480,
        "record_capture_stage_snapshots": True,
        "raise_cap": False,
    }:
        raise ValueError("V5-D v04 memory contract changed")
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
        raise ValueError("V5-D v04 authorization boundary changed")


def resolve_v04(v03: Mapping[str, Any], recovery: Mapping[str, Any]) -> dict[str, Any]:
    validate_v04_recovery(recovery, v03)
    resolved = deepcopy(dict(v03))
    memory = deepcopy(dict(resolved["memory"]))
    memory.update(recovery["memory"])
    memory["shared_graph_pool_with_proven_lifetime_safety"] = True
    resolved.update(
        {
            "schema_version": V04_RESOLVED_SCHEMA,
            "status": recovery["status"],
            "authorized_at": recovery["authorized_at"],
            "authorized_scope": recovery["authorized_scope"],
            "protocol": recovery["protocol"],
            "run_id": recovery["run_id"],
            "raw_cuda_graph": deepcopy(recovery["raw_cuda_graph"]),
            "memory": memory,
            "recovery_v04": {
                "base_resolved_configuration_semantic_sha256": recovery[
                    "base_resolved_configuration_semantic_sha256"
                ],
                "v03_run_id": recovery["v03_run_id"],
                "v03_technical_stop_semantic_sha256": recovery[
                    "v03_technical_stop_semantic_sha256"
                ],
                "permitted_changes": recovery["permitted_changes"],
            },
            "current_authorization": recovery["current_authorization"],
            "advance_only_to": recovery["advance_only_to"],
            "semantic_sha256": recovery["resolved_configuration_semantic_sha256"],
        }
    )
    validate_v04_resolved(resolved)
    return resolved


def validate_v04_resolved(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != V04_RESOLVED_SCHEMA:
        raise ValueError("V5-D v04 resolved schema changed")
    if config.get("run_id") != V04_RUN_ID:
        raise ValueError("V5-D v04 resolved run identity changed")
    if config.get("semantic_sha256") != semantic_sha256(config):
        raise ValueError("V5-D v04 resolved semantic hash mismatch")
    if config.get("recovery_v04", {}).get("v03_technical_stop_semantic_sha256") != V03_STOP_SHA256:
        raise ValueError("V5-D v04 resolved provenance changed")
    if config.get("raw_cuda_graph", {}).get("shared_private_pool") is not True:
        raise ValueError("V5-D v04 resolved pool contract changed")
    if config.get("memory", {}).get("peak_reserved_bytes_max") != 23 * 1024**3:
        raise ValueError("V5-D v04 resolved byte cap changed")


def load_v04(project_root: Path) -> dict[str, Any]:
    v03 = load_v5_d_freeze(project_root)
    recovery = json.loads((project_root / V04_RECOVERY_RELATIVE).read_text(encoding="utf-8"))
    return resolve_v04(v03, recovery)
