"""Isolated V5-D V10 downstream-only graph recovery; V09 is immutable."""

from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from savr.acr.v5_d_runtime import semantic_sha256
from savr.acr.v5_d_v08_runtime import INFERENCE_SEMANTICS
from savr.acr.v5_d_v09_runtime import DEFAULT_ALLOCATOR, load_v09


V10_RUN_ID = "acr-v5d-real-tensor-feasibility-v10"
V10_RESOLVED_SCHEMA = "acr.v5d-gpu-feasibility-resolved.v10"
V10_RECOVERY_SCHEMA = "acr.v5d-downstream-only-graph-recovery.v10"
V10_RECOVERY_RELATIVE = Path("configs/acr/v5_d_downstream_only_graph_recovery_v10.json")
V09_CONFIG_SHA256 = "e10abef2b2b78472ec7257ed08cd9f9723d20df8335b7ffc1702867464493408"
V09_STOP_SHA256 = "2113acaad46550b26da8bbfcfe25de4e78312e55e7047974d5b555dd88316209"
V09_RAW_SHA256 = "e2d5058732ed93fc9a4c10b327279af14de5e8c62dcda003f3bb48d9ba01214a"
BASE_SCIENCE_SHA256 = "f445cf5d1a5ec6877ebea46ccc3883a11a676b38cb33a711ee4b74baf22f53f8"

HYBRID_ARCHITECTURE = {
    "wrist_backend": "eager-static-buffer",
    "combined_token_materialization": "eager-scene-first-owned-buffer",
    "downstream_backend": "raw-cudagraph",
    "graph_object_count": 1,
    "capture_count": 1,
    "capture_label": "downstream",
    "wrist_capture_count": 0,
    "shared_private_pool": False,
    "supplied_pool_token": False,
    "capture_stream": "one-non-default-side-stream",
    "capture_error_mode": "global",
    "concurrent_capture_or_replay": False,
    "automatic_retries": 0,
    "empty_cache_calls": 0,
}
PRE_CAPTURE_WARMUP = {
    "order": ["wrist", "downstream"],
    "wrist_eager_calls": 3,
    "downstream_eager_calls": 3,
    "all_warmups_before_capture": True,
    "capture_order": ["downstream"],
}
LIVE_QUERY = {
    "order": [
        "copy-owned-inputs",
        "eager-wrist",
        "scene-first-cat",
        "downstream-graph-replay",
        "host-transfer-and-unnormalize",
    ],
    "scene_core_calls": 0,
    "wrist_core_calls": 1,
    "downstream_core_calls": 1,
    "stable_owned_pointers": True,
    "stable_replay_stream": True,
}
PRIOR_TIMING_RATIONALE = {
    "source": "reports/runtime/acr_v3_c.json",
    "timed_reuse_queries": 12,
    "median_total_cuda_ms": 1151.7415771484375,
    "median_wrist_visual_cuda_ms": 75.66808032989502,
    "median_derived_downstream_cuda_ms": 1076.0382170677185,
    "derived_downstream_fraction": 0.9342705329192416,
}
PERMITTED_CHANGES = [
    "replace-wrist-cudagraph-with-eager-static-buffer-core",
    "retain-one-downstream-only-cudagraph",
    "record-hybrid-architecture-provenance",
]


def validate_v10_recovery(recovery: Mapping[str, Any], v09: Mapping[str, Any]) -> None:
    if recovery.get("schema_version") != V10_RECOVERY_SCHEMA:
        raise ValueError("V5-D V10 recovery schema changed")
    if recovery.get("semantic_sha256") != semantic_sha256(recovery):
        raise ValueError("V5-D V10 recovery semantic hash mismatch")
    if (
        v09.get("semantic_sha256") != V09_CONFIG_SHA256
        or recovery.get("base_v09_resolved_configuration_semantic_sha256") != V09_CONFIG_SHA256
    ):
        raise ValueError("V5-D V10 base V09 identity changed")
    if (
        recovery.get("v09_run_id") != "acr-v5d-real-tensor-feasibility-v09"
        or recovery.get("run_id") != V10_RUN_ID
        or recovery.get("v09_technical_stop_semantic_sha256") != V09_STOP_SHA256
        or recovery.get("v09_raw_attempt_semantic_sha256") != V09_RAW_SHA256
        or recovery.get("base_scientific_freeze_semantic_sha256") != BASE_SCIENCE_SHA256
    ):
        raise ValueError("V5-D V10 provenance changed")
    if recovery.get("permitted_changes") != PERMITTED_CHANGES:
        raise ValueError("V5-D V10 change scope changed")
    if recovery.get("architecture") != HYBRID_ARCHITECTURE:
        raise ValueError("V5-D V10 architecture changed")
    if recovery.get("pre_capture_warmup") != PRE_CAPTURE_WARMUP:
        raise ValueError("V5-D V10 warm-up lifecycle changed")
    if recovery.get("live_query") != LIVE_QUERY:
        raise ValueError("V5-D V10 live-query contract changed")
    if recovery.get("prior_timing_rationale") != PRIOR_TIMING_RATIONALE:
        raise ValueError("V5-D V10 timing rationale changed")
    authorization = recovery.get("current_authorization", {})
    if authorization != {
        "protocol_documentation": True,
        "pre_gpu_implementation": True,
        "gpu_inspection_or_selection": False,
        "model_queries": 0,
        "simulator_use": False,
        "protected_outcome_access": False,
        "manuscript_changes": False,
    }:
        raise ValueError("V5-D V10 authorization boundary changed")


def resolve_v10(v09: Mapping[str, Any], recovery: Mapping[str, Any]) -> dict[str, Any]:
    validate_v10_recovery(recovery, v09)
    resolved = deepcopy(dict(v09))
    resolved.update(
        {
            "schema_version": V10_RESOLVED_SCHEMA,
            "status": recovery["status"],
            "authorized_at": recovery["authorized_at"],
            "authorized_scope": recovery["authorized_scope"],
            "protocol": recovery["protocol"],
            "run_id": recovery["run_id"],
            "raw_cuda_graph": deepcopy(recovery["architecture"]),
            "pre_capture_warmup": deepcopy(recovery["pre_capture_warmup"]),
            "live_query": deepcopy(recovery["live_query"]),
            "recovery_v10": {
                "base_v09_resolved_configuration_semantic_sha256": recovery[
                    "base_v09_resolved_configuration_semantic_sha256"
                ],
                "v09_run_id": recovery["v09_run_id"],
                "v09_technical_stop_semantic_sha256": recovery[
                    "v09_technical_stop_semantic_sha256"
                ],
                "v09_raw_attempt_semantic_sha256": recovery["v09_raw_attempt_semantic_sha256"],
                "base_scientific_freeze_semantic_sha256": recovery[
                    "base_scientific_freeze_semantic_sha256"
                ],
                "permitted_changes": deepcopy(recovery["permitted_changes"]),
                "prior_timing_rationale": deepcopy(recovery["prior_timing_rationale"]),
            },
            "current_authorization": deepcopy(recovery["current_authorization"]),
            "advance_only_to": recovery["advance_only_to"],
            "semantic_sha256": recovery["resolved_configuration_semantic_sha256"],
        }
    )
    validate_v10_resolved(resolved)
    return resolved


def validate_v10_resolved(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != V10_RESOLVED_SCHEMA:
        raise ValueError("V5-D V10 resolved schema changed")
    if config.get("run_id") != V10_RUN_ID:
        raise ValueError("V5-D V10 resolved run identity changed")
    if config.get("semantic_sha256") != semantic_sha256(config):
        raise ValueError("V5-D V10 resolved semantic hash mismatch")
    recovery = config.get("recovery_v10", {})
    if (
        recovery.get("v09_technical_stop_semantic_sha256") != V09_STOP_SHA256
        or recovery.get("v09_raw_attempt_semantic_sha256") != V09_RAW_SHA256
    ):
        raise ValueError("V5-D V10 resolved provenance changed")
    if config.get("raw_cuda_graph") != HYBRID_ARCHITECTURE:
        raise ValueError("V5-D V10 resolved architecture changed")
    if config.get("pre_capture_warmup") != PRE_CAPTURE_WARMUP:
        raise ValueError("V5-D V10 resolved warm-up changed")
    if config.get("live_query") != LIVE_QUERY:
        raise ValueError("V5-D V10 resolved live-query contract changed")
    if config.get("allocator") != DEFAULT_ALLOCATOR:
        raise ValueError("V5-D V10 allocator changed")
    if config.get("inference_semantics") != INFERENCE_SEMANTICS:
        raise ValueError("V5-D V10 inference semantics changed")
    if config.get("memory", {}).get("peak_reserved_bytes_max") != 23 * 1024**3:
        raise ValueError("V5-D V10 memory cap changed")
    if config.get("resource_caps", {}).get("full_model_query_hard_cap") != 111:
        raise ValueError("V5-D V10 query cap changed")


def load_v10(project_root: Path) -> dict[str, Any]:
    v09 = load_v09(project_root)
    recovery = json.loads((project_root / V10_RECOVERY_RELATIVE).read_text(encoding="utf-8"))
    return resolve_v10(v09, recovery)
