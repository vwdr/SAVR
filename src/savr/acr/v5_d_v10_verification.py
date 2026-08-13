"""Independent architecture checks for complete V10 raw/final records."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def verify_v10_architecture(record: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    evidence = record.get("hybrid_architecture")
    if not isinstance(evidence, Mapping):
        return ["missing V10 hybrid architecture provenance"]
    expected_scalars = {
        "wrist_backend": "eager-static-buffer",
        "downstream_backend": "raw-cudagraph",
        "graph_objects_created": 1,
        "graph_objects_retained": 1,
        "wrist_capture_count": 0,
        "shared_pool_api_calls": 0,
        "supplied_pool_token": False,
        "empty_cache_calls": 0,
    }
    for key, expected in expected_scalars.items():
        if evidence.get(key) != expected:
            errors.append(f"V10 architecture field {key} changed")
    if evidence.get("pre_capture_warmup_order") != ["wrist", "downstream"]:
        errors.append("V10 pre-capture warm-up order changed")
    if evidence.get("pre_capture_warmup_calls") != {"wrist": 3, "downstream": 3}:
        errors.append("V10 pre-capture warm-up counts changed")
    if evidence.get("capture_attempt_order") != ["downstream"]:
        errors.append("V10 capture attempt order changed")
    if evidence.get("capture_order") != ["downstream"]:
        errors.append("V10 completed capture order changed")
    expected_preparation_labels = [
        "raw-wrist-warmup-0",
        "raw-wrist-warmup-1",
        "raw-wrist-warmup-2",
        "raw-downstream-warmup-0",
        "raw-downstream-warmup-1",
        "raw-downstream-warmup-2",
        "raw-downstream-capture-0",
    ]
    if evidence.get("preparation_labels") != expected_preparation_labels:
        errors.append("V10 preparation labels changed")
    if record.get("preparation_labels") != expected_preparation_labels:
        errors.append("V10 published preparation labels changed")
    stages = [row.get("stage") for row in evidence.get("memory_trace", [])]
    if stages != [
        "wrist-after-pre-capture-warmup",
        "downstream-after-pre-capture-warmup",
        "downstream-after-capture",
    ]:
        errors.append("V10 memory-stage evidence changed")
    return errors
