"""Install isolated V06 adapters while preserving V05 transition handling."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any


def install_v06_adapters() -> None:
    from savr.acr.v5_d_v05_adapter import install_v05_adapters

    install_v05_adapters()
    import savr.acr.records as records
    import savr.acr.v5_d_recovery as recovery
    import savr.acr.v5_d_runtime as runtime
    import savr.acr.v5_d_torch_backend as backend
    from savr.acr.v5_d_v06_runtime import V06_RESOLVED_SCHEMA, load_v06, validate_v06_resolved
    from savr.acr.v5_d_v06_torch_backend import (
        V06PreCaptureWarmupRawCudaGraphCorePair,
        last_v06_raw_graph_evidence,
    )

    previous_validate = runtime.validate_v5_d_freeze
    previous_write_once = records.ImmutableRecordStore.write_once

    def validate(config: Mapping[str, Any]) -> None:
        if config.get("schema_version") == V06_RESOLVED_SCHEMA:
            validate_v06_resolved(config)
        else:
            previous_validate(config)

    def load(project_root):
        return load_v06(project_root)

    def write_once(self, identity: str, record: Mapping[str, Any]):
        evidence = last_v06_raw_graph_evidence()
        if evidence is not None and (
            identity.endswith("/backend-attempt-raw-cudagraph") or identity.endswith("/final")
        ):
            augmented = deepcopy(dict(record))
            augmented["raw_graph_pre_capture_warmup_order"] = evidence["pre_capture_warmup_order"]
            augmented["raw_graph_empty_cache_calls"] = evidence["empty_cache_calls"]
            augmented.pop("semantic_sha256", None)
            augmented["semantic_sha256"] = runtime.semantic_sha256(augmented)
            record = augmented
        return previous_write_once(self, identity, record)

    setattr(runtime, "validate_v5_d_freeze", validate)
    setattr(runtime, "load_v5_d_freeze", load)
    setattr(backend, "RawCudaGraphCorePair", V06PreCaptureWarmupRawCudaGraphCorePair)
    setattr(recovery, "V03_RUN_ID", "acr-v5d-real-tensor-feasibility-v06")
    setattr(records.ImmutableRecordStore, "write_once", write_once)
