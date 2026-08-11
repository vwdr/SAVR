"""Install isolated V04 adapters before invoking the immutable V03 tooling."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any


def install_v04_adapters() -> None:
    import savr.acr.records as records
    import savr.acr.v5_d_recovery as recovery
    import savr.acr.v5_d_runtime as runtime
    import savr.acr.v5_d_torch_backend as backend
    from savr.acr.v5_d_v04_runtime import V04_RESOLVED_SCHEMA, load_v04, validate_v04_resolved
    from savr.acr.v5_d_v04_torch_backend import (
        V04SharedPoolRawCudaGraphCorePair,
        last_raw_graph_evidence,
    )

    original_validate = runtime.validate_v5_d_freeze
    original_write_once = records.ImmutableRecordStore.write_once

    def validate(config: Mapping[str, Any]) -> None:
        if config.get("schema_version") == V04_RESOLVED_SCHEMA:
            validate_v04_resolved(config)
        else:
            original_validate(config)

    def load(project_root):
        return load_v04(project_root)

    def write_once(self, identity: str, record: Mapping[str, Any]):
        evidence = last_raw_graph_evidence()
        if evidence is not None and (
            identity.endswith("/backend-attempt-raw-cudagraph") or identity.endswith("/final")
        ):
            augmented = deepcopy(dict(record))
            augmented["raw_graph_shared_pool"] = True
            augmented["raw_graph_capture_order"] = evidence["capture_order"]
            augmented["raw_graph_memory_trace"] = evidence["memory_trace"]
            augmented.pop("semantic_sha256", None)
            augmented["semantic_sha256"] = runtime.semantic_sha256(augmented)
            record = augmented
        return original_write_once(self, identity, record)

    setattr(runtime, "validate_v5_d_freeze", validate)
    setattr(runtime, "load_v5_d_freeze", load)
    setattr(backend, "RawCudaGraphCorePair", V04SharedPoolRawCudaGraphCorePair)
    setattr(recovery, "V03_RUN_ID", "acr-v5d-real-tensor-feasibility-v04")
    setattr(records.ImmutableRecordStore, "write_once", write_once)
