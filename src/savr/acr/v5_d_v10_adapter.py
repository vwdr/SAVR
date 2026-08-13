"""Install isolated V10 hybrid-backend identity and provenance adapters."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any


def install_v10_adapters() -> None:
    from savr.acr.v5_d_v09_adapter import install_v09_adapters

    install_v09_adapters()
    import savr.acr.records as records
    import savr.acr.v5_d_recovery as recovery
    import savr.acr.v5_d_runtime as runtime
    import savr.acr.v5_d_torch_backend as backend
    from savr.acr.v5_d_v10_runtime import (
        V10_RESOLVED_SCHEMA,
        load_v10,
        validate_v10_resolved,
    )
    from savr.acr.v5_d_v10_torch_backend import (
        V10DownstreamOnlyCudaGraphCorePair,
        last_v10_hybrid_evidence,
    )

    previous_validate = runtime.validate_v5_d_freeze
    previous_write_once = records.ImmutableRecordStore.write_once

    def validate(config: Mapping[str, Any]) -> None:
        if config.get("schema_version") == V10_RESOLVED_SCHEMA:
            validate_v10_resolved(config)
        else:
            previous_validate(config)

    def load(project_root):
        return load_v10(project_root)

    def write_once(self, identity: str, record: Mapping[str, Any]):
        evidence = last_v10_hybrid_evidence()
        if evidence is not None and (
            identity.endswith("/backend-attempt-raw-cudagraph") or identity.endswith("/final")
        ):
            augmented = deepcopy(dict(record))
            augmented["hybrid_architecture"] = deepcopy(evidence)
            augmented["preparation_labels"] = deepcopy(evidence["preparation_labels"])
            augmented.pop("semantic_sha256", None)
            augmented["semantic_sha256"] = runtime.semantic_sha256(augmented)
            record = augmented
        return previous_write_once(self, identity, record)

    setattr(runtime, "validate_v5_d_freeze", validate)
    setattr(runtime, "load_v5_d_freeze", load)
    setattr(backend, "RawCudaGraphCorePair", V10DownstreamOnlyCudaGraphCorePair)
    setattr(recovery, "V03_RUN_ID", "acr-v5d-real-tensor-feasibility-v10")
    setattr(records.ImmutableRecordStore, "write_once", write_once)
