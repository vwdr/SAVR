"""Install isolated V07 identity and allocator-provenance adapters."""

from __future__ import annotations

import os
from collections.abc import Mapping
from copy import deepcopy
from typing import Any


def install_v07_adapters() -> None:
    from savr.acr.v5_d_v06_adapter import install_v06_adapters

    install_v06_adapters()
    import savr.acr.records as records
    import savr.acr.v5_d_recovery as recovery
    import savr.acr.v5_d_runtime as runtime
    from savr.acr.v5_d_v07_runtime import V07_RESOLVED_SCHEMA, load_v07, validate_v07_resolved

    previous_validate = runtime.validate_v5_d_freeze
    previous_write_once = records.ImmutableRecordStore.write_once

    def validate(config: Mapping[str, Any]) -> None:
        if config.get("schema_version") == V07_RESOLVED_SCHEMA:
            validate_v07_resolved(config)
        else:
            previous_validate(config)

    def load(project_root):
        return load_v07(project_root)

    def write_once(self, identity: str, record: Mapping[str, Any]):
        if identity.endswith("/backend-attempt-raw-cudagraph") or identity.endswith("/final"):
            augmented = deepcopy(dict(record))
            augmented["cuda_allocator_environment"] = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
            augmented.pop("semantic_sha256", None)
            augmented["semantic_sha256"] = runtime.semantic_sha256(augmented)
            record = augmented
        return previous_write_once(self, identity, record)

    setattr(runtime, "validate_v5_d_freeze", validate)
    setattr(runtime, "load_v5_d_freeze", load)
    setattr(recovery, "V03_RUN_ID", "acr-v5d-real-tensor-feasibility-v07")
    setattr(records.ImmutableRecordStore, "write_once", write_once)
