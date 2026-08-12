"""Install isolated V09 identity and default-allocator provenance adapters."""

from __future__ import annotations

import os
from collections.abc import Mapping
from copy import deepcopy
from typing import Any


def install_v09_adapters() -> None:
    from savr.acr.v5_d_v08_adapter import install_v08_adapters

    install_v08_adapters()
    import savr.acr.records as records
    import savr.acr.v5_d_recovery as recovery
    import savr.acr.v5_d_runtime as runtime
    from savr.acr.v5_d_v09_runtime import V09_RESOLVED_SCHEMA, load_v09, validate_v09_resolved

    previous_validate = runtime.validate_v5_d_freeze
    previous_write_once = records.ImmutableRecordStore.write_once

    def validate(config: Mapping[str, Any]) -> None:
        if config.get("schema_version") == V09_RESOLVED_SCHEMA:
            validate_v09_resolved(config)
        else:
            previous_validate(config)

    def load(project_root):
        return load_v09(project_root)

    def write_once(self, identity: str, record: Mapping[str, Any]):
        if identity.endswith("/backend-attempt-raw-cudagraph") or identity.endswith("/final"):
            if os.environ.get("PYTORCH_CUDA_ALLOC_CONF") is not None:
                raise RuntimeError("V09 raw evidence was produced with an allocator override")
            from torch.cuda.memory import get_allocator_backend

            observed_backend = str(get_allocator_backend())
            if observed_backend != "native":
                raise RuntimeError(f"V09 raw allocator backend changed: {observed_backend}")
            augmented = deepcopy(dict(record))
            augmented["allocator_reversion"] = {
                "environment_variable": "PYTORCH_CUDA_ALLOC_CONF",
                "observed_value": None,
                "default_native_allocator": True,
                "observed_backend": observed_backend,
            }
            augmented.pop("semantic_sha256", None)
            augmented["semantic_sha256"] = runtime.semantic_sha256(augmented)
            record = augmented
        return previous_write_once(self, identity, record)

    setattr(runtime, "validate_v5_d_freeze", validate)
    setattr(runtime, "load_v5_d_freeze", load)
    setattr(recovery, "V03_RUN_ID", "acr-v5d-real-tensor-feasibility-v09")
    setattr(records.ImmutableRecordStore, "write_once", write_once)
