"""Install isolated V08 identity and inference-provenance adapters."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any


_ACTIVE_LIFECYCLE: Any = None


def set_active_lifecycle(value: Any) -> None:
    global _ACTIVE_LIFECYCLE
    _ACTIVE_LIFECYCLE = value


def install_v08_adapters() -> None:
    from savr.acr.v5_d_v07_adapter import install_v07_adapters

    install_v07_adapters()
    import savr.acr.records as records
    import savr.acr.v5_d_recovery as recovery
    import savr.acr.v5_d_runtime as runtime
    from savr.acr.v5_d_v08_runtime import V08_RESOLVED_SCHEMA, load_v08, validate_v08_resolved

    previous_validate = runtime.validate_v5_d_freeze
    previous_write_once = records.ImmutableRecordStore.write_once

    def validate(config: Mapping[str, Any]) -> None:
        if config.get("schema_version") == V08_RESOLVED_SCHEMA:
            validate_v08_resolved(config)
        else:
            previous_validate(config)

    def load(project_root):
        return load_v08(project_root)

    def write_once(self, identity: str, record: Mapping[str, Any]):
        if identity.endswith("/backend-attempt-raw-cudagraph") or identity.endswith("/final"):
            if _ACTIVE_LIFECYCLE is None:
                raise RuntimeError("V08 raw evidence lacks an active inference lifecycle")
            augmented = deepcopy(dict(record))
            augmented["inference_semantics"] = _ACTIVE_LIFECYCLE.active_attestation()
            if augmented["inference_semantics"] != {
                "entered": True,
                "grad_enabled": False,
                "inference_mode_enabled": True,
            }:
                raise RuntimeError("V08 raw record was not written under exact inference semantics")
            augmented.pop("semantic_sha256", None)
            augmented["semantic_sha256"] = runtime.semantic_sha256(augmented)
            record = augmented
        return previous_write_once(self, identity, record)

    setattr(runtime, "validate_v5_d_freeze", validate)
    setattr(runtime, "load_v5_d_freeze", load)
    setattr(recovery, "V03_RUN_ID", "acr-v5d-real-tensor-feasibility-v08")
    setattr(records.ImmutableRecordStore, "write_once", write_once)
