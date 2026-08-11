"""Install isolated V05 adapters while retaining the V04 shared-pool backend."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def install_v05_adapters() -> None:
    from savr.acr.v5_d_v04_adapter import install_v04_adapters

    install_v04_adapters()
    import savr.acr.v5_d_recovery as recovery
    import savr.acr.v5_d_runtime as runtime
    from savr.acr.v5_d_v05_runtime import V05_RESOLVED_SCHEMA, load_v05, validate_v05_resolved

    previous_validate = runtime.validate_v5_d_freeze

    def validate(config: Mapping[str, Any]) -> None:
        if config.get("schema_version") == V05_RESOLVED_SCHEMA:
            validate_v05_resolved(config)
        else:
            previous_validate(config)

    def load(project_root):
        return load_v05(project_root)

    setattr(runtime, "validate_v5_d_freeze", validate)
    setattr(runtime, "load_v5_d_freeze", load)
    setattr(recovery, "V03_RUN_ID", "acr-v5d-real-tensor-feasibility-v05")
