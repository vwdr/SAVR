#!/usr/bin/env python3
"""Invoke the frozen V5-D runner through the isolated V10 adapters."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


def requested_backend() -> str:
    try:
        return sys.argv[sys.argv.index("--backend") + 1]
    except (ValueError, IndexError):
        return "unknown"


def main() -> int:
    backend = requested_backend()
    if os.environ.get("PYTORCH_CUDA_ALLOC_CONF") is not None:
        raise SystemExit(f"V10 {backend} process must use the default allocator environment")

    from savr.acr.v5_d_v08_adapter import set_active_lifecycle
    from savr.acr.v5_d_v10_adapter import install_v10_adapters
    from savr.acr.v5_d_v10_runtime import V10_RUN_ID, load_v10

    install_v10_adapters()
    config = load_v10(ROOT)
    authorization = config["current_authorization"]
    if authorization.get("gpu_inspection_or_selection") is not True:
        raise SystemExit("V10 GPU execution is not authorized")
    if authorization.get("model_queries_max") != 111:
        raise SystemExit("V10 GPU execution lacks the exact query authorization")

    import run_acr_v5_d as runner
    from savr.acr.v5_d_recovery import semantic_sha256, write_json_once
    from savr.acr.v5_d_v05_transition import V05TransitionSampler
    from savr.acr.v5_d_v08_inference import V08InferenceLifecycle

    runner.V03_RUN_ID = V10_RUN_ID
    if backend == "raw-cudagraph":
        import torch

        run_root = ROOT / "results" / V10_RUN_ID
        launch = json.loads((run_root / "launch" / "record.json").read_text(encoding="utf-8"))
        attempt = json.loads(
            (run_root / "backend-attempt-torch-compile" / "record.json").read_text(encoding="utf-8")
        )
        permit = json.loads(
            (run_root / "raw-transition-permit" / "record.json").read_text(encoding="utf-8")
        )
        if any(
            record.get("semantic_sha256") != semantic_sha256(record)
            for record in (launch, attempt, permit)
        ):
            raise SystemExit("V10 transition provenance semantic hash mismatch")
        if (
            permit.get("permitted") is not True
            or permit.get("requires_fresh_process") is not True
            or attempt.get("correctness_records") != 0
            or attempt.get("timing_records") != 0
            or attempt.get("raw_transition_permitted") is not True
        ):
            raise SystemExit("V10 transition lacks an authorized zero-output compiler permit")
        original_snapshot = runner.selected_gpu_snapshot

        def write_transition(record):
            write_json_once(run_root / "transition-revalidation" / "record.json", record)

        transition = V05TransitionSampler(
            run_id=V10_RUN_ID,
            rule=config["transition_revalidation"],
            expected_index=int(launch["selected_gpu"]["index"]),
            expected_uuid=str(launch["selected_gpu"]["uuid"]),
            snapshot=original_snapshot,
            sleep=time.sleep,
            write_once=write_transition,
        )
        lifecycle = V08InferenceLifecycle(torch_module=torch, transition=transition)
        set_active_lifecycle(lifecycle)
        runner.selected_gpu_snapshot = lifecycle.snapshot_and_enter
        original_attempt = runner._run_attempt

        def guarded_attempt() -> int:
            try:
                return original_attempt()
            finally:
                lifecycle.close()
                record = {
                    "schema_version": "acr.v5d-inference-lifecycle.v10",
                    "run_id": V10_RUN_ID,
                    "backend": "raw-cudagraph",
                    **lifecycle.lifecycle_record(),
                }
                record["semantic_sha256"] = semantic_sha256(record)
                write_json_once(run_root / "inference-lifecycle" / "record.json", record)
                set_active_lifecycle(None)

        runner._run_attempt = guarded_attempt
    return runner.main()


if __name__ == "__main__":
    raise SystemExit(main())
