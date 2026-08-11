#!/usr/bin/env python3
"""Invoke the frozen V5-D runner through the isolated V06 adapters."""

from __future__ import annotations

import json
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
    from savr.acr.v5_d_v06_adapter import install_v06_adapters

    install_v06_adapters()
    import run_acr_v5_d as runner
    from savr.acr.v5_d_recovery import semantic_sha256, write_json_once
    from savr.acr.v5_d_v05_transition import V05TransitionSampler
    from savr.acr.v5_d_v06_runtime import V06_RUN_ID, load_v06

    runner.V03_RUN_ID = V06_RUN_ID
    if requested_backend() == "raw-cudagraph":
        config = load_v06(ROOT)
        run_root = ROOT / "results" / V06_RUN_ID
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
            raise SystemExit("V06 transition provenance semantic hash mismatch")
        if (
            permit.get("permitted") is not True
            or permit.get("requires_fresh_process") is not True
            or attempt.get("correctness_records") != 0
            or attempt.get("timing_records") != 0
            or attempt.get("raw_transition_permitted") is not True
        ):
            raise SystemExit("V06 transition lacks an authorized zero-output compiler permit")
        original_snapshot = runner.selected_gpu_snapshot

        def write_transition(record):
            write_json_once(run_root / "transition-revalidation" / "record.json", record)

        runner.selected_gpu_snapshot = V05TransitionSampler(
            run_id=V06_RUN_ID,
            rule=config["transition_revalidation"],
            expected_index=int(launch["selected_gpu"]["index"]),
            expected_uuid=str(launch["selected_gpu"]["uuid"]),
            snapshot=original_snapshot,
            sleep=time.sleep,
            write_once=write_transition,
        )
    return runner.main()


if __name__ == "__main__":
    raise SystemExit(main())
