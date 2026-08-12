#!/usr/bin/env python3
"""Analyze V08 twice and publish independent verification."""

from __future__ import annotations

import json
import sys
from pathlib import Path


EXPECTED_ROOT = Path("/home/ved/SAVR")


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    if root != EXPECTED_ROOT:
        raise SystemExit(f"Refusing to finalize V5-D V08 outside {EXPECTED_ROOT}: {root}")
    sys.path.insert(0, str(root / "src"))
    sys.path.insert(0, str(root / "scripts"))
    from analyze_acr_v5_d import analyze, canonical_bytes, semantic_sha256
    from savr.acr.records import ImmutableRecordStore
    from savr.acr.v5_d_v08_runtime import load_v08
    from verify_acr_v5_d import verify

    config = load_v08(root)
    run = json.loads(
        (root / "results" / config["run_id"] / "final" / "record.json").read_text(
            encoding="utf-8"
        )
    )
    first = analyze(config, run)
    second = analyze(config, run)
    if canonical_bytes(first) != canonical_bytes(second):
        raise RuntimeError("V5-D V08 repeated analysis is not byte-identical")
    errors = verify(config, run, first)
    if errors:
        raise RuntimeError(f"V5-D V08 independent verification failed: {errors}")
    store = ImmutableRecordStore(root / "results")
    store.write_once("acr-v5d-analysis-v08", first)
    verification = {
        "schema_version": "acr.v5d-verification.v1",
        "run_id": config["run_id"],
        "configuration_semantic_sha256": config["semantic_sha256"],
        "run_semantic_sha256": run["semantic_sha256"],
        "analysis_semantic_sha256": first["semantic_sha256"],
        "repeated_analysis_byte_identical": True,
        "independent_errors": [],
        "verified": True,
        "passed": first["passed"],
        "disposition": first["disposition"],
    }
    verification["semantic_sha256"] = semantic_sha256(verification)
    store.write_once("acr-v5d-verification-v08", verification)
    print(json.dumps(verification, indent=2, sort_keys=True))
    return 0 if first["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
