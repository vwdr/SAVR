#!/usr/bin/env python3
"""Run the V5-D analyzer twice and publish independent verification."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any


EXPECTED_ROOT = Path("/home/ved/SAVR")


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def semantic_sha256(value: dict[str, Any]) -> str:
    payload = dict(value)
    payload.pop("semantic_sha256", None)
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    if root != EXPECTED_ROOT:
        raise SystemExit(f"Refusing to finalize V5-D outside {EXPECTED_ROOT}: {root}")
    sys.path.insert(0, str(root / "src"))
    sys.path.insert(0, str(root / "scripts"))
    from analyze_acr_v5_d import analyze
    from savr.acr.records import ImmutableRecordStore
    from savr.acr.v5_d_runtime import load_v5_d_freeze
    from verify_acr_v5_d import verify

    config = load_v5_d_freeze(root)
    run = json.loads(
        (root / "results/acr-v5d-real-tensor-feasibility-v03/final/record.json").read_text(
            encoding="utf-8"
        )
    )
    first = analyze(config, run)
    second = analyze(config, run)
    if canonical_bytes(first) != canonical_bytes(second):
        raise RuntimeError("V5-D repeated analysis is not byte-identical")
    errors = verify(config, run, first)
    if errors:
        raise RuntimeError(f"V5-D independent verification failed: {errors}")
    store = ImmutableRecordStore(root / "results")
    store.write_once("acr-v5d-analysis-v03", first)
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
    store.write_once("acr-v5d-verification-v03", verification)
    print(json.dumps(verification, indent=2, sort_keys=True))
    return 0 if first["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
