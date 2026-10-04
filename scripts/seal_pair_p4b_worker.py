#!/usr/bin/env python3
"""Authenticate completed P4B accounting before any scientific unblinding."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ROOT = Path("/home/ved/SAVR")


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def file_sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def line_count(path: Path) -> int:
    with path.open("rb") as stream:
        return sum(1 for _line in stream)


def write_once(path: Path, value: Mapping[str, Any]) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(canonical_bytes(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    if ROOT != EXPECTED_ROOT or os.environ.get("CUDA_VISIBLE_DEVICES", ""):
        raise SystemExit("P4B worker sealing requires the project root and hidden CUDA")
    sys.path.insert(0, str(ROOT / "src"))
    from savr.pair.p4b import expected_counts, semantic_sha256, validate_config

    config = json.loads((ROOT / args.config).read_text())
    validate_config(config, ROOT)
    run_root = args.run_root.resolve()
    if run_root != (ROOT / config["artifacts"]["run_root"]).resolve():
        raise SystemExit("P4B worker seal run root changed")
    seal_path = run_root / "worker_seal.json"
    paths = {
        "interventions": run_root / "interventions.sealed.jsonl",
        "features": run_root / "features.sealed.jsonl",
        "contracts": run_root / "contracts.sealed.jsonl",
    }
    summary_path = run_root / "worker_summary.json"
    if seal_path.exists() or not summary_path.is_file() or any(not path.is_file() for path in paths.values()):
        raise SystemExit("P4B worker sealing inputs are missing or already sealed")
    summary = json.loads(summary_path.read_text())
    counts = expected_counts(int(config["schedule"]["seed"]))
    expected_lines = {
        "interventions": counts["intervention_records"],
        "features": counts["feature_records"],
        "contracts": counts["contract_records"],
    }
    hashes = {key: file_sha256(path) for key, path in paths.items()}
    if (
        summary.get("status") != "completed"
        or summary.get("configuration_semantic_sha256") != config["semantic_sha256"]
        or summary.get("model_calls") != counts["planned_model_calls"]
        or summary.get("intervention_records") != counts["intervention_records"]
        or summary.get("feature_records") != counts["feature_records"]
        or summary.get("contract_records") != counts["contract_records"]
        or summary.get("artifact_sha256") != hashes
        or {key: line_count(path) for key, path in paths.items()} != expected_lines
    ):
        raise SystemExit("P4B worker evidence failed accounting or hash reconciliation")
    seal = {
        "schema_version": "pair-p4b-worker-seal-v1",
        "run_id": config["run_id"],
        "status": "sealed_complete",
        "configuration_semantic_sha256": config["semantic_sha256"],
        "worker_summary_sha256": file_sha256(summary_path),
        "artifact_sha256": hashes,
        "line_counts": expected_lines,
        "model_calls": counts["planned_model_calls"],
        "scientific_records_opened": False,
        "terminal_outcomes_accessed": False,
        "locked_test_values_accessed": False,
        "cuda_visible": False,
    }
    seal["semantic_sha256"] = semantic_sha256(seal)
    write_once(seal_path, seal)
    print(json.dumps({"status": "sealed_complete"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
