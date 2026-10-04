#!/usr/bin/env python3
"""CUDA-hidden implementation or full data preflight for PAIR P4B."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import py_compile
import sys
from collections import Counter
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


def write_once(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(canonical_bytes(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--implementation-only", action="store_true")
    args = parser.parse_args()
    if ROOT != EXPECTED_ROOT:
        raise SystemExit(f"P4B preflight refuses to run outside {EXPECTED_ROOT}")
    if os.environ.get("CUDA_VISIBLE_DEVICES", ""):
        raise SystemExit("P4B preflight requires CUDA to be hidden")
    if Path(sys.executable).resolve() != Path(
        "/home/ved/SAVR/envs/vla-cache-compat/bin/python"
    ).resolve():
        raise SystemExit("P4B preflight runtime changed")
    sys.path.insert(0, str(ROOT / "src"))
    from jsonschema import Draft202012Validator
    from savr.pair.p4b import expected_counts, semantic_sha256, validate_config

    config = json.loads((ROOT / args.config).read_text())
    validate_config(config, ROOT)
    report_path = (ROOT / args.report).resolve()
    if not report_path.is_relative_to(ROOT / "reports/pair_p4b") or report_path.exists():
        raise SystemExit("P4B preflight report must be new under reports/pair_p4b")
    required_code = (
        "scripts/freeze_pair_p4b_schedule.py",
        "scripts/run_pair_p4b_worker.py",
        "scripts/seal_pair_p4b_worker.py",
        "scripts/analyze_pair_p4b.py",
        "scripts/preflight_pair_p4b.py",
        "scripts/freeze_pair_p4b_launch.py",
        "src/savr/pair/p4b.py",
        "src/savr/pair/p4_openvla.py",
    )
    for relative in required_code:
        py_compile.compile(str(ROOT / relative), doraise=True)
    schemas = (
        "intervention_record.schema.json",
        "p4_feature_record.schema.json",
        "p4_contract_record.schema.json",
    )
    for name in schemas:
        Draft202012Validator.check_schema(
            json.loads((ROOT / "schemas/pair" / name).read_text())
        )
    schedule_path = ROOT / config["artifacts"]["schedule"]
    summary_path = ROOT / config["artifacts"]["schedule_summary"]
    run_root = ROOT / config["artifacts"]["run_root"]
    schedule_validation: dict[str, Any] = {"mode": "not_opened"}
    if args.implementation_only:
        if schedule_path.exists() or summary_path.exists() or run_root.exists():
            raise SystemExit("P4B implementation-only preflight requires unopened outputs")
    else:
        import h5py

        schedule_bytes = schedule_path.read_bytes()
        schedule = [json.loads(line) for line in schedule_bytes.splitlines()]
        summary = json.loads(summary_path.read_text())
        counts = expected_counts(int(config["schedule"]["seed"]))
        p4_rows = [
            json.loads(line)
            for line in (
                ROOT / config["authenticated_inputs"]["p4_schedule"]["path"]
            ).read_text().splitlines()
        ]
        p4_ids = {row["trajectory_id"] for row in p4_rows}
        cells = Counter(
            (row["suite"], row["horizon"])
            for row in schedule if row["category"] == "structured"
        )
        if (
            summary["status"] != "completed"
            or summary["schedule_sha256"] != hashlib.sha256(schedule_bytes).hexdigest()
            or summary["counts"] != counts
            or len(schedule) != counts["anchors"]
            or len({row["trajectory_id"] for row in schedule}) != counts["anchors"]
            or {row["trajectory_id"] for row in schedule} & p4_ids
            or any(row["split"] == "locked_test" for row in schedule)
            or len(cells) != 12
            or set(cells.values()) != {20}
        ):
            raise SystemExit("P4B full schedule validation failed")
        data_root = ROOT / "data/pair/libero_hdf5/f13aa24a3da8c43c7225569f28c562979fa0e35a"
        source_hashes = {}
        for source in sorted({row["source_path"] for row in schedule}):
            expected = {row["source_sha256"] for row in schedule if row["source_path"] == source}
            if len(expected) != 1 or file_sha256(data_root / source) != next(iter(expected)):
                raise SystemExit("P4B source identity changed")
            source_hashes[source] = next(iter(expected))
        for row in schedule:
            with h5py.File(data_root / row["source_path"], "r") as handle:
                names = [
                    row["action_dataset"], row["primary_camera_dataset"],
                    row["wrist_camera_dataset"], *row["proprio_datasets"],
                ]
                if any(name not in handle for name in names):
                    raise SystemExit("P4B scheduled dataset is missing")
                if handle[row["action_dataset"]].shape != (row["step_count"], 7):
                    raise SystemExit("P4B action shape changed")
                if max(row["future_original_steps"]) + 8 > row["step_count"]:
                    raise SystemExit("P4B expert window is incomplete")
        schedule_validation = {
            "mode": "full",
            "schedule_sha256": summary["schedule_sha256"],
            "sources": len(source_hashes),
            "source_manifest_sha256": hashlib.sha256(canonical_bytes(source_hashes)).hexdigest(),
            "p4_overlap": 0,
            "locked_test_use": 0,
        }
    free_bytes = os.statvfs(ROOT).f_bavail * os.statvfs(ROOT).f_frsize
    if free_bytes < int(config["resource_caps"]["artifact_bytes"]):
        raise SystemExit("P4B lacks its frozen artifact allowance")
    result = {
        "schema_version": "pair-p4b-preflight-v1",
        "run_id": config["run_id"],
        "status": "passed",
        "mode": "implementation_only" if args.implementation_only else "full",
        "configuration_semantic_sha256": config["semantic_sha256"],
        "counts": expected_counts(int(config["schedule"]["seed"])),
        "schedule_validation": schedule_validation,
        "code_sha256": {relative: file_sha256(ROOT / relative) for relative in required_code},
        "schema_sha256": {name: file_sha256(ROOT / "schemas/pair" / name) for name in schemas},
        "schedule_generated": schedule_path.exists(),
        "unused_calibration_values_accessed": False,
        "locked_test_values_accessed": False,
        "model_accessed": False,
        "simulator_accessed": False,
        "cuda_visible": False,
        "free_bytes": free_bytes,
    }
    result["semantic_sha256"] = semantic_sha256(result)
    write_once(report_path, result)
    print(json.dumps({"status": "passed", "mode": result["mode"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
