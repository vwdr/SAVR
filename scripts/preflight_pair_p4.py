#!/usr/bin/env python3
"""CPU-only fail-closed preflight for the frozen PAIR P4 worker."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ROOT = Path("/home/ved/SAVR")


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
    parser.add_argument("--recovery-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if ROOT != EXPECTED_ROOT:
        raise SystemExit(f"P4 preflight refuses to run outside {EXPECTED_ROOT}")
    if os.environ.get("CUDA_VISIBLE_DEVICES", ""):
        raise SystemExit("P4 preflight requires CUDA to be hidden")
    if Path(sys.executable).resolve() != Path(
        "/home/ved/SAVR/envs/vla-cache-compat/bin/python"
    ).resolve():
        raise SystemExit("P4 preflight runtime changed")
    sys.path.insert(0, str(ROOT / "src"))
    import h5py
    from jsonschema import Draft202012Validator

    from savr.pair.p4 import expected_counts, semantic_sha256, validate_config

    config = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    recovery = json.loads((ROOT / args.recovery_config).read_text(encoding="utf-8"))
    validate_config(config, ROOT)
    if recovery.get("semantic_sha256") != semantic_sha256(recovery):
        raise SystemExit("P4 schedule recovery semantic hash mismatch")
    if recovery["base_config_semantic_sha256"] != config["semantic_sha256"]:
        raise SystemExit("P4 base/recovery semantic identity changed")
    schedule_path = ROOT / recovery["schedule_artifacts"]["schedule"]
    summary_path = ROOT / recovery["schedule_artifacts"]["summary"]
    schedule_bytes = schedule_path.read_bytes()
    schedule = [json.loads(line) for line in schedule_bytes.splitlines()]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    counts = expected_counts()
    if (
        summary["status"] != "completed"
        or summary["schedule_sha256"] != hashlib.sha256(schedule_bytes).hexdigest()
        or summary["counts"] != counts
        or len(schedule) != 400
        or len({row["trajectory_id"] for row in schedule}) != 400
    ):
        raise SystemExit("P4 recovered schedule accounting changed")
    cells = Counter(
        (row["split"], row["category"], row["observed_gripper_transition"])
        for row in schedule
    )
    for split in ("train", "calibration"):
        for category in ("all_fresh_control", "atomic", "structured"):
            if not cells[(split, category, False)] or not cells[(split, category, True)]:
                raise SystemExit("P4 schedule lost transition support in a protected cell")
    if args.output_root.resolve().exists():
        raise SystemExit("P4 immutable output root already exists")
    for schema_name in (
        "intervention_record.schema.json",
        "p4_feature_record.schema.json",
        "p4_contract_record.schema.json",
        "router_artifact.schema.json",
    ):
        schema = json.loads((ROOT / "schemas/pair" / schema_name).read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(schema)
    required_code = (
        "scripts/run_pair_p4_worker.py",
        "scripts/train_pair_p4_router.py",
        "scripts/analyze_pair_p4.py",
        "src/savr/pair/p4.py",
        "src/savr/pair/p4_openvla.py",
    )
    code_hashes = {path: file_sha256(ROOT / path) for path in required_code}
    data_root = ROOT / "data/pair/libero_hdf5/f13aa24a3da8c43c7225569f28c562979fa0e35a"
    source_hashes = {}
    for source_path in sorted({row["source_path"] for row in schedule}):
        expected = {row["source_sha256"] for row in schedule if row["source_path"] == source_path}
        if len(expected) != 1:
            raise SystemExit("P4 source identity is inconsistent")
        path = data_root / source_path
        observed = file_sha256(path)
        if observed != next(iter(expected)):
            raise SystemExit("P4 HDF5 source hash mismatch")
        source_hashes[source_path] = observed
    for row in schedule:
        with h5py.File(data_root / row["source_path"], "r") as handle:
            datasets = [
                row["action_dataset"],
                row["primary_camera_dataset"],
                row["wrist_camera_dataset"],
                *row["proprio_datasets"],
            ]
            if any(path not in handle for path in datasets):
                raise SystemExit("P4 scheduled HDF5 dataset is missing")
            if handle[row["action_dataset"]].shape != (row["step_count"], 7):
                raise SystemExit("P4 action dataset shape changed")
            if max(row["future_original_steps"]) + 8 > row["step_count"]:
                raise SystemExit("P4 scheduled expert chunk is incomplete")
    free_bytes = os.statvfs(ROOT).f_bavail * os.statvfs(ROOT).f_frsize
    if free_bytes < int(config["resource_caps"]["artifact_bytes"]):
        raise SystemExit("P4 lacks its frozen artifact allowance")
    report = {
        "schema_version": "pair-p4-preflight-v1",
        "run_id": recovery["run_id"],
        "status": "passed",
        "configuration_semantic_sha256": recovery["semantic_sha256"],
        "schedule_sha256": summary["schedule_sha256"],
        "counts": counts,
        "transition_cell_counts": {
            f"{split}/{category}/{str(transition).lower()}": count
            for (split, category, transition), count in sorted(cells.items())
        },
        "source_file_count": len(source_hashes),
        "source_manifest_sha256": hashlib.sha256(canonical_bytes(source_hashes)).hexdigest(),
        "code_sha256": code_hashes,
        "schema_validation": "passed",
        "output_root_absent": True,
        "expert_values_accessed": False,
        "calibration_regret_accessed": False,
        "locked_test_labels_accessed": False,
        "simulator_accessed": False,
        "cuda_visible": False,
        "model_accessed": False,
        "free_bytes": free_bytes,
    }
    report["semantic_sha256"] = semantic_sha256(report)
    write_once(ROOT / args.report, report)
    print(json.dumps({"status": "passed", "source_file_count": len(source_hashes)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
