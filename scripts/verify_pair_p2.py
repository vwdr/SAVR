#!/usr/bin/env python3
"""Run and publish the CUDA-hidden PAIR P2 correctness gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = ""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def semantic_sha256(value: dict[str, Any]) -> str:
    payload = dict(value)
    payload.pop("semantic_sha256", None)
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def tracked_inputs(root: Path) -> tuple[Path, ...]:
    files = []
    for relative in ("src/savr/pair", "tests/pair"):
        files.extend(
            path for path in (root / relative).rglob("*.py") if "__pycache__" not in path.parts
        )
    files.extend(
        (
            root / "schemas/pair/technical_record.schema.json",
            root / "configs/pair/p2_cpu_v1.json",
            root / "scripts/verify_pair_p2.py",
        )
    )
    return tuple(sorted(files))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.project_root.resolve()
    config_path = root / "configs/pair/p2_cpu_v1.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("semantic_sha256") != semantic_sha256(config):
        raise RuntimeError("P2 configuration semantic hash mismatch")
    scope = config["scope"]
    if scope != {
        "synthetic_only": True,
        "gpu_count": 0,
        "cuda_visible_devices": "",
        "model_access": False,
        "checkpoint_access": False,
        "demonstration_access": False,
        "simulator_access": False,
        "outcome_access": False,
        "network_access": False,
        "wall_hours_maximum": 4,
        "artifact_bytes_maximum": 1073741824,
    }:
        raise RuntimeError("P2 resource or protected-population boundary changed")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("P2 requires CUDA to remain hidden")

    started = datetime.now(timezone.utc).isoformat()
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment["PYTHONPATH"] = ".:src"
    command = [sys.executable, "-m", "pytest", "-q", *config["tests"]["focused"]]
    process = subprocess.run(
        command,
        cwd=root,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    summary_line = process.stdout.strip().splitlines()[-1] if process.stdout.strip() else ""
    if process.returncode != 0:
        raise RuntimeError(f"P2 focused regression gate failed: {summary_line}")
    counts = {
        name: int(value) for value, name in re.findall(r"(\d+) (passed|skipped)", summary_line)
    }
    if counts.get("passed", 0) < 50:
        raise RuntimeError("P2 focused test count is unexpectedly small")

    if "torch" in sys.modules:
        import torch

        if torch.cuda.is_initialized():
            raise RuntimeError("CUDA initialized during P2")

    project_revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    inputs = tracked_inputs(root)
    manifest_rows = [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in inputs
    ]
    manifest_payload = {
        "schema_version": "pair-p2-artifact-manifest-v1",
        "run_id": config["run_id"],
        "project_revision": project_revision,
        "files": manifest_rows,
        "total_bytes": sum(row["bytes"] for row in manifest_rows),
    }
    manifest = dict(manifest_payload)
    manifest["semantic_sha256"] = semantic_sha256(manifest_payload)

    sys.path.insert(0, str(root / "src"))
    from savr.pair.records import freeze_technical_record, records_sha256, validate_schema

    technical = freeze_technical_record(
        {
            "schema_version": "pair-technical-v1",
            "run_id": config["run_id"],
            "phase": "P2",
            "check_id": "focused-pair-and-brace-regression",
            "status": "passed",
            "synthetic": True,
            "cuda_visible": False,
            "model_accessed": False,
            "checkpoint_accessed": False,
            "simulator_accessed": False,
            "details": {
                "passed": counts.get("passed", 0),
                "skipped": counts.get("skipped", 0),
                "summary": summary_line,
                "artifact_manifest_sha256": manifest["semantic_sha256"],
            },
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    validate_schema(technical, root / "schemas/pair/technical_record.schema.json")
    completed = datetime.now(timezone.utc).isoformat()
    run_summary = {
        "schema_version": "pair-run-summary-v1",
        "run_id": config["run_id"],
        "phase": "P2",
        "status": "completed",
        "expected_terminal_records": 1,
        "observed_terminal_records": 1,
        "records_sha256": records_sha256([technical]),
        "config_sha256": file_sha256(config_path),
        "project_revision": project_revision,
        "gpu_ids": [],
        "peak_gpu_memory_mib": None,
        "artifact_bytes": manifest["total_bytes"],
        "stop_reason": None,
        "started_at_utc": started,
        "completed_at_utc": completed,
    }
    from savr.pair.records import validate_schema as validate

    validate(run_summary, root / "schemas/pair/run_summary.schema.json")
    output = config["outputs"]
    write_json(root / output["artifact_manifest"], manifest)
    write_json(root / output["technical_record"], technical)
    write_json(root / output["run_summary"], run_summary)
    print(summary_line)
    print(f"PAIR P2 PASS: {manifest['semantic_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
