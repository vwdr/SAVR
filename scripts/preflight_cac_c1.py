#!/usr/bin/env python3
"""Read-only fail-closed preflight for CAC C1 Recovery 01."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ROOT = Path("/home/ved/SAVR")


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def semantic_sha256(value: Mapping[str, Any]) -> str:
    body = dict(value); body.pop("semantic_sha256", None)
    return hashlib.sha256(canonical_bytes(body)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def gpu_telemetry(gpu: int) -> tuple[int, int]:
    output = subprocess.check_output(
        ["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.used,utilization.gpu",
         "--format=csv,noheader,nounits"], text=True,
    ).strip()
    memory, utilization = (int(value.strip()) for value in output.split(","))
    return memory, utilization


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--allow-pending-authorization", action="store_true")
    args = parser.parse_args()
    if ROOT != EXPECTED_ROOT:
        raise SystemExit(f"CAC C1 preflight refuses to run outside {EXPECTED_ROOT}")
    config = json.loads((ROOT / args.config).read_text())
    checks: dict[str, bool] = {}
    checks["semantic_hash"] = config["semantic_sha256"] == semantic_sha256(config)
    checks["schema"] = config["schema_version"] == "cac-c1-tensor-feasibility-v1"
    expected_authorization = "User explicitly approved C1 Recovery 01 on 2026-08-31"
    checks["authorization"] = config["authorization"] == expected_authorization
    if not checks["authorization"] and not args.allow_pending_authorization:
        raise SystemExit("CAC C1 Recovery 01 authorization is absent")
    checks["authenticated_files"] = all(
        (ROOT / relative).is_file() and file_sha256(ROOT / relative) == expected
        for relative, expected in config["authenticated_files"].items()
    )
    recovery = config["recovery"]
    checks["source_stop_preserved"] = (
        (ROOT / config["source_attempt"] / "technical_stop.json").is_file()
        and not (ROOT / recovery["output_root"]).exists()
    )
    model_config = json.loads((ROOT / config["model_config"]).read_text())
    checkpoint = ROOT / model_config["model"]["checkpoint_relative"]
    checks["checkpoint_metadata"] = all(
        file_sha256(checkpoint / name) == expected
        for name, expected in config["checkpoint_metadata_sha256"].items()
    )
    protected = ("config.json", "configuration_prismatic.py", "modeling_prismatic.py")
    checks["checkpoint_no_stale_backups"] = not any(
        path.name.startswith(tuple(f"{name}.back." for name in protected))
        or path.name.startswith(tuple(f"{name}.backup" for name in protected))
        or path.name in {f"{name}.bak" for name in protected}
        for path in checkpoint.iterdir()
    )
    inputs = json.loads((ROOT / config["input_manifest"]).read_text())
    data_root = ROOT / inputs["data_root_relative"]
    selected = (inputs["inputs"][2], inputs["inputs"][3])
    checks["selected_observation_sources"] = all(
        file_sha256(data_root / row["source_path"]) == row["source_sha256"] for row in selected
    )
    schedule = config["schedule"]
    checks["call_accounting"] = (
        sum(int(schedule[key]) for key in (
            "warmup_calls", "sidecar_and_allfresh_controls", "hook_control_calls",
            "recursive_repeat_and_reset_calls", "isolation_calls", "timing_calls",
        )) == 95
        and int(schedule["timing_calls"]) == sum(
            2 * (horizon + 1) * int(schedule["timing_repetitions_per_horizon"])
            for horizon in schedule["timing_horizons"]
        )
    )
    memory, utilization = gpu_telemetry(args.physical_gpu)
    checks["selected_gpu_idle"] = memory <= 1024 and utilization <= 10
    checks["storage"] = shutil.disk_usage(ROOT).free >= 8 * 1024**3
    checks["runtime"] = Path(sys.executable).resolve() == Path(
        "/home/ved/SAVR/envs/vla-cache-compat/bin/python"
    ).resolve()
    scientific_ready = all(value for key, value in checks.items() if key != "authorization")
    result = {
        "schema_version": "cac-c1-preflight-v1", "checks": checks,
        "ready_except_authorization": scientific_ready,
        "fully_ready": scientific_ready and checks["authorization"],
        "selected_physical_gpu": args.physical_gpu,
        "selected_gpu_aggregate_memory_mib": memory,
        "selected_gpu_utilization_percent": utilization,
        "writes_performed": False,
    }
    print(json.dumps(result, sort_keys=True))
    return 0 if scientific_ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
