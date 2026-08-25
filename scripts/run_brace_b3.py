#!/usr/bin/env python3
"""Run the single bounded BRACE-B3 attempt and preserve immutable evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ROOT = Path("/home/ved/SAVR")
DEFAULT_CONFIG = Path("configs/brace/b3_physical_v1.json")
ALLOCATIONS = {"core_fr": 22, "cache_suite": 302, "vla_adp": 32, "vla_pruner": 32}


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def write_once(path: Path, value: dict[str, Any]) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(canonical_bytes(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def gpu_snapshot(index: int) -> dict[str, int]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            f"--id={index}",
            "--query-gpu=memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    row = next(csv.reader(io.StringIO(output), skipinitialspace=True))
    return {"memory_total_mib": int(row[0]), "memory_used_mib": int(row[1]), "utilization_percent": int(row[2])}


def tree_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def environment(run: Path, gpu: int, pythonpath: str) -> dict[str, str]:
    env = dict(os.environ)
    cache = run / "cache"
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "SAVR_PHYSICAL_GPU_ID": str(gpu),
            "HF_HOME": str(cache / "huggingface"),
            "HF_HUB_CACHE": str(cache / "huggingface/hub"),
            "TORCH_HOME": str(cache / "torch"),
            "TORCHINDUCTOR_CACHE_DIR": str(cache / "torchinductor"),
            "TRITON_CACHE_DIR": str(cache / "triton"),
            "LIBERO_CONFIG_PATH": str(cache / "libero"),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTHONNOUSERSITE": "1",
            "WANDB_MODE": "disabled",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONPATH": pythonpath,
        }
    )
    env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    return env


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    if ROOT != EXPECTED_ROOT or Path.cwd().resolve() != EXPECTED_ROOT:
        raise SystemExit(f"B3 runner is restricted to {EXPECTED_ROOT}")
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from savr.brace.b3 import allowed_project_status, load_config_file

    config = load_config_file(ROOT, args.config)
    run_path = ROOT / "results" / config["run_id"]
    launch = json.loads((run_path / "launch.json").read_text())
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if launch["source_revision"] != revision or launch["configuration_semantic_sha256"] != config["semantic_sha256"]:
        raise SystemExit("B3 launch identity changed")
    pinned_repositories = {
        "openvla_oft_revision": ROOT / "third_party/openvla-oft",
        "libero_revision": ROOT / "third_party/LIBERO",
        "vla_cache_revision": ROOT / "third_party/vla-cache",
        "vla_adp_revision": ROOT / "third_party/vla-adp",
        "vla_pruner_revision": ROOT / "third_party/vla-pruner",
        "specprune_vla_revision": ROOT / "third_party/specprune-vla",
    }
    for key, repository in pinned_repositories.items():
        observed = subprocess.check_output(
            ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
        ).strip()
        dirt = subprocess.check_output(
            ["git", "-C", str(repository), "status", "--porcelain"], text=True
        ).strip()
        if observed != config["provenance"][key] or dirt:
            raise SystemExit(f"Pinned B3 repository changed: {key}")
    if any((run_path / name).exists() for name in ("run_summary.json", "technical_stop.json")):
        raise SystemExit("B3 attempt already has a terminal record")
    raw_status = subprocess.check_output(
        ["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"]
    )
    if not allowed_project_status(raw_status, config["run_id"]):
        raise SystemExit("B3 source tree is not clean apart from preserved tmp/")
    gpu = int(launch["selected_gpu"]["index"])
    snapshot = gpu_snapshot(gpu)
    if snapshot["memory_used_mib"] > 512 or snapshot["utilization_percent"] > 5:
        raise SystemExit("Selected GPU ceased to meet the frozen idle rule; no model was loaded")
    for directory in (
        run_path / "workers",
        run_path / "logs",
        run_path / "cache/huggingface/hub",
        run_path / "cache/torch",
        run_path / "cache/torchinductor",
        run_path / "cache/triton",
        run_path / "cache/libero",
    ):
        directory.mkdir(parents=True, exist_ok=True)
    libero = {
        "assets": str(ROOT / "third_party/LIBERO/libero/libero/assets"),
        "bddl_files": str(ROOT / "third_party/LIBERO/libero/libero/bddl_files"),
        "benchmark_root": str(ROOT / "third_party/LIBERO/libero/libero"),
        "datasets": str(run_path / "cache/libero/datasets"),
        "init_states": str(ROOT / "third_party/LIBERO/libero/libero/init_files"),
    }
    (run_path / "cache/libero/config.yaml").write_text(
        json.dumps(libero, indent=2, sort_keys=True) + "\n"
    )
    specifications = [
        ("core_fr", ROOT / "envs/openvla-oft/bin/python", f"{ROOT / 'src'}"),
        (
            "cache_suite",
            ROOT / "envs/vla-cache-compat/bin/python",
            f"{ROOT / 'third_party/vla-cache/src/openvla-oft'}:{ROOT / 'src'}",
        ),
        ("vla_adp", ROOT / "envs/openvla-oft/bin/python", f"{ROOT / 'third_party/vla-adp'}:{ROOT / 'src'}"),
        (
            "vla_pruner",
            ROOT / "envs/vla-cache-compat/bin/python",
            f"{ROOT / 'third_party/vla-pruner/src/openvla-oft/transformers/src'}:"
            f"{ROOT / 'third_party/vla-pruner/src/openvla-oft'}:{ROOT / 'src'}",
        ),
    ]
    telemetry = []
    completed = []
    used_queries: dict[str, int] = {}
    started = time.monotonic()
    try:
        for method, python, pythonpath in specifications:
            log_path = run_path / "logs" / f"{method}.log"
            output_path = run_path / "workers" / f"{method}.json"
            with log_path.open("xb") as log:
                process = subprocess.Popen(
                    [
                        str(python),
                        "scripts/run_brace_b3_worker.py",
                        "--method",
                        method,
                        "--output",
                        str(output_path),
                        "--config",
                        str(args.config),
                    ],
                    cwd=ROOT,
                    env=environment(run_path, gpu, pythonpath),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
                while process.poll() is None:
                    point = gpu_snapshot(gpu)
                    point.update({"method": method, "elapsed_seconds": time.monotonic() - started})
                    telemetry.append(point)
                    if point["memory_used_mib"] >= 23 * 1024:
                        process.terminate()
                        process.wait(timeout=30)
                        raise RuntimeError("selected GPU aggregate memory reached the 23 GiB stop boundary")
                    if time.monotonic() - started > int(config["resource_caps"]["wall_seconds"]):
                        process.terminate()
                        process.wait(timeout=30)
                        raise RuntimeError("B3 wall-time cap reached")
                    if len(telemetry) % 10 == 0 and tree_bytes(run_path) > int(
                        config["resource_caps"]["artifact_bytes"]
                    ):
                        process.terminate()
                        process.wait(timeout=30)
                        raise RuntimeError("B3 artifact-byte cap reached")
                    time.sleep(1)
                if process.returncode != 0 or not output_path.is_file():
                    raise RuntimeError(f"{method} worker stopped with status {process.returncode}")
                worker = json.loads(output_path.read_text())
                used_queries[method] = int(worker["queries"])
                if used_queries[method] > ALLOCATIONS[method]:
                    raise RuntimeError(f"{method} exceeded its frozen query allocation")
            completed.append(method)
        summary = {
            "schema_version": "brace.b3-run.v1",
            "run_id": config["run_id"],
            "status": "completed",
            "configuration_semantic_sha256": config["semantic_sha256"],
            "source_revision": revision,
            "completed_methods": completed,
            "planned_queries": sum(ALLOCATIONS.values()),
            "queries": sum(used_queries.values()),
            "used_queries_by_method": used_queries,
            "unused_queries_not_reassigned": sum(ALLOCATIONS.values()) - sum(used_queries.values()),
            "peak_aggregate_gpu_memory_used_mib": max(item["memory_used_mib"] for item in telemetry),
            "telemetry_samples": telemetry,
            "artifact_bytes": tree_bytes(run_path),
            "wall_seconds": time.monotonic() - started,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        summary["semantic_sha256"] = hashlib.sha256(canonical_bytes(summary)).hexdigest()
        write_once(run_path / "run_summary.json", summary)
        return 0
    except Exception as error:
        stop = {
            "schema_version": "brace.b3-technical-stop.v1",
            "run_id": config["run_id"],
            "status": "technical_stop",
            "reason": str(error),
            "completed_methods": completed,
            "conservative_queries_charged": sum(ALLOCATIONS[name] for name in completed)
            + (ALLOCATIONS[specifications[len(completed)][0]] if len(completed) < len(specifications) else 0),
            "automatic_retry": False,
            "telemetry_samples": telemetry,
            "wall_seconds": time.monotonic() - started,
            "stopped_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        stop["semantic_sha256"] = hashlib.sha256(canonical_bytes(stop)).hexdigest()
        write_once(run_path / "technical_stop.json", stop)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
