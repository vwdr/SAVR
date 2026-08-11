#!/usr/bin/env python3
"""Closed-stdin, CUDA-hidden import/API preflight for V5-D v05."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


EXPECTED_ROOT = Path("/home/ved/SAVR")
OPENVLA_REVISION = "e4287e94541f459edc4feabc4e181f537cd569a8"
LIBERO_REVISION = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
PREFLIGHT_ID = "acr-v5d-v05-transition-import-preflight-v02"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_output(path: Path, *arguments: str) -> str:
    return subprocess.check_output(["git", "-C", str(path), *arguments], text=True).strip()


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    if root != EXPECTED_ROOT:
        raise SystemExit(f"Refusing V5-D v05 import preflight outside {EXPECTED_ROOT}: {root}")
    sys.path.insert(0, str(root / "src"))
    from savr.acr.v5_d_recovery import create_libero_config_once, semantic_sha256, write_json_once
    from savr.acr.v5_d_v05_runtime import load_v05

    config = load_v05(root)
    upstream = root / "third_party/openvla-oft"
    libero = root / "third_party/LIBERO"
    if git_output(upstream, "rev-parse", "HEAD") != OPENVLA_REVISION or git_output(
        libero, "rev-parse", "HEAD"
    ) != LIBERO_REVISION:
        raise SystemExit("V5-D v05 import preflight source revision changed")
    if git_output(upstream, "status", "--porcelain") or git_output(
        libero, "status", "--porcelain"
    ):
        raise SystemExit("V5-D v05 import preflight source tree is dirty")

    preflight_root = root / "results" / PREFLIGHT_ID
    if preflight_root.exists():
        raise SystemExit(f"Immutable V5-D v05 import preflight already exists: {preflight_root}")
    attestation = create_libero_config_once(root, preflight_root, config["recovery_v02"])
    environment = os.environ.copy()
    environment.update(
        {
            "LIBERO_CONFIG_PATH": str(Path(attestation["path"]).parent),
            "CUDA_VISIBLE_DEVICES": "",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTHONPATH": str(root / "src"),
            "WANDB_MODE": "disabled",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    code = r'''
import builtins
import inspect
import json

def forbidden_input(*args, **kwargs):
    raise RuntimeError("V5-D v05 interactive input was attempted")

builtins.input = forbidden_input
import torch
before = bool(torch.cuda.is_initialized())
graph_signature = inspect.signature(torch.cuda.graph)
capture_signature = inspect.signature(torch.cuda.CUDAGraph.capture_begin)
from experiments.robot.libero import run_libero_eval
from savr.acr.v5_d_v05_transition import V05TransitionSampler
after = bool(torch.cuda.is_initialized())
print(json.dumps({
    "torch_version": torch.__version__,
    "graph_has_pool_parameter": "pool" in graph_signature.parameters,
    "capture_begin_has_pool_parameter": "pool" in capture_signature.parameters,
    "cudagraph_has_pool_method": hasattr(torch.cuda.CUDAGraph, "pool"),
    "transition_sampler_imported": V05TransitionSampler.__name__ == "V05TransitionSampler",
    "cuda_initialized_before": before,
    "cuda_initialized_after": after,
    "cuda_available": bool(torch.cuda.is_available()),
    "visible_device_count": int(torch.cuda.device_count()),
    "module": run_libero_eval.__name__,
    "interactive_input_attempted": False,
}, sort_keys=True))
'''
    started = time.monotonic()
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=upstream,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=180,
        check=False,
    )
    elapsed = time.monotonic() - started
    if completed.returncode != 0:
        raise RuntimeError("V5-D v05 closed-stdin import failed: " + completed.stderr[-2000:])
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    imported = json.loads(lines[-1]) if lines else None
    expected = {
        "capture_begin_has_pool_parameter": True,
        "cuda_available": False,
        "cuda_initialized_after": False,
        "cuda_initialized_before": False,
        "cudagraph_has_pool_method": True,
        "graph_has_pool_parameter": True,
        "interactive_input_attempted": False,
        "module": "experiments.robot.libero.run_libero_eval",
        "torch_version": "2.2.0+cu118",
        "transition_sampler_imported": True,
        "visible_device_count": 0,
    }
    if imported != expected:
        raise RuntimeError("V5-D v05 CUDA-hidden import/API attestation changed")
    record = {
        "schema_version": "acr.v5d-transition-import-preflight.v5",
        "status": "pass",
        "preflight_id": PREFLIGHT_ID,
        "recorded_at_utc": utc_now(),
        "execution_revision": git_output(root, "rev-parse", "HEAD"),
        "configuration_semantic_sha256": config["semantic_sha256"],
        "openvla_revision": OPENVLA_REVISION,
        "libero_revision": LIBERO_REVISION,
        "libero_config_sha256": attestation["sha256"],
        "stdin": "closed-devnull",
        "subprocess_returncode": completed.returncode,
        "elapsed_seconds": elapsed,
        "import_attestation": imported,
        "stdout_sha256": hashlib.sha256(completed.stdout.encode()).hexdigest(),
        "stderr_sha256": hashlib.sha256(completed.stderr.encode()).hexdigest(),
        "gpu_inspection": 0,
        "cuda_initialized": False,
        "model_loaded": False,
        "model_queries": 0,
        "simulator_instances": 0,
        "simulator_episodes": 0,
        "simulator_resets": 0,
        "downloads": 0,
        "new_task_outcomes": 0,
    }
    record["semantic_sha256"] = semantic_sha256(record)
    write_json_once(root / "reports/runtime/acr_v5_d_v05_import_preflight.json", record)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
