#!/usr/bin/env python3
"""GPU micro-qualification for every Recovery 03 comparison representation."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


ROOT = Path("/home/ved/SAVR")
DEFAULT_CONFIG = Path("configs/openvla/semantic_parity_s3_v04_recovery03.json")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    if Path.cwd().resolve() != ROOT:
        raise SystemExit(f"comparator qualification must start in {ROOT}")
    import importlib.util

    worker_path = ROOT / "scripts/run_openvla_semantic_parity_recovery03.py"
    spec = importlib.util.spec_from_file_location("s3_recovery03_for_qualification", worker_path)
    if spec is None or spec.loader is None:
        raise SystemExit("Recovery 03 worker cannot be loaded")
    recovery03 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(recovery03)
    recovery02 = recovery03.load_recovery02()
    recovery01 = recovery02.load_recovery01()
    v1 = recovery01.load_v1()
    config = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    recovery03.validate_recovery03(config, recovery02, recovery01, v1)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    physical = os.environ.get("OPENVLA_PHYSICAL_GPU_ID", "")
    if not physical.isdigit() or visible != physical or "," in visible:
        raise SystemExit("comparator qualification requires one selected GPU")
    initial = v1.selected_gpu_snapshot(int(physical))
    if initial["memory_used_mib"] > 1024 or initial["utilization_percent"] > 5:
        raise SystemExit("selected GPU is not sufficiently idle")
    output_root = ROOT / config["comparator_qualification"]["output_root"]
    if output_root.exists():
        raise SystemExit("immutable comparator qualification output already exists")
    output_root.mkdir(parents=True)
    started = time.monotonic()
    try:
        import numpy as np
        import torch

        if torch.cuda.device_count() != 1:
            raise RuntimeError("comparator qualification can see an invalid GPU count")
        records = []
        for boundary in sorted(recovery03.ALL_BOUNDARIES):
            pair = recovery03.expected_representation_pair(boundary)
            if pair == ("tensor_cuda", "tensor_cuda"):
                reference = torch.tensor([0.0, 1.0], device="cuda:0", dtype=torch.bfloat16)
                candidate = reference.clone()
            elif pair == ("tensor_cuda", "ndarray_cpu"):
                reference = torch.tensor([0.0, 1.0], device="cuda:0", dtype=torch.bfloat16)
                candidate = np.asarray([0.0, 1.0], dtype=np.float32)
            else:
                reference = np.asarray([0.0, 1.0], dtype=np.float32)
                candidate = reference.copy()
            record = recovery03.compare_with_contract(
                v1.compare_values,
                boundary=boundary,
                reference=reference,
                candidate=candidate,
                tolerance=0.0,
                torch_module=torch,
                np_module=np,
            )
            if not record["passed"]:
                raise RuntimeError(f"representative boundary failed: {boundary}")
            records.append(record)

        mismatch = recovery03.compare_with_contract(
            v1.compare_values,
            boundary="unnormalized_actions",
            reference=np.zeros(8, dtype=np.float32),
            candidate=np.zeros((1, 8), dtype=np.float32),
            tolerance=0.0,
            torch_module=torch,
            np_module=np,
        )
        nonfinite = recovery03.compare_with_contract(
            v1.compare_values,
            boundary="unnormalized_actions",
            reference=np.asarray([np.nan], dtype=np.float32),
            candidate=np.asarray([0.0], dtype=np.float32),
            tolerance=0.0,
            torch_module=torch,
            np_module=np,
        )
        if mismatch["max_abs_nonfinite"] != "positive_infinity":
            raise RuntimeError("shape mismatch was not sealed")
        if nonfinite["max_abs_nonfinite"] != "nan":
            raise RuntimeError("non-finite mismatch was not sealed")
        try:
            recovery03.compare_with_contract(
                v1.compare_values,
                boundary="normalized_actions",
                reference=np.zeros((8, 7), dtype=np.float32),
                candidate=np.zeros((8, 7), dtype=np.float32),
                tolerance=0.0,
                torch_module=torch,
                np_module=np,
            )
        except RuntimeError as error:
            if "representation changed" not in str(error):
                raise
        else:
            raise RuntimeError("representation drift was not rejected")
        mock_manifest = {
            "records": records + [mismatch, nonfinite],
            "raw_values_persisted": False,
        }
        json.dumps(mock_manifest, sort_keys=True, separators=(",", ":"), allow_nan=False)
        del reference, candidate
        torch.cuda.empty_cache()
        final = v1.selected_gpu_snapshot(int(physical))
        summary = {
            "schema_version": "openvla-comparator-qualification-result-v1",
            "run_id": "openvla-comparator-qualification-s3q-v01",
            "complete": True,
            "passed": True,
            "boundary_contracts_exercised": len(records),
            "shape_mismatch_sealed": True,
            "nonfinite_mismatch_sealed": True,
            "representation_drift_rejected": True,
            "cuda_tensor_to_numpy_float32_passed": True,
            "model_loads": 0,
            "model_calls": 0,
            "simulator_outcomes_accessed": False,
            "initial_selected_gpu": initial,
            "final_selected_gpu": final,
            "elapsed_seconds": time.monotonic() - started,
            "automatic_retry": False,
            "advance": {
                "next_stage": "S3_RECOVERY03_MODEL_ATTEMPT",
                "authorized": True,
                "stop_before_next_stage": False,
            },
        }
        summary["semantic_sha256"] = v1.semantic_sha256(summary)
        v1.write_once(output_root / "worker_summary.json", summary)
        return 0
    except BaseException as error:
        stop = {
            "schema_version": "openvla-comparator-qualification-stop-v1",
            "run_id": "openvla-comparator-qualification-s3q-v01",
            "complete": False,
            "error_type": type(error).__name__,
            "error": str(error),
            "model_loads": 0,
            "model_calls": 0,
            "simulator_outcomes_accessed": False,
            "automatic_retry": False,
        }
        stop["semantic_sha256"] = v1.semantic_sha256(stop)
        v1.write_once(output_root / "technical_stop.json", stop)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
