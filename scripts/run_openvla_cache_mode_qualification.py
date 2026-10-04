#!/usr/bin/env python3
"""Selected-GPU synthetic proof of pinned None/False/True cache semantics."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import time
from pathlib import Path


ROOT = Path("/home/ved/SAVR")
DEFAULT_CONFIG = Path("configs/openvla/semantic_parity_s3_v05_recovery04.json")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    if Path.cwd().resolve() != ROOT:
        raise SystemExit(f"cache-mode qualification must start in {ROOT}")
    worker_path = ROOT / "scripts/run_openvla_semantic_parity_recovery04.py"
    spec = importlib.util.spec_from_file_location("s3_recovery04_for_cache_qualification", worker_path)
    if spec is None or spec.loader is None:
        raise SystemExit("Recovery 04 worker cannot be loaded")
    recovery04 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(recovery04)
    recovery03 = recovery04.load_recovery03()
    recovery02 = recovery03.load_recovery02()
    recovery01 = recovery02.load_recovery01()
    v1 = recovery01.load_v1()
    config = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    recovery04.validate_recovery04(config, recovery03, recovery02, recovery01, v1)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    physical = os.environ.get("OPENVLA_PHYSICAL_GPU_ID", "")
    if not physical.isdigit() or visible != physical or "," in visible:
        raise SystemExit("cache-mode qualification requires one selected GPU")
    initial = v1.selected_gpu_snapshot(int(physical))
    if initial["memory_used_mib"] > 1024 or initial["utilization_percent"] > 5:
        raise SystemExit("selected GPU is not sufficiently idle")
    output_root = ROOT / config["cache_mode_qualification"]["output_root"]
    if output_root.exists():
        raise SystemExit("immutable cache-mode qualification output already exists")
    output_root.mkdir(parents=True)
    started = time.monotonic()
    try:
        import torch
        import transformers
        from transformers import LlamaConfig, LlamaModel

        if torch.cuda.device_count() != 1:
            raise RuntimeError("cache-mode qualification can see an invalid GPU count")
        llama_config = LlamaConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=32,
        )
        llama_config.proportion_attn_var = None
        llama_config.reusable_patches = None
        model = LlamaModel(llama_config).to("cuda:0").eval()
        input_ids = torch.tensor([[1, 2, 3]], device="cuda:0")
        with torch.inference_mode():
            none_output = model(input_ids=input_ids, use_cache=None, return_dict=True)
            false_output = model(input_ids=input_ids, use_cache=False, return_dict=True)
            true_output = model(input_ids=input_ids, use_cache=True, return_dict=True)
        results = {
            "none_returns_cache": none_output.past_key_values is not None,
            "false_returns_cache": false_output.past_key_values is not None,
            "true_returns_cache": true_output.past_key_values is not None,
        }
        if results != {
            "none_returns_cache": True,
            "false_returns_cache": False,
            "true_returns_cache": True,
        }:
            raise RuntimeError(f"pinned cache semantics changed: {results}")
        received = []

        def probe(*values, **kwargs):
            received.append(kwargs["use_cache"])
            return {"cache": kwargs["use_cache"]}

        if recovery04.explicit_no_cache_forward(probe, use_cache=None)["cache"] is not False:
            raise RuntimeError("Recovery 04 did not translate None to False")
        if recovery04.explicit_no_cache_forward(probe, use_cache=True)["cache"] is not True:
            raise RuntimeError("Recovery 04 changed the True cache control")
        if received != [False, True]:
            raise RuntimeError("Recovery 04 cache wrapper call sequence changed")
        del model, input_ids, none_output, false_output, true_output
        torch.cuda.empty_cache()
        final = v1.selected_gpu_snapshot(int(physical))
        summary = {
            "schema_version": "openvla-cache-mode-qualification-result-v1",
            "run_id": "openvla-cache-mode-qualification-s3q-v01",
            "complete": True,
            "passed": True,
            "transformers_version": transformers.__version__,
            "config_use_cache": bool(llama_config.use_cache),
            **results,
            "wrapper_none_translates_to_false": True,
            "wrapper_true_preserved": True,
            "synthetic_llama_calls": 3,
            "openvla_model_loads": 0,
            "openvla_model_calls": 0,
            "simulator_outcomes_accessed": False,
            "initial_selected_gpu": initial,
            "final_selected_gpu": final,
            "elapsed_seconds": time.monotonic() - started,
            "automatic_retry": False,
            "advance": {
                "next_stage": "S3_RECOVERY04_MODEL_ATTEMPT",
                "authorized": True,
                "stop_before_next_stage": False,
            },
        }
        summary["semantic_sha256"] = v1.semantic_sha256(summary)
        v1.write_once(output_root / "worker_summary.json", summary)
        return 0
    except BaseException as error:
        stop = {
            "schema_version": "openvla-cache-mode-qualification-stop-v1",
            "run_id": "openvla-cache-mode-qualification-s3q-v01",
            "complete": False,
            "error_type": type(error).__name__,
            "error": str(error),
            "openvla_model_loads": 0,
            "openvla_model_calls": 0,
            "simulator_outcomes_accessed": False,
            "automatic_retry": False,
        }
        stop["semantic_sha256"] = v1.semantic_sha256(stop)
        v1.write_once(output_root / "technical_stop.json", stop)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
