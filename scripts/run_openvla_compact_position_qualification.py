#!/usr/bin/env python3
"""Qualify the pinned VLA-Cache compact-position output contract."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/home/ved/SAVR")
DEFAULT_OUTPUT = Path("results/openvla-compact-position-qualification-s4q-v01")


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def semantic_sha256(value: dict[str, Any]) -> str:
    body = dict(value)
    body.pop("semantic_sha256", None)
    return hashlib.sha256(canonical_bytes(body)).hexdigest()


def write_once(path: Path, value: dict[str, Any]) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False))
        stream.write("\n")


def aggregate_gpu(physical: int) -> dict[str, int]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            str(physical),
            "--query-gpu=memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    fields = [int(value.strip()) for value in result.stdout.strip().split(",")]
    if len(fields) != 2:
        raise RuntimeError("aggregate GPU telemetry shape changed")
    return {"memory_used_mib": fields[0], "utilization_percent": fields[1]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config-sha256", required=True)
    args = parser.parse_args()
    if Path.cwd().resolve() != ROOT:
        raise SystemExit(f"qualification must start in {ROOT}")
    if len(args.config_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in args.config_sha256
    ):
        raise SystemExit("qualification config identity is not a SHA-256 value")
    output = ROOT / args.output
    if output.exists():
        raise SystemExit("immutable compact-position qualification output exists")
    output.mkdir(parents=True, exist_ok=False)
    physical = int(os.environ["OPENVLA_PHYSICAL_GPU_ID"])
    started = time.monotonic()
    calls = 0
    before = aggregate_gpu(physical)
    torch = None
    try:
        import torch as torch_module
        from transformers import LlamaConfig, LlamaForCausalLM

        from savr.openvla.official_semantics import (
            OpenVLASemanticError,
            cache_fork_active_positions,
            derive_official_layout,
            select_official_action_hidden,
        )

        torch = torch_module
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("qualification requires exactly one visible GPU")
        torch.manual_seed(20260901)
        torch.cuda.manual_seed_all(20260901)
        config = LlamaConfig(
            vocab_size=128,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=12,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=1024,
            use_cache=True,
            attention_dropout=0.0,
        )
        config.proportion_attn_var = None
        config.reusable_patches = None
        model = LlamaForCausalLM(config).to("cuda:0").eval()
        action_mask = torch.zeros((1, 79), dtype=torch.bool, device="cuda:0")
        action_mask[:, 22:78] = True
        layout = derive_official_layout(
            action_mask=action_mask,
            projected_tokens=513,
            instruction_token_indices=(4, 5, 6),
        )
        embeddings = torch.randn(
            (1, layout.full_sequence_tokens, 16),
            dtype=torch.float32,
            device="cuda:0",
        )
        attention_mask = torch.ones(
            (1, layout.full_sequence_tokens), dtype=torch.long, device="cuda:0"
        )
        with torch.inference_mode():
            anchor = model(
                inputs_embeds=embeddings,
                attention_mask=attention_mask,
                use_cache=True,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            calls += 1
            reusable = torch.arange(1, 129, dtype=torch.long, device="cuda:0")
            schedule = torch.zeros(12, dtype=torch.float32, device="cuda:0")
            schedule[2], schedule[6], schedule[9], schedule[11] = 0.25, 0.5, 0.75, 1.0
            model.config.reusable_patches = reusable
            model.config.proportion_attn_var = schedule
            compact = model(
                inputs_embeds=embeddings,
                attention_mask=attention_mask,
                past_key_values=anchor.past_key_values,
                use_cache=True,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            calls += 1
        active = cache_fork_active_positions(compact)
        hidden = compact.hidden_states[-1]
        selected = select_official_action_hidden(hidden, layout, active)
        observed = tuple(int(value) for value in active.detach().cpu().tolist())
        expected = tuple(
            position
            for position in range(layout.full_sequence_tokens)
            if position not in set(range(1, 129))
        )
        if observed != expected or int(hidden.shape[1]) != len(expected):
            raise RuntimeError("pinned fork compact-position map differs from exact expectation")
        if tuple(selected.shape) != (1, 56, 16):
            raise RuntimeError("position-mapped action selection shape changed")
        sentinel = active.to(dtype=torch.float32).reshape(1, -1, 1)
        sentinel_selected = select_official_action_hidden(sentinel, layout, active)
        expected_action = tuple(float(value) for value in layout.action_readout_positions)
        if tuple(sentinel_selected.flatten().detach().cpu().tolist()) != expected_action:
            raise RuntimeError("sentinel position mapping is not exact")
        expected_errors = {
            "missing": "removed an official action state",
            "duplicate": "sorted unique",
            "length": "does not match hidden states",
        }
        for mutation, expected_error in expected_errors.items():
            try:
                if mutation == "missing":
                    altered = active[active != layout.action_readout_positions[0]]
                    select_official_action_hidden(
                        torch.zeros((1, len(altered), 1), device="cuda:0"), layout, altered
                    )
                elif mutation == "duplicate":
                    altered = active.clone()
                    altered[1] = altered[0]
                    select_official_action_hidden(sentinel, layout, altered)
                else:
                    select_official_action_hidden(sentinel, layout, active[:-1])
            except OpenVLASemanticError as error:
                if expected_error in str(error):
                    continue
                raise RuntimeError(
                    f"malformed {mutation} map raised the wrong contract error"
                ) from error
            raise RuntimeError(f"malformed {mutation} position map was accepted")
        active_hash = hashlib.sha256(active.detach().cpu().numpy().tobytes()).hexdigest()
        del selected, sentinel_selected, sentinel, compact, anchor, embeddings, model
        gc.collect()
        torch.cuda.empty_cache()
        after = aggregate_gpu(physical)
        summary = {
            "schema_version": "openvla-compact-position-qualification-s4q-v1",
            "run_id": "openvla-compact-position-qualification-s4q-v01",
            "complete": True,
            "passed": True,
            "config_sha256": args.config_sha256,
            "model_calls": calls,
            "openvla_checkpoint_loads": 0,
            "simulator_calls": 0,
            "outcomes_accessed": False,
            "expert_actions_accessed": False,
            "raw_policy_actions_persisted": False,
            "full_sequence_length": layout.full_sequence_tokens,
            "active_sequence_length": len(observed),
            "removed_visual_positions": layout.full_sequence_tokens - len(observed),
            "action_positions_survived": True,
            "position_map_sorted_unique": True,
            "sentinel_mapping_exact": True,
            "malformed_contracts_rejected": len(expected_errors),
            "active_position_sha256": active_hash,
            "gpu_before": before,
            "gpu_after_cleanup": after,
            "elapsed_seconds": time.monotonic() - started,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        summary["semantic_sha256"] = semantic_sha256(summary)
        write_once(output / "worker_summary.json", summary)
        return 0
    except BaseException as error:
        if torch is not None:
            try:
                gc.collect()
                torch.cuda.empty_cache()
            except BaseException:
                pass
        stop = {
            "schema_version": "openvla-compact-position-qualification-s4q-stop-v1",
            "run_id": "openvla-compact-position-qualification-s4q-v01",
            "complete": False,
            "error_type": type(error).__name__,
            "error": str(error),
            "model_calls": calls,
            "openvla_checkpoint_loads": 0,
            "simulator_calls": 0,
            "outcomes_accessed": False,
            "automatic_retry": False,
            "elapsed_seconds": time.monotonic() - started,
            "stopped_at": datetime.now(timezone.utc).isoformat(),
        }
        stop["semantic_sha256"] = semantic_sha256(stop)
        write_once(output / "technical_stop.json", stop)
        descriptor = os.open(
            output / "technical_traceback.log", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
