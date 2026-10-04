#!/usr/bin/env python3
"""Selected-GPU full dense-helper cache-mode qualification for Recovery 05."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace


ROOT = Path("/home/ved/SAVR")
DEFAULT_CONFIG = Path("configs/openvla/semantic_parity_s3_v06_recovery05.json")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    if Path.cwd().resolve() != ROOT:
        raise SystemExit(f"dense-helper qualification must start in {ROOT}")
    worker_path = ROOT / "scripts/run_openvla_semantic_parity_recovery05.py"
    spec = importlib.util.spec_from_file_location("s3_recovery05_for_helper_qualification", worker_path)
    if spec is None or spec.loader is None:
        raise SystemExit("Recovery 05 worker cannot be loaded")
    recovery05 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(recovery05)
    recovery04 = recovery05.load_recovery04()
    recovery03 = recovery04.load_recovery03()
    recovery02 = recovery03.load_recovery02()
    recovery01 = recovery02.load_recovery01()
    v1 = recovery01.load_v1()
    config = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    recovery05.validate_recovery05(
        config, recovery04, recovery03, recovery02, recovery01, v1
    )
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    physical = os.environ.get("OPENVLA_PHYSICAL_GPU_ID", "")
    if not physical.isdigit() or visible != physical or "," in visible:
        raise SystemExit("dense-helper qualification requires one selected GPU")
    initial = v1.selected_gpu_snapshot(int(physical))
    if initial["memory_used_mib"] > 1024 or initial["utilization_percent"] > 5:
        raise SystemExit("selected GPU is not sufficiently idle")
    output_root = ROOT / config["full_helper_qualification"]["output_root"]
    if output_root.exists():
        raise SystemExit("immutable dense-helper qualification output already exists")
    output_root.mkdir(parents=True)
    started = time.monotonic()
    try:
        import numpy as np
        import torch
        from savr.openvla import official_semantics

        if torch.cuda.device_count() != 1:
            raise RuntimeError("dense-helper qualification can see an invalid GPU count")
        prompt_tokens = 22
        input_tokens = prompt_tokens + 56 + 1
        action_mask = torch.zeros((1, input_tokens), dtype=torch.bool, device="cuda:0")
        action_mask[:, prompt_tokens : prompt_tokens + 56] = True
        prepared = SimpleNamespace(
            action_mask=action_mask,
            projected_patches=torch.arange(
                513, dtype=torch.float32, device="cuda:0"
            ).reshape(1, 513, 1),
            input_embeddings=torch.arange(
                input_tokens, dtype=torch.float32, device="cuda:0"
            ).reshape(1, input_tokens, 1),
            attention_mask=torch.ones(
                (1, input_tokens), dtype=torch.long, device="cuda:0"
            ),
            instruction_token_indices=(4, 5, 6),
        )
        observed_modes = []

        class Language:
            def __call__(self, **kwargs):
                observed_modes.append(kwargs["use_cache"])
                return SimpleNamespace(
                    hidden_states=(kwargs["inputs_embeds"],),
                    past_key_values=("synthetic-cache",) if kwargs["use_cache"] else None,
                )

        class Model:
            def __init__(self):
                self.language_model = Language()

            def _build_multimodal_attention(self, inputs, patches, mask):
                return (
                    torch.cat([inputs[:, :1], patches, inputs[:, 1:]], dim=1),
                    torch.cat(
                        [
                            mask[:, :1],
                            torch.ones((1, 513), dtype=mask.dtype, device=mask.device),
                            mask[:, 1:],
                        ],
                        dim=1,
                    ),
                )

            def _unnormalize_actions(self, actions, key):
                if key != "suite":
                    raise RuntimeError("synthetic normalization key changed")
                return actions

        class Head:
            def predict_action(self, hidden):
                return hidden[..., 0]

        class Tap:
            def __init__(self):
                self.entered = False
                self.exited = False

            def __enter__(self):
                self.entered = True
                return self

            def __exit__(self, exc_type, exc, traceback):
                self.exited = True

        model = Model()
        common = {
            "torch_module": torch,
            "np_module": np,
            "model": model,
            "action_head": Head(),
            "cfg": SimpleNamespace(unnorm_key="suite"),
            "prepared": prepared,
        }
        direct_none = official_semantics.structurally_aligned_dense_forward(
            **common, use_cache=None
        )
        control = recovery04.explicit_no_cache_forward(
            official_semantics.structurally_aligned_dense_forward,
            **common,
            use_cache=None,
        )
        cached = recovery04.explicit_no_cache_forward(
            official_semantics.structurally_aligned_dense_forward,
            **common,
            use_cache=True,
        )
        sidecar = recovery04.explicit_no_cache_forward(
            official_semantics.structurally_aligned_dense_forward,
            **common,
            use_cache=True,
            tap_factory=Tap,
        )
        expected_modes = [None, False, True, True]
        if observed_modes != expected_modes:
            raise RuntimeError(f"full-helper forwarded modes changed: {observed_modes}")
        if direct_none["cache"] is not None or control["cache"] is not None:
            raise RuntimeError("synthetic direct/control helper unexpectedly returned a cache")
        if cached["cache"] is None or sidecar["cache"] is None:
            raise RuntimeError("synthetic cache helper returned no cache")
        tap = sidecar["tap"]
        if tap is None or tap.entered is not True or tap.exited is not True:
            raise RuntimeError("synthetic sidecar context was not completed")
        actions = [row["actions"] for row in (direct_none, control, cached, sidecar)]
        actions_identical = all(np.array_equal(actions[0], row) for row in actions[1:])
        if not actions_identical:
            raise RuntimeError("synthetic actions changed across cache modes")
        del prepared, model, direct_none, control, cached, sidecar, actions
        torch.cuda.empty_cache()
        final = v1.selected_gpu_snapshot(int(physical))
        summary = {
            "schema_version": "openvla-dense-helper-qualification-result-v1",
            "run_id": "openvla-dense-helper-qualification-s3q-v01",
            "complete": True,
            "passed": True,
            "official_semantics_sha256": recovery05.file_sha256(
                ROOT / "src/savr/openvla/official_semantics.py"
            ),
            "observed_forwarded_modes": observed_modes,
            "direct_none_preserved": True,
            "control_returns_cache": False,
            "cache_returns_cache": True,
            "sidecar_returns_cache": True,
            "sidecar_context_entered": True,
            "actions_identical_across_modes": True,
            "full_helper_calls": 4,
            "openvla_model_loads": 0,
            "openvla_model_calls": 0,
            "simulator_outcomes_accessed": False,
            "initial_selected_gpu": initial,
            "final_selected_gpu": final,
            "elapsed_seconds": time.monotonic() - started,
            "automatic_retry": False,
            "advance": {
                "next_stage": "S3_RECOVERY05_MODEL_ATTEMPT",
                "authorized": True,
                "stop_before_next_stage": False,
            },
        }
        summary["semantic_sha256"] = v1.semantic_sha256(summary)
        v1.write_once(output_root / "worker_summary.json", summary)
        return 0
    except BaseException as error:
        stop = {
            "schema_version": "openvla-dense-helper-qualification-stop-v1",
            "run_id": "openvla-dense-helper-qualification-s3q-v01",
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
