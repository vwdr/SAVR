#!/usr/bin/env python3
"""Run one bounded, unmodified OpenVLA-OFT Full Refresh LIBERO episode."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_ROOT = Path("/home/ved/SAVR")
CHECKPOINT_REVISION = "638918f3d1c2e43a39a8a20772bdb8b91835e4b7"
CHECKPOINT_RELATIVE = Path("checkpoints/openvla-7b-oft-libero-four-suite")
SUITE = "libero_spatial"
TASK_ID = 0
INITIAL_STATE_ID = 0
SEED = 0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    if project_root != EXPECTED_ROOT:
        raise SystemExit(f"Refusing to run outside {EXPECTED_ROOT}: {project_root}")

    physical_gpu_id = os.environ.get("SAVR_PHYSICAL_GPU_ID")
    visible_gpu = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not physical_gpu_id or visible_gpu != physical_gpu_id:
        raise SystemExit("SAVR_PHYSICAL_GPU_ID must exactly match CUDA_VISIBLE_DEVICES")

    os.environ["HF_HOME"] = str(project_root / "cache" / "huggingface")
    os.environ["HF_HUB_CACHE"] = str(project_root / "cache" / "huggingface" / "hub")
    os.environ["LIBERO_CONFIG_PATH"] = str(project_root / "cache" / "libero")
    os.environ["TORCH_HOME"] = str(project_root / "cache" / "torch")
    os.environ["MUJOCO_GL"] = "osmesa"
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.environ["WANDB_MODE"] = "disabled"
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    upstream_root = project_root / "third_party" / "openvla-oft"
    checkpoint = project_root / CHECKPOINT_RELATIVE
    if not checkpoint.is_dir():
        raise SystemExit(f"Missing checkpoint: {checkpoint}")
    sys.path.insert(0, str(upstream_root))

    run_id = datetime.now(timezone.utc).strftime("phase2a-fr-%Y%m%dT%H%M%SZ")
    run_dir = project_root / "results" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = run_dir / "manifest.json"

    protected_names = (
        "config.json",
        "configuration_prismatic.py",
        "modeling_prismatic.py",
    )
    protected_bytes = {
        name: (checkpoint / name).read_bytes() for name in protected_names
    }
    files_before = {item.name for item in checkpoint.iterdir()}
    hashes_before = {name: sha256(checkpoint / name) for name in protected_names}

    manifest: dict[str, Any] = {
        "run_id": run_id,
        "status": "RUNNING",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_commit": subprocess.check_output(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"], text=True
        ).strip(),
        "upstream_commit": subprocess.check_output(
            ["git", "-C", str(upstream_root), "rev-parse", "HEAD"], text=True
        ).strip(),
        "checkpoint_revision": CHECKPOINT_REVISION,
        "checkpoint_path": str(checkpoint),
        "physical_gpu_id": physical_gpu_id,
        "cuda_visible_devices": visible_gpu,
        "suite": SUITE,
        "task_id": TASK_ID,
        "initial_state_id": INITIAL_STATE_ID,
        "seed": SEED,
        "policy": "FR",
        "num_trials_per_task": 1,
        "num_open_loop_steps": 8,
        "use_l1_regression": True,
        "use_diffusion": False,
        "use_film": False,
        "num_images_in_input": 2,
        "use_proprio": True,
        "center_crop": True,
        "checkpoint_hashes_before": hashes_before,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    exit_code = 1
    try:
        os.chdir(upstream_root)

        import torch
        from libero.libero import benchmark

        from experiments.robot.libero import run_libero_eval as upstream_eval
        from experiments.robot.robot_utils import (
            get_image_resize_size,
            set_seed_everywhere,
        )

        if not torch.cuda.is_available():
            raise RuntimeError("Selected CUDA device is not available")
        if torch.cuda.device_count() != 1:
            raise RuntimeError(
                f"Expected exactly one visible GPU, found {torch.cuda.device_count()}"
            )

        manifest["visible_gpu_name"] = torch.cuda.get_device_name(0)
        manifest["visible_gpu_capability"] = list(torch.cuda.get_device_capability(0))
        torch.cuda.reset_peak_memory_stats()

        cfg = upstream_eval.GenerateConfig(
            pretrained_checkpoint=str(checkpoint),
            task_suite_name=SUITE,
            num_trials_per_task=1,
            seed=SEED,
            local_log_dir=str(run_dir / "logs"),
            use_wandb=False,
            center_crop=True,
            num_open_loop_steps=8,
            num_images_in_input=2,
            use_proprio=True,
            use_l1_regression=True,
            use_diffusion=False,
            use_film=False,
        )
        upstream_eval.validate_config(cfg)
        set_seed_everywhere(SEED)

        load_started = time.perf_counter()
        model, action_head, proprio_projector, noisy_action_projector, processor = (
            upstream_eval.initialize_model(cfg)
        )
        torch.cuda.synchronize()
        manifest["model_load_seconds"] = time.perf_counter() - load_started

        query_seconds: list[float] = []
        action_chunk_lengths: list[int] = []
        original_get_action = upstream_eval.get_action

        def measured_get_action(*args: Any, **kwargs: Any) -> Any:
            torch.cuda.synchronize()
            started = time.perf_counter()
            actions = original_get_action(*args, **kwargs)
            torch.cuda.synchronize()
            query_seconds.append(time.perf_counter() - started)
            action_chunk_lengths.append(len(actions))
            return actions

        upstream_eval.get_action = measured_get_action
        resize_size = get_image_resize_size(cfg)
        task_suite = benchmark.get_benchmark_dict()[SUITE]()
        log_file, log_path, upstream_run_id = upstream_eval.setup_logging(cfg)

        os.chdir(run_dir)
        episode_started = time.perf_counter()
        total_episodes, total_successes = upstream_eval.run_task(
            cfg,
            task_suite,
            TASK_ID,
            model,
            resize_size,
            processor,
            action_head,
            proprio_projector,
            noisy_action_projector,
            0,
            0,
            log_file,
        )
        torch.cuda.synchronize()
        manifest["episode_seconds"] = time.perf_counter() - episode_started
        log_file.close()

        manifest.update(
            {
                "status": "SUCCESS",
                "episodes": total_episodes,
                "successes": total_successes,
                "task_success": bool(total_successes == 1),
                "policy_query_count": len(query_seconds),
                "policy_query_seconds": query_seconds,
                "action_chunk_lengths": action_chunk_lengths,
                "upstream_log_path": log_path,
                "upstream_run_id": upstream_run_id,
                "peak_gpu_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_gpu_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
            }
        )
        exit_code = 0
    except Exception as error:
        manifest["status"] = "FAILED"
        manifest["error_type"] = type(error).__name__
        manifest["error"] = str(error)
        raise
    finally:
        for name, content in protected_bytes.items():
            (checkpoint / name).write_bytes(content)
        files_after_restore = {item.name for item in checkpoint.iterdir()}
        new_files = sorted(files_after_restore - files_before)
        removed_backups = []
        for name in new_files:
            if ".back." in name:
                (checkpoint / name).unlink()
                removed_backups.append(name)
        manifest["removed_upstream_backup_files"] = removed_backups
        manifest["checkpoint_hashes_after_restore"] = {
            name: sha256(checkpoint / name) for name in protected_names
        }
        manifest["checkpoint_restored"] = (
            manifest["checkpoint_hashes_after_restore"] == hashes_before
        )
        manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    print(json.dumps(manifest, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
