#!/usr/bin/env python3
"""Run the approved resumable 50-episode LIBERO-Spatial FR pilot."""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_ROOT = Path("/home/ved/SAVR")
CHECKPOINT_REVISION = "638918f3d1c2e43a39a8a20772bdb8b91835e4b7"
CHECKPOINT_RELATIVE = Path("checkpoints/openvla-7b-oft-libero-four-suite")
RUN_ID = "phase2b-fr-spatial-pilot-v1"
SUITE = "libero_spatial"
TASK_IDS = tuple(range(10))
INITIAL_STATE_IDS = tuple(range(5))
SEED = 0
EXPECTED_EPISODES = 50
ARTIFACT_CAP_BYTES = 2 * 1024**3
MAX_TERMINAL_ERRORS = 3


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


class CudaModuleTimer:
    def __init__(self, torch_module: Any, torch: Any) -> None:
        self.torch = torch
        self.events: list[tuple[Any, Any]] = []
        self.stack: list[Any] = []
        self.pre_handle = torch_module.register_forward_pre_hook(self._before)
        self.post_handle = torch_module.register_forward_hook(self._after)

    def _before(self, _module: Any, _inputs: Any) -> None:
        start = self.torch.cuda.Event(enable_timing=True)
        start.record()
        self.stack.append(start)

    def _after(self, _module: Any, _inputs: Any, _output: Any) -> None:
        if not self.stack:
            raise RuntimeError("CUDA timing hook stack underflow")
        end = self.torch.cuda.Event(enable_timing=True)
        end.record()
        self.events.append((self.stack.pop(), end))

    def count(self) -> int:
        return len(self.events)

    def elapsed_since(self, prior_count: int) -> list[float]:
        return [
            start.elapsed_time(end)
            for start, end in self.events[prior_count:]
        ]

    def remove(self) -> None:
        self.pre_handle.remove()
        self.post_handle.remove()


def terminal_records(episodes_dir: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(episodes_dir.glob("task_*_state_*.json")):
        records.append(json.loads(path.read_text()))
    return records


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    if project_root != EXPECTED_ROOT:
        raise SystemExit(f"Refusing to run outside {EXPECTED_ROOT}: {project_root}")

    physical_gpu_id = os.environ.get("SAVR_PHYSICAL_GPU_ID")
    if not physical_gpu_id or os.environ.get("CUDA_VISIBLE_DEVICES") != physical_gpu_id:
        raise SystemExit("SAVR_PHYSICAL_GPU_ID must exactly match CUDA_VISIBLE_DEVICES")

    def handle_term(_signum: int, _frame: Any) -> None:
        raise TimeoutError("Received termination signal")

    signal.signal(signal.SIGTERM, handle_term)
    signal.signal(signal.SIGINT, handle_term)

    os.environ["HF_HOME"] = str(project_root / "cache" / "huggingface")
    os.environ["HF_HUB_CACHE"] = str(project_root / "cache" / "huggingface" / "hub")
    os.environ["LIBERO_CONFIG_PATH"] = str(project_root / "cache" / "libero")
    os.environ["TORCH_HOME"] = str(project_root / "cache" / "torch")
    os.environ["MUJOCO_GL"] = "osmesa"
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    upstream_root = project_root / "third_party" / "openvla-oft"
    checkpoint = project_root / CHECKPOINT_RELATIVE
    run_dir = project_root / "results" / RUN_ID
    episodes_dir = run_dir / "episodes"
    videos_dir = run_dir / "videos"
    manifest_path = run_dir / "manifest.json"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(upstream_root))

    project_commit = subprocess.check_output(
        ["git", "-C", str(project_root), "rev-parse", "HEAD"], text=True
    ).strip()
    upstream_commit = subprocess.check_output(
        ["git", "-C", str(upstream_root), "rev-parse", "HEAD"], text=True
    ).strip()

    protected_names = (
        "config.json",
        "configuration_prismatic.py",
        "modeling_prismatic.py",
    )
    protected_bytes = {
        name: (checkpoint / name).read_bytes() for name in protected_names
    }
    protected_hashes = {name: sha256(checkpoint / name) for name in protected_names}
    checkpoint_files_before = {item.name for item in checkpoint.iterdir()}

    frozen_config = {
        "run_id": RUN_ID,
        "checkpoint_revision": CHECKPOINT_REVISION,
        "upstream_commit": upstream_commit,
        "suite": SUITE,
        "task_ids": list(TASK_IDS),
        "initial_state_ids": list(INITIAL_STATE_IDS),
        "seed": SEED,
        "expected_episodes": EXPECTED_EPISODES,
        "policy": "FR",
        "num_open_loop_steps": 8,
        "num_images_in_input": 2,
        "use_proprio": True,
        "use_l1_regression": True,
        "use_diffusion": False,
        "use_film": False,
        "center_crop": True,
    }
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["frozen_config"] != frozen_config:
            raise RuntimeError("Existing run manifest does not match frozen config")
    else:
        manifest = {
            "status": "RUNNING",
            "started_at_utc": utc_now(),
            "project_commit": project_commit,
            "physical_gpu_id": physical_gpu_id,
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "frozen_config": frozen_config,
            "checkpoint_hashes_before": protected_hashes,
        }
        atomic_json(manifest_path, manifest)

    import imageio.v2 as imageio
    import numpy as np
    import torch
    from libero.libero import benchmark

    from experiments.robot.libero import run_libero_eval as upstream_eval
    from experiments.robot.libero.libero_utils import (
        get_libero_dummy_action,
        get_libero_env,
    )
    from experiments.robot.robot_utils import get_image_resize_size, set_seed_everywhere
    from prismatic.vla.constants import NUM_ACTIONS_CHUNK

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one selected CUDA device must be visible")

    cfg = upstream_eval.GenerateConfig(
        pretrained_checkpoint=str(checkpoint),
        task_suite_name=SUITE,
        num_trials_per_task=5,
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
    torch.cuda.reset_peak_memory_stats()

    vision_timer = projector_timer = action_head_timer = None
    current_env = None
    exit_code = 1
    try:
        os.chdir(upstream_root)
        load_started = time.perf_counter()
        model, action_head, proprio_projector, noisy_action_projector, processor = (
            upstream_eval.initialize_model(cfg)
        )
        torch.cuda.synchronize()
        manifest["model_load_seconds"] = time.perf_counter() - load_started
        manifest["visible_gpu_name"] = torch.cuda.get_device_name(0)
        manifest["visible_gpu_capability"] = list(torch.cuda.get_device_capability(0))

        vision_timer = CudaModuleTimer(model.vision_backbone, torch)
        projector_timer = CudaModuleTimer(model.projector, torch)
        action_head_timer = CudaModuleTimer(action_head.model, torch)
        resize_size = get_image_resize_size(cfg)
        task_suite = benchmark.get_benchmark_dict()[SUITE]()

        existing = terminal_records(episodes_dir)
        global_query_index = sum(
            len(record.get("queries", [])) for record in existing
        )
        terminal_errors = sum(record["status"] == "ERROR" for record in existing)
        saved_success_tasks = {
            record["task_id"]
            for record in existing
            if record.get("video_saved") and record.get("success")
        }

        for task_id in TASK_IDS:
            task = task_suite.get_task(task_id)
            initial_states = task_suite.get_task_init_states(task_id)
            current_env, task_description = get_libero_env(
                task, cfg.model_family, resolution=cfg.env_img_res
            )
            try:
                for state_id in INITIAL_STATE_IDS:
                    episode_path = (
                        episodes_dir / f"task_{task_id:02d}_state_{state_id:02d}.json"
                    )
                    if episode_path.exists():
                        continue
                    if terminal_errors >= MAX_TERMINAL_ERRORS:
                        raise RuntimeError("Reached predeclared terminal-error stop rule")
                    if directory_size(run_dir) > ARTIFACT_CAP_BYTES:
                        raise RuntimeError("Reached two-GiB artifact cap")

                    episode = {
                        "status": "RUNNING",
                        "started_at_utc": utc_now(),
                        "task_id": task_id,
                        "initial_state_id": state_id,
                        "seed": SEED,
                        "task_description": task_description,
                        "queries": [],
                        "preprocessing": [],
                        "success": False,
                        "video_saved": False,
                    }
                    episode_started = time.perf_counter()
                    replay_images = []
                    current_env.reset()
                    observation = current_env.set_init_state(initial_states[state_id])
                    action_queue: deque[Any] = deque(maxlen=cfg.num_open_loop_steps)
                    environment_step = 0
                    max_steps = upstream_eval.TASK_MAX_STEPS[cfg.task_suite_name]

                    try:
                        while environment_step < max_steps + cfg.num_steps_wait:
                            if environment_step < cfg.num_steps_wait:
                                observation, _, _, _ = current_env.step(
                                    get_libero_dummy_action(cfg.model_family)
                                )
                                environment_step += 1
                                continue

                            prep_started = time.perf_counter()
                            policy_observation, image = upstream_eval.prepare_observation(
                                observation, resize_size
                            )
                            episode["preprocessing"].append(
                                {
                                    "environment_step": environment_step,
                                    "wall_seconds": time.perf_counter() - prep_started,
                                }
                            )
                            replay_images.append(image)

                            if not action_queue:
                                counts_before = {
                                    "vision": vision_timer.count(),
                                    "projector": projector_timer.count(),
                                    "action_head": action_head_timer.count(),
                                }
                                total_start = torch.cuda.Event(enable_timing=True)
                                total_end = torch.cuda.Event(enable_timing=True)
                                torch.cuda.synchronize()
                                query_started = time.perf_counter()
                                total_start.record()
                                actions = upstream_eval.get_action(
                                    cfg,
                                    model,
                                    policy_observation,
                                    task_description,
                                    processor=processor,
                                    action_head=action_head,
                                    proprio_projector=proprio_projector,
                                    noisy_action_projector=noisy_action_projector,
                                    use_film=cfg.use_film,
                                )
                                total_end.record()
                                torch.cuda.synchronize()
                                query_wall = time.perf_counter() - query_started
                                if not np.isfinite(actions).all():
                                    raise RuntimeError("Policy produced non-finite actions")

                                component_ms = {
                                    "vision_backbone": vision_timer.elapsed_since(
                                        counts_before["vision"]
                                    ),
                                    "visual_projector": projector_timer.elapsed_since(
                                        counts_before["projector"]
                                    ),
                                    "action_head": action_head_timer.elapsed_since(
                                        counts_before["action_head"]
                                    ),
                                }
                                if any(len(values) != 1 for values in component_ms.values()):
                                    raise RuntimeError(
                                        f"Timing hook count mismatch: {component_ms}"
                                    )
                                total_cuda_ms = total_start.elapsed_time(total_end)
                                component_scalar_ms = {
                                    name: values[0]
                                    for name, values in component_ms.items()
                                }
                                residual_ms = max(
                                    0.0,
                                    total_cuda_ms - sum(component_scalar_ms.values()),
                                )
                                episode["queries"].append(
                                    {
                                        "global_query_index": global_query_index,
                                        "episode_query_index": len(episode["queries"]),
                                        "environment_step": environment_step,
                                        "warmup": global_query_index < 3,
                                        "wall_seconds": query_wall,
                                        "total_cuda_ms": total_cuda_ms,
                                        **{
                                            f"{name}_cuda_ms": value
                                            for name, value in component_scalar_ms.items()
                                        },
                                        "downstream_residual_cuda_ms": residual_ms,
                                        "action_chunk_length": len(actions),
                                        "gpu_memory_allocated_bytes": torch.cuda.memory_allocated(),
                                        "gpu_memory_reserved_bytes": torch.cuda.memory_reserved(),
                                    }
                                )
                                global_query_index += 1
                                action_queue.extend(actions)

                            action = upstream_eval.process_action(
                                action_queue.popleft(), cfg.model_family
                            )
                            observation, _, done, _ = current_env.step(action.tolist())
                            if done:
                                episode["success"] = True
                                break
                            environment_step += 1

                        episode["status"] = "COMPLETED"
                    except Exception as error:
                        episode["status"] = "ERROR"
                        episode["error_type"] = type(error).__name__
                        episode["error"] = str(error)
                        terminal_errors += 1

                    should_save_video = (
                        episode["status"] == "ERROR"
                        or not episode["success"]
                        or task_id not in saved_success_tasks
                    )
                    if should_save_video and replay_images:
                        video_path = (
                            videos_dir
                            / f"task_{task_id:02d}_state_{state_id:02d}"
                            f"_success_{episode['success']}.mp4"
                        )
                        with imageio.get_writer(video_path, fps=30) as writer:
                            for frame in replay_images:
                                writer.append_data(frame)
                        episode["video_saved"] = True
                        episode["video_path"] = str(video_path)
                        if episode["success"]:
                            saved_success_tasks.add(task_id)

                    episode["environment_steps"] = environment_step
                    episode["episode_seconds"] = time.perf_counter() - episode_started
                    episode["completed_at_utc"] = utc_now()
                    atomic_json(episode_path, episode)
                    del replay_images
                    if directory_size(run_dir) > ARTIFACT_CAP_BYTES:
                        raise RuntimeError("Reached two-GiB artifact cap")
            finally:
                current_env.close()
                current_env = None

        records = terminal_records(episodes_dir)
        planned_pairs = {
            (task_id, state_id)
            for task_id in TASK_IDS
            for state_id in INITIAL_STATE_IDS
        }
        observed_pairs = {
            (record["task_id"], record["initial_state_id"]) for record in records
        }
        manifest.update(
            {
                "status": "COMPLETE" if observed_pairs == planned_pairs else "INCOMPLETE",
                "terminal_episode_count": len(records),
                "completed_episode_count": sum(
                    record["status"] == "COMPLETED" for record in records
                ),
                "error_episode_count": sum(
                    record["status"] == "ERROR" for record in records
                ),
                "success_count": sum(bool(record["success"]) for record in records),
                "missing_pairs": sorted(planned_pairs - observed_pairs),
                "peak_gpu_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_gpu_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
                "artifact_bytes": directory_size(run_dir),
            }
        )
        exit_code = 0 if manifest["status"] == "COMPLETE" else 1
    except Exception as error:
        manifest["status"] = "INTERRUPTED"
        manifest["error_type"] = type(error).__name__
        manifest["error"] = str(error)
        raise
    finally:
        if current_env is not None:
            current_env.close()
        for timer in (vision_timer, projector_timer, action_head_timer):
            if timer is not None:
                timer.remove()
        for name, content in protected_bytes.items():
            (checkpoint / name).write_bytes(content)
        new_files = {item.name for item in checkpoint.iterdir()} - checkpoint_files_before
        removed_backups = []
        for name in sorted(new_files):
            if ".back." in name:
                (checkpoint / name).unlink()
                removed_backups.append(name)
        manifest["removed_upstream_backup_files"] = removed_backups
        manifest["checkpoint_hashes_after_restore"] = {
            name: sha256(checkpoint / name) for name in protected_names
        }
        manifest["checkpoint_restored"] = (
            manifest["checkpoint_hashes_after_restore"] == protected_hashes
        )
        manifest["completed_at_utc"] = utc_now()
        atomic_json(manifest_path, manifest)

    print(json.dumps(manifest, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
