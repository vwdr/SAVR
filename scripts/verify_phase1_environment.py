#!/usr/bin/env python3
"""Verify SAVR Phase 1 imports and CPU-only LIBERO rendering."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


EXPECTED_ROOT = Path("/home/ved/SAVR")


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    if project_root != EXPECTED_ROOT:
        raise SystemExit(f"Refusing to run outside {EXPECTED_ROOT}: {project_root}")

    os.environ["MUJOCO_GL"] = "osmesa"
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"
    os.environ["PYTHONNOUSERSITE"] = "1"

    environment_library = project_root / "envs" / "openvla-oft" / "lib"
    library_path = os.environ.get("LD_LIBRARY_PATH", "")
    library_entries = [entry for entry in library_path.split(":") if entry]
    if str(environment_library) not in library_entries:
        os.environ["LD_LIBRARY_PATH"] = ":".join([str(environment_library), *library_entries])
        os.execve(sys.executable, [sys.executable, *sys.argv], os.environ)

    import bddl
    import gym
    import mujoco
    import numpy as np
    import robosuite
    import torch
    import transformers
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    versions = {
        "bddl": getattr(bddl, "__version__", "unknown"),
        "gym": getattr(gym, "__version__", "unknown"),
        "mujoco": getattr(mujoco, "__version__", "unknown"),
        "numpy": np.__version__,
        "robosuite": getattr(robosuite, "__version__", "unknown"),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }

    task_suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    task = task_suite.get_task(0)
    bddl_path = os.path.join(
        get_libero_path("bddl_files"),
        task.problem_folder,
        task.bddl_file,
    )
    environment = OffScreenRenderEnv(
        bddl_file_name=bddl_path,
        camera_heights=128,
        camera_widths=128,
    )
    try:
        environment.seed(0)
        environment.reset()
        initial_states = task_suite.get_task_init_states(0)
        observation = environment.set_init_state(initial_states[0])
        observation, reward, done, _ = environment.step(np.zeros(7, dtype=np.float32))
        image_keys = sorted(key for key in observation if key.endswith("_image"))
        if not image_keys:
            raise RuntimeError("LIBERO observation did not contain rendered images")
        image_shapes = {key: list(observation[key].shape) for key in image_keys}
    finally:
        environment.close()

    report = {
        "cpu_only_requested": True,
        "cuda_initialized": torch.cuda.is_initialized(),
        "mujoco_gl": os.environ["MUJOCO_GL"],
        "pyopengl_platform": os.environ["PYOPENGL_PLATFORM"],
        "suite": "libero_spatial",
        "task_id": 0,
        "initial_state_id": 0,
        "reward_after_one_zero_action": float(reward),
        "done_after_one_zero_action": bool(done),
        "image_shapes": image_shapes,
        "versions": versions,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
