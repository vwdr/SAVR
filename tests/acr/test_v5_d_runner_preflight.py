from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


def load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def row(index: int, *, memory: int, utilization: int):
    return {
        "index": index,
        "uuid": f"gpu-{index}",
        "name": "NVIDIA TITAN RTX",
        "driver_version": "test",
        "memory_total_mib": 24576,
        "memory_used_mib": memory,
        "utilization_percent": utilization,
        "recorded_at_utc": "test",
    }


def test_gpu_selection_requires_every_sample_and_chooses_lowest_eligible() -> None:
    selector = load_script("select_acr_v5_d_gpu")
    runtime = load_script("analyze_acr_v5_d")
    del runtime
    import json

    config = json.loads(
        (ROOT / "configs/acr/v5_d_gpu_feasibility_freeze.json").read_text(encoding="utf-8")
    )
    samples = [
        [row(0, memory=100, utilization=0), row(1, memory=200, utilization=2)],
        [row(0, memory=600, utilization=0), row(1, memory=200, utilization=2)],
        [row(0, memory=100, utilization=0), row(1, memory=200, utilization=2)],
    ]
    assert selector.select_index(samples, config) == (1, [1])
    with pytest.raises(RuntimeError, match="No GPU"):
        selector.select_index([[row(0, memory=600, utilization=6)] for _ in range(3)], config)


def test_runner_is_synthetic_only_and_has_no_simulator_or_outcome_path() -> None:
    runner = (ROOT / "scripts/run_acr_v5_d.py").read_text(encoding="utf-8")
    assert "num_trials_per_task=0" in runner
    assert '"simulator_episodes": 0' in runner
    assert '"simulator_resets": 0' in runner
    assert '"new_task_outcomes": 0' in runner
    forbidden = (
        "benchmark.get_task",
        "env.reset",
        "env.step",
        "episode_success",
        '"success":',
        '"reward":',
        "kill(",
        "terminate(",
        "sudo",
    )
    assert not any(token in runner for token in forbidden)


def test_backend_hot_cores_contain_no_host_or_audit_side_effects() -> None:
    module = __import__(
        "savr.acr.v5_d_torch_backend",
        fromlist=["build_openvla_core_functions"],
    )
    source = inspect.getsource(module.build_openvla_core_functions)
    forbidden = (
        "hashlib",
        "json",
        "Path(",
        "open(",
        ".cpu(",
        ".numpy(",
        "synchronize",
        "isfinite",
        "tolist",
    )
    assert not any(token in source for token in forbidden)


def test_launch_script_uses_project_local_caches_and_only_frozen_waterfall() -> None:
    launch = (ROOT / "scripts/launch_acr_v5_d.sh").read_text(encoding="utf-8")
    for name in (
        "HF_HOME",
        "HF_HUB_CACHE",
        "TORCH_HOME",
        "TORCHINDUCTOR_CACHE_DIR",
        "TRITON_CACHE_DIR",
    ):
        assert f"export {name}=" in launch
    assert launch.count("scripts/run_acr_v5_d.py") == 2
    assert "--backend torch-compile" in launch
    assert "--backend raw-cudagraph" in launch
    assert "if [[ ${status} -eq 20 ]]" in launch
    assert "scripts/finalize_acr_v5_d.py" in launch
