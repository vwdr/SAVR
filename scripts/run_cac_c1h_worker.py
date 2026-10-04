#!/usr/bin/env python3
"""Run the frozen CAC C1H paired dense-versus-D62 simulator screen."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
import random
import signal
import shutil
import subprocess
import sys
import threading
import time
import traceback
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path("/home/ved/SAVR")
DEFAULT_CONFIG = Path("configs/cac/c1h_headroom_v1.json")
S3_RECOVERY05_WORKER = ROOT / "scripts/run_openvla_semantic_parity_recovery05.py"


class TechnicalStop(RuntimeError):
    """Fail-closed C1H technical stop."""


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise TechnicalStop(f"cannot load {path.name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def install_qualified_official_loader() -> tuple[Any, dict[str, Any]]:
    recovery05 = load_module(S3_RECOVERY05_WORKER, "s3_recovery05_for_cac_c1h_s6")
    recovery04 = recovery05.load_recovery04()
    recovery03 = recovery04.load_recovery03()
    recovery02 = recovery03.load_recovery02()
    recovery01 = recovery02.load_recovery01()
    if (
        recovery02.function_source_sha256(recovery01.install_official_loader_guard)
        != recovery02.QUALIFIED_LOADER_GUARD_SHA256
    ):
        raise TechnicalStop("C1H qualified official loader guard changed")
    return recovery01.install_official_loader_guard()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def semantic_sha256(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("semantic_sha256", None)
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def write_once(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(canonical_bytes(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def write_text_once(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())


def write_jsonl_once(path: Path, values: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        for value in values:
            stream.write(canonical_bytes(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def append_progress(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("ab") as stream:
        stream.write(canonical_bytes(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def official_control_observation(np_module: Any, observation: Mapping[str, Any]) -> dict[str, Any]:
    required = {"full_image", "wrist_image", "state"}
    if not required.issubset(observation):
        raise TechnicalStop("C1H official control observation contract changed")
    cloned = {key: np_module.asarray(value).copy() for key, value in observation.items()}
    cloned["prev_images"] = [
        cloned["full_image"].copy(),
        cloned["wrist_image"].copy(),
    ]
    return cloned


def unpack_official_action_result(np_module: Any, result: Any) -> tuple[Any, Any, Any, Any]:
    if not isinstance(result, tuple) or len(result) != 4:
        raise TechnicalStop("C1H official action helper return contract changed")
    actions, cache, images, metrics = result
    action_array = np_module.asarray(actions, dtype=np_module.float32)
    if action_array.shape != (8, 7) or not np_module.isfinite(action_array).all():
        raise TechnicalStop("C1H official action helper produced an invalid action chunk")
    return action_array, cache, images, metrics


def build_policy_query_output(
    np_module: Any,
    result: Mapping[str, Any],
    prepared: Any,
    salience: Any,
    *,
    require_cac: bool,
) -> dict[str, Any]:
    """Validate and propagate the exact tensors required by C1H controls."""

    actions = np_module.asarray(result.get("actions"), dtype=np_module.float32)
    if actions.shape != (8, 7) or not np_module.isfinite(actions).all():
        raise TechnicalStop("C1H custom action output changed")
    output = {
        "actions": actions,
        "cache": result.get("cache"),
        "prepared": prepared,
        "salience": salience,
    }
    if require_cac:
        action_hidden = result.get("action_hidden")
        base_action = result.get("base_action")
        if action_hidden is None or tuple(action_hidden.shape) != (1, 56, 4096):
            raise TechnicalStop("C1H captured action-hidden contract changed")
        if base_action is None or tuple(base_action.shape) != (1, 8, 7):
            raise TechnicalStop("C1H captured normalized-action contract changed")
        output["action_hidden"] = action_hidden
        output["base_action"] = base_action
    return output


def aggregate_gpu_memory_mib(physical_gpu: int) -> int:
    value = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            str(physical_gpu),
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    return int(value)


def selected_gpu_snapshot(physical_gpu: int) -> dict[str, int]:
    value = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            str(physical_gpu),
            "--query-gpu=memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    fields = [int(item.strip()) for item in value.split(",")]
    if len(fields) != 2:
        raise TechnicalStop("selected-GPU snapshot did not return exactly one row")
    return {"memory_used_mib": fields[0], "utilization_percent": fields[1]}


class AggregateMemorySampler:
    """Continuously sample only aggregate memory on the selected GPU."""

    def __init__(self, gpu: int, initial: int) -> None:
        self.gpu = gpu
        self.peak = initial
        self.error: BaseException | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, name="cac-c1h-gpu-memory", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(0.20):
            try:
                value = aggregate_gpu_memory_mib(self.gpu)
                with self._lock:
                    self.peak = max(self.peak, value)
            except BaseException as error:
                self.error = error
                return

    def snapshot(self) -> int:
        with self._lock:
            return self.peak

    def close(self) -> int:
        self._stop.set()
        self._thread.join()
        return self.snapshot()


def validate_config(config: Mapping[str, Any]) -> None:
    schema = config.get("schema_version")
    corrected_s6 = schema in {
        "cac-c1h-corrected-s6-v1", "cac-c1h-corrected-s6-recovery01-v1"
    }
    if schema not in {
        "cac-c1h-config-v1",
        "cac-c1h-corrected-s6-v1",
        "cac-c1h-corrected-s6-recovery01-v1",
    }:
        raise TechnicalStop("C1H config schema changed")
    if config.get("semantic_sha256") != semantic_sha256(config):
        raise TechnicalStop("C1H config semantic hash mismatch")
    authenticated_inputs = [
        "protocol", "c0_freeze", "population_manifest", "profile_config", "c1_result"
    ]
    if corrected_s6:
        authenticated_inputs.append("parent_protocol")
    for key in authenticated_inputs:
        path = ROOT / config[key]
        if not path.is_file() or file_sha256(path) != config[f"{key}_sha256"]:
            raise TechnicalStop(f"C1H authenticated input changed: {key}")
    for relative, expected in config["code"].items():
        path = ROOT / relative
        if not path.is_file() or file_sha256(path) != expected:
            raise TechnicalStop(f"C1H frozen code changed: {relative}")
    c1 = json.loads((ROOT / config["c1_result"]).read_text(encoding="utf-8"))
    if (
        c1.get("complete") is not True
        or not all(c1.get("gates", {}).values())
        or c1.get("advance") != {
            "next_phase": "C1H",
            "authorized": False,
            "stop_before_next_phase": True,
        }
    ):
        raise TechnicalStop("C1 did not authenticate a complete pass into C1H")
    if corrected_s6:
        if config.get("corrected_foundation") != {
            "s3_summary": "results/openvla-semantic-parity-s3-v06-recovery05/worker_summary.json",
            "s3_summary_sha256": config["corrected_foundation"]["s3_summary_sha256"],
            "s3_semantic_sha256": "eae02332c936c7816e2de7870b14fb78122e1acc603cc0dd04b39683f1695380",
            "s4_summary": "results/openvla-d62-requalification-s4-v02-recovery01/worker_summary.json",
            "s4_summary_sha256": config["corrected_foundation"]["s4_summary_sha256"],
            "s4_semantic_sha256": "988df0eaafb6eaf03549ee9fce8c66677bb9c524ec6eebba12ce6df5b922f38d",
            "s5_result": "results/cac-c1-s5-requalification-v01/result.json",
            "s5_result_sha256": config["corrected_foundation"]["s5_result_sha256"],
            "s5_semantic_sha256": "c4cd5bc67e181fb0e18e0e5f4b78750a6eccb635c9b73aadd0a6709f570fd46c",
        }:
            raise TechnicalStop("C1H corrected-foundation identity changed")
        for stage in ("s3", "s4"):
            path = ROOT / config["corrected_foundation"][f"{stage}_summary"]
            value = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
            if (
                not path.is_file()
                or file_sha256(path)
                != config["corrected_foundation"][f"{stage}_summary_sha256"]
                or value.get("semantic_sha256")
                != config["corrected_foundation"][f"{stage}_semantic_sha256"]
                or value.get("complete") is not True
                or value.get("passed") is not True
            ):
                raise TechnicalStop(f"C1H corrected {stage.upper()} foundation did not pass")
        if c1.get("substrate") != {
            "action_readout_corrected": True,
            "compact_positions_explicitly_mapped": True,
            "corrected_id": "D62_BAL_PT1_S4C_V1",
            "historical_profile_id": "D62_BAL_PT1",
            "historical_profile_values_reused": True,
            "s4_selection_changed_observations": 0,
            "s4_selection_materially_changed": False,
        } or c1.get("official_parity", {}).get("unnormalized_action_max_abs") != 0.0:
            raise TechnicalStop("C1H corrected S5 substrate did not pass")
    if config["authorization"] != {
        "c1h_authorized": True,
        "c2_authorized": False,
        "terminal_outcomes": True,
        "locked_state_ids_10_49": False,
    }:
        raise TechnicalStop("C1H authorization boundary changed")
    if config["populations"] != {
        "stage1": "headroom_stage1",
        "extension": "headroom_extension",
        "conditions_per_stage": 120,
        "episodes_per_stage": 240,
    }:
        raise TechnicalStop("C1H population boundary changed")
    expected_profile = "D62_BAL_PT1_S4C_V1" if corrected_s6 else "D62_BAL_PT1"
    expected_substrate = {
        "profile": expected_profile,
        "maximum_cached_query_intervals": 4,
        "reset_rule": "complete dense reset after at most four cached query intervals",
        "actions_per_query": 8,
    }
    if corrected_s6:
        expected_substrate.update(
            {
                "historical_profile_id": "D62_BAL_PT1",
                "historical_profile_values_reused": True,
                "action_readout_corrected": True,
                "compact_positions_explicitly_mapped": True,
            }
        )
    if config["cache_substrate"] != expected_substrate:
        raise TechnicalStop("C1H D62 substrate changed")
    query_accounting = config["query_accounting"]
    expected_worst = 4 + 2 * 30 * 2 * sum(
        math.ceil(int(value) / 8) for value in config["model"]["suite_max_steps"].values()
    )
    if query_accounting != {
        "technical_control_queries": 4,
        "stage1_worst_case_queries": 9964,
        "cumulative_worst_case_queries": 19924,
    } or expected_worst != 19924:
        raise TechnicalStop("C1H exact query accounting changed")
    if config["resource_caps"] != {
        "gpu_count": 1,
        "model_processes": 1,
        "episode_attempts": 480,
        "model_queries": 20000,
        "wall_seconds": 36000,
        "artifact_bytes": 1073741824,
        "peak_gpu_memory_mib_strict_max": 23552,
        "downloads": 0,
        "automatic_retry": False,
    }:
        raise TechnicalStop("C1H resource boundary changed")
    if config["advance"] != {
        "next_phase": "C2",
        "authorized": False,
        "stop_before_next_phase": True,
    }:
        raise TechnicalStop("C1H advance boundary changed")
    recovery = config.get("technical_recovery")
    if recovery is not None:
        stop_path = ROOT / recovery["source_technical_stop"]
        source = json.loads(stop_path.read_text(encoding="utf-8")) if stop_path.is_file() else {}
        if (
            not stop_path.is_file()
            or file_sha256(stop_path) != recovery["source_technical_stop_sha256"]
            or source.get("semantic_sha256") != recovery["source_technical_stop_semantic_sha256"]
            or source.get("run_id") != recovery["source_run_id"]
            or source.get("episode_attempts") != recovery["source_episode_attempts"]
            or source.get("model_queries") != recovery["source_model_queries"]
            or source.get("partial_outcomes_opened")
            != recovery["source_partial_outcomes_opened"]
            or source.get("automatic_retry") is not False
            or recovery["automatic_retry"] is not False
        ):
            raise TechnicalStop("C1H recovery evidence or boundary changed")


def base_config(evaluation: Any, checkpoint: Path, run_root: Path) -> Any:
    return evaluation.GenerateConfig(
        pretrained_checkpoint=str(checkpoint),
        task_suite_name="libero_spatial",
        num_trials_per_task=0,
        seed=7,
        local_log_dir=str(run_root / "logs"),
        use_wandb=False,
        center_crop=True,
        num_open_loop_steps=8,
        num_images_in_input=2,
        use_proprio=True,
        use_l1_regression=True,
        use_diffusion=False,
        use_film=False,
        use_vla_cache=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--qualification-only", action="store_true")
    args = parser.parse_args()
    if Path.cwd().resolve() != ROOT:
        raise SystemExit(f"C1H worker must start in {ROOT}")
    output_root = args.output_root.resolve()
    if not output_root.is_relative_to(ROOT / "results") or output_root.exists():
        raise SystemExit("C1H requires a new immutable output below project results")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    physical = os.environ.get("CAC_PHYSICAL_GPU_ID", "")
    if not visible or "," in visible or not physical.isdigit() or visible != physical:
        raise SystemExit("C1H requires one explicit selected GPU")
    project_libero = ROOT / "configs/pair/libero_runtime"
    if not (project_libero / "config.yaml").is_file():
        raise SystemExit("project-local LIBERO configuration is missing")
    os.environ["LIBERO_CONFIG_PATH"] = str(project_libero)

    config_path = (ROOT / args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    corrected_s6 = config.get("schema_version") in {
        "cac-c1h-corrected-s6-v1", "cac-c1h-corrected-s6-recovery01-v1"
    }
    if args.qualification_only:
        if config.get("schema_version") != "cac-c1h-corrected-s6-recovery01-v1":
            raise SystemExit("C1H qualification requires the Recovery 01 schema")
        expected_output = (ROOT / config["qualification_output_root"]).resolve()
    else:
        expected_output = (ROOT / config.get("output_root", str(args.output_root))).resolve()
    if corrected_s6 and output_root != expected_output:
        raise SystemExit("C1H S6 output identity changed")
    if (
        config.get("schema_version") == "cac-c1h-corrected-s6-recovery01-v1"
        and not args.qualification_only
    ):
        qualification_path = ROOT / config["qualification_output_root"] / "worker_summary.json"
        qualification = (
            json.loads(qualification_path.read_text(encoding="utf-8"))
            if qualification_path.is_file()
            else {}
        )
        controls = qualification.get("technical_controls", {})
        if (
            qualification.get("semantic_sha256") != semantic_sha256(qualification)
            or qualification.get("complete") is not True
            or qualification.get("passed") is not True
            or qualification.get("mode") != "outcome_free_pre_episode_qualification"
            or qualification.get("model_queries") != 4
            or qualification.get("terminal_episode_attempts") != 0
            or qualification.get("terminal_outcomes_opened") is not False
            or controls.get("independent_official_oracle") is not True
            or controls.get("observation_isolated") is not True
            or max(
                float(controls.get("official_hidden_max_abs", 1.0)),
                float(controls.get("official_normalized_action_max_abs", 1.0)),
                float(controls.get("official_helper_max_abs", 1.0)),
            ) > 1e-6
            or controls.get("d62_anchor_and_reuse") is not True
        ):
            raise SystemExit("C1H Recovery 01 outcome-free qualification did not pass")
    if shutil.disk_usage(ROOT).free < int(config["forecast"]["project_disk_margin_required_bytes"]):
        raise SystemExit("C1H project disk margin is insufficient")
    initial_gpu = selected_gpu_snapshot(int(physical))
    if initial_gpu["memory_used_mib"] > 1024 or initial_gpu["utilization_percent"] > 5:
        raise SystemExit("C1H selected GPU is not sufficiently idle")
    output_root.mkdir(parents=True)
    runtime_cache = output_root / "runtime-cache"
    runtime_paths = {
        "MPLCONFIGDIR": runtime_cache / "matplotlib",
        "HF_MODULES_CACHE": runtime_cache / "hf-modules",
        "HF_HOME": runtime_cache / "hf-home",
        "TORCH_HOME": runtime_cache / "torch",
        "XDG_CACHE_HOME": runtime_cache / "xdg",
        "TMPDIR": runtime_cache / "tmp",
        "WANDB_DIR": runtime_cache / "wandb",
    }
    for key, path in runtime_paths.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    os.environ.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "WANDB_MODE": "disabled",
            "PYTHONNOUSERSITE": "1",
            "MUJOCO_GL": "osmesa",
            "PYOPENGL_PLATFORM": "osmesa",
        }
    )
    sys.pycache_prefix = str(runtime_cache / "pycache")
    progress_path = output_root / "progress.jsonl"
    started_at = utc_now()
    started = time.monotonic()
    attempts = queries = 0
    peak_aggregate = initial_gpu["memory_used_mib"]
    memory_sampler = AggregateMemorySampler(int(physical), peak_aggregate)
    memory_sampler.start()
    current_stage = "stage1"
    protected_outcomes_opened = False
    caught: BaseException | None = None
    checkpoint = checkpoint_baseline = None
    model = action_head = proprio_projector = processor = None
    restoration: dict[str, Any] | None = None
    evaluation = None
    loader_originals: dict[str, Any] | None = None
    old_handlers: dict[int, Any] = {}

    def interrupted(signum: int, _frame: Any) -> None:
        raise TechnicalStop(f"C1H received signal {signum}")

    for signum in (signal.SIGINT, signal.SIGTERM):
        old_handlers[signum] = signal.signal(signum, interrupted)

    try:
        sys.path.insert(0, str(ROOT / "src"))
        from savr.cac.c1h import cumulative_decision, frozen_schedule, stage1_decision, summarize
        from savr.pair.p3 import P3Profile

        conditions = read_jsonl(ROOT / config["population_manifest"])
        if any(row.get("semantic_sha256") != semantic_sha256(row) for row in conditions):
            raise TechnicalStop("C0 simulator population contains a semantic-hash mismatch")
        schedules = {
            "stage1": frozen_schedule(
                conditions,
                population=config["populations"]["stage1"],
                seed=int(config["schedule"]["seed"]),
            ),
            "extension": frozen_schedule(
                conditions,
                population=config["populations"]["extension"],
                seed=int(config["schedule"]["seed"]) + 1,
            ),
        }
        write_jsonl_once(output_root / "stage1_schedule.jsonl", schedules["stage1"])
        write_jsonl_once(output_root / "extension_schedule.jsonl", schedules["extension"])
        manifest = {
            "schema_version": "cac-c1h-worker-manifest-v1",
            "run_id": config["run_id"],
            "started_at_utc": started_at,
            "config": str(args.config),
            "config_sha256": file_sha256(config_path),
            "physical_gpu_id": int(physical),
            "initial_aggregate_gpu_memory_mib": peak_aggregate,
            "process_id": os.getpid(),
            "outcomes_sealed_until_exact_stage_completion": True,
        }
        manifest["semantic_sha256"] = semantic_sha256(manifest)
        write_once(output_root / "worker_manifest.json", manifest)

        corrected_s6 = config.get("schema_version") in {
            "cac-c1h-corrected-s6-v1", "cac-c1h-corrected-s6-recovery01-v1"
        }
        if corrected_s6:
            evaluation, loader_originals = install_qualified_official_loader()
            source = ROOT / "third_party/openvla-oft"
        else:
            source = ROOT / "third_party/vla-cache/src/openvla-oft"
        os.chdir(source)
        sys.path.insert(0, str(source))
        import numpy as np
        import torch
        from experiments.robot.libero import run_libero_eval as evaluation
        from experiments.robot.openvla_utils import normalize_proprio, prepare_images_for_vla
        from experiments.robot.libero.libero_utils import (
            get_libero_dummy_action,
            get_libero_env,
        )
        from experiments.robot.robot_utils import (
            get_image_resize_size,
            set_seed_everywhere,
        )
        from libero.libero import benchmark
        from savr.acr.v5_d_recovery import capture_checkpoint_baseline, restore_checkpoint_exact
        from savr.openvla.official_semantics import OfficialBoundaryCapture
        from savr.pair.p3_openvla import (
            PhysicalSourceTracker,
            configure_dense,
            configure_profile,
            forward_query,
            ordered_tile_profile_vectorized,
            prepare_query,
            runtime_positions,
        )

        set_seed_everywhere(7)
        checkpoint = ROOT / config["model"]["checkpoint_relative"]
        protected_names = ("config.json", "configuration_prismatic.py", "modeling_prismatic.py")
        stale_backups = sorted(
            path.name
            for path in checkpoint.iterdir()
            if any(
                path.name.startswith(f"{name}.back.")
                or path.name.startswith(f"{name}.backup")
                or path.name == f"{name}.bak"
                for name in protected_names
            )
        )
        if stale_backups:
            raise TechnicalStop(f"C1H checkpoint contains stale loader backups: {stale_backups}")
        for name, expected in config["model"]["checkpoint_metadata_sha256"].items():
            if file_sha256(checkpoint / name) != expected:
                raise TechnicalStop(f"C1H checkpoint metadata changed: {name}")
        checkpoint_baseline = capture_checkpoint_baseline(checkpoint, protected_names)
        cfg = base_config(evaluation, checkpoint, output_root)
        evaluation.validate_config(cfg)
        model, action_head, proprio_projector, _noisy, processor = evaluation.initialize_model(cfg)
        model.eval()
        if torch.cuda.device_count() != 1 or action_head is None or proprio_projector is None:
            raise TechnicalStop("C1H model/device initialization contract failed")
        normalization_keys = {
            suite: suite if suite in model.norm_stats else f"{suite}_no_noops"
            for suite in config["suites"]
        }
        if any(key not in model.norm_stats for key in normalization_keys.values()):
            raise TechnicalStop("C1H checkpoint lacks a required suite normalization key")
        profile_config = json.loads((ROOT / config["profile_config"]).read_text(encoding="utf-8"))
        profile_source_id = (
            config["cache_substrate"].get("historical_profile_id", "D62_BAL_PT1")
        )
        profile = P3Profile.from_mapping(next(
            row for row in profile_config["profiles"] if row["profile_id"] == profile_source_id
        ))
        resize_size = get_image_resize_size(cfg)
        suites = {name: benchmark.get_benchmark_dict()[name]() for name in config["suites"]}
        suite_task_indices = {
            name: {suites[name].get_task(index).name: index for index in range(10)}
            for name in config["suites"]
        }
        if any(len(indices) != 10 for indices in suite_task_indices.values()):
            raise TechnicalStop("LIBERO suite task identities are not one-to-one")
        max_steps_expected = config["model"]["suite_max_steps"]
        if {name: int(evaluation.TASK_MAX_STEPS[name]) for name in config["suites"]} != max_steps_expected:
            raise TechnicalStop("official LIBERO suite horizons changed")
        cap_mib = int(config["resource_caps"]["peak_gpu_memory_mib_strict_max"])

        def guard(*, check_artifact: bool = False, check_memory: bool = False) -> None:
            nonlocal peak_aggregate
            if time.monotonic() - started >= int(config["resource_caps"]["wall_seconds"]):
                raise TechnicalStop("C1H wall-time cap reached")
            if attempts > int(config["resource_caps"]["episode_attempts"]):
                raise TechnicalStop("C1H episode cap exceeded")
            if queries > int(config["resource_caps"]["model_queries"]):
                raise TechnicalStop("C1H query cap exceeded")
            if check_artifact and directory_size(output_root) >= int(
                config["resource_caps"]["artifact_bytes"]
            ):
                raise TechnicalStop("C1H artifact cap reached")
            if memory_sampler.error is not None:
                raise TechnicalStop("C1H aggregate-memory sampler failed") from memory_sampler.error
            if check_memory:
                reserved = initial_gpu["memory_used_mib"] + int(
                    torch.cuda.max_memory_reserved() / (1024 * 1024)
                )
                peak_aggregate = max(peak_aggregate, memory_sampler.snapshot(), reserved)
                if peak_aggregate >= cap_mib:
                    raise TechnicalStop("C1H strict selected-GPU memory boundary reached")

        @torch.inference_mode()
        def policy_query(
            policy_observation: Mapping[str, Any],
            instruction: str,
            *,
            cache: Any = None,
            tracker: Any = None,
            anchor_salience: Any = None,
            ordinal: int = 0,
            anchor: bool = False,
            capture_cac: bool = False,
        ) -> dict[str, Any]:
            nonlocal queries
            prepared = prepare_query(
                torch_module=torch,
                np=np,
                model=model,
                processor=processor,
                proprio_projector=proprio_projector,
                prepare_images=prepare_images_for_vla,
                normalize_proprio=normalize_proprio,
                cfg=cfg,
                raw_scene=policy_observation["full_image"],
                raw_wrist=policy_observation["wrist_image"],
                raw_state=policy_observation["state"],
                instruction=instruction,
            )
            salience = groups = None
            if tracker is None:
                configure_dense(model)
            else:
                tracker.add_record(ordinal, prepared)
                ordered, proportions, groups, _metadata = ordered_tile_profile_vectorized(
                    current=prepared,
                    tracker=tracker,
                    profile=profile,
                    previous_salience=anchor_salience,
                    torch_module=torch,
                )
                configure_profile(model, ordered, proportions, torch)
            if queries >= int(config["resource_caps"]["model_queries"]):
                raise TechnicalStop("C1H query cap exhausted before model invocation")
            queries += 1
            result = forward_query(
                torch_module=torch,
                np=np,
                model=model,
                action_head=action_head,
                cfg=cfg,
                prepared=prepared,
                past_key_values=cache,
                capture_layers=config["model"]["sidecar_layers"] if anchor else (),
                capture_cac=capture_cac,
                return_actions=True,
            )
            guard(check_memory=True)
            if anchor:
                positions = runtime_positions(prepared, torch)
                salience = result["tap"].salience(
                    instruction_positions=positions["instruction"],
                    action_positions=positions["action"],
                    visual_positions=positions["primary"] + positions["wrist"],
                )
            if tracker is not None:
                tracker.advance(ordinal, groups)
            return build_policy_query_output(
                np, result, prepared, salience, require_cac=capture_cac
            )

        def run_pre_episode_controls() -> dict[str, Any]:
            """Exercise the exact simulator-to-action path before any terminal episode."""

            nonlocal queries
            item = schedules["stage1"][0]
            suite_name = str(item["suite"])
            local_task = suite_task_indices[suite_name][str(item["task_id"])]
            task_suite = suites[suite_name]
            task = task_suite.get_task(local_task)
            cfg.task_suite_name = suite_name
            cfg.unnorm_key = normalization_keys[suite_name]
            set_seed_everywhere(int(item["seed"]))
            environment, instruction = get_libero_env(
                task, cfg.model_family, resolution=cfg.env_img_res
            )
            try:
                environment.reset()
                states = task_suite.get_task_init_states(local_task)
                observation = environment.set_init_state(
                    states[int(item["initial_state_id"])].copy()
                )
                for _ in range(cfg.num_steps_wait):
                    observation, _, _, _ = environment.step(
                        get_libero_dummy_action(cfg.model_family)
                    )
                policy_observation, _ = evaluation.prepare_observation(observation, resize_size)
                official_observation = official_control_observation(np, policy_observation)
                configure_dense(model)
                if queries >= int(config["resource_caps"]["model_queries"]):
                    raise TechnicalStop("C1H query cap exhausted before parity control")
                queries += 1
                source_before = {
                    key: np.asarray(policy_observation[key]).copy()
                    for key in ("full_image", "wrist_image", "state")
                }
                with torch.inference_mode(), OfficialBoundaryCapture(model, action_head) as capture:
                    official_result = evaluation.get_action(
                        cfg,
                        model,
                        official_observation,
                        instruction,
                        processor=processor,
                        action_head=action_head,
                        proprio_projector=proprio_projector,
                        noisy_action_projector=_noisy,
                        use_film=False,
                    )
                official, official_cache, _official_images, _official_metrics = (
                    unpack_official_action_result(np, official_result)
                )
                official_values = capture.exact_values()
                del official_result, official_cache, _official_images, _official_metrics
                guard(check_memory=True)
                helper = policy_query(
                    policy_observation, instruction, capture_cac=corrected_s6
                )
                helper_actions = helper["actions"]
                if official.shape != (8, 7) or helper_actions.shape != (8, 7):
                    raise TechnicalStop("C1H parity-control action layout changed")
                maximum = float(np.max(np.abs(official - helper_actions)))
                if not np.isfinite(maximum) or maximum > 1e-6:
                    raise TechnicalStop(
                        f"C1H helper differs from official dense inference: {maximum}"
                    )
                hidden_maximum = normalized_maximum = 0.0
                if corrected_s6:
                    hidden_maximum = float(
                        (
                            official_values["action_hidden"].float()
                            - helper["action_hidden"].float()
                        ).abs().max().item()
                    )
                    normalized_maximum = float(
                        (
                            official_values["normalized_actions"].reshape(1, 8, 7).float()
                            - helper["base_action"].float()
                        ).abs().max().item()
                    )
                    if hidden_maximum > 1e-6 or normalized_maximum > 1e-6:
                        raise TechnicalStop(
                            "C1H corrected hidden/action parity control failed: "
                            f"hidden={hidden_maximum}, normalized={normalized_maximum}"
                        )
                observation_isolated = all(
                    np.array_equal(policy_observation[key], source_before[key])
                    for key in source_before
                )
                if not observation_isolated:
                    raise TechnicalStop("C1H official parity mutated the policy observation")
                processed = evaluation.process_action(helper_actions[0], cfg.model_family)
                if np.asarray(processed).shape != (7,) or not np.isfinite(processed).all():
                    raise TechnicalStop("C1H action-processing control failed")
                d62_anchor = policy_query(policy_observation, instruction, anchor=True)
                d62_tracker = PhysicalSourceTracker(0, d62_anchor["prepared"])
                d62_reuse = policy_query(
                    policy_observation,
                    instruction,
                    cache=d62_anchor["cache"],
                    tracker=d62_tracker,
                    anchor_salience=d62_anchor["salience"],
                    ordinal=1,
                )
                if (
                    d62_anchor["actions"].shape != (8, 7)
                    or d62_reuse["actions"].shape != (8, 7)
                    or not np.isfinite(d62_anchor["actions"]).all()
                    or not np.isfinite(d62_reuse["actions"]).all()
                    or d62_anchor["cache"] is None
                    or d62_reuse["cache"] is None
                ):
                    raise TechnicalStop("C1H D62 entry control failed")
                configure_dense(model)
                del helper, d62_anchor, d62_reuse, d62_tracker
                gc.collect()
                torch.cuda.empty_cache()
                return {
                    "queries": 4,
                    "official_helper_max_abs": maximum,
                    "official_hidden_max_abs": hidden_maximum,
                    "official_normalized_action_max_abs": normalized_maximum,
                    "independent_official_oracle": corrected_s6,
                    "observation_isolated": observation_isolated,
                    "official_observation_contract": "current images copied into prev_images",
                    "official_return_contract": "actions-cache-images-metrics",
                    "d62_anchor_and_reuse": True,
                    "simulator_setup": True,
                    "observation_shapes": {
                        "scene": list(np.asarray(policy_observation["full_image"]).shape),
                        "wrist": list(np.asarray(policy_observation["wrist_image"]).shape),
                        "state": list(np.asarray(policy_observation["state"]).shape),
                    },
                    "terminal_outcome_access": False,
                }
            finally:
                environment.close()

        def run_episode(item: Mapping[str, Any]) -> dict[str, Any]:
            nonlocal attempts
            guard(check_artifact=True, check_memory=True)
            if attempts >= int(config["resource_caps"]["episode_attempts"]):
                raise TechnicalStop("C1H episode cap exhausted before scheduling")
            attempts += 1
            suite_name = str(item["suite"])
            cfg.task_suite_name = suite_name
            cfg.unnorm_key = normalization_keys[suite_name]
            set_seed_everywhere(int(item["seed"]))
            try:
                local_task = suite_task_indices[suite_name][str(item["task_id"])]
            except KeyError as error:
                raise TechnicalStop("C1H C0 task identity is absent from its LIBERO suite") from error
            task_suite = suites[suite_name]
            task = task_suite.get_task(local_task)
            if task.name != item["task_id"]:
                raise TechnicalStop("C1H task identity does not match the C0 manifest")
            states = task_suite.get_task_init_states(local_task)
            state_id = int(item["initial_state_id"])
            environment, instruction = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)
            episode_started = time.perf_counter()
            episode_queries = control_steps = 0
            success = False
            try:
                environment.reset()
                observation = environment.set_init_state(states[state_id].copy())
                queue: deque[Any] = deque(maxlen=cfg.num_open_loop_steps)
                environment_step = 0
                cache = tracker = anchor_salience = None
                cached_intervals = 0
                while environment_step < max_steps_expected[suite_name] + cfg.num_steps_wait:
                    guard()
                    if environment_step < cfg.num_steps_wait:
                        observation, _, _, _ = environment.step(
                            get_libero_dummy_action(cfg.model_family)
                        )
                        environment_step += 1
                        continue
                    if not queue:
                        prepared_observation, _ = evaluation.prepare_observation(observation, resize_size)
                        if item["arm"] == "dense":
                            result = policy_query(prepared_observation, instruction)
                        elif cache is None:
                            result = policy_query(
                                prepared_observation, instruction, anchor=True
                            )
                            cache = result["cache"]
                            tracker = PhysicalSourceTracker(0, result["prepared"])
                            anchor_salience = result["salience"]
                            cached_intervals = 0
                        elif cached_intervals >= 4:
                            cache = tracker = anchor_salience = None
                            gc.collect()
                            torch.cuda.empty_cache()
                            result = policy_query(
                                prepared_observation, instruction, anchor=True
                            )
                            cache = result["cache"]
                            tracker = PhysicalSourceTracker(0, result["prepared"])
                            anchor_salience = result["salience"]
                            cached_intervals = 0
                        else:
                            result = policy_query(
                                prepared_observation,
                                instruction,
                                cache=cache,
                                tracker=tracker,
                                anchor_salience=anchor_salience,
                                ordinal=cached_intervals + 1,
                            )
                            cache = result["cache"]
                            cached_intervals += 1
                        actions = result["actions"]
                        if actions.shape != (8, 7) or not np.isfinite(actions).all():
                            raise TechnicalStop("C1H produced an invalid action chunk")
                        queue.extend(actions)
                        episode_queries += 1
                        if item["arm"] == "dense":
                            del result
                    action = evaluation.process_action(queue.popleft(), cfg.model_family)
                    if not np.isfinite(action).all():
                        raise TechnicalStop("C1H processed action is nonfinite")
                    observation, _, done, _ = environment.step(action.tolist())
                    control_steps += 1
                    if done:
                        success = True
                        break
                    environment_step += 1
            finally:
                environment.close()
            duration = time.perf_counter() - episode_started
            record = {
                "schema_version": "cac-c1h-terminal-episode-v1",
                "run_id": config["run_id"],
                "stage": current_stage,
                **dict(item),
                "suite_task_index": local_task,
                "status": "completed",
                "success": success,
                "query_count": episode_queries,
                "control_steps": control_steps,
                "wall_seconds": duration,
            }
            record["semantic_sha256"] = semantic_sha256(record)
            append_progress(
                progress_path,
                {
                    "schema_version": "cac-c1h-progress-v1",
                    "run_id": config["run_id"],
                    "stage": current_stage,
                    "schedule_id": item["schedule_id"],
                    "terminal_record_number": attempts,
                    "query_count_cumulative": queries,
                    "artifact_bytes": directory_size(output_root),
                    "elapsed_seconds": time.monotonic() - started,
                    "recorded_at_utc": utc_now(),
                },
            )
            guard(check_artifact=True, check_memory=True)
            return record

        def run_stage(name: str, schedule: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
            nonlocal current_stage, protected_outcomes_opened
            current_stage = name
            rows = [run_episode(item) for item in schedule]
            if len(rows) != 240:
                raise TechnicalStop(f"{name} did not complete exactly 240 episodes")
            write_jsonl_once(output_root / f"{name}_terminal_records.jsonl", rows)
            protected_outcomes_opened = True
            return rows

        technical_controls = run_pre_episode_controls()
        if queries != int(config["query_accounting"]["technical_control_queries"]):
            raise TechnicalStop("C1H technical-control query accounting changed")
        if args.qualification_only:
            guard(check_artifact=True, check_memory=True)
            model = action_head = proprio_projector = processor = _noisy = None
            gc.collect()
            torch.cuda.empty_cache()
            if checkpoint is None or checkpoint_baseline is None:
                raise TechnicalStop("C1H qualification restoration baseline is unavailable")
            restoration = restore_checkpoint_exact(checkpoint, checkpoint_baseline)
            if not all(
                restoration[key]
                for key in ("protected_bytes_restored", "backup_cleanup_complete", "inventory_equal")
            ):
                raise TechnicalStop("C1H qualification checkpoint restoration failed")
            peak_aggregate = max(peak_aggregate, memory_sampler.close())
            qualification_summary = {
                "schema_version": "cac-c1h-pre-episode-qualification-v1",
                "run_id": config["run_id"],
                "mode": "outcome_free_pre_episode_qualification",
                "complete": True,
                "passed": True,
                "model_queries": queries,
                "terminal_episode_attempts": attempts,
                "terminal_outcomes_opened": False,
                "technical_controls": technical_controls,
                "peak_aggregate_gpu_memory_mib": peak_aggregate,
                "checkpoint_restoration": restoration,
                "completed_at_utc": utc_now(),
                "automatic_retry": False,
                "next_step_authorized": False,
            }
            qualification_summary["semantic_sha256"] = semantic_sha256(
                qualification_summary
            )
            write_once(output_root / "worker_summary.json", qualification_summary)
            return 0
        stage1_rows = run_stage("stage1", schedules["stage1"])
        stage1_summary = summarize(stage1_rows, expected_conditions=120)
        decision = stage1_decision(stage1_summary)
        stage1_report = {
            "schema_version": "cac-c1h-stage-summary-v1",
            "stage": "stage1",
            "summary": stage1_summary,
            "decision": decision,
            "terminal_records_sha256": file_sha256(output_root / "stage1_terminal_records.jsonl"),
        }
        stage1_report["semantic_sha256"] = semantic_sha256(stage1_report)
        write_once(output_root / "stage1_summary.json", stage1_report)

        all_rows = list(stage1_rows)
        extension_opened = decision == "extend"
        if extension_opened:
            extension_rows = run_stage("extension", schedules["extension"])
            all_rows.extend(extension_rows)
            cumulative = summarize(all_rows, expected_conditions=240)
            decision = cumulative_decision(cumulative)
            extension_report = {
                "schema_version": "cac-c1h-stage-summary-v1",
                "stage": "extension",
                "summary": cumulative,
                "decision": decision,
                "terminal_records_sha256": file_sha256(
                    output_root / "extension_terminal_records.jsonl"
                ),
            }
            extension_report["semantic_sha256"] = semantic_sha256(extension_report)
            write_once(output_root / "extension_summary.json", extension_report)
        else:
            cumulative = stage1_summary

        guard(check_artifact=True, check_memory=True)
        model = action_head = proprio_projector = processor = _noisy = None
        gc.collect()
        torch.cuda.empty_cache()
        if checkpoint is None or checkpoint_baseline is None:
            raise TechnicalStop("C1H checkpoint restoration baseline is unavailable")
        restoration = restore_checkpoint_exact(checkpoint, checkpoint_baseline)
        if not all(
            restoration[key]
            for key in ("protected_bytes_restored", "backup_cleanup_complete", "inventory_equal")
        ):
            raise TechnicalStop("C1H checkpoint restoration failed")
        peak_aggregate = max(peak_aggregate, memory_sampler.close())
        summary = {
            "schema_version": "cac-c1h-worker-summary-v1",
            "run_id": config["run_id"],
            "status": "completed",
            "gate_h_decision": decision,
            "extension_opened": extension_opened,
            "terminal_episode_records": len(all_rows),
            "conditions": len(all_rows) // 2,
            "model_queries": queries,
            "technical_controls": technical_controls,
            "peak_aggregate_gpu_memory_mib": peak_aggregate,
            "elapsed_seconds": time.monotonic() - started,
            "completed_at_utc": utc_now(),
            "cumulative_summary": cumulative,
            "stage1_terminal_records_sha256": file_sha256(
                output_root / "stage1_terminal_records.jsonl"
            ),
            "extension_terminal_records_sha256": (
                file_sha256(output_root / "extension_terminal_records.jsonl")
                if extension_opened
                else None
            ),
            "c2_started": False,
            "checkpoint_restoration": restoration,
            "outcomes_opened_only_after_exact_stage_completion": True,
        }
        summary["semantic_sha256"] = semantic_sha256(summary)
        write_once(output_root / "worker_summary.json", summary)
        return 0
    except BaseException as error:
        caught = error
        restoration_error = None
        try:
            model = action_head = proprio_projector = processor = None
            gc.collect()
            if "torch" in locals():
                torch.cuda.empty_cache()
            if checkpoint is not None and checkpoint_baseline is not None:
                restoration = restore_checkpoint_exact(checkpoint, checkpoint_baseline)
        except BaseException as restore_error:
            restoration_error = f"{type(restore_error).__name__}: {restore_error}"
        peak_aggregate = max(peak_aggregate, memory_sampler.close())
        stop = {
            "schema_version": "cac-c1h-technical-stop-v1",
            "run_id": config.get("run_id"),
            "status": "technical_stop",
            "stage": current_stage,
            "error_type": type(error).__name__,
            "error": str(error),
            "episode_attempts": attempts,
            "model_queries": queries,
            "progress_records": (
                len(progress_path.read_text(encoding="utf-8").splitlines())
                if progress_path.is_file()
                else 0
            ),
            "partial_outcomes_opened": protected_outcomes_opened,
            "automatic_retry": False,
            "checkpoint_restoration": restoration,
            "checkpoint_restoration_error": restoration_error,
            "peak_aggregate_gpu_memory_mib": peak_aggregate,
            "recorded_at_utc": utc_now(),
        }
        stop["semantic_sha256"] = semantic_sha256(stop)
        try:
            write_once(output_root / "technical_stop.json", stop)
        except FileExistsError:
            pass
        try:
            write_text_once(
                output_root / "technical_traceback.log",
                "".join(traceback.format_exception(type(error), error, error.__traceback__)),
            )
        except FileExistsError:
            pass
        return 1
    finally:
        peak_aggregate = max(peak_aggregate, memory_sampler.close())
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
        if evaluation is not None and loader_originals is not None:
            for name, value in loader_originals.items():
                setattr(evaluation, name, value)
        if caught is not None:
            print(f"C1H technical stop: {type(caught).__name__}: {caught}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
