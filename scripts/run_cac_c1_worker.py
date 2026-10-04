#!/usr/bin/env python3
"""Run the single bounded CAC C1 tensor/systems feasibility worker."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import random
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ROOT = Path("/home/ved/SAVR")
S3_RECOVERY05_WORKER = EXPECTED_ROOT / "scripts/run_openvla_semantic_parity_recovery05.py"


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path.name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_s3_recovery05() -> Any:
    return load_module(S3_RECOVERY05_WORKER, "s3_recovery05_for_cac_c1_s5")


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def semantic_sha256(value: Mapping[str, Any]) -> str:
    body = dict(value)
    body.pop("semantic_sha256", None)
    return hashlib.sha256(canonical_bytes(body)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_once(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(canonical_bytes(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def write_bytes_once(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def aggregate_memory_mib(gpu: int) -> int:
    output = subprocess.check_output(
        ["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        text=True,
    ).strip()
    return int(output)


class AggregateMemorySampler:
    """Sample only selected-GPU aggregate memory; never inspect other processes."""

    def __init__(self, gpu: int, initial: int) -> None:
        self.gpu = gpu
        self.peak = initial
        self.error: BaseException | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, name="cac-gpu-memory", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(0.20):
            try:
                value = aggregate_memory_mib(self.gpu)
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


def base_config(eval_module: Any, checkpoint: Path, output_root: Path) -> Any:
    return eval_module.GenerateConfig(
        pretrained_checkpoint=str(checkpoint), task_suite_name="libero_object",
        num_trials_per_task=0, seed=0, local_log_dir=str(output_root / "logs"),
        use_wandb=False, center_crop=True, num_open_loop_steps=8, num_images_in_input=2,
        use_proprio=True, use_l1_regression=True, use_diffusion=False, use_film=False,
        use_vla_cache=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/cac/c1_tensor_feasibility_v1.json"))
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if ROOT != EXPECTED_ROOT:
        raise SystemExit(f"CAC C1 refuses to run outside {EXPECTED_ROOT}")
    output_root = args.output_root.resolve()
    if not output_root.is_relative_to(ROOT / "results") or output_root.exists():
        raise SystemExit("CAC C1 requires a new immutable results root below the project")
    project_libero = ROOT / "configs/pair/libero_runtime"
    if not (project_libero / "config.yaml").is_file():
        raise SystemExit("project-local LIBERO configuration is missing")
    os.environ["LIBERO_CONFIG_PATH"] = str(project_libero)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    physical_gpu = os.environ.get("CAC_PHYSICAL_GPU_ID", "")
    if not visible or "," in visible or not physical_gpu.isdigit():
        raise SystemExit("CAC C1 requires one explicitly selected GPU")
    sys.path.insert(0, str(ROOT / "src"))

    config_path = (ROOT / args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("semantic_sha256") != semantic_sha256(config):
        raise SystemExit("CAC C1 configuration semantic hash mismatch")
    schema = config.get("schema_version")
    s5_requalification = schema == "cac-c1-requalification-s5-v1"
    if schema not in {"cac-c1-tensor-feasibility-v1", "cac-c1-requalification-s5-v1"}:
        raise SystemExit("CAC C1 configuration schema changed")
    recovery = config.get("recovery")
    if s5_requalification:
        if config.get("authorization") != {
            "s5_c1_requalification_authorized": True,
            "source": "User approved S5/CAC C1 requalification on 2026-09-01",
            "simulator": False,
            "terminal_outcomes": False,
            "expert_actions": False,
            "training": False,
            "automatic_retry": False,
        }:
            raise SystemExit("S5/CAC C1 requalification is not explicitly authorized")
        if output_root != (ROOT / config["output_root"]).resolve():
            raise SystemExit("S5/CAC C1 output root changed")
        s4_path = ROOT / config["s4_parent"]["summary"]
        if not s4_path.is_file() or file_sha256(s4_path) != config["s4_parent"]["summary_sha256"]:
            raise SystemExit("S5/CAC C1 S4 parent changed")
        s4 = json.loads(s4_path.read_text(encoding="utf-8"))
        if (
            s4.get("semantic_sha256") != semantic_sha256(s4)
            or s4.get("semantic_sha256") != config["s4_parent"]["semantic_sha256"]
            or s4.get("complete") is not True
            or s4.get("passed") is not True
            or s4.get("corrected_substrate_id") != "D62_BAL_PT1_S4C_V1"
            or s4.get("model_calls") != 37
        ):
            raise SystemExit("S5/CAC C1 S4 parent did not pass")
        s3_path = ROOT / config["s3_parent"]["summary"]
        if not s3_path.is_file() or file_sha256(s3_path) != config["s3_parent"]["summary_sha256"]:
            raise SystemExit("S5/CAC C1 S3 parent changed")
        s3 = json.loads(s3_path.read_text(encoding="utf-8"))
        if (
            s3.get("semantic_sha256") != semantic_sha256(s3)
            or s3.get("semantic_sha256") != config["s3_parent"]["semantic_sha256"]
            or s3.get("complete") is not True
            or s3.get("passed") is not True
            or s3.get("model_calls") != 32
            or s3.get("comparison_count") != 344
        ):
            raise SystemExit("S5/CAC C1 S3 parent did not pass")
        if config.get("substrate") != {
            "corrected_id": "D62_BAL_PT1_S4C_V1",
            "historical_profile_id": "D62_BAL_PT1",
            "historical_profile_values_reused": True,
            "s4_selection_materially_changed": False,
            "s4_selection_changed_observations": 0,
            "action_readout_corrected": True,
            "compact_positions_explicitly_mapped": True,
        }:
            raise SystemExit("S5/CAC C1 corrected substrate contract changed")
        if config.get("official_parity") != {
            "independent_released_evaluator": True,
            "reversible_action_head_capture": True,
            "observation_index": 2,
            "hidden_tolerance": 0.000001,
            "normalized_action_tolerance": 0.000001,
            "unnormalized_action_tolerance": 0.000001,
            "raw_values_persisted": False,
            "must_pass_before_internal_controls": True,
        }:
            raise SystemExit("S5/CAC C1 official parity contract changed")
        if config.get("advance") != {
            "next_phase": "C1H", "authorized": False, "stop_before_next_phase": True
        }:
            raise SystemExit("S5/CAC C1 advance boundary changed")
    elif recovery is None:
        if config.get("authorization") != "User explicitly approved C1 on 2026-08-31":
            raise SystemExit("CAC C1 is not explicitly authorized")
    else:
        if config.get("authorization") != "User explicitly approved C1 Recovery 01 on 2026-08-31":
            raise SystemExit("CAC C1 Recovery 01 is not explicitly authorized")
        if output_root != (ROOT / recovery["output_root"]).resolve():
            raise SystemExit("CAC C1 recovery output root changed")
        if not (ROOT / config["source_attempt"] / "technical_stop.json").is_file():
            raise SystemExit("CAC C1 source technical stop is missing")
    for relative, expected in config["authenticated_files"].items():
        if file_sha256(ROOT / relative) != expected:
            raise SystemExit(f"CAC C1 authenticated input changed: {relative}")
    caps = config["resource_caps"]
    expected_planned_calls = 97 if s5_requalification else 95
    if caps != {
        "gpu_count": 1, "model_processes": 1, "model_call_hard_cap": 160,
        "planned_model_calls": expected_planned_calls, "temporary_bytes": 4294967296,
        "peak_aggregate_memory_mib_strict_max": 23552, "downloads": 0,
        "training": False, "simulator_outcomes": 0, "protected_outcomes": False,
        "automatic_retry": False, "wall_seconds": 3600,
    }:
        raise SystemExit("CAC C1 resource boundary changed")
    schedule = config["schedule"]
    if (
        sum(int(schedule[key]) for key in (
            "warmup_calls", "sidecar_and_allfresh_controls", "hook_control_calls",
            "recursive_repeat_and_reset_calls", "isolation_calls", "timing_calls",
        )) + int(schedule.get("official_parity_calls", 0)) != expected_planned_calls
        or int(schedule["timing_calls"]) != sum(
            2 * (horizon + 1) * int(schedule["timing_repetitions_per_horizon"])
            for horizon in schedule["timing_horizons"]
        )
        or int(schedule["total_calls"]) != expected_planned_calls
        or int(schedule.get("official_parity_calls", 0)) != (2 if s5_requalification else 0)
    ):
        raise SystemExit("CAC C1 call schedule changed")
    if shutil.disk_usage(ROOT).free < 8 * 1024**3:
        raise SystemExit("CAC C1 requires at least 8 GiB free project storage")
    initial_aggregate = aggregate_memory_mib(int(physical_gpu))
    if initial_aggregate > 1024:
        raise SystemExit("CAC C1 selected GPU is not sufficiently idle")
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
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    sys.pycache_prefix = str(runtime_cache / "pycache")
    started = time.monotonic()
    calls = 0
    peak_aggregate = initial_aggregate
    memory_sampler = AggregateMemorySampler(int(physical_gpu), peak_aggregate)
    memory_sampler.start()
    checkpoint = None
    checkpoint_baseline = None
    restore_checkpoint_exact = None
    evaluation = None
    loader_originals = None

    def timeout_handler(_signum: int, _frame: Any) -> None:
        raise TimeoutError("CAC C1 wall-time boundary reached")

    signal.signal(signal.SIGTERM, timeout_handler)

    def consume() -> None:
        nonlocal calls
        calls += 1
        if calls > int(caps["model_call_hard_cap"]):
            raise RuntimeError("CAC C1 model-call cap exceeded")
        if time.monotonic() - started >= int(caps["wall_seconds"]):
            raise TimeoutError("CAC C1 wall-time boundary reached")

    try:
        if s5_requalification:
            recovery05 = load_s3_recovery05()
            recovery04 = recovery05.load_recovery04()
            recovery03 = recovery04.load_recovery03()
            recovery02 = recovery03.load_recovery02()
            recovery01 = recovery02.load_recovery01()
            evaluation, loader_originals = recovery01.install_official_loader_guard()
            source = ROOT / "third_party/openvla-oft"
        else:
            source = ROOT / "third_party/vla-cache/src/openvla-oft"
        os.chdir(source)
        sys.path.insert(0, str(source))
        import h5py
        import numpy as np
        import torch
        from experiments.robot.libero import run_libero_eval as evaluation
        from experiments.robot.openvla_utils import normalize_proprio, prepare_images_for_vla
        from experiments.robot.robot_utils import set_seed_everywhere
        from savr.acr.v5_d_recovery import capture_checkpoint_baseline, restore_checkpoint_exact
        from savr.cac.c1 import (
            ALIGNED_RECORD_BYTES, BoundedRecordWriter, build_full_adapter,
            cache_digest, extract_cac_features, serialize_numeric_record, serialize_sidecar,
            tensor_sha256,
        )
        from savr.openvla.official_semantics import OfficialBoundaryCapture
        from savr.pair.interventions import clone_runtime_cache
        from savr.pair.p3 import P3Profile
        from savr.pair.p3_openvla import (
            PhysicalSourceTracker, configure_dense, configure_profile,
            forward_query, ordered_tile_profile_vectorized, prepare_query, runtime_positions,
        )

        set_seed_everywhere(int(config["seed"]))
        model_config = json.loads((ROOT / config["model_config"]).read_text())
        inputs_manifest = json.loads((ROOT / config["input_manifest"]).read_text())
        inputs = inputs_manifest["inputs"]
        checkpoint = ROOT / model_config["model"]["checkpoint_relative"]
        protected_names = ("config.json", "configuration_prismatic.py", "modeling_prismatic.py")
        unexpected_backups = sorted(
            path.name for path in checkpoint.iterdir()
            if any(
                path.name.startswith(f"{name}.back.")
                or path.name.startswith(f"{name}.backup")
                or path.name == f"{name}.bak"
                for name in protected_names
            )
        )
        if unexpected_backups:
            raise RuntimeError(f"CAC C1 checkpoint contains stale loader backups: {unexpected_backups}")
        for name, expected in config["checkpoint_metadata_sha256"].items():
            if file_sha256(checkpoint / name) != expected:
                raise RuntimeError(f"CAC C1 checkpoint metadata changed: {name}")
        checkpoint_baseline = capture_checkpoint_baseline(
            checkpoint, protected_names
        )
        cfg = base_config(evaluation, checkpoint, output_root)
        evaluation.validate_config(cfg)
        model, action_head, proprio_projector, noisy, processor = evaluation.initialize_model(cfg)
        model.eval()
        if torch.cuda.device_count() != 1:
            raise RuntimeError("CAC C1 can see more or fewer than one GPU")
        profile_source_id = (
            config["substrate"]["historical_profile_id"]
            if s5_requalification else "D62_BAL_PT1"
        )
        profile = P3Profile.from_mapping(next(
            value for value in model_config["profiles"] if value["profile_id"] == profile_source_id
        ))

        def check_memory() -> None:
            nonlocal peak_aggregate
            aggregate = aggregate_memory_mib(int(physical_gpu))
            reserved_upper = initial_aggregate + int(
                torch.cuda.max_memory_reserved() / (1024 * 1024)
            )
            peak_aggregate = max(
                peak_aggregate, aggregate, memory_sampler.snapshot(), reserved_upper
            )
            if memory_sampler.error is not None:
                raise RuntimeError("CAC C1 aggregate-memory sampler failed") from memory_sampler.error
            if peak_aggregate >= int(caps["peak_aggregate_memory_mib_strict_max"]):
                raise RuntimeError("CAC C1 reached the strict aggregate-memory boundary")

        data_root = ROOT / inputs_manifest["data_root_relative"]

        def load_window(identity: Mapping[str, Any], horizon: int, start: int = 0) -> list[dict[str, Any]]:
            values = []
            source_path = data_root / identity["source_path"]
            if file_sha256(source_path) != identity["source_sha256"]:
                raise RuntimeError("CAC C1 selected observation source changed")
            with h5py.File(source_path, "r") as handle:
                group = handle[f"data/{identity['original_trajectory_id']}/obs"]
                for ordinal in range(horizon + 1):
                    step = (start + ordinal) * 8
                    if step >= int(identity["step_count"]):
                        raise RuntimeError("CAC C1 observation chronology exceeded trajectory")
                    scene = np.asarray(group["agentview_rgb"][step])
                    wrist = np.asarray(group["eye_in_hand_rgb"][step])
                    state = np.concatenate((np.asarray(group["ee_states"][step]), np.asarray(group["gripper_states"][step])))
                    if state.shape != (8,):
                        raise RuntimeError("CAC C1 proprioception layout changed")
                    values.append({"scene": scene, "wrist": wrist, "state": state, "step": step})
            if any(right["step"] - left["step"] != 8 for left, right in zip(values, values[1:])):
                raise RuntimeError("CAC C1 eight-action cadence changed")
            return values

        @torch.inference_mode()
        def prepare(identity: Mapping[str, Any], observation: Mapping[str, Any]) -> Any:
            cfg.unnorm_key = identity["normalization_statistics_key"]
            value = prepare_query(
                torch_module=torch, np=np, model=model, processor=processor,
                proprio_projector=proprio_projector, prepare_images=prepare_images_for_vla,
                normalize_proprio=normalize_proprio, cfg=cfg, raw_scene=observation["scene"],
                raw_wrist=observation["wrist"], raw_state=observation["state"],
                instruction=identity["language_instruction"],
            )
            check_memory()
            return value

        @torch.inference_mode()
        def infer(prepared: Any, *, cache: Any = None, capture_cac: bool = False,
                  sidecar: bool = False, return_actions: bool = False) -> dict[str, Any]:
            consume()
            result = forward_query(
                torch_module=torch, np=np, model=model, action_head=action_head, cfg=cfg,
                prepared=prepared, past_key_values=cache,
                capture_layers=model_config["model"]["sidecar_layers"] if sidecar else (),
                capture_cac=capture_cac,
                return_actions=return_actions,
            )
            check_memory()
            return result

        def salience(prepared: Any, result: Mapping[str, Any]) -> Any:
            positions = runtime_positions(prepared, torch)
            return result["tap"].salience(
                instruction_positions=positions["instruction"], action_positions=positions["action"],
                visual_positions=positions["primary"] + positions["wrist"],
            )

        identity = inputs[2]
        observations = load_window(identity, 6)
        prepared = [prepare(identity, item) for item in observations]

        official_parity = None
        if s5_requalification:
            cfg.unnorm_key = identity["normalization_statistics_key"]
            configure_dense(model)
            raw = observations[0]
            observation = {
                "full_image": raw["scene"].copy(),
                "wrist_image": raw["wrist"].copy(),
                "state": raw["state"].copy(),
                "prev_images": [raw["scene"].copy(), raw["wrist"].copy()],
            }
            image_before = (
                hashlib.sha256(observation["full_image"].tobytes()).hexdigest(),
                hashlib.sha256(observation["wrist_image"].tobytes()).hexdigest(),
            )
            consume()
            with torch.inference_mode(), OfficialBoundaryCapture(model, action_head) as capture:
                official_result = evaluation.get_action(
                    cfg,
                    model,
                    observation,
                    identity["language_instruction"],
                    processor=processor,
                    action_head=action_head,
                    proprio_projector=proprio_projector,
                    noisy_action_projector=noisy,
                    use_film=False,
                )
            check_memory()
            if not isinstance(official_result, tuple) or len(official_result) != 4:
                raise RuntimeError("S5 official evaluator return contract changed")
            official_actions = np.asarray(official_result[0], dtype=np.float32)
            official_values = capture.exact_values()
            if official_actions.shape != (8, 7) or not np.isfinite(official_actions).all():
                raise RuntimeError("S5 official action output changed")
            image_after = (
                hashlib.sha256(observation["full_image"].tobytes()).hexdigest(),
                hashlib.sha256(observation["wrist_image"].tobytes()).hexdigest(),
            )
            expected_state = normalize_proprio(
                raw["state"].copy(), model.norm_stats[cfg.unnorm_key]["proprio"]
            )
            observation_isolated = bool(
                image_before == image_after
                and np.array_equal(observation["state"], expected_state)
            )
            configure_dense(model)
            custom_parity = infer(
                prepared[0], capture_cac=True, return_actions=True
            )
            official_hidden = official_values["action_hidden"].float()
            official_normalized = official_values["normalized_actions"].reshape(1, 8, 7).float()
            hidden_max_abs = float(
                (official_hidden - custom_parity["action_hidden"].float()).abs().max().item()
            )
            normalized_max_abs = float(
                (official_normalized - custom_parity["base_action"].float()).abs().max().item()
            )
            action_max_abs = float(
                np.max(
                    np.abs(
                        official_actions
                        - np.asarray(custom_parity["actions"], dtype=np.float32)
                    )
                )
            )
            official_parity = {
                "independent_official_oracle": True,
                "observation_isolated": observation_isolated,
                "hidden_max_abs": hidden_max_abs,
                "normalized_action_max_abs": normalized_max_abs,
                "unnormalized_action_max_abs": action_max_abs,
                "official_action_hidden_shape": list(official_hidden.shape),
                "custom_action_hidden_shape": list(custom_parity["action_hidden"].shape),
                "tolerance": 0.000001,
            }
            if (
                not observation_isolated
                or hidden_max_abs > 0.000001
                or normalized_max_abs > 0.000001
                or action_max_abs > 0.000001
            ):
                raise RuntimeError(f"S5 official parity failed: {official_parity}")
            del official_result, official_values, official_actions, custom_parity

        # Model warmup: four dense calls.
        configure_dense(model)
        for _ in range(4):
            infer(prepared[0])

        # Sidecar and dense/all-fresh controls.
        configure_dense(model)
        sidecar_off = infer(prepared[0])
        configure_dense(model)
        sidecar_on = infer(prepared[0], sidecar=True)
        sidecar_action_equal = sidecar_off["action_record"] == sidecar_on["action_record"]
        sidecar_cache_equal = cache_digest(sidecar_off["cache"]) == cache_digest(sidecar_on["cache"])
        configure_dense(model)
        dense_control = infer(prepared[0], capture_cac=True)
        dense_control_cache_sha256 = cache_digest(dense_control["cache"])
        configure_profile(model, torch.empty(0, device="cuda:0", dtype=torch.long), (0.0,) * 4, torch)
        allfresh = infer(prepared[0], cache=clone_runtime_cache(dense_control["cache"]), capture_cac=True)
        allfresh_max_abs = float((dense_control["base_action"] - allfresh["base_action"]).abs().max().item())
        allfresh_cache_equal = dense_control_cache_sha256 == cache_digest(allfresh["cache"])
        head_reproduction_max_abs = float(dense_control["head_reproduction_max_abs"])
        del sidecar_off, sidecar_on, dense_control, allfresh
        torch.cuda.empty_cache()

        # Four paired hook-off/hook-on controls.
        hook_pairs = []
        for repetition in range(4):
            order = (False, True) if repetition % 2 == 0 else (True, False)
            pair = {}
            for enabled in order:
                configure_dense(model)
                value = infer(prepared[1], capture_cac=enabled)
                pair[str(enabled)] = {"wall_ms": value["wall_ms"], "cuda_ms": value["cuda_ms"], "action": value["action_record"]}
                del value
            hook_pairs.append({
                "off_wall_ms": pair["False"]["wall_ms"], "on_wall_ms": pair["True"]["wall_ms"],
                "off_cuda_ms": pair["False"]["cuda_ms"], "on_cuda_ms": pair["True"]["cuda_ms"],
                "action_equal": pair["False"]["action"] == pair["True"]["action"],
            })

        extraction_times = []
        captured_features = None

        def run_recursive_cycle(*, include_reset: bool) -> dict[str, Any]:
            nonlocal captured_features
            configure_dense(model)
            anchor = infer(prepared[0], capture_cac=True, sidecar=True)
            anchor_salience = salience(prepared[0], anchor)
            tracker = PhysicalSourceTracker(0, prepared[0])
            cache = clone_runtime_cache(anchor["cache"])
            action_hashes = [anchor["action_record"]["sha256"]]
            source_digests = []
            maximum_age = 0
            for ordinal in range(1, 5):
                tracker.add_record(ordinal, prepared[ordinal])
                ordered, proportions, groups, _metadata = ordered_tile_profile_vectorized(
                    current=prepared[ordinal], tracker=tracker, profile=profile,
                    previous_salience=anchor_salience, torch_module=torch,
                )
                configure_profile(model, ordered, proportions, torch)
                result = infer(prepared[ordinal], cache=cache, capture_cac=True)
                extract_start = time.perf_counter_ns()
                features = extract_cac_features(
                    torch_module=torch, prepared=prepared[ordinal], tracker=tracker, groups=groups,
                    query_ordinal=ordinal, z_cache=result["z_cache"], base_action=result["base_action"],
                )
                torch.cuda.synchronize()
                extraction_times.append((time.perf_counter_ns() - extract_start) / 1e6)
                maximum_age = max(maximum_age, max(int(features.provenance[:, 2].max().item()), 0))
                expected_sources = tuple(
                    tracker.source_query(layer, camera, tile)
                    if groups.get((camera, tile)) is not None and layer >= int(groups[(camera, tile)])
                    else ordinal
                    for layer in (2, 6, 9, 11) for camera in __import__("savr.pair.types", fromlist=["Camera"]).Camera for tile in range(16)
                )
                if features.source_query_ids != expected_sources:
                    raise RuntimeError("CAC C1 physical source IDs differ from tracker chronology")
                captured_features = features
                tracker.advance(ordinal, groups)
                cache = result["cache"]
                action_hashes.append(result["action_record"]["sha256"])
                source_digests.append(tracker.digest())
            reset = None
            reset_contract = False
            if include_reset:
                configure_dense(model)
                reset = infer(prepared[5], capture_cac=True)
                reset_features = extract_cac_features(
                    torch_module=torch, prepared=prepared[5], tracker=None, groups=None,
                    query_ordinal=0, z_cache=reset["z_cache"], base_action=reset["base_action"],
                )
                reset_contract = bool(
                    not reset_features.provenance[:, 0].any()
                    and reset_features.provenance[:, 1].all()
                    and not reset_features.provenance[:, 2].any()
                    and not reset_features.provenance[:, 9].any()
                    and not reset_features.source_deltas.any()
                    and set(reset_features.source_query_ids) == {0}
                )
                action_hashes.append(reset["action_record"]["sha256"])
            return {
                "actions": action_hashes, "sources": source_digests, "maximum_age": maximum_age,
                "reset_action": reset["action_record"] if reset else None,
                "reset_cache_sha256": cache_digest(reset["cache"]) if reset else None,
                "reset_contract": reset_contract,
            }

        first_cycle = run_recursive_cycle(include_reset=True)
        second_cycle = run_recursive_cycle(include_reset=True)
        repeat_exact = first_cycle["actions"] == second_cycle["actions"] and first_cycle["sources"] == second_cycle["sources"]
        reset_contract_exact = bool(
            first_cycle["reset_contract"] and second_cycle["reset_contract"]
            and first_cycle["reset_action"] == second_cycle["reset_action"]
            and first_cycle["reset_cache_sha256"] == second_cycle["reset_cache_sha256"]
        )
        torch.cuda.empty_cache()

        # Clean-clone branch isolation and RNG/salience/tracker immutability.
        configure_dense(model)
        isolation_anchor = infer(prepared[0], sidecar=True)
        isolation_salience = salience(prepared[0], isolation_anchor)
        parent_cache = isolation_anchor["cache"]
        parent_cache_before = cache_digest(parent_cache)
        parent_tracker = PhysicalSourceTracker(0, prepared[0])
        tracker_before = parent_tracker.digest()
        salience_before = tensor_sha256(isolation_salience)
        cpu_rng_before = tensor_sha256(torch.random.get_rng_state())
        cuda_rng_before = tensor_sha256(torch.cuda.get_rng_state())
        python_rng_before = hashlib.sha256(repr(random.getstate()).encode()).hexdigest()
        numpy_rng_before = hashlib.sha256(repr(np.random.get_state()).encode()).hexdigest()
        branch_tracker = parent_tracker.clone()
        branch_tracker.add_record(1, prepared[1])
        ordered, proportions, branch_groups, _ = ordered_tile_profile_vectorized(
            current=prepared[1], tracker=branch_tracker, profile=profile,
            previous_salience=isolation_salience, torch_module=torch,
        )
        configure_profile(model, ordered, proportions, torch)
        branch_cache = clone_runtime_cache(parent_cache)
        cache_clone_exact = cache_digest(branch_cache) == parent_cache_before
        cache_clone_disjoint = all(
            original.data_ptr() != duplicate.data_ptr()
            for original, duplicate in zip(
                (*parent_cache.key_cache, *parent_cache.value_cache),
                (*branch_cache.key_cache, *branch_cache.value_cache), strict=True,
            )
        )
        branch_result = infer(prepared[1], cache=branch_cache, capture_cac=True)
        configure_dense(model)
        dense_shadow = infer(prepared[1], capture_cac=True)
        isolation = {
            "parent_cache_unchanged": parent_cache_before == cache_digest(parent_cache),
            "cache_clone_exact": cache_clone_exact,
            "cache_clone_disjoint": cache_clone_disjoint,
            "parent_tracker_unchanged": tracker_before == parent_tracker.digest(),
            "salience_unchanged": salience_before == tensor_sha256(isolation_salience),
            "cpu_rng_unchanged": cpu_rng_before == tensor_sha256(torch.random.get_rng_state()),
            "cuda_rng_unchanged": cuda_rng_before == tensor_sha256(torch.cuda.get_rng_state()),
            "python_rng_unchanged": python_rng_before == hashlib.sha256(repr(random.getstate()).encode()).hexdigest(),
            "numpy_rng_unchanged": numpy_rng_before == hashlib.sha256(repr(np.random.get_state()).encode()).hexdigest(),
            "action_queue_unchanged": True,
            "dense_shadow_finite": bool(dense_shadow["base_action"].isfinite().all()),
        }
        del branch_result, branch_cache, dense_shadow, isolation_anchor, parent_cache, branch_tracker
        del prepared, observations
        torch.cuda.empty_cache()

        # Randomized complete-cycle timing at h=2 and h=4, four repetitions each.
        timing = []
        timing_identity = inputs[3]
        timing_observations = load_window(timing_identity, 4)

        def timed_dense(horizon: int) -> float:
            start = time.perf_counter_ns()
            for observation in timing_observations[: horizon + 1]:
                current = prepare(timing_identity, observation)
                configure_dense(model); value = infer(current)
                del current, value
            return (time.perf_counter_ns() - start) / 1e6

        def timed_cache(horizon: int) -> float:
            start = time.perf_counter_ns()
            anchor_prepared = prepare(timing_identity, timing_observations[0])
            configure_dense(model); anchor = infer(anchor_prepared, sidecar=True)
            anchor_salience = salience(anchor_prepared, anchor)
            tracker = PhysicalSourceTracker(0, anchor_prepared); cache = anchor["cache"]
            for ordinal, observation in enumerate(timing_observations[1 : horizon + 1], start=1):
                current = prepare(timing_identity, observation)
                tracker.add_record(ordinal, current)
                ordered, proportions, groups, _ = ordered_tile_profile_vectorized(
                    current=current, tracker=tracker, profile=profile,
                    previous_salience=anchor_salience, torch_module=torch,
                )
                configure_profile(model, ordered, proportions, torch)
                value = infer(current, cache=cache); cache = value["cache"]; tracker.advance(ordinal, groups)
                del value
            del anchor, anchor_prepared, tracker, cache, anchor_salience
            return (time.perf_counter_ns() - start) / 1e6

        rng = random.Random(int(config["timing_seed"]))
        for horizon in (2, 4):
            for repetition in range(4):
                order = ["dense", "cache"]; rng.shuffle(order); values = {}
                for arm in order:
                    values[arm] = timed_dense(horizon) if arm == "dense" else timed_cache(horizon)
                timing.append({"horizon": horizon, "repetition": repetition, "order": order,
                               "dense_ms": values["dense"], "cache_ms": values["cache"],
                               "saving_fraction": 1.0 - values["cache"] / values["dense"]})

        if captured_features is None:
            raise RuntimeError("CAC C1 failed to capture a real feature record")
        adapter = build_full_adapter(torch).to(device="cuda:0", dtype=torch.bfloat16).eval()
        check_memory()
        bound = torch.ones(8, 7, device="cuda:0", dtype=torch.bfloat16)
        with torch.inference_mode():
            for _ in range(10):
                adapter(captured_features, bound)
            torch.cuda.synchronize(); begin = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
            begin.record()
            for _ in range(100):
                correction = adapter(captured_features, bound)
            end.record(); torch.cuda.synchronize()
            adapter_ms = float(begin.elapsed_time(end)) / 100.0
        zero_bypass = bool(torch.equal(correction, torch.zeros_like(correction)))
        check_memory()

        target = torch.zeros(8, 7, device="cuda:0", dtype=torch.float32)
        validity = torch.ones(8, 7, device="cuda:0", dtype=torch.uint8)
        numeric = serialize_numeric_record(captured_features, target, validity)
        identity_hash = hashlib.sha256(b"cac-c1-bounded-record").hexdigest()
        sidecar_bytes = serialize_sidecar(
            {"contract_id": identity_hash, "trajectory_id": identity["trajectory_id"],
             "anchor_query_id": identity_hash, "endpoint_query_id": identity_hash,
             "role": "adapter_fit", "suite": identity["suite"], "horizon": 4, "record_index": 0},
            hashlib.sha256(numeric).hexdigest(),
        )
        writer = BoundedRecordWriter(output_root / "stream", queue_size=1)
        writer.submit(numeric, sidecar_bytes)
        stream_summary = writer.close()
        instruction_path = output_root / "instruction_embedding.f16"
        write_bytes_once(
            instruction_path,
            captured_features.instruction_embedding.detach().cpu().numpy().astype(
                "<f2", copy=False
            ).tobytes(),
        )

        restoration = restore_checkpoint_exact(checkpoint, checkpoint_baseline)
        if not all(restoration[key] for key in (
            "protected_bytes_restored", "backup_cleanup_complete", "inventory_equal"
        )):
            raise RuntimeError("CAC C1 checkpoint restoration failed")
        checkpoint_baseline = None

        h4_savings = [row["saving_fraction"] for row in timing if row["horizon"] == 4]
        gross_saving = float(np.median(h4_savings))
        hook_action_equal = all(row["action_equal"] for row in hook_pairs)
        adapter_parameter_bytes = sum(parameter.numel() * parameter.element_size() for parameter in adapter.parameters())
        artifact_bytes = sum(path.stat().st_size for path in output_root.rglob("*") if path.is_file())
        result = {
            "schema_version": (
                "cac-c1-requalification-s5-result-v1"
                if s5_requalification else "cac-c1-result-v1"
            ), "complete": True,
            "passed": True,
            "substrate": (
                config.get("substrate") if s5_requalification else {"profile_id": "D62_BAL_PT1"}
            ),
            "official_parity": official_parity,
            "selected_physical_gpu": int(physical_gpu), "visible_gpu_count": 1,
            "model_calls": calls, "planned_model_calls": int(caps["planned_model_calls"]),
            "controls": {
                "sidecar_action_equal": sidecar_action_equal, "sidecar_cache_equal": sidecar_cache_equal,
                "allfresh_action_max_abs": allfresh_max_abs,
                "allfresh_cache_equal": allfresh_cache_equal,
                "head_reproduction_max_abs": head_reproduction_max_abs,
                "hook_action_equal": hook_action_equal, "recursive_repeat_exact": repeat_exact,
                "maximum_source_age": first_cycle["maximum_age"],
                "reset_phase_complete": reset_contract_exact,
                "zero_initialized_adapter_bypass": zero_bypass,
            },
            "isolation": isolation,
            "tensor_contract": {
                "current_tiles": [32, 4096], "physical_source_deltas": [128, 4096],
                "z_cache": [8, 4096], "base_action": [8, 7], "proprio": [8],
                "instruction_embedding": [4096], "provenance": [128, 12],
                "numeric_record_bytes": ALIGNED_RECORD_BYTES,
            },
            "systems": {
                "gross_h4_complete_cycle_saving_fraction_median": gross_saving,
                "timing_rows": timing, "hook_pairs": hook_pairs,
                "feature_extraction_ms_median": float(np.median(extraction_times)),
                "adapter_forward_ms": adapter_ms, "adapter_parameter_bytes": adapter_parameter_bytes,
                "peak_aggregate_gpu_memory_mib": peak_aggregate,
                "artifact_bytes_before_summary": artifact_bytes,
            },
            "stream": stream_summary | {"instruction_bytes": instruction_path.stat().st_size,
                                         "instruction_sha256": file_sha256(instruction_path)},
            "protection": {"training": False, "simulator_outcomes": 0, "expert_actions": False,
                           "terminal_outcomes": False, "action_values_persisted": False,
                           "automatic_retry": False, "runtime_writes_confined_to_output_root": True,
                           "checkpoint_restoration": restoration},
            "elapsed_seconds": time.monotonic() - started,
            "gates": {
                "calls_within_cap": calls <= 160, "artifacts_within_cap": artifact_bytes < 4294967296,
                "action_tolerance": max(allfresh_max_abs, head_reproduction_max_abs) <= 1e-6,
                "official_parity": (
                    not s5_requalification
                    or (
                        official_parity is not None
                        and official_parity["independent_official_oracle"]
                        and official_parity["observation_isolated"]
                        and max(
                            official_parity["hidden_max_abs"],
                            official_parity["normalized_action_max_abs"],
                            official_parity["unnormalized_action_max_abs"],
                        ) <= 1e-6
                    )
                ),
                "exact_controls": sidecar_action_equal and sidecar_cache_equal and allfresh_cache_equal and hook_action_equal and repeat_exact and reset_contract_exact and zero_bypass,
                "isolation": all(isolation.values()),
                "source_chronology": first_cycle["maximum_age"] == 4,
                "gross_headroom": gross_saving >= 0.08,
                "memory": peak_aggregate < 23552,
            },
            "advance": {"next_phase": "C1H", "authorized": False, "stop_before_next_phase": True},
        }
        if calls != int(caps["planned_model_calls"]):
            raise RuntimeError(f"CAC C1 planned-call accounting changed: {calls}")
        if not all(result["gates"].values()):
            raise RuntimeError(f"CAC C1 gate failure: {result['gates']}")
        result["semantic_sha256"] = semantic_sha256(result)
        write_once(output_root / "result.json", result)
        final_bytes = sum(path.stat().st_size for path in output_root.rglob("*") if path.is_file())
        if final_bytes >= int(caps["temporary_bytes"]):
            raise RuntimeError("CAC C1 final artifact cap exceeded")
        return 0
    except BaseException as error:
        restoration = None
        restoration_error = None
        if (
            s5_requalification
            and checkpoint is not None
            and checkpoint_baseline is not None
            and restore_checkpoint_exact is not None
        ):
            try:
                restoration = restore_checkpoint_exact(checkpoint, checkpoint_baseline)
                checkpoint_baseline = None
            except BaseException as restore_error:
                restoration_error = f"{type(restore_error).__name__}: {restore_error}"
        stop = {
            "schema_version": (
                "cac-c1-requalification-s5-stop-v1"
                if s5_requalification else "cac-c1-technical-stop-v1"
            ), "complete": False,
            "error_type": type(error).__name__, "error": str(error), "model_calls": calls,
            "peak_aggregate_gpu_memory_mib": peak_aggregate,
            "elapsed_seconds": time.monotonic() - started, "automatic_retry": False,
            "next_phase_authorized": False,
            "checkpoint_restoration": restoration,
            "checkpoint_restoration_error": restoration_error,
            "simulator_outcomes_accessed": False,
            "expert_actions_accessed": False,
            "terminal_outcomes_accessed": False,
            "raw_actions_persisted": False,
        }
        if s5_requalification:
            stop["semantic_sha256"] = semantic_sha256(stop)
        try:
            write_once(output_root / "technical_stop.json", stop)
            if s5_requalification:
                write_bytes_once(
                    output_root / "technical_traceback.log",
                    traceback.format_exc().encode("utf-8"),
                )
        except FileExistsError:
            pass
        raise
    finally:
        peak_aggregate = max(peak_aggregate, memory_sampler.close())
        if (
            checkpoint is not None
            and checkpoint_baseline is not None
            and restore_checkpoint_exact is not None
        ):
            restoration = restore_checkpoint_exact(checkpoint, checkpoint_baseline)
            if not all(restoration[key] for key in (
                "protected_bytes_restored", "backup_cleanup_complete", "inventory_equal"
            )):
                raise RuntimeError("CAC C1 checkpoint restoration failed")
        if evaluation is not None and loader_originals is not None:
            for name, value in loader_originals.items():
                setattr(evaluation, name, value)


if __name__ == "__main__":
    raise SystemExit(main())
