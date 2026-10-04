#!/usr/bin/env python3
"""Run the frozen, outcome-free OpenVLA official/custom semantic parity gate."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import signal
import shutil
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


ROOT = Path("/home/ved/SAVR")
DEFAULT_CONFIG = Path("configs/openvla/semantic_parity_s3_v1.json")


class ParityTechnicalStop(RuntimeError):
    """A technical condition prevented a valid semantic decision."""


class SemanticMismatch(RuntimeError):
    """The corrected path disagreed with the independent official oracle."""

    def __init__(self, record: Mapping[str, Any]) -> None:
        super().__init__(f"semantic mismatch at {record['boundary']}")
        self.record = dict(record)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def semantic_sha256(value: Mapping[str, Any]) -> str:
    body = dict(value)
    body.pop("semantic_sha256", None)
    return hashlib.sha256(canonical_bytes(body)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_once(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(canonical_bytes(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def tensor_sha256(value: Any) -> str:
    tensor = value.detach().contiguous().cpu()
    raw = tensor.view(dtype=__import__("torch").uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def array_sha256(value: Any, np_module: Any) -> str:
    array = np_module.ascontiguousarray(value)
    return hashlib.sha256(array.tobytes()).hexdigest()


def compare_values(
    *,
    boundary: str,
    reference: Any,
    candidate: Any,
    tolerance: float,
    torch_module: Any,
    np_module: Any,
) -> dict[str, Any]:
    is_tensor = torch_module.is_tensor(reference) and torch_module.is_tensor(candidate)
    if is_tensor:
        reference_shape = list(reference.shape)
        candidate_shape = list(candidate.shape)
        reference_hash = tensor_sha256(reference)
        candidate_hash = tensor_sha256(candidate)
        finite = bool(reference.isfinite().all() and candidate.isfinite().all())
        if reference_shape == candidate_shape:
            if reference.dtype == torch_module.bool or not reference.dtype.is_floating_point:
                maximum = 0.0 if bool(torch_module.equal(reference, candidate)) else float("inf")
            else:
                maximum = float(
                    (reference.float() - candidate.float()).abs().max().item()
                )
        else:
            maximum = float("inf")
    else:
        reference_array = np_module.asarray(reference)
        candidate_array = np_module.asarray(candidate)
        reference_shape = list(reference_array.shape)
        candidate_shape = list(candidate_array.shape)
        reference_hash = array_sha256(reference_array, np_module)
        candidate_hash = array_sha256(candidate_array, np_module)
        finite = bool(
            np_module.isfinite(reference_array).all()
            and np_module.isfinite(candidate_array).all()
        )
        maximum = (
            float(np_module.max(np_module.abs(reference_array - candidate_array)))
            if reference_shape == candidate_shape
            else float("inf")
        )
    passed = bool(
        reference_shape == candidate_shape
        and finite
        and maximum <= tolerance
    )
    return {
        "boundary": boundary,
        "reference_shape": reference_shape,
        "candidate_shape": candidate_shape,
        "reference_sha256": reference_hash,
        "candidate_sha256": candidate_hash,
        "max_abs": maximum,
        "tolerance": tolerance,
        "finite": finite,
        "passed": passed,
    }


def validate_static(config: Mapping[str, Any], root: Path = ROOT) -> None:
    if config.get("schema_version") != "openvla-semantic-parity-s3-v1":
        raise ParityTechnicalStop("semantic-parity config schema changed")
    if config.get("semantic_sha256") != semantic_sha256(config):
        raise ParityTechnicalStop("semantic-parity config hash mismatch")
    if config.get("authorization") != {
        "s3_authorized": True,
        "source": "User approved continuation on 2026-08-31",
        "simulator": False,
        "terminal_outcomes": False,
        "training": False,
        "automatic_retry": False,
    }:
        raise ParityTechnicalStop("semantic-parity authorization boundary changed")
    if config.get("resource_caps") != {
        "gpu_count": 1,
        "model_processes": 1,
        "observations": 8,
        "calls_per_observation": 4,
        "model_call_hard_cap": 32,
        "wall_seconds": 1800,
        "artifact_bytes": 268435456,
        "peak_gpu_memory_mib_strict_max": 23552,
        "downloads": 0,
        "simulator_outcomes": 0,
        "raw_actions_persisted": False,
        "automatic_retry": False,
    }:
        raise ParityTechnicalStop("semantic-parity resource boundary changed")
    if config.get("advance") != {
        "next_stage": "S4_D62_REQUALIFICATION",
        "authorized": False,
        "stop_before_next_stage": True,
    }:
        raise ParityTechnicalStop("semantic-parity advance boundary changed")
    if config.get("output_root") != "results/openvla-semantic-parity-s3-v01":
        raise ParityTechnicalStop("semantic-parity output root changed")
    for relative, expected in config["authenticated_files"].items():
        path = root / relative
        if not path.is_file() or file_sha256(path) != expected:
            raise ParityTechnicalStop(f"authenticated input changed: {relative}")
    input_manifest = json.loads((root / config["input_manifest"]).read_text(encoding="utf-8"))
    if (
        input_manifest.get("semantic_sha256") != semantic_sha256(input_manifest)
        or input_manifest.get("terminal_outcome_fields_accessed") is not False
        or input_manifest.get("expert_action_fields_accessed") is not False
        or len(input_manifest.get("inputs", [])) != 8
        or {row["suite"] for row in input_manifest["inputs"]}
        != {"libero_spatial", "libero_object", "libero_goal", "libero_10"}
        or any(
            sum(row["suite"] == suite for row in input_manifest["inputs"]) != 2
            for suite in {"libero_spatial", "libero_object", "libero_goal", "libero_10"}
        )
    ):
        raise ParityTechnicalStop("frozen parity population changed")
    data_root = root / input_manifest["data_root_relative"]
    for row in input_manifest["inputs"]:
        source = data_root / row["source_path"]
        if not source.is_file() or file_sha256(source) != row["source_sha256"]:
            raise ParityTechnicalStop("frozen observation source changed")
    checkpoint = root / config["model"]["checkpoint_relative"]
    for name, expected in config["model"]["checkpoint_metadata_sha256"].items():
        if not (checkpoint / name).is_file() or file_sha256(checkpoint / name) != expected:
            raise ParityTechnicalStop(f"checkpoint metadata changed: {name}")
    if (root / config["output_root"]).exists():
        raise ParityTechnicalStop("immutable semantic-parity output root already exists")


def selected_gpu_snapshot(physical_gpu: int) -> dict[str, int]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            str(physical_gpu),
            "--query-gpu=memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    fields = [int(item.strip()) for item in output.split(",")]
    if len(fields) != 2:
        raise ParityTechnicalStop("selected-GPU telemetry returned an invalid row")
    return {"memory_used_mib": fields[0], "utilization_percent": fields[1]}


class AggregateMemorySampler:
    def __init__(self, gpu: int, initial: int) -> None:
        self.gpu = gpu
        self.peak = initial
        self.error: BaseException | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.wait(0.2):
            try:
                value = selected_gpu_snapshot(self.gpu)["memory_used_mib"]
                with self._lock:
                    self.peak = max(self.peak, value)
            except BaseException as error:
                self.error = error
                return

    def start(self) -> None:
        self._thread.start()

    def close(self) -> int:
        self._stop.set()
        self._thread.join()
        with self._lock:
            return self.peak


def base_config(evaluation: Any, checkpoint: Path, output_root: Path) -> Any:
    return evaluation.GenerateConfig(
        pretrained_checkpoint=str(checkpoint),
        task_suite_name="libero_spatial",
        num_trials_per_task=0,
        seed=20260831,
        local_log_dir=str(output_root / "logs"),
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
    args = parser.parse_args()
    if Path.cwd().resolve() != ROOT:
        raise SystemExit(f"semantic parity must start in {ROOT}")
    config_path = (ROOT / args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_static(config)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    physical = os.environ.get("OPENVLA_PHYSICAL_GPU_ID", "")
    if not physical.isdigit() or visible != physical or "," in visible:
        raise SystemExit("semantic parity requires one explicitly selected GPU")
    initial_gpu = selected_gpu_snapshot(int(physical))
    if initial_gpu["memory_used_mib"] > 1024 or initial_gpu["utilization_percent"] > 5:
        raise SystemExit("selected GPU is not sufficiently idle")
    output_root = ROOT / config["output_root"]
    output_root.mkdir(parents=True)
    runtime_cache = output_root / "runtime-cache"
    for key, relative in {
        "MPLCONFIGDIR": "matplotlib",
        "HF_MODULES_CACHE": "hf-modules",
        "HF_HOME": "hf-home",
        "TORCH_HOME": "torch",
        "XDG_CACHE_HOME": "xdg",
        "TMPDIR": "tmp",
        "WANDB_DIR": "wandb",
    }.items():
        path = runtime_cache / relative
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    os.environ.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "WANDB_MODE": "disabled",
            "PYTHONNOUSERSITE": "1",
        }
    )
    sys.pycache_prefix = str(runtime_cache / "pycache")
    sys.path.insert(0, str(ROOT / "src"))
    project_libero = ROOT / "configs/pair/libero_runtime"
    if not (project_libero / "config.yaml").is_file():
        raise SystemExit("project-local LIBERO configuration is missing")
    os.environ["LIBERO_CONFIG_PATH"] = str(project_libero)
    started = time.monotonic()
    calls = 0
    comparisons: list[dict[str, Any]] = []
    observations_completed = 0
    sampler = AggregateMemorySampler(int(physical), initial_gpu["memory_used_mib"])
    sampler.start()
    checkpoint = ROOT / config["model"]["checkpoint_relative"]
    checkpoint_baseline = None
    restore_checkpoint_exact_fn = None
    model = action_head = proprio_projector = processor = None
    config_baseline: tuple[Any, Any] | None = None

    def restore_checkpoint() -> dict[str, Any] | None:
        nonlocal checkpoint_baseline
        if checkpoint_baseline is None or restore_checkpoint_exact_fn is None:
            return None
        result = restore_checkpoint_exact_fn(checkpoint, checkpoint_baseline)
        checkpoint_baseline = None
        return result

    def timed_out(_signum: int, _frame: Any) -> None:
        raise TimeoutError("semantic-parity wall-time limit reached")

    signal.signal(signal.SIGALRM, timed_out)
    signal.alarm(int(config["resource_caps"]["wall_seconds"]))

    def consume() -> None:
        nonlocal calls
        calls += 1
        if calls > int(config["resource_caps"]["model_call_hard_cap"]):
            raise ParityTechnicalStop("semantic-parity call cap exceeded")

    def record_or_stop(record: Mapping[str, Any], *, observation_id: str, mode: str) -> None:
        enriched = {"observation_id": observation_id, "mode": mode, **dict(record)}
        comparisons.append(enriched)
        if not enriched["passed"]:
            raise SemanticMismatch(enriched)

    try:
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
        from savr.cac.c1 import instruction_token_indices
        from savr.openvla.official_semantics import (
            OfficialBoundaryCapture,
            SemanticSDPASidecarTap,
            prepare_semantic_query,
            structurally_aligned_dense_forward,
        )
        restore_checkpoint_exact_fn = restore_checkpoint_exact

        set_seed_everywhere(int(config["seed"]))
        protected_names = ("config.json", "configuration_prismatic.py", "modeling_prismatic.py")
        stale = sorted(
            path.name
            for path in checkpoint.iterdir()
            if any(
                path.name.startswith(f"{name}.back.")
                or path.name.startswith(f"{name}.backup")
                or path.name == f"{name}.bak"
                for name in protected_names
            )
        )
        if stale:
            raise ParityTechnicalStop(f"checkpoint contains stale loader backups: {stale}")
        checkpoint_baseline = capture_checkpoint_baseline(checkpoint, protected_names)
        cfg = base_config(evaluation, checkpoint, output_root)
        evaluation.validate_config(cfg)
        model, action_head, proprio_projector, _noisy, processor = evaluation.initialize_model(cfg)
        model.eval()
        if torch.cuda.device_count() != 1 or action_head is None or proprio_projector is None:
            raise ParityTechnicalStop("model/device initialization contract failed")
        config_baseline = (
            model.language_model.config.proportion_attn_var,
            model.language_model.config.reusable_patches,
        )
        manifest = json.loads((ROOT / config["input_manifest"]).read_text(encoding="utf-8"))
        data_root = ROOT / manifest["data_root_relative"]

        def dense_config() -> None:
            model.language_model.config.proportion_attn_var = None
            model.language_model.config.reusable_patches = None

        def load_observation(identity: Mapping[str, Any]) -> dict[str, Any]:
            source_path = data_root / identity["source_path"]
            with h5py.File(source_path, "r") as handle:
                group = handle[f"data/{identity['original_trajectory_id']}/obs"]
                scene = np.asarray(group["agentview_rgb"][0]).copy()
                wrist = np.asarray(group["eye_in_hand_rgb"][0]).copy()
                state = np.concatenate(
                    (
                        np.asarray(group["ee_states"][0]),
                        np.asarray(group["gripper_states"][0]),
                    )
                ).copy()
            if scene.ndim != 3 or wrist.ndim != 3 or state.shape != (8,):
                raise ParityTechnicalStop("frozen observation layout changed")
            return {"scene": scene, "wrist": wrist, "state": state}

        @torch.inference_mode()
        def custom_prepare(identity: Mapping[str, Any], raw: Mapping[str, Any]) -> Any:
            cfg.unnorm_key = identity["normalization_statistics_key"]
            dense_config()
            return prepare_semantic_query(
                torch_module=torch,
                np_module=np,
                model=model,
                processor=processor,
                proprio_projector=proprio_projector,
                prepare_images=prepare_images_for_vla,
                normalize_proprio=normalize_proprio,
                instruction_indexer=instruction_token_indices,
                cfg=cfg,
                raw_scene=raw["scene"],
                raw_wrist=raw["wrist"],
                raw_state=raw["state"],
                instruction=identity["language_instruction"],
            )

        @torch.inference_mode()
        def custom_forward(prepared: Any, mode: str) -> dict[str, Any]:
            consume()
            dense_config()
            tap_factory = (
                lambda: SemanticSDPASidecarTap(
                    torch, tuple(config["model"]["sidecar_layers"])
                )
                if mode == "sidecar"
                else None
            )
            return structurally_aligned_dense_forward(
                torch_module=torch,
                np_module=np,
                model=model,
                action_head=action_head,
                cfg=cfg,
                prepared=prepared,
                use_cache=None if mode == "none" else True,
                tap_factory=tap_factory,
            )

        @torch.inference_mode()
        def official_forward(identity: Mapping[str, Any], raw: Mapping[str, Any]) -> tuple[Any, Any, dict[str, Any], dict[str, Any]]:
            consume()
            cfg.unnorm_key = identity["normalization_statistics_key"]
            dense_config()
            official_observation = {
                "full_image": raw["scene"].copy(),
                "wrist_image": raw["wrist"].copy(),
                "state": raw["state"].copy(),
                "prev_images": [raw["scene"].copy(), raw["wrist"].copy()],
            }
            image_hashes = {
                "full": array_sha256(official_observation["full_image"], np),
                "wrist": array_sha256(official_observation["wrist_image"], np),
                "prev0": array_sha256(official_observation["prev_images"][0], np),
                "prev1": array_sha256(official_observation["prev_images"][1], np),
            }
            with OfficialBoundaryCapture(model, action_head) as capture:
                result = evaluation.get_action(
                    cfg,
                    model,
                    official_observation,
                    identity["language_instruction"],
                    processor=processor,
                    action_head=action_head,
                    proprio_projector=proprio_projector,
                    noisy_action_projector=_noisy,
                    use_film=False,
                )
            if not isinstance(result, tuple) or len(result) != 4:
                raise ParityTechnicalStop("official evaluator return contract changed")
            actions, cache, _images, _metrics = result
            action_array = np.asarray(actions, dtype=np.float32)
            if action_array.shape != (8, 7) or not np.isfinite(action_array).all():
                raise ParityTechnicalStop("official evaluator returned invalid actions")
            expected_state = normalize_proprio(
                raw["state"].copy(), model.norm_stats[cfg.unnorm_key]["proprio"]
            )
            isolation = {
                "images_unchanged": image_hashes
                == {
                    "full": array_sha256(official_observation["full_image"], np),
                    "wrist": array_sha256(official_observation["wrist_image"], np),
                    "prev0": array_sha256(official_observation["prev_images"][0], np),
                    "prev1": array_sha256(official_observation["prev_images"][1], np),
                },
                "state_mutation_documented": bool(
                    np.array_equal(np.asarray(official_observation["state"]), np.asarray(expected_state))
                ),
            }
            if not all(isolation.values()):
                raise ParityTechnicalStop("official observation-copy isolation failed")
            return action_array, cache, capture.exact_values(), isolation

        boundary_map = {
            "input_ids": "input_ids",
            "input_embeddings": "input_embeddings",
            "action_mask": "action_mask",
            "pixel_values": "preprocessed_pixels",
            "language_embeddings": "language_embeddings",
            "vision_output": "vision_output",
            "pre_proprio_projected": "vision_output",
            "normalized_proprio": "normalized_proprio",
            "projected_with_proprio": "projected_patches",
        }
        for index, identity in enumerate(manifest["inputs"]):
            observation_id = identity["trajectory_id"]
            raw = load_observation(identity)
            if index % 2 == 0:
                official_actions, official_cache, official_values, isolation = official_forward(identity, raw)
                prepared = custom_prepare(identity, raw)
                variants = {mode: custom_forward(prepared, mode) for mode in ("none", "cache", "sidecar")}
            else:
                prepared = custom_prepare(identity, raw)
                variants = {mode: custom_forward(prepared, mode) for mode in ("none", "cache", "sidecar")}
                official_actions, official_cache, official_values, isolation = official_forward(identity, raw)
            for official_name, prepared_name in boundary_map.items():
                record_or_stop(
                    compare_values(
                        boundary=official_name,
                        reference=official_values[official_name],
                        candidate=getattr(prepared, prepared_name),
                        tolerance=0.0,
                        torch_module=torch,
                        np_module=np,
                    ),
                    observation_id=observation_id,
                    mode="prepared",
                )
            baseline = variants["none"]
            dynamic_map = {
                "masked_input_embeddings": "masked_input_embeddings",
                "multimodal_embeddings": "multimodal_embeddings",
                "multimodal_attention_mask": "multimodal_attention_mask",
                "action_hidden": "action_hidden",
            }
            for mode, variant in variants.items():
                record_or_stop(
                    compare_values(
                        boundary="input_attention_mask",
                        reference=official_values["input_attention_mask"],
                        candidate=prepared.attention_mask,
                        tolerance=0.0,
                        torch_module=torch,
                        np_module=np,
                    ),
                    observation_id=observation_id,
                    mode=mode,
                )
                record_or_stop(
                    compare_values(
                        boundary="multimodal_projected",
                        reference=official_values["multimodal_projected"],
                        candidate=prepared.projected_patches,
                        tolerance=0.0,
                        torch_module=torch,
                        np_module=np,
                    ),
                    observation_id=observation_id,
                    mode=mode,
                )
                for boundary, key in dynamic_map.items():
                    record_or_stop(
                        compare_values(
                            boundary=boundary,
                            reference=official_values[boundary],
                            candidate=variant[key],
                            tolerance=float(config["tolerance"]),
                            torch_module=torch,
                            np_module=np,
                        ),
                        observation_id=observation_id,
                        mode=mode,
                    )
                record_or_stop(
                    compare_values(
                        boundary="normalized_actions",
                        reference=official_values["normalized_actions"].reshape(8, 7),
                        candidate=variant["normalized_actions"],
                        tolerance=float(config["tolerance"]),
                        torch_module=torch,
                        np_module=np,
                    ),
                    observation_id=observation_id,
                    mode=mode,
                )
                record_or_stop(
                    compare_values(
                        boundary="unnormalized_actions",
                        reference=official_actions,
                        candidate=variant["actions"],
                        tolerance=float(config["tolerance"]),
                        torch_module=torch,
                        np_module=np,
                    ),
                    observation_id=observation_id,
                    mode=mode,
                )
            for mode in ("cache", "sidecar"):
                for key in (
                    "multimodal_embeddings",
                    "multimodal_attention_mask",
                    "action_hidden",
                    "normalized_actions",
                    "actions",
                ):
                    record_or_stop(
                        compare_values(
                            boundary=f"determinism_{key}",
                            reference=baseline[key],
                            candidate=variants[mode][key],
                            tolerance=0.0,
                            torch_module=torch,
                            np_module=np,
                        ),
                        observation_id=observation_id,
                        mode=mode,
                    )
            if baseline["cache"] is not None:
                raise ParityTechnicalStop("use_cache=None unexpectedly returned a cache")
            if variants["cache"]["cache"] is None or variants["sidecar"]["cache"] is None:
                raise ParityTechnicalStop("cache-producing dense mode returned no cache")
            if variants["sidecar"]["tap"] is None:
                raise ParityTechnicalStop("sidecar mode failed to install its capture")
            layout = baseline["layout"]
            if (
                len(layout.action_readout_positions) != 56
                or layout.action_readout_positions[0]
                != layout.placeholder_multimodal_positions[0] - 1
                or layout.action_readout_positions[-1]
                != layout.placeholder_multimodal_positions[-1] - 1
            ):
                raise ParityTechnicalStop("official structural span invariant failed")
            observations_completed += 1
            del variants, baseline, official_cache, official_values, prepared, raw
            gc.collect()
            torch.cuda.empty_cache()
            if directory_size(output_root) > int(
                config["resource_caps"]["artifact_bytes"]
            ):
                raise ParityTechnicalStop("semantic-parity artifact cap exceeded")
        peak = sampler.close()
        if sampler.error is not None:
            raise ParityTechnicalStop("aggregate-memory sampler failed") from sampler.error
        peak = max(peak, selected_gpu_snapshot(int(physical))["memory_used_mib"])
        if peak >= int(config["resource_caps"]["peak_gpu_memory_mib_strict_max"]):
            raise ParityTechnicalStop("strict aggregate-memory boundary reached")
        if calls != 32 or observations_completed != 8:
            raise ParityTechnicalStop("semantic-parity completion arithmetic changed")
        comparison_manifest = {
            "schema_version": "openvla-semantic-parity-comparisons-v1",
            "run_id": config["run_id"],
            "records": comparisons,
            "raw_values_persisted": False,
        }
        comparison_manifest["semantic_sha256"] = semantic_sha256(comparison_manifest)
        write_once(output_root / "comparison_manifest.json", comparison_manifest)
        restoration = restore_checkpoint()
        summary = {
            "schema_version": "openvla-semantic-parity-result-v1",
            "run_id": config["run_id"],
            "complete": True,
            "passed": True,
            "observations_completed": observations_completed,
            "model_calls": calls,
            "comparison_count": len(comparisons),
            "maximum_abs_error": max(float(row["max_abs"]) for row in comparisons),
            "first_mismatch": None,
            "comparison_manifest_sha256": file_sha256(output_root / "comparison_manifest.json"),
            "checkpoint_restoration": restoration,
            "peak_aggregate_gpu_memory_mib": peak,
            "elapsed_seconds": time.monotonic() - started,
            "simulator_outcomes_accessed": False,
            "expert_actions_accessed": False,
            "raw_actions_persisted": False,
            "automatic_retry": False,
            "advance": config["advance"],
            "completed_at": utc_now(),
        }
        summary["semantic_sha256"] = semantic_sha256(summary)
        write_once(output_root / "worker_summary.json", summary)
        return 0
    except SemanticMismatch as error:
        peak = sampler.close()
        comparison_manifest = {
            "schema_version": "openvla-semantic-parity-comparisons-v1",
            "run_id": config["run_id"],
            "records": comparisons,
            "raw_values_persisted": False,
        }
        comparison_manifest["semantic_sha256"] = semantic_sha256(comparison_manifest)
        write_once(output_root / "comparison_manifest.json", comparison_manifest)
        restoration = restore_checkpoint()
        summary = {
            "schema_version": "openvla-semantic-parity-result-v1",
            "run_id": config["run_id"],
            "complete": True,
            "passed": False,
            "observations_completed": observations_completed,
            "model_calls": calls,
            "comparison_count": len(comparisons),
            "first_mismatch": error.record,
            "comparison_manifest_sha256": file_sha256(output_root / "comparison_manifest.json"),
            "checkpoint_restoration": restoration,
            "peak_aggregate_gpu_memory_mib": peak,
            "elapsed_seconds": time.monotonic() - started,
            "simulator_outcomes_accessed": False,
            "expert_actions_accessed": False,
            "raw_actions_persisted": False,
            "automatic_retry": False,
            "advance": {"next_stage": None, "authorized": False, "stop_before_next_stage": True},
            "completed_at": utc_now(),
        }
        summary["semantic_sha256"] = semantic_sha256(summary)
        write_once(output_root / "worker_summary.json", summary)
        return 2
    except BaseException as error:
        peak = sampler.close()
        restoration_error = None
        restoration = None
        try:
            restoration = restore_checkpoint()
        except BaseException as restore_error:
            restoration_error = f"{type(restore_error).__name__}: {restore_error}"
        stop = {
            "schema_version": "openvla-semantic-parity-technical-stop-v1",
            "run_id": config.get("run_id"),
            "complete": False,
            "error_type": type(error).__name__,
            "error": str(error),
            "observations_completed": observations_completed,
            "model_calls": calls,
            "peak_aggregate_gpu_memory_mib": peak,
            "elapsed_seconds": time.monotonic() - started,
            "simulator_outcomes_accessed": False,
            "raw_actions_persisted": False,
            "automatic_retry": False,
            "checkpoint_restoration": restoration,
            "checkpoint_restoration_error": restoration_error,
            "stopped_at": utc_now(),
        }
        stop["semantic_sha256"] = semantic_sha256(stop)
        write_once(output_root / "technical_stop.json", stop)
        descriptor = os.open(
            output_root / "technical_traceback.log",
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(traceback.format_exc())
            stream.flush()
            os.fsync(stream.fileno())
        return 1
    finally:
        signal.alarm(0)
        if config_baseline is not None and model is not None:
            model.language_model.config.proportion_attn_var = config_baseline[0]
            model.language_model.config.reusable_patches = config_baseline[1]
        model = action_head = proprio_projector = processor = None
        gc.collect()
        try:
            if "torch" in locals() and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except BaseException:
            pass
        if checkpoint_baseline is not None and restore_checkpoint_exact_fn is not None:
            try:
                restore_checkpoint_exact_fn(checkpoint, checkpoint_baseline)
            except BaseException:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
