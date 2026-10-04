#!/usr/bin/env python3
"""Fixed forty-task original-runtime dense evaluation; no cache or training."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import traceback

ROOT = Path("/home/ved/SAVR")


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else hash_stream(stream)


def hash_stream(stream):
    h = hashlib.sha256()
    for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
        h.update(block)
    return h.hexdigest()


def write_once(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def snapshot(gpu):
    value = subprocess.check_output([
        "nvidia-smi", "-i", str(gpu),
        "--query-gpu=uuid,memory.used,utilization.gpu", "--format=csv,noheader,nounits",
    ], text=True, timeout=10).strip().split(",")
    if len(value) != 3:
        raise RuntimeError("invalid selected-GPU telemetry")
    return {"uuid": value[0].strip(), "memory_mib": int(value[1]), "utilization": int(value[2])}


class Resources:
    def __init__(self, config):
        self.config = config
        self.queries = 0
        self.peak = 0
        self.error = None
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.sample, daemon=True)

    def sample(self):
        while not self.stop.wait(2):
            try:
                value = snapshot(self.config["gpu"]["index"])
                self.peak = max(self.peak, value["memory_mib"])
                if value["uuid"] != self.config["gpu"]["uuid"]:
                    raise RuntimeError("GPU identity changed")
                if self.peak >= self.config["caps"]["memory_mib"]:
                    raise RuntimeError("aggregate GPU memory cap reached")
            except Exception as error:
                self.error = str(error)
                return

    def consume(self):
        if self.error:
            raise RuntimeError(self.error)
        if self.queries >= self.config["caps"]["queries"]:
            raise RuntimeError("query cap reached")
        self.queries += 1

    def close(self):
        self.stop.set()
        self.thread.join(timeout=12)
        if self.thread.is_alive():
            raise RuntimeError("resource sampler failed to close")
        if self.error:
            raise RuntimeError(self.error)


def environment(run):
    for name, directory in {
        "HF_HOME": "hf", "HF_HUB_CACHE": "hf/hub", "HF_MODULES_CACHE": "hf/modules",
        "TRANSFORMERS_CACHE": "hf/transformers", "TORCH_HOME": "torch",
        "XDG_CACHE_HOME": "xdg", "MPLCONFIGDIR": "matplotlib", "TMPDIR": "tmp",
        "WANDB_DIR": "wandb", "NUMBA_CACHE_DIR": "numba",
    }.items():
        target = run / "runtime-cache" / directory
        target.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(target)
    os.environ.update({
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "PYTHONDONTWRITEBYTECODE": "1", "WANDB_MODE": "disabled",
        "TOKENIZERS_PARALLELISM": "false", "MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl",
        "LIBERO_CONFIG_PATH": str(ROOT / "configs/pair/libero_runtime"),
        "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
    })
    sys.dont_write_bytecode = True


def select_conditions(prior, suites):
    selected = []
    for suite in suites:
        rows = sorted([r for r in prior if r["suite"] == suite and
                       r["population"] == "headroom_stage1" and r["initial_state_id"] == 0],
                      key=lambda r: r["task_id"])
        if len(rows) != 10 or len({r["task_id"] for r in rows}) != 10:
            raise RuntimeError("expected exactly ten unique consumed tasks per suite")
        if any(r["seed"] != 7 for r in rows):
            raise RuntimeError("development seed changed")
        selected.extend(dict(row, arm_order=["original"]) for row in rows)
    if len({r["condition_id"] for r in selected}) != 40:
        raise RuntimeError("duplicate condition identity")
    return selected


def verify(config):
    if config["schema_version"] != "openvla-original-baseline-40task-v1":
        raise RuntimeError("invalid configuration schema")
    for relative, expected in config["authenticated_files"].items():
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT) or sha(path) != expected:
            raise RuntimeError(f"source/input identity mismatch: {relative}")
    if config["offline_queries"] != 32 or len(config["episode_conditions"]) != 40:
        raise RuntimeError("frozen population count changed")
    suites = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
    if config["suites"] != suites:
        raise RuntimeError("suite order changed")
    prior = [json.loads(line) for line in
             (ROOT / "reports/cac_c0_recovery01/simulator_populations_v1.jsonl").read_text().splitlines()]
    selected = select_conditions(prior, suites)
    if config["episode_conditions"] != selected:
        raise RuntimeError("frozen deterministic development selection differs")
    if any(row["initial_state_id"] != 0 for row in config["episode_conditions"]):
        raise RuntimeError("unapproved initial state")
    if config["caps"] != {"queries": 1692, "episodes": 40, "seconds": 7200,
                           "memory_mib": 23552, "artifact_bytes": 536870912}:
        raise RuntimeError("frozen resource limits changed")
    manifest = json.loads((ROOT / "configs/pair/p3_inputs_v1.json").read_text())
    for row in manifest["inputs"]:
        if row["split"] != "train" or sha(ROOT / manifest["data_root_relative"] / row["source_path"]) != row["source_sha256"]:
            raise RuntimeError("offline observation identity mismatch")
    return manifest


def run(config, run_root, manifest, resources):
    import h5py
    import numpy as np
    import torch
    import transformers
    import tensorflow as tf

    # TensorFlow is used only by image preprocessing. Do not let it claim
    # GPU memory alongside the PyTorch policy.
    tf.config.set_visible_devices([], "GPU")
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)

    if transformers.__version__ != "4.40.1" or torch.cuda.device_count() != 1:
        raise RuntimeError("expected original runtime and exactly one visible GPU")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(23000 / 24576, 0)
    sys.path.insert(0, str(ROOT / "third_party/openvla-oft"))
    sys.path.insert(0, str(ROOT / "src"))
    from experiments.robot import openvla_utils as utils
    from experiments.robot.libero import run_libero_eval as evaluation
    from experiments.robot.libero.libero_utils import get_libero_env
    from experiments.robot.robot_utils import get_image_resize_size, set_seed_everywhere
    from libero.libero import benchmark
    from savr.cac.c1 import instruction_token_indices
    from savr.openvla.attention_diagnostic import LlamaAttentionDiagnostic, execute_episode
    from savr.openvla.official_semantics import (
        OfficialBoundaryCapture, prepare_semantic_query, structurally_aligned_dense_forward,
    )

    checkpoint = ROOT / config["checkpoint"]
    checkpoint_names = {p.name for p in checkpoint.iterdir()}
    cfg = evaluation.GenerateConfig(
        pretrained_checkpoint=str(checkpoint), task_suite_name="libero_spatial",
        num_trials_per_task=0, seed=7, local_log_dir=str(run_root / "logs"),
        use_wandb=False, center_crop=True, num_open_loop_steps=8,
        num_images_in_input=2, use_proprio=True, use_l1_regression=True,
        use_diffusion=False, use_film=False,
    )
    evaluation.validate_config(cfg)
    set_seed_everywhere(7)
    original_update, original_sync = utils.update_auto_map, utils.check_model_logic_mismatch
    utils.update_auto_map = lambda _path: None
    utils.check_model_logic_mismatch = lambda _path: None
    try:
        model, head, proprio, noisy, processor = evaluation.initialize_model(cfg)
    finally:
        utils.update_auto_map, utils.check_model_logic_mismatch = original_update, original_sync
    if head is None or proprio is None or noisy is not None:
        raise RuntimeError("released L1 model components differ")
    if model.training or head.training or proprio.training:
        raise RuntimeError("a model component is still in training mode")
    layers = model.language_model.model.layers
    if len(layers) != 32 or any(type(layer.self_attn).__name__ != "LlamaSdpaAttention" for layer in layers):
        raise RuntimeError("live model is not using 32 original SDPA layers")
    resize_size = get_image_resize_size(cfg)
    write_once(run_root / "loaded_runtime.json", {
        "model_class": type(model).__module__ + "." + type(model).__name__,
        "transformers_version": transformers.__version__, "torch_version": torch.__version__,
        "attention_classes": [type(layer.self_attn).__name__ for layer in layers],
        "resize_size": resize_size, "configuration": vars(cfg),
        "checkpoint_metadata_mutation_disabled": True,
        "tensorflow_gpu_disabled": True,
    })

    def cpu(value):
        if isinstance(value, torch.Tensor):
            value = value.detach().float().cpu().numpy()
        return np.asarray(value, dtype=np.float64)

    def difference(a, b, name):
        a, b = cpu(a), cpu(b)
        if name == "normalized_proprio":
            if a.size != 8 or b.size != 8:
                raise RuntimeError("invalid proprio shape")
            a, b = a.reshape(1, 8), b.reshape(1, 8)
        if name == "normalized_actions":
            if a.size != 56 or b.size != 56:
                raise RuntimeError("invalid normalized-action shape")
            a, b = a.reshape(8, 7), b.reshape(8, 7)
        if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise RuntimeError(f"invalid parity boundary {name}")
        return float(np.max(np.abs(a - b)))

    @torch.inference_mode()
    def official(observation, instruction, mode="original", capture=False):
        resources.consume()
        observation = {key: np.asarray(value).copy() for key, value in observation.items()}
        with LlamaAttentionDiagnostic(torch, layers, mode) as attention:
            if capture:
                with OfficialBoundaryCapture(model, head) as boundaries:
                    result = evaluation.get_action(
                        cfg, model, observation, instruction, processor=processor,
                        action_head=head, proprio_projector=proprio,
                        noisy_action_projector=noisy, use_film=False,
                    )
                values = boundaries.exact_values()
            else:
                result = evaluation.get_action(
                    cfg, model, observation, instruction, processor=processor,
                    action_head=head, proprio_projector=proprio,
                    noisy_action_projector=noisy, use_film=False,
                )
                values = None
        array = np.asarray(result, dtype=np.float32)
        if array.shape != (8, 7) or not np.isfinite(array).all():
            raise RuntimeError("official evaluator did not return finite 8x7 actions")
        if capture:
            return array, values, attention.records
        return array

    @torch.inference_mode()
    def custom(observation, instruction):
        resources.consume()
        prepared = prepare_semantic_query(
            torch_module=torch, np_module=np, model=model, processor=processor,
            proprio_projector=proprio, prepare_images=utils.prepare_images_for_vla,
            normalize_proprio=utils.normalize_proprio, instruction_indexer=instruction_token_indices,
            cfg=cfg, raw_scene=observation["full_image"].copy(),
            raw_wrist=observation["wrist_image"].copy(), raw_state=observation["state"].copy(),
            instruction=instruction,
        )
        with LlamaAttentionDiagnostic(torch, layers, "original"):
            output = structurally_aligned_dense_forward(
                torch_module=torch, np_module=np, model=model, action_head=head,
                cfg=cfg, prepared=prepared, use_cache=False,
            )
        return prepared, output

    parity_map = {
        "input_ids": "input_ids", "input_embeddings": "input_embeddings",
        "action_mask": "action_mask", "pixel_values": "preprocessed_pixels",
        "language_embeddings": "language_embeddings", "vision_output": "vision_output",
        "pre_proprio_projected": "vision_output", "normalized_proprio": "normalized_proprio",
        "projected_with_proprio": "projected_patches",
    }
    offline = []
    for index, identity in enumerate(manifest["inputs"]):
        with h5py.File(ROOT / manifest["data_root_relative"] / identity["source_path"], "r") as handle:
            group = handle[f"data/{identity['original_trajectory_id']}/obs"]
            observation = {
                "full_image": np.asarray(group["agentview_rgb"][0]).copy(),
                "wrist_image": np.asarray(group["eye_in_hand_rgb"][0]).copy(),
                "state": np.concatenate((group["ee_states"][0], group["gripper_states"][0])).copy(),
            }
        cfg.unnorm_key = identity["normalization_statistics_key"]
        instruction = identity["language_instruction"]
        reference, ref, native_masks = official(observation, instruction, capture=True)
        prepared, own = custom(observation, instruction)
        differences = {name: difference(ref[name], getattr(prepared, attribute), name)
                       for name, attribute in parity_map.items()}
        for name in ("masked_input_embeddings", "multimodal_embeddings", "multimodal_attention_mask",
                     "action_hidden", "normalized_actions"):
            differences[name] = difference(ref[name], own[name], name)
        differences["executed_action_chunk"] = difference(reference, own["actions"], "actions")
        if any(value > 1e-6 for value in differences.values()):
            write_once(run_root / "reference_mismatch.json", {"observation": index, "differences": differences})
            raise RuntimeError("original-runtime official/custom parity failed")
        causal, cause, causal_masks = official(observation, instruction, mode="causal_control", capture=True)
        restored, restore, _ = official(observation, instruction, capture=True)
        restoration = {name: difference(ref[name], restore[name], name) for name in ref}
        restoration["executed_action_chunk"] = difference(reference, restored, "actions")
        if any(value > 1e-6 for value in restoration.values()):
            raise RuntimeError("original policy was not restored after causal control")
        offline.append({
            "observation_id": identity["trajectory_id"], "suite": identity["suite"],
            "parity_max_abs": differences, "restoration_max_abs": restoration,
            "causal_vs_original_normalized_max_abs": difference(ref["normalized_actions"], cause["normalized_actions"], "normalized_actions"),
            "causal_vs_original_action_max_abs": difference(reference, causal, "actions"),
            "causal_vs_original_action_mean_abs": float(np.mean(np.abs(reference.astype(np.float64) - causal))),
            "original_masks": native_masks, "causal_masks": causal_masks,
        })
        del prepared, own, ref, cause, restore
        print(json.dumps({"stage": "reference_check", "observations_complete": index + 1,
                          "model_queries": resources.queries}), flush=True)
    if resources.queries != 32:
        raise RuntimeError("incorrect offline query count")
    write_once(run_root / "reference_check.json", {"complete": True, "records": offline, "queries": 32})
    if resources.error:
        raise RuntimeError(resources.error)

    suites = {name: benchmark.get_benchmark_dict()[name]() for name in config["suites"]}
    for name, suite in suites.items():
        actual = {suite.get_task(i).name for i in range(suite.n_tasks)}
        frozen = {r["task_id"] for r in config["episode_conditions"] if r["suite"] == name}
        if suite.n_tasks != 10 or actual != frozen:
            raise RuntimeError("actual benchmark task inventory differs from frozen population")
    episodes = []
    for ordinal, row in enumerate(config["episode_conditions"]):
        cfg.task_suite_name = row["suite"]
        cfg.unnorm_key = row["suite"] if row["suite"] in model.norm_stats else row["suite"] + "_no_noops"
        suite = suites[row["suite"]]
        matches = [i for i in range(suite.n_tasks) if suite.get_task(i).name == row["task_id"]]
        if len(matches) != 1:
            raise RuntimeError("frozen task name did not resolve uniquely")
        task_index = matches[0]
        state = suite.get_task_init_states(task_index)[row["initial_state_id"]].copy()
        initial_state_sha256 = hashlib.sha256(state.tobytes()).hexdigest()
        for mode in row["arm_order"]:
            set_seed_everywhere(7)
            env, instruction = get_libero_env(suite.get_task(task_index), cfg.model_family, resolution=cfg.env_img_res)
            before = resources.queries
            try:
                record = execute_episode(
                    evaluation, cfg, env, state, instruction,
                    lambda obs, text: official(obs, text, mode=mode), resize_size,
                )
            finally:
                env.close()
            if record["policy_queries"] != resources.queries - before:
                raise RuntimeError("episode query count mismatch")
            episodes.append({**row, "mode": mode, "initial_state_sha256": initial_state_sha256, **record})
            with (run_root / "progress.jsonl").open("a") as stream:
                stream.write(json.dumps({"episodes_completed": len(episodes),
                                         "model_queries": resources.queries}) + "\n")
            print(json.dumps({"stage": "closed_loop", "episodes_complete": len(episodes),
                              "model_queries": resources.queries}), flush=True)
    if len(episodes) != 40:
        raise RuntimeError("incomplete forty-task dense population")
    if checkpoint_names != {p.name for p in checkpoint.iterdir()}:
        raise RuntimeError("checkpoint inventory changed")
    for relative, expected in config["authenticated_files"].items():
        if sha(ROOT / relative) != expected:
            raise RuntimeError("authenticated source/checkpoint/input bytes changed")
    resources.close()
    if sum(p.stat().st_size for p in run_root.rglob("*") if p.is_file()) >= config["caps"]["artifact_bytes"]:
        raise RuntimeError("diagnostic artifact cap reached")
    write_once(run_root / "episodes.json", {"complete": True, "records": episodes})
    return {"complete": True, "reference_check_passed": True, "conditions": 40,
            "episode_count": 40, "model_queries": resources.queries,
            "successes": {mode: sum(r["success"] for r in episodes if r["mode"] == mode)
                          for mode in ("original",)},
            "peak_aggregate_gpu_memory_mib": resources.peak,
            "torch_peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
            "checkpoint_unchanged": True, "authenticated_files_unchanged": True, "training_performed": False,
            "positive_method_result": False, "automatic_retry": False,
            "artifact_sha256": {name: sha(run_root / name) for name in
                                ("reference_check.json", "episodes.json", "loaded_runtime.json")}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if Path.cwd().resolve() != ROOT or not Path(sys.prefix).resolve().is_relative_to(ROOT / "envs/openvla-oft"):
        raise SystemExit("diagnostic requires the original project-local runtime")
    config = json.loads(args.config.read_text())
    gpu = config["gpu"]["index"]
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(gpu):
        raise SystemExit("selected GPU mismatch")
    manifest = verify(config)
    if args.preflight_only:
        print(json.dumps({"preflight_passed": True, "conditions": 40, "authenticated_files": len(config["authenticated_files"])}))
        return 0
    initial = snapshot(gpu)
    if initial["uuid"] != config["gpu"]["uuid"] or initial["memory_mib"] > 1024 or initial["utilization"] > 5:
        raise SystemExit("selected GPU is not available; no model launched")
    run_root = (ROOT / config["output_root"]).resolve()
    if not run_root.is_relative_to(ROOT / "results"):
        raise SystemExit("invalid output path")
    run_root.mkdir(exist_ok=False)
    environment(run_root)
    write_once(run_root / "launch.json", {"config": config, "config_sha256": sha(args.config),
                                         "pid": os.getpid(), "initial_gpu": initial,
                                         "started_at": datetime.now(timezone.utc).isoformat()})
    resources = Resources(config)
    resources.peak = initial["memory_mib"]
    started = time.monotonic()
    resources.thread.start()

    def timeout(_signal, _frame):
        raise RuntimeError("diagnostic wall-clock cap reached")

    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(config["caps"]["seconds"])
    try:
        result = run(config, run_root, manifest, resources)
        result["elapsed_seconds"] = time.monotonic() - started
        result["completed_at"] = datetime.now(timezone.utc).isoformat()
        write_once(run_root / "worker_summary.json", result)
        print(json.dumps({"complete": True, "output_root": str(run_root)}), flush=True)
        return 0
    except BaseException as error:
        resources.stop.set()
        with (run_root / "technical_traceback.log").open("x") as stream:
            traceback.print_exc(file=stream)
        write_once(run_root / "technical_stop.json", {
            "complete": False, "error_type": type(error).__name__, "error": str(error),
            "model_queries": resources.queries, "peak_aggregate_gpu_memory_mib": resources.peak,
            "elapsed_seconds": time.monotonic() - started, "automatic_retry": False,
            "partial_outcomes_disclosed": False,
        })
        print(json.dumps({"complete": False, "technical_stop": type(error).__name__}), flush=True)
        return 1
    finally:
        signal.alarm(0)
        resources.stop.set()
        resources.thread.join(timeout=12)


if __name__ == "__main__":
    raise SystemExit(main())

