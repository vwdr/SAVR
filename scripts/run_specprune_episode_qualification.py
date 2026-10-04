#!/usr/bin/env python3
"""Frozen multi-query and episode integration check; not a performance benchmark."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback
import run_openvla_original_baseline as base
import run_specprune_qualification as prior

ROOT = Path("/home/ved/SAVR")
CAPS = dict(queries=512, episodes=2, seconds=1800, memory_mib=23552, artifact_bytes=268435456)
PRIOR_CONFIG = "configs/openvla/specprune_real_qualification_v1.json"
PRIOR_SHA = "8adc9dde86d08f8d717c96e4017250229c71d7f5062ebe697cbb4c53eb75e168"
CONDITION = "d78f949af14d367e4a3057d13b211cc12f9ea7cbe00b6401c9bc5e1f74787bb9"
STATE_SHA = "4e3eaf7a315bc3bafb6db4837c8eef0e5f8aea717782a2a3ea33139392d966ed"

def validate_contract(config):
    if (config["schema_version"] != "specprune-episode-qualification-v1"
            or config["caps"] != CAPS or config["tolerance"] != 1e-6
            or config["output_root"] != "results/specprune-episode-qualification-v01"
            or config["offline_calls"] != 56 or config["frame_indices"] != [0, 1]
            or config["episode_modes"] != ["dense", "compressed"]
            or config["condition_id"] != CONDITION or config["initial_state_sha256"] != STATE_SHA
            or config["automatic_retry"] is not False):
        raise ValueError("frozen integration contract differs")

def verify(config):
    validate_contract(config)
    if base.sha(ROOT / PRIOR_CONFIG) != PRIOR_SHA:
        raise ValueError("prior qualification configuration changed")
    reference, manifest = prior.verify(json.loads((ROOT / PRIOR_CONFIG).read_text()))
    if config["observation_ids"] != [r["trajectory_id"] for r in manifest["inputs"]]:
        raise ValueError("offline observation identities differ")
    if config["gpu"] != reference["gpu"] or reference["episode_conditions"][0]["condition_id"] != CONDITION:
        raise ValueError("GPU or development condition differs")
    for relative, expected in config["authenticated_files"].items():
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT) or base.sha(path) != expected:
            raise ValueError(f"source identity differs: {relative}")
    import h5py
    for row in manifest["inputs"]:
        with h5py.File(ROOT / manifest["data_root_relative"] / row["source_path"], "r") as handle:
            obs = handle[f"data/{row['original_trajectory_id']}/obs"]
            if any(len(obs[k]) < 2 for k in ("agentview_rgb", "eye_in_hand_rgb", "ee_states", "gripper_states")):
                raise ValueError("two fixed frames unavailable")
    return reference, manifest

def reconcile(config, summary, records):
    validate_contract(config)
    if (summary.get("complete") is not True or summary.get("offline_calls") != 56
            or len(records["offline"]) != 8 or len(records["episodes"]) != 2
            or not 56 < summary["model_queries"] <= CAPS["queries"]
            or not 0 <= summary["elapsed_seconds"] < CAPS["seconds"]
            or not 0 <= summary["peak_aggregate_gpu_memory_mib"] < CAPS["memory_mib"]
            or summary.get("checkpoint_unchanged") is not True
            or summary.get("authenticated_files_unchanged") is not True
            or summary.get("automatic_retry") is not False
            or summary.get("training_performed") is not False
            or summary.get("positive_method_result") is not False):
        raise ValueError("incomplete or invalid summary")
    if [r["observation_id"] for r in records["offline"]] != config["observation_ids"]:
        raise ValueError("offline observation accounting differs")
    for row in records["offline"]:
        if (row["calls"] != 7 or row["compressed_query_counts"] != [1, 2, 1]
                or row["all_original_layers"] is not True
                or row["persistent_indices_and_confidence"] is not True
                or len(row["dense_parity"]) != 2):
            raise ValueError("multi-query contract failed")
        for errors in row["dense_parity"] + [row["reset_parity"]]:
            if set(errors) != {"hidden", "normalized", "actions"} or any(
                    not isinstance(v, (int, float)) or not 0 <= v <= 1e-6 for v in errors.values()):
                raise ValueError("identity parity failed")
    if [r["mode"] for r in records["episodes"]] != ["dense", "compressed"]:
        raise ValueError("episode arm order differs")
    for row in records["episodes"]:
        if (row["condition_id"] != CONDITION or row["initial_state_sha256"] != STATE_SHA
                or not 1 <= row["executed_steps"] <= 220
                or not 1 <= row["policy_queries"] <= row["executed_steps"]
                or type(row["success"]) is not bool or row["all_original_layers"] is not True):
            raise ValueError("invalid episode accounting")
    dense = records["episodes"][0]
    if (dense["success"], dense["executed_steps"], dense["policy_queries"]) != (True, 78, 10):
        raise ValueError("dense bridge changed the consumed reference episode")
    if dense["controller_enabled"] or not records["episodes"][1]["controller_enabled"]:
        raise ValueError("controller mode differs")
    if summary["model_queries"] != 56 + sum(r["policy_queries"] for r in records["episodes"]):
        raise ValueError("query count mismatch")
    return True

def run(config, reference, manifest, run_root, resources):
    import h5py
    import numpy as np
    import torch
    import transformers
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    if transformers.__version__ != "4.40.1" or torch.cuda.device_count() != 1:
        raise RuntimeError("expected original Transformers and one visible GPU")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(23000 / 24576, 0)
    sys.path.insert(0, str(ROOT / "third_party/openvla-oft"))
    sys.path.insert(0, str(ROOT / "src"))
    from experiments.robot import openvla_utils as utils
    from experiments.robot.libero import run_libero_eval as evaluation
    from experiments.robot.robot_utils import set_seed_everywhere
    from savr.cac.c1 import instruction_token_indices
    from savr.openvla.attention_diagnostic import LlamaAttentionDiagnostic
    from savr.openvla.official_semantics import (
        OfficialActionHeadCapture, prepare_semantic_query, derive_official_layout,
        select_official_action_hidden,
    )
    from savr.openvla.specprune import EpisodeState, SpecPruneSelection, low_change_indices, pruned_decoder_forward

    checkpoint = ROOT / reference["checkpoint"]
    names = {p.name for p in checkpoint.iterdir()}
    cfg = evaluation.GenerateConfig(
        pretrained_checkpoint=str(checkpoint), task_suite_name="libero_spatial",
        num_trials_per_task=0, seed=7, local_log_dir=str(run_root / "logs"),
        use_wandb=False, center_crop=True, num_open_loop_steps=8,
        num_images_in_input=2, use_proprio=True, use_l1_regression=True,
        use_diffusion=False, use_film=False,
    )
    evaluation.validate_config(cfg)
    set_seed_everywhere(7)
    update, sync = utils.update_auto_map, utils.check_model_logic_mismatch
    utils.update_auto_map = lambda _path: None
    utils.check_model_logic_mismatch = lambda _path: None
    try:
        model, head, proprio, noisy, processor = evaluation.initialize_model(cfg)
    finally:
        utils.update_auto_map, utils.check_model_logic_mismatch = update, sync
    if head is None or proprio is None or noisy is not None or any(m.training for m in (model, head, proprio)):
        raise RuntimeError("released regression components differ")
    decoder = model.language_model.model
    if len(decoder.layers) != 32 or any(type(l.self_attn).__name__ != "LlamaSdpaAttention" for l in decoder.layers):
        raise RuntimeError("32 original SDPA layers required")
    base.write_once(run_root / "loaded_runtime.json", dict(
        torch_version=torch.__version__, transformers_version=transformers.__version__,
        model_class=type(model).__module__ + "." + type(model).__name__,
        attention_classes=[type(l.self_attn).__name__ for l in decoder.layers],
        tensorflow_gpu_disabled=True, checkpoint_metadata_mutation_disabled=True,
    ))

    from savr.openvla.specprune_episode import CameraPair, SpecPruneEpisodeQuery, execute_specprune_episode
    from experiments.robot.libero.libero_utils import get_libero_env
    from experiments.robot.robot_utils import get_image_resize_size
    from libero.libero import benchmark
    queries = {mode: SpecPruneEpisodeQuery(
        model=model, head=head, proprio=proprio, processor=processor, cfg=cfg, utils=utils,
        instruction_indexer=instruction_token_indices, enabled=mode == "compressed")
        for mode in ("dense", "compressed")}

    def outputs(capture, actions):
        values = dict(hidden=capture.exact_hidden().detach().float().cpu().numpy(),
                      normalized=capture.exact_output().detach().float().cpu().numpy().reshape(8, 7),
                      actions=np.asarray(actions))
        if values["hidden"].shape != (1, 56, 4096) or values["actions"].shape != (8, 7) or any(
                not np.isfinite(v).all() for v in values.values()):
            raise ValueError("action boundary differs")
        return values

    @torch.inference_mode()
    def call(observation, instruction, mode, previous=None, state=None, capture=True):
        resources.consume()
        with LlamaAttentionDiagnostic(torch, decoder.layers, "original") as audit:
            with OfficialActionHeadCapture(head) as head_capture:
                if mode == "native":
                    actions = evaluation.get_action(cfg, model, {k: np.asarray(v).copy() for k,v in observation.items()},
                        instruction, processor=processor, action_head=head, proprio_projector=proprio,
                        noisy_action_projector=noisy, use_film=False)
                else:
                    actions = queries[mode](observation, instruction, previous, state)
        if len(audit.records) != 32:
            raise ValueError("native layer count differs")
        return outputs(head_capture, actions) if capture else actions

    def parity(left, right):
        differences = {k: float(np.max(np.abs(left[k].astype(np.float64)-right[k].astype(np.float64)))) for k in left}
        if any(not np.isfinite(v) or v > 1e-6 for v in differences.values()):
            raise ValueError("action boundary parity failed")
        return differences

    def progress(**kw):
        value = dict(model_queries=resources.queries, **kw)
        with (run_root / "progress.jsonl").open("a") as stream:
            stream.write(json.dumps(value) + "\n")
        print(json.dumps(value), flush=True)

    offline = []
    for index, identity in enumerate(manifest["inputs"]):
        observations = []
        with h5py.File(ROOT / manifest["data_root_relative"] / identity["source_path"], "r") as handle:
            group = handle[f"data/{identity['original_trajectory_id']}/obs"]
            for frame in (0, 1):
                observations.append(dict(full_image=np.asarray(group["agentview_rgb"][frame]).copy(),
                    wrist_image=np.asarray(group["eye_in_hand_rgb"][frame]).copy(),
                    state=np.concatenate((group["ee_states"][frame], group["gripper_states"][frame])).copy()))
        cfg.unnorm_key = identity["normalization_statistics_key"]
        instruction = identity["language_instruction"]
        previous = CameraPair(0, observations[0]["full_image"].copy(), observations[0]["wrist_image"].copy())
        dense_state = EpisodeState(); dense_state.reset(identity["trajectory_id"])
        errors = []
        for observation in observations:
            native = call(observation, instruction, "native")
            dense = call(observation, instruction, "dense", previous, dense_state)
            errors.append(parity(native, dense))
        state = EpisodeState(); state.reset(identity["trajectory_id"])
        first = call(observations[0], instruction, "compressed", previous, state)
        qcounts = [state.query]
        if not state.confidence or not state.previous_indices:
            raise ValueError("first compressed query did not create history")
        confidence_ids = {k: id(v) for k,v in state.confidence.items()}
        call(observations[1], instruction, "compressed", previous, state)
        qcounts.append(state.query)
        if any(id(state.confidence[k]) != v for k,v in confidence_ids.items()) or not state.previous_indices:
            raise ValueError("within-episode confidence or indices lost")
        state.reset(identity["trajectory_id"])
        if state.confidence or state.previous_indices or state.query or state.precise:
            raise ValueError("episode reset leaked state")
        reset = call(observations[0], instruction, "compressed", previous, state)
        qcounts.append(state.query)
        offline.append(dict(observation_id=identity["trajectory_id"], calls=7,
            dense_parity=errors, reset_parity=parity(first, reset), compressed_query_counts=qcounts,
            persistent_indices_and_confidence=True, all_original_layers=True))
        progress(offline_observations_complete=index+1)
    if resources.queries != 56:
        raise ValueError("offline count differs")

    row = reference["episode_conditions"][0]
    cfg.task_suite_name = row["suite"]
    cfg.unnorm_key = row["suite"] if row["suite"] in model.norm_stats else row["suite"] + "_no_noops"
    suite = benchmark.get_benchmark_dict()[row["suite"]]()
    matches = [i for i in range(suite.n_tasks) if suite.get_task(i).name == row["task_id"]]
    if len(matches) != 1 or evaluation.TASK_MAX_STEPS[row["suite"]] != 220:
        raise ValueError("task identity or original horizon differs")
    task_index = matches[0]
    initial_state = suite.get_task_init_states(task_index)[0].copy()
    if hashlib.sha256(initial_state.tobytes()).hexdigest() != STATE_SHA:
        raise ValueError("initial simulator state differs")
    episodes = []
    for mode in config["episode_modes"]:
        set_seed_everywhere(7)
        env, instruction = get_libero_env(suite.get_task(task_index), cfg.model_family, resolution=cfg.env_img_res)
        before = resources.queries
        try:
            record = execute_specprune_episode(evaluation, cfg, env, initial_state, instruction,
                lambda obs, text, previous, state: call(obs, text, mode, previous, state, capture=False),
                get_image_resize_size(cfg), episode_id=row["condition_id"], controller=mode == "compressed")
        finally:
            env.close()
        if record["policy_queries"] != resources.queries-before:
            raise ValueError("simulator query accounting differs")
        episodes.append(dict(record, condition_id=CONDITION, initial_state_sha256=STATE_SHA,
                             mode=mode, all_original_layers=True))
        progress(episodes_complete=len(episodes))
    if names != {p.name for p in checkpoint.iterdir()}:
        raise ValueError("checkpoint inventory changed")
    verify(config)
    resources.close()
    base.write_once(run_root / "records.json", dict(offline=offline, episodes=episodes))
    return dict(complete=True, offline_calls=56, model_queries=resources.queries,
        checkpoint_unchanged=True, authenticated_files_unchanged=True, automatic_retry=False,
        training_performed=False, positive_method_result=False,
        peak_aggregate_gpu_memory_mib=resources.peak,
        artifacts_sha256={p: base.sha(run_root / p) for p in ("records.json", "loaded_runtime.json", "progress.jsonl")})

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if Path.cwd().resolve() != ROOT or not Path(sys.prefix).resolve().is_relative_to(ROOT / "envs/openvla-oft"):
        raise SystemExit("original project runtime and root required")
    config = json.loads(args.config.read_text())
    reference, manifest = verify(config)
    if args.preflight_only:
        print(json.dumps(dict(preflight_passed=True, observations=8, calls=56, episodes=2,
                             authenticated_files=len(config["authenticated_files"]) + len(reference["authenticated_files"]))))
        return 0
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(config["gpu"]["index"]):
        raise SystemExit("selected GPU mismatch")
    initial = base.snapshot(config["gpu"]["index"])
    if initial["uuid"] != config["gpu"]["uuid"] or initial["memory_mib"] > 1024 or initial["utilization"] > 5:
        raise SystemExit("selected GPU is not idle; nothing launched")
    run_root = ROOT / config["output_root"]
    run_root.mkdir(exist_ok=False)
    base.environment(run_root)
    base.write_once(run_root / "launch.json", dict(config=config, config_sha256=base.sha(args.config),
                    pid=os.getpid(), started_at=datetime.now(timezone.utc).isoformat(), initial_gpu=initial))
    resources = base.Resources(config)
    resources.peak = initial["memory_mib"]
    resources.thread.start()
    start = time.monotonic()
    def timeout(*_args):
        raise RuntimeError("qualification wall-clock cap reached")
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(CAPS["seconds"])
    try:
        result = run(config, reference, manifest, run_root, resources)
        result["elapsed_seconds"] = time.monotonic() - start
        result["completed_at"] = datetime.now(timezone.utc).isoformat()
        reconcile(config, result, json.loads((run_root / "records.json").read_text()))
        if sum(p.stat().st_size for p in run_root.rglob("*") if p.is_file()) >= CAPS["artifact_bytes"] - 65536:
            raise RuntimeError("artifact cap reached")
        base.write_once(run_root / "worker_summary.json", result)
        print(json.dumps(dict(complete=True, output_root=str(run_root))), flush=True)
        return 0
    except BaseException as error:
        with (run_root / "technical_traceback.log").open("x") as stream:
            traceback.print_exc(file=stream)
        base.write_once(run_root / "technical_stop.json", dict(complete=False, error_type=type(error).__name__,
                        error=str(error), model_queries=resources.queries, automatic_retry=False,
                        elapsed_seconds=time.monotonic() - start, peak_aggregate_gpu_memory_mib=resources.peak))
        print(json.dumps(dict(complete=False, technical_stop=type(error).__name__)), flush=True)
        return 1
    finally:
        signal.alarm(0)
        resources.stop.set()
        resources.thread.join(timeout=12)


if __name__ == "__main__":
    raise SystemExit(main())
