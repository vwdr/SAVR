#!/usr/bin/env python3
"""New prospective reference evaluation; old workers and evidence stay unchanged."""
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
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from savr.openvla.contemporary_contract import CAPS, DESIGN_SHA, HORIZONS, reconcile, require, timing_slots, validate_design

HOOK_CAPS = dict(model_calls=80, episodes=0, seconds=1800,
                 aggregate_memory_mib=23552, artifact_bytes=268435456)
REQUIRED_FILES = (
    "scripts/run_contemporary_reference.py", "scripts/analyze_contemporary_reference.py",
    "scripts/run_openvla_original_baseline.py", "scripts/run_specprune_qualification.py",
    "src/savr/openvla/contemporary_contract.py", "src/savr/openvla/contemporary_episode.py",
    "src/savr/openvla/specprune.py", "src/savr/openvla/specprune_episode.py",
    "src/savr/openvla/official_semantics.py", "src/savr/openvla/attention_diagnostic.py",
    "src/savr/openvla/execution_precision.py", "src/savr/cac/c1.py",
    "configs/openvla/contemporary_reference_design_v1.json", "configs/pair/p3_inputs_v1.json",
    "results/openvla-original-baseline-40task-v01/episodes.json",
)


def scoped(relative):
    path = (ROOT / relative).resolve()
    require(not Path(relative).is_absolute() and path.is_relative_to(ROOT), "path outside project")
    return path


def verify(config):
    mode = config["mode"]
    require(config["schema_version"] == "contemporary-reference-executable-v1"
            and config["launch_ready"] is True and config["automatic_retry"] is False
            and mode in ("hook_qualification", "evaluation") and config["tolerance"] == 1e-6,
            "unfrozen execution contract")
    expected_caps = HOOK_CAPS if mode == "hook_qualification" else CAPS
    expected_output = "results/contemporary-hook-qualification-v01" if mode == "hook_qualification" else "results/contemporary-reference-v01"
    require(config["caps"] == expected_caps and config["output_root"] == expected_output,
            "versioned output/caps differ")
    require(set(REQUIRED_FILES) <= set(config["authenticated_files"]), "required source not frozen")
    for relative, expected in config["authenticated_files"].items():
        require(base.sha(scoped(relative)) == expected, f"source identity differs: {relative}")
    require(config["worker_sha256"] == base.sha(Path(__file__))
            and config["analyzer_sha256"] == base.sha(ROOT / "scripts/analyze_contemporary_reference.py"),
            "worker/analyzer identity differs")
    designpath = ROOT / "configs/openvla/contemporary_reference_design_v1.json"
    require(base.sha(designpath) == DESIGN_SHA, "frozen design changed")
    design = json.loads(designpath.read_text())
    qualification = ROOT / "configs/openvla/specprune_real_qualification_v1.json"
    require(base.sha(qualification) == "8adc9dde86d08f8d717c96e4017250229c71d7f5062ebe697cbb4c53eb75e168", "qualification source changed")
    reference, manifest = prior.verify(json.loads(qualification.read_text()))
    validate_design(design, reference)
    original_episodes = ROOT / REQUIRED_FILES[-1]
    require(base.sha(original_episodes) == "4fb13ee64d1698fa6be14ae083d437db0e36ab10659924e88e65c611dad4dcb6",
            "reference state provenance differs")
    original_bundle = json.loads(original_episodes.read_text())
    require(original_bundle["complete"] is True and len(original_bundle["records"]) == 40,
            "reference episode bundle incomplete")
    states = {e["condition_id"]: e["initial_state_sha256"] for e in original_bundle["records"]}
    require(config["initial_state_sha256"] == states and len(states) == 40, "state manifest differs")
    require(config["observation_ids"] == [r["trajectory_id"] for r in manifest["inputs"]], "timing inputs differ")
    if mode == "evaluation":
        evidence = scoped(config["hook_qualification_summary"])
        require(base.sha(evidence) == config["hook_qualification_summary_sha256"], "hook qualification changed")
        q = json.loads(evidence.read_text())
        require(q.get("complete") is True and q.get("hook_neutrality_passed") is True
                and q.get("model_queries") == 80 and q.get("worker_sha256") == config["worker_sha256"],
                "matching completed hook qualification required")
        require(not (evidence.parent / "technical_stop.json").exists(), "stopped hook qualification")
        for relative, expected in q["artifacts_sha256"].items():
            path = (evidence.parent / relative).resolve()
            require(path.is_relative_to(evidence.parent) and base.sha(path) == expected, "hook artifact mismatch")
    return design, reference, manifest


def run(config, design, reference, manifest, run_root, resources):
    import h5py
    import numpy as np
    import torch
    import transformers
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    require(transformers.__version__ == "4.40.1" and torch.__version__ == "2.2.0+cu118"
            and torch.cuda.device_count() == 1, "original runtime/one GPU required")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(23000 / 24576, 0)
    sys.path.insert(0, str(ROOT / "third_party/openvla-oft"))
    from experiments.robot import openvla_utils as utils
    from experiments.robot.libero import run_libero_eval as evaluation
    from experiments.robot.robot_utils import set_seed_everywhere, get_image_resize_size
    from experiments.robot.libero.libero_utils import get_libero_env
    from libero.libero import benchmark
    from savr.cac.c1 import instruction_token_indices
    from savr.openvla.attention_diagnostic import LlamaAttentionDiagnostic
    from savr.openvla.official_semantics import OfficialActionHeadCapture
    from savr.openvla.execution_precision import execution_chunk_float32, processed_command_chunk
    from savr.openvla.specprune import EpisodeState
    from savr.openvla.specprune_episode import SpecPruneEpisodeQuery, CameraPair
    from savr.openvla.contemporary_episode import execute_measured_episode

    checkpoint = ROOT / reference["checkpoint"]
    names = {p.name for p in checkpoint.iterdir()}
    cfg = evaluation.GenerateConfig(pretrained_checkpoint=str(checkpoint), task_suite_name="libero_spatial",
        num_trials_per_task=0, seed=7, local_log_dir=str(run_root / "logs"), use_wandb=False,
        center_crop=True, num_open_loop_steps=8, num_images_in_input=2, use_proprio=True,
        use_l1_regression=True, use_diffusion=False, use_film=False)
    evaluation.validate_config(cfg)
    set_seed_everywhere(7)
    update, sync = utils.update_auto_map, utils.check_model_logic_mismatch
    utils.update_auto_map = lambda _path: None
    utils.check_model_logic_mismatch = lambda _path: None
    try:
        model, head, proprio, noisy, processor = evaluation.initialize_model(cfg)
    finally:
        utils.update_auto_map, utils.check_model_logic_mismatch = update, sync
    require(head is not None and proprio is not None and noisy is None
            and not any(m.training for m in (model, head, proprio)), "released inference components differ")
    decoder = model.language_model.model
    require(len(decoder.layers) == 32 and all(type(l.self_attn).__name__ == "LlamaSdpaAttention"
            for l in decoder.layers), "original 32-layer SDPA runtime required")
    require(all(evaluation.TASK_MAX_STEPS[s] == h for s, h in HORIZONS.items()), "original horizons changed")
    require(get_image_resize_size(cfg) == 224 and cfg.num_steps_wait == 10, "image/settling contract differs")
    base.write_once(run_root / "loaded_runtime.json", dict(torch_version=torch.__version__,
        transformers_version=transformers.__version__, model_class=type(model).__module__ + "." + type(model).__name__,
        attention_classes=[type(l.self_attn).__name__ for l in decoder.layers], configuration=vars(cfg),
        tensorflow_gpu_disabled=True, checkpoint_metadata_mutation_disabled=True))
    bridges = {arm: SpecPruneEpisodeQuery(model=model, head=head, proprio=proprio, processor=processor,
               cfg=cfg, utils=utils, instruction_indexer=instruction_token_indices, enabled=arm == "specprune")
               for arm in ("dense", "specprune")}
    records = dict(episodes=[], timing=[], parity=[], hook_checks=[])

    def append(name, value):
        if sum(p.stat().st_size for p in run_root.rglob("*") if p.is_file()) >= config["caps"]["artifact_bytes"] - 65536:
            raise RuntimeError("artifact cap reached")
        with (run_root / name).open("a") as stream:
            stream.write(json.dumps(value, allow_nan=False) + "\n")

    def progress():
        value = dict(episodes=len(records["episodes"]), timing_calls=len(records["timing"]),
                     hook_checks=len(records["hook_checks"]), model_queries=resources.queries)
        append("progress.jsonl", value)
        print(json.dumps(value), flush=True)

    def obs_hash(obs):
        digest = hashlib.sha256()
        for key in ("full_image", "wrist_image", "state"):
            a = np.asarray(obs[key])
            digest.update(key.encode()); digest.update(str((a.shape, a.dtype.str)).encode()); digest.update(a.tobytes())
        return digest.hexdigest()

    @torch.inference_mode()
    def call(arm, obs, text, previous, state):
        resources.consume()
        if arm == "native":
            raw = evaluation.get_action(cfg, model, {k: np.asarray(v).copy() for k, v in obs.items()}, text,
                processor=processor, action_head=head, proprio_projector=proprio,
                noisy_action_projector=noisy, use_film=False)
            return np.asarray(raw), dict(retained_visual_tokens=512, precise=False)
        raw = bridges[arm](obs, text, previous, state)
        return raw, bridges[arm].last_query.copy()

    def audited(arm, obs, text, previous, state):
        with LlamaAttentionDiagnostic(torch, decoder.layers, "original") as audit:
            with OfficialActionHeadCapture(head) as capture:
                raw, details = call(arm, obs, text, previous, state)
        value = dict(hidden=capture.exact_hidden().detach().float().cpu().numpy(),
                     normalized=capture.exact_output().detach().float().cpu().numpy(), raw_actions=raw)
        return raw, details, value, len(audit.records)

    def commands(raw):
        return processed_command_chunk(raw, evaluation.process_action, cfg.model_family)

    def native_shadow(slot, obs, text, previous, state):
        before_hash = obs_hash(obs)
        raw, details, a, na = audited("native", obs, text, previous, state)
        other, _, b, nb = audited("dense", obs, text, previous, state)
        require(obs_hash(obs) == before_hash, "shadow changed observation")
        errors = {}
        for key in a:
            require(a[key].shape == b[key].shape and np.isfinite(a[key]).all()
                    and np.isfinite(b[key]).all(), "invalid parity tensor")
            errors[key] = float(np.max(np.abs(a[key].astype(np.float64) - b[key].astype(np.float64))))
        native_commands, dense_commands = commands(raw), commands(other)
        trace = dict(slot_id=slot["slot_id"], query=state.query, errors=errors, layers=[na, nb],
                     observation_sha256=before_hash,
                     command_sha256=hashlib.sha256(native_commands.tobytes()).hexdigest(),
                     command_bytes_equal=native_commands.tobytes() == dense_commands.tobytes())
        append("parity.jsonl", trace)
        require(trace["command_bytes_equal"] and [na, nb] == [32, 32]
                and all(v <= 1e-6 for v in errors.values()), "same-input native/dense parity failed")
        records["parity"].append(trace)
        return raw, details

    # Only already authenticated train trajectories are read; no expert actions or locked data.
    observations = {}
    for identity in manifest["inputs"]:
        require(identity["split"] == "train", "non-training timing input")
        with h5py.File(scoped(manifest["data_root_relative"] + "/" + identity["source_path"]), "r") as handle:
            group = handle[f"data/{identity['original_trajectory_id']}/obs"]
            frames = [dict(full_image=np.asarray(group["agentview_rgb"][i]).copy(),
                           wrist_image=np.asarray(group["eye_in_hand_rgb"][i]).copy(),
                           state=np.concatenate((group["ee_states"][i], group["gripper_states"][i])).copy())
                      for i in (0, 1)]
        observations[identity["trajectory_id"]] = (identity, frames)

    if config["mode"] == "hook_qualification":
        # New bounded preflight: 8 * (2 native + 2 arms * 2 hooks * 2 frames) = 80 calls.
        for identity, frames in observations.values():
            cfg.unnorm_key = identity["normalization_statistics_key"]
            text = identity["language_instruction"]
            for arm in ("native", "dense", "specprune"):
                outputs = []
                for hooks in (False, True):
                    state = EpisodeState(); state.reset(identity["trajectory_id"])
                    values = []
                    for i, obs in enumerate(frames[:1] if arm == "native" else frames):
                        previous = CameraPair.capture(frames[0], 0)
                        fingerprint = obs_hash(obs)
                        if hooks:
                            raw, details, _, layers = audited(arm, obs, text, previous, state)
                            require(layers == 32, "hook layer count differs")
                        else:
                            raw, details = call(arm, obs, text, previous, state)
                        require(obs_hash(obs) == fingerprint, "query mutated its observation")
                        values.append((commands(raw), details, tuple(state.previous_indices)))
                    outputs.append(values)
                equal = all(a[0].tobytes() == b[0].tobytes() and a[1:] == b[1:]
                            for a, b in zip(*outputs))
                check = dict(observation_id=identity["trajectory_id"], arm=arm, hooks_neutral=equal)
                append("hook_checks.jsonl", check)
                require(equal, "audit hooks change commands or selection path")
                records["hook_checks"].append(check)
            progress()
        require(resources.queries == 80 and len(records["hook_checks"]) == 24, "hook preflight count differs")
    else:
        current_trace = None
        for slot in timing_slots(config["observation_ids"]):
            identity, frames = observations[slot["observation_id"]]
            cfg.unnorm_key = identity["normalization_statistics_key"]
            trace = (slot["warmup"], slot["round"], slot["observation_id"], slot["arm"])
            if trace != current_trace:
                require(slot["frame"] == 0, "trace must begin at frame zero")
                state = EpisodeState(); state.reset(trace); current_trace = trace
            require(state.precise is False, "short timing trace must remain coarse")
            torch.cuda.synchronize(); started = time.perf_counter()
            previous = CameraPair.capture(frames[0], 0)
            raw, details = call(slot["arm"], frames[slot["frame"]], identity["language_instruction"], previous, state)
            execution_chunk_float32(raw)
            torch.cuda.synchronize(); seconds = time.perf_counter() - started
            row = dict(slot, seconds=seconds, query=state.query, precise=details["precise"],
                       retained_visual_tokens=details["retained_visual_tokens"])
            append("timing.jsonl", row); records["timing"].append(row)
        progress()
        for slot in design["episode_slots"]:
            condition = slot["condition"]
            cfg.task_suite_name = condition["suite"]
            cfg.unnorm_key = condition["suite"] if condition["suite"] in model.norm_stats else condition["suite"] + "_no_noops"
            suite = benchmark.get_benchmark_dict()[condition["suite"]]()
            matches = [i for i in range(suite.n_tasks) if suite.get_task(i).name == condition["task_id"]]
            require(len(matches) == 1, "task name must resolve uniquely")
            initial = suite.get_task_init_states(matches[0])[condition["initial_state_id"]].copy()
            require(hashlib.sha256(initial.tobytes()).hexdigest() == config["initial_state_sha256"][condition["condition_id"]],
                    "initial state differs")
            set_seed_everywhere(7)
            env, instruction = get_libero_env(suite.get_task(matches[0]), cfg.model_family, resolution=cfg.env_img_res)
            try:
                query = (lambda obs, text, previous, state: native_shadow(slot, obs, text, previous, state)) if slot["arm"] == "native" else (
                    lambda obs, text, previous, state: call(slot["arm"], obs, text, previous, state))
                result = execute_measured_episode(evaluation, cfg, env, initial, instruction, query, 224,
                    episode_id=slot["slot_id"], controller=slot["controller_enabled"], synchronize=torch.cuda.synchronize)
            finally:
                env.close()
            row = dict(result, slot_id=slot["slot_id"], condition_id=condition["condition_id"],
                       suite=condition["suite"], arm=slot["arm"])
            append("episodes.jsonl", row); records["episodes"].append(row); progress()
    require(names == {p.name for p in checkpoint.iterdir()}, "checkpoint inventory changed")
    verify(config)
    resources.close()
    base.write_once(run_root / "records.json", records)
    return dict(complete=True, model_queries=resources.queries, episodes=len(records["episodes"]),
                hook_neutrality_passed=config["mode"] == "hook_qualification", worker_sha256=config["worker_sha256"],
                checkpoint_unchanged=True, authenticated_files_unchanged=True, training_performed=False,
                automatic_retry=False, positive_method_result=False, peak_aggregate_gpu_memory_mib=resources.peak)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--preflight-only", action="store_true")
    args = p.parse_args()
    require(Path.cwd().resolve() == ROOT and Path(sys.prefix).resolve().is_relative_to(ROOT / "envs/openvla-oft"),
            "original project runtime required")
    require(args.config.resolve().is_relative_to(ROOT / "configs/openvla"), "project config required")
    config = json.loads(args.config.read_text())
    design, reference, manifest = verify(config)
    if args.preflight_only:
        print(json.dumps(dict(preflight_passed=True, mode=config["mode"], caps=config["caps"])))
        return 0
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == str(config["gpu"]["index"]), "selected GPU mismatch")
    initial = base.snapshot(config["gpu"]["index"])
    require(initial["uuid"] == config["gpu"]["uuid"] and initial["memory_mib"] <= 1024 and initial["utilization"] <= 5,
            "selected GPU not idle; no launch")
    run_root = scoped(config["output_root"])
    run_root.mkdir(exist_ok=False)
    base.environment(run_root)
    base.write_once(run_root / "launch.json", dict(config=config, config_sha256=base.sha(args.config),
        pid=os.getpid(), started_at=datetime.now(timezone.utc).isoformat(), initial_gpu=initial))
    resource_config = dict(gpu=config["gpu"], caps=dict(queries=config["caps"]["model_calls"], memory_mib=23552))
    resources = base.Resources(resource_config); resources.peak = initial["memory_mib"]
    resources.thread.start(); started = time.monotonic()
    def timeout(*_args):
        raise RuntimeError("frozen wall-clock cap reached")
    signal.signal(signal.SIGALRM, timeout); signal.alarm(config["caps"]["seconds"])
    try:
        summary = run(config, design, reference, manifest, run_root, resources)
        summary.update(elapsed_seconds=time.monotonic() - started, completed_at=datetime.now(timezone.utc).isoformat())
        require(summary["elapsed_seconds"] < config["caps"]["seconds"]
                and summary["peak_aggregate_gpu_memory_mib"] < 23552, "resource cap reached")
        if config["mode"] == "evaluation":
            reconcile(design, reference, summary, json.loads((run_root / "records.json").read_text()), config["observation_ids"])
        artifacts = [p for p in run_root.iterdir() if p.is_file() and p.suffix in (".json", ".jsonl")]
        summary["artifacts_sha256"] = {p.name: base.sha(p) for p in artifacts}
        require(sum(p.stat().st_size for p in run_root.rglob("*") if p.is_file()) < config["caps"]["artifact_bytes"] - 65536,
                "artifact cap reached")
        base.write_once(run_root / "worker_summary.json", summary)
        print(json.dumps(dict(complete=True, output_root=config["output_root"])), flush=True)
        return 0
    except BaseException as error:
        with (run_root / "technical_traceback.log").open("x") as stream:
            traceback.print_exc(file=stream)
        base.write_once(run_root / "technical_stop.json", dict(complete=False, error_type=type(error).__name__,
            error=str(error), model_queries=resources.queries, automatic_retry=False,
            elapsed_seconds=time.monotonic() - started, peak_aggregate_gpu_memory_mib=resources.peak))
        print(json.dumps(dict(complete=False, technical_stop=type(error).__name__)), flush=True)
        return 1
    finally:
        signal.alarm(0); resources.stop.set(); resources.thread.join(timeout=12)


if __name__ == "__main__":
    raise SystemExit(main())
