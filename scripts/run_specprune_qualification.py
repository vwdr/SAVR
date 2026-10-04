#!/usr/bin/env python3
"""Frozen, original-runtime SpecPrune qualification; no simulator or training."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback

import run_openvla_original_baseline as base

ROOT = Path("/home/ved/SAVR")
MODES = ["native", "disabled", "keep_all", "compressed", "restored"]
CAPS = dict(queries=40, episodes=0, seconds=1800, memory_mib=23552, artifact_bytes=268435456)
BASE_CONFIG = "configs/openvla/original_baseline_40task_v1.json"
BASE_SHA = "90a65540dfade976ad58924b8318576b38c310ac1d8831db31ceb3d4682daa27"


def validate_contract(config):
    if (config["schema_version"] != "specprune-real-qualification-v1"
            or config["modes"] != MODES or config["caps"] != CAPS
            or config["tolerance"] != 1e-6 or config["observations"] != 8
            or config["reference_config"] != BASE_CONFIG
            or config["reference_config_sha256"] != BASE_SHA
            or config["output_root"] != "results/specprune-real-qualification-v01"
            or config["compression_preset"] != "source-default-coarse-first-query"
            or config["automatic_retry"] is not False):
        raise ValueError("frozen qualification contract differs")


def verify(config):
    validate_contract(config)
    if base.sha(ROOT / BASE_CONFIG) != BASE_SHA:
        raise ValueError("reference configuration identity differs")
    reference = json.loads((ROOT / BASE_CONFIG).read_text())
    manifest = base.verify(reference)
    for relative, expected in config["authenticated_files"].items():
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT) or base.sha(path) != expected:
            raise ValueError(f"source identity differs: {relative}")
    if (len(manifest["inputs"]) != 8
            or [r["trajectory_id"] for r in manifest["inputs"]] != config["observation_ids"]
            or config["gpu"] != reference["gpu"]):
        raise ValueError("observation/GPU identity differs")
    return reference, manifest


def reconcile(config, summary, records):
    """CPU-only completeness and semantic gate, also exercised with corrupt fixtures."""
    validate_contract(config)
    if (summary.get("complete") is not True or summary.get("model_queries") != 40
            or summary.get("checkpoint_unchanged") is not True
            or summary.get("authenticated_files_unchanged") is not True
            or summary.get("training_performed") is not False
            or summary.get("simulator_episodes") != 0
            or summary.get("automatic_retry") is not False
            or summary.get("positive_method_result") is not False
            or not 0 <= summary["peak_aggregate_gpu_memory_mib"] < CAPS["memory_mib"]
            or not 0 <= summary["elapsed_seconds"] < CAPS["seconds"]
            or len(records) != 8):
        raise ValueError("incomplete or invalid qualification summary")
    if [r["observation_id"] for r in records] != config["observation_ids"]:
        raise ValueError("observation accounting differs")
    for row in records:
        if list(row["modes"]) != MODES:
            # JSON writers sort keys; require the independently stored order.
            if sorted(row["modes"]) != sorted(MODES):
                raise ValueError("mode set differs")
        if row["mode_order"] != MODES:
            raise ValueError("mode order differs")
        for mode in MODES:
            item = row["modes"][mode]
            if item["layers"] != 32 or item["finite"] is not True:
                raise ValueError("invalid layer accounting or actions")
            if mode in ("disabled", "keep_all", "restored"):
                errors = item["parity_max_abs"]
                if set(errors) != {"hidden", "normalized", "actions"} or any(
                    not isinstance(v, (int, float)) or not 0 <= v <= config["tolerance"] for v in errors.values()
                ):
                    raise ValueError("identity parity failed")
            if mode in ("disabled", "keep_all", "compressed"):
                positions = item["positions"]
                total = row["full_tokens"]
                if positions != sorted(set(positions)) or not positions or positions[-1] >= total or positions[0] != 0:
                    raise ValueError("invalid final position map")
                if not {0, *range(513, total)}.issubset(positions):
                    raise ValueError("required nonvisual token missing")
                expected_head = list(range(total - 58, total - 2))
                if item["action_positions"] != expected_head:
                    raise ValueError("action-head positions differ")
                if (mode == "compressed" and len(positions) >= total) or (mode != "compressed" and len(positions) != total):
                    raise ValueError("pruning/identity token accounting differs")
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

    def array(value):
        if isinstance(value, torch.Tensor):
            value = value.detach().float().cpu().numpy()
        return np.asarray(value, dtype=np.float32)

    def outputs(hidden, normalized, actions):
        values = dict(hidden=array(hidden), normalized=array(normalized).reshape(8, 7), actions=array(actions))
        if values["hidden"].shape != (1, 56, 4096) or values["actions"].shape != (8, 7) or any(not np.isfinite(v).all() for v in values.values()):
            raise RuntimeError("invalid action boundary")
        return values

    def parity(ref, own):
        return {name: float(np.max(np.abs(ref[name].astype(np.float64) - own[name].astype(np.float64)))) for name in ref}

    @torch.inference_mode()
    def native(observation, instruction):
        resources.consume()
        with LlamaAttentionDiagnostic(torch, decoder.layers, "original") as audit:
            with OfficialActionHeadCapture(head) as capture:
                actions = evaluation.get_action(cfg, model,
                    {k: np.asarray(v).copy() for k, v in observation.items()}, instruction,
                    processor=processor, action_head=head, proprio_projector=proprio,
                    noisy_action_projector=noisy, use_film=False)
        return outputs(capture.exact_hidden(), capture.exact_output(), actions), dict(layers=len(audit.records), finite=True)

    @torch.inference_mode()
    def adapted(observation, instruction, mode, identity):
        resources.consume()
        prepared = prepare_semantic_query(
            torch_module=torch, np_module=np, model=model, processor=processor,
            proprio_projector=proprio, prepare_images=utils.prepare_images_for_vla,
            normalize_proprio=utils.normalize_proprio, instruction_indexer=instruction_token_indices,
            cfg=cfg, raw_scene=observation["full_image"].copy(), raw_wrist=observation["wrist_image"].copy(),
            raw_state=observation["state"].copy(), instruction=instruction)
        layout = derive_official_layout(action_mask=prepared.action_mask, projected_tokens=513,
                                        instruction_token_indices=prepared.instruction_token_indices)
        masked = prepared.input_embeddings * ~prepared.action_mask.unsqueeze(-1)
        embeddings, mask = model._build_multimodal_attention(masked, prepared.projected_patches, prepared.attention_mask)
        if mask.ndim != 2 or not bool((mask == 1).all()):
            raise RuntimeError("qualification only supports unpadded current queries")
        low = ()
        if mode == "compressed":
            images = utils.prepare_images_for_vla([observation["full_image"].copy(), observation["wrist_image"].copy()], cfg)
            low = tuple(np.concatenate([
                low_change_indices(images[0], images[0], top_k=236, threshold=.986),
                low_change_indices(images[1], images[1], top_k=230, threshold=.98, wrist=True)]).tolist())
        episode = EpisodeState(); episode.reset(identity)
        selector = SpecPruneSelection(layout, episode, low, enabled=mode != "disabled",
                                      dynamic=mode == "compressed", device=embeddings.device)
        with LlamaAttentionDiagnostic(torch, decoder.layers, "original") as audit:
            last_hidden, positions = pruned_decoder_forward(decoder, embeddings, selector)
        hidden = select_official_action_hidden(last_hidden, layout, positions)
        normalized = head.predict_action(hidden).reshape(8, 7).float().cpu().numpy()
        actions = model._unnormalize_actions(normalized, cfg.unnorm_key)
        return outputs(hidden, normalized, actions), dict(
            layers=len(audit.records), finite=True, positions=positions.cpu().tolist(),
            action_positions=list(layout.action_readout_positions), trace=selector.trace,
            full_tokens=layout.full_sequence_tokens, episode_query_count=episode.query,
            retained_visual_tokens=int(((positions >= 1) & (positions <= 512)).sum()),
        )

    records = []
    for index, identity in enumerate(manifest["inputs"]):
        with h5py.File(ROOT / manifest["data_root_relative"] / identity["source_path"], "r") as handle:
            group = handle[f"data/{identity['original_trajectory_id']}/obs"]
            obs = dict(full_image=np.asarray(group["agentview_rgb"][0]).copy(),
                       wrist_image=np.asarray(group["eye_in_hand_rgb"][0]).copy(),
                       state=np.concatenate((group["ee_states"][0], group["gripper_states"][0])).copy())
        cfg.unnorm_key = identity["normalization_statistics_key"]
        instruction = identity["language_instruction"]
        row = dict(observation_id=identity["trajectory_id"], mode_order=MODES, modes={})
        ref, details = native(obs, instruction)
        row["modes"]["native"] = details
        for mode in ("disabled", "keep_all", "compressed"):
            own, details = adapted(obs, instruction, mode, identity["trajectory_id"])
            row["full_tokens"] = details.pop("full_tokens")
            if mode != "compressed":
                details["parity_max_abs"] = parity(ref, own)
                if any(v > config["tolerance"] for v in details["parity_max_abs"].values()):
                    raise RuntimeError(f"identity parity failed: observation {index}, {mode}, {details['parity_max_abs']}")
            row["modes"][mode] = details
        restored, details = native(obs, instruction)
        details["parity_max_abs"] = parity(ref, restored)
        if any(v > config["tolerance"] for v in details["parity_max_abs"].values()):
            raise RuntimeError("reference restoration failed")
        row["modes"]["restored"] = details
        records.append(row)
        progress = dict(observations_complete=index + 1, model_queries=resources.queries)
        with (run_root / "progress.jsonl").open("a") as stream:
            stream.write(json.dumps(progress) + "\n")
        print(json.dumps(progress), flush=True)
    if resources.queries != 40 or names != {p.name for p in checkpoint.iterdir()}:
        raise RuntimeError("query count or checkpoint inventory differs")
    verify(config)
    resources.close()
    base.write_once(run_root / "records.json", records)
    return dict(complete=True, model_queries=40, checkpoint_unchanged=True,
                authenticated_files_unchanged=True, training_performed=False, simulator_episodes=0,
                automatic_retry=False, positive_method_result=False,
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
        print(json.dumps(dict(preflight_passed=True, observations=8, calls=40,
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
