#!/usr/bin/env python3
"""Dense-only targeted execution-precision check; no compressed outcome access."""
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
ROOT=Path("/home/ved/SAVR")
CAPS=dict(queries=112,episodes=2,seconds=1200,memory_mib=23552,artifact_bytes=268435456)
CONDITION="d78f949af14d367e4a3057d13b211cc12f9ea7cbe00b6401c9bc5e1f74787bb9"
STATE_SHA="4e3eaf7a315bc3bafb6db4837c8eef0e5f8aea717782a2a3ea33139392d966ed"

def validate_contract(c):
    if (c["schema_version"]!="dense-precision-trace-v1" or c["caps"]!=CAPS
        or c["output_root"]!="results/dense-precision-trace-v01"
        or c["episode_modes"]!=["native","bridge"] or c["tolerance"]!=1e-6
        or c["condition_id"]!=CONDITION or c["automatic_retry"] is not False):
        raise ValueError("frozen precision contract differs")

def verify(c):
    validate_contract(c)
    p=ROOT/"configs/openvla/specprune_real_qualification_v1.json"
    if base.sha(p)!="8adc9dde86d08f8d717c96e4017250229c71d7f5062ebe697cbb4c53eb75e168":
        raise ValueError("original qualification identity differs")
    ref,manifest=prior.verify(json.loads(p.read_text()))
    if c["gpu"]!=ref["gpu"] or ref["episode_conditions"][0]["condition_id"]!=CONDITION:
        raise ValueError("GPU or condition differs")
    for relative,expected in c["authenticated_files"].items():
        path=(ROOT/relative).resolve()
        if not path.is_relative_to(ROOT) or base.sha(path)!=expected:
            raise ValueError(f"source identity differs: {relative}")
    return ref,manifest

def reconcile(c,s,r):
    validate_contract(c)
    if (s.get("complete") is not True or not 0<s["model_queries"]<=112
        or not 0<=s["elapsed_seconds"]<1200 or not 0<=s["peak_aggregate_gpu_memory_mib"]<23552
        or s.get("checkpoint_unchanged") is not True or s.get("authenticated_files_unchanged") is not True
        or s.get("training_performed") is not False or s.get("automatic_retry") is not False
        or s.get("positive_method_result") is not False
        or [e["mode"] for e in r["episodes"]]!=["native","bridge"]):
        raise ValueError("invalid completed diagnostic")
    for e in r["episodes"]:
        if (e["success"],e["executed_steps"],e["policy_queries"])!=(True,78,10):
            raise ValueError("precision-aligned episode differs from consumed reference")
    if s["model_queries"]!=2*sum(e["policy_queries"] for e in r["episodes"]) or len(r["traces"])!=20:
        raise ValueError("trace/call accounting differs")
    for t in r["traces"]:
        if (t["command_bytes_equal"] is not True or t["layers"]!=[32,32]
            or set(t["errors"])!={"hidden","normalized","raw_actions"}
            or any(not 0<=v<=1e-6 for v in t["errors"].values())):
            raise ValueError("command or model parity failed")
    a=[(t["observation_sha256"],t["command_sha256"]) for t in r["traces"] if t["mode"]=="native"]
    b=[(t["observation_sha256"],t["command_sha256"]) for t in r["traces"] if t["mode"]=="bridge"]
    if a!=b or len(a)!=10:
        raise ValueError("separate episode observation/command traces differ")
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

    from savr.openvla.specprune_episode import SpecPruneEpisodeQuery, execute_specprune_episode
    from savr.openvla.execution_precision import execution_chunk_float32, processed_command_chunk
    from experiments.robot.libero.libero_utils import get_libero_env
    from experiments.robot.robot_utils import get_image_resize_size
    from libero.libero import benchmark
    bridge=SpecPruneEpisodeQuery(model=model,head=head,proprio=proprio,processor=processor,
        cfg=cfg,utils=utils,instruction_indexer=instruction_token_indices,enabled=False)

    def observation_hash(obs):
        digest=hashlib.sha256()
        for key in ("full_image","wrist_image","state"):
            a=np.asarray(obs[key])
            digest.update(key.encode());digest.update(str((a.shape,a.dtype.str)).encode())
            digest.update(a.tobytes())
        return digest.hexdigest()

    @torch.inference_mode()
    def query_pair(obs,text,previous,state,mode):
        values=[];layers=[];raws=[]
        for arm in ("native","bridge"):
            resources.consume()
            with LlamaAttentionDiagnostic(torch,decoder.layers,"original") as audit:
                with OfficialActionHeadCapture(head) as capture:
                    if arm=="native":
                        actions=evaluation.get_action(cfg,model,{k:np.asarray(v).copy() for k,v in obs.items()},
                            text,processor=processor,action_head=head,proprio_projector=proprio,
                            noisy_action_projector=noisy,use_film=False)
                    else:
                        actions=bridge(obs,text,previous,state)
            raw=np.asarray(actions);raws.append(raw)
            values.append(dict(hidden=capture.exact_hidden().detach().float().cpu().numpy(),
                normalized=capture.exact_output().detach().float().cpu().numpy(),
                raw_actions=raw))
            layers.append(len(audit.records))
        errors={}
        for key in values[0]:
            a,b=values[0][key],values[1][key]
            if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
                raise ValueError("invalid paired model outputs")
            errors[key]=float(np.max(np.abs(a.astype(np.float64)-b.astype(np.float64))))
        commands=[processed_command_chunk(raw,evaluation.process_action,cfg.model_family) for raw in raws]
        same=commands[0].tobytes()==commands[1].tobytes()
        trace=dict(mode=mode,query=state.query,observation_sha256=observation_hash(obs),
            command_sha256=hashlib.sha256(commands[0].tobytes()).hexdigest(),
            command_bytes_equal=same,errors=errors,layers=layers,
            raw_dtypes=[str(a.dtype) for a in raws],executed_dtype="float32")
        traces.append(trace)
        with (run_root/"trace.jsonl").open("a") as stream:
            stream.write(json.dumps(trace,allow_nan=False)+"\n")
        if not same or layers!=[32,32] or any(v>1e-6 for v in errors.values()):
            base.write_once(run_root/"parity_mismatch.json",trace)
            raise ValueError("same-observation command/model parity failed")
        return execution_chunk_float32(raws[int(mode=="bridge")])

    row=reference["episode_conditions"][0]
    cfg.task_suite_name=row["suite"]
    cfg.unnorm_key=row["suite"] if row["suite"] in model.norm_stats else row["suite"]+"_no_noops"
    suite=benchmark.get_benchmark_dict()[row["suite"]]()
    matches=[i for i in range(suite.n_tasks) if suite.get_task(i).name==row["task_id"]]
    if len(matches)!=1 or evaluation.TASK_MAX_STEPS[row["suite"]]!=220:
        raise ValueError("original task/horizon differs")
    task_index=matches[0]
    initial_state=suite.get_task_init_states(task_index)[0].copy()
    if hashlib.sha256(initial_state.tobytes()).hexdigest()!=STATE_SHA:
        raise ValueError("simulator state identity differs")
    traces=[];episodes=[]
    for mode in config["episode_modes"]:
        set_seed_everywhere(7)
        env,instruction=get_libero_env(suite.get_task(task_index),cfg.model_family,resolution=cfg.env_img_res)
        try:
            episode=execute_specprune_episode(evaluation,cfg,env,initial_state,instruction,
                lambda obs,text,previous,state:query_pair(obs,text,previous,state,mode),
                get_image_resize_size(cfg),episode_id=CONDITION,controller=False)
        finally:
            env.close()
        episodes.append(dict(episode,mode=mode))
        value=dict(episodes_complete=len(episodes),model_queries=resources.queries)
        with (run_root/"progress.jsonl").open("a") as stream:stream.write(json.dumps(value)+"\n")
        print(json.dumps(value),flush=True)
    if names!={p.name for p in checkpoint.iterdir()}:raise ValueError("checkpoint inventory changed")
    verify(config);resources.close()
    base.write_once(run_root/"records.json",dict(episodes=episodes,traces=traces))
    return dict(complete=True,model_queries=resources.queries,checkpoint_unchanged=True,
        authenticated_files_unchanged=True,training_performed=False,positive_method_result=False,
        automatic_retry=False,peak_aggregate_gpu_memory_mib=resources.peak,
        artifacts_sha256={p:base.sha(run_root/p) for p in ("loaded_runtime.json","progress.jsonl","trace.jsonl","records.json")})

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
        print(json.dumps(dict(preflight_passed=True, max_calls=112, episodes=2,
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

