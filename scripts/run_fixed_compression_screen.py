#!/usr/bin/env python3
"""Frozen 128-episode current-frame screen; never train or retry automatically."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback
import run_fixed_compression_qualification as qualification
import run_openvla_original_baseline as base

ROOT=Path('/home/ved/SAVR')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from savr.openvla.compression_screen_contract import CAPS,reconcile,validate_config
from savr.openvla.contemporary_contract import require,HORIZONS

FILES=('scripts/run_fixed_compression_screen.py','scripts/analyze_fixed_compression_screen.py',
       'src/savr/openvla/compression_screen_contract.py','tests/openvla/test_compression_screen.py')
QUAL_CONFIG='configs/openvla/fixed_compression_qualification_v1.json'
QUAL_SHA='d4cb82d508b16a923270d4e979eced5f243b4a902d851b71772bc9702aa862b4'


def verify(c):
    require(base.sha(ROOT/QUAL_CONFIG)==QUAL_SHA,'qualification config changed')
    q=json.loads((ROOT/QUAL_CONFIG).read_text());old,ref,manifest=qualification.verify(q)
    validate_config(c,ref,q['observation_ids'])
    require(c['gpu']==q['gpu'] and c['initial_state_sha256']==old['initial_state_sha256']
            and c['observation_ids']==q['observation_ids'],'population/GPU differs')
    require(set(FILES)<=set(c['authenticated_files']),'screen sources incomplete')
    for n,h in c['authenticated_files'].items():require(base.sha(qualification.scoped(n))==h,'screen source changed: '+n)
    require(c['worker_sha256']==base.sha(Path(__file__)) and c['analyzer_sha256']==base.sha(ROOT/FILES[1]),'worker/analyzer identity differs')
    p=qualification.scoped(c['qualification_summary'])
    require(str(p.relative_to(ROOT))=='results/fixed-compression-qualification-v01/worker_summary.json'
            and base.sha(p)==c['qualification_summary_sha256'] and not (p.parent/'technical_stop.json').exists(),'qualification changed/stopped')
    s=json.loads(p.read_text());require(s['complete'] is True,'qualification incomplete')
    for n,h in s['artifacts_sha256'].items():
        x=(p.parent/n).resolve();require(x.is_relative_to(p.parent) and base.sha(x)==h,'qualification artifact differs')
    qualification.reconcile(q,s,json.loads((p.parent/'records.json').read_text()))
    return old,ref,manifest


def run(c,ref,observations,p,resources):
    import numpy as np
    import torch
    import transformers
    import tensorflow as tf
    tf.config.set_visible_devices([],'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(1);tf.config.threading.set_inter_op_parallelism_threads(1)
    require(torch.__version__=='2.2.0+cu118' and transformers.__version__=='4.40.1'
            and torch.cuda.device_count()==1,'original one-GPU runtime required')
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(23000/24576,0)
    sys.path.insert(0,str(ROOT/'third_party/openvla-oft'))
    from experiments.robot import openvla_utils as utils
    from experiments.robot.libero import run_libero_eval as evaluation
    from experiments.robot.robot_utils import set_seed_everywhere,get_image_resize_size
    from experiments.robot.libero.libero_utils import get_libero_env
    from libero.libero import benchmark
    from savr.cac.c1 import instruction_token_indices
    from savr.openvla.fixed_compression import FixedCompressionQuery
    from savr.openvla.specprune import EpisodeState
    from savr.openvla.offline_camera_inputs import OfflineCameraPair
    from savr.openvla.official_semantics import OfficialActionHeadCapture
    from savr.openvla.attention_diagnostic import LlamaAttentionDiagnostic
    from savr.openvla.execution_precision import processed_command_chunk,execution_chunk_float32
    from savr.openvla.contemporary_episode import execute_measured_episode
    checkpoint=ROOT/ref['checkpoint'];names={x.name for x in checkpoint.iterdir()}
    cfg=evaluation.GenerateConfig(pretrained_checkpoint=str(checkpoint),task_suite_name='libero_spatial',
        num_trials_per_task=0,seed=7,local_log_dir=str(p/'logs'),use_wandb=False,center_crop=True,
        num_open_loop_steps=8,num_images_in_input=2,use_proprio=True,use_l1_regression=True,use_diffusion=False,use_film=False)
    evaluation.validate_config(cfg);set_seed_everywhere(7)
    update,sync=utils.update_auto_map,utils.check_model_logic_mismatch
    utils.update_auto_map=lambda _:None;utils.check_model_logic_mismatch=lambda _:None
    try:model,head,proprio,noisy,processor=evaluation.initialize_model(cfg)
    finally:utils.update_auto_map,utils.check_model_logic_mismatch=update,sync
    require(head is not None and proprio is not None and noisy is None and
            not any(m.training for m in (model,head,proprio)),'released components differ')
    decoder=model.language_model.model
    require(len(decoder.layers)==32 and all(type(l.self_attn).__name__=='LlamaSdpaAttention' for l in decoder.layers),'original SDPA required')
    require(all(evaluation.TASK_MAX_STEPS[s]==h for s,h in HORIZONS.items())
            and get_image_resize_size(cfg)==224 and cfg.num_steps_wait==10,'episode contract changed')
    base.write_once(p/'loaded_runtime.json',dict(torch=torch.__version__,transformers=transformers.__version__,
        configuration=vars(cfg),tensorflow_gpu_disabled=True,checkpoint_metadata_mutation_disabled=True))
    bridges={str(n):FixedCompressionQuery(model=model,head=head,proprio=proprio,processor=processor,cfg=cfg,
        utils=utils,instruction_indexer=instruction_token_indices,budget=n) for n in (512,384,256)}
    records=dict(episodes=[],timing=[],parity=[])

    def append(name,value):
        require(sum(x.stat().st_size for x in p.rglob('*') if x.is_file())<CAPS['artifact_bytes']-65536,'artifact cap reached')
        with (p/name).open('a') as f:f.write(json.dumps(value,allow_nan=False)+'\n')

    def progress():append('progress.jsonl',dict(episodes=len(records['episodes']),timing_records=len(records['timing']),model_queries=resources.queries))

    def obs_hash(obs):
        h=hashlib.sha256()
        for k in ('full_image','wrist_image','state'):
            a=np.asarray(obs[k]);h.update(k.encode());h.update(str((a.shape,a.dtype.str)).encode());h.update(a.tobytes())
        return h.hexdigest()

    @torch.inference_mode()
    def call(arm,obs,text,previous,state):
        resources.consume()
        if arm=='native':
            raw=evaluation.get_action(cfg,model,{k:np.asarray(v).copy() for k,v in obs.items()},text,
                processor=processor,action_head=head,proprio_projector=proprio,noisy_action_projector=noisy,use_film=False)
            return np.asarray(raw),dict(retained_visual_tokens=512,precise=False)
        raw=bridges[arm](obs,text,previous,state)
        return raw,dict(bridges[arm].last_query)

    def audited(arm,obs,text,previous,state):
        with LlamaAttentionDiagnostic(torch,decoder.layers,'original') as audit:
            with OfficialActionHeadCapture(head) as capture:raw,details=call(arm,obs,text,previous,state)
        values=dict(hidden=capture.exact_hidden().detach().float().cpu().numpy(),
                    normalized=capture.exact_output().detach().float().cpu().numpy(),raw=raw)
        return raw,details,values,len(audit.records)

    def shadow(slot,obs,text,previous,state):
        fingerprint=obs_hash(obs)
        raw,details,a,na=audited('native',obs,text,previous,state)
        other,_,b,nb=audited('512',obs,text,previous,state)
        require(obs_hash(obs)==fingerprint and all(a[k].shape==b[k].shape and np.isfinite(a[k]).all()
                    and np.isfinite(b[k]).all() for k in a),'shadow observation/output changed')
        errors={k:float(np.max(np.abs(a[k].astype(np.float64)-b[k].astype(np.float64)))) for k in a}
        commands=processed_command_chunk(raw,evaluation.process_action,cfg.model_family)
        other_commands=processed_command_chunk(other,evaluation.process_action,cfg.model_family)
        row=dict(slot_id=slot['slot_id'],query=state.query,errors=errors,layers=[na,nb],
            observation_sha256=fingerprint,command_sha256=hashlib.sha256(commands.tobytes()).hexdigest(),
            command_bytes_equal=commands.tobytes()==other_commands.tobytes())
        append('parity.jsonl',row)
        require(row['command_bytes_equal'] and [na,nb]==[32,32] and all(v<=1e-6 for v in errors.values()),'native/512 parity failed')
        records['parity'].append(row);return raw,details

    current_trace=None
    for slot in c['timing_slots']:
        identity,frames=observations[slot['observation_id']];cfg.unnorm_key=identity['normalization_statistics_key']
        trace=(slot['warmup'],slot['round'],slot['observation_id'],slot['arm'])
        if trace!=current_trace:
            require(slot['frame']==0,'trace must start at frame zero');state=EpisodeState();state.reset(trace);current_trace=trace
        torch.cuda.synchronize();started=time.perf_counter()
        previous=OfflineCameraPair.capture(frames[0],0)
        raw,details=call(slot['arm'],frames[slot['frame']],identity['language_instruction'],previous,state)
        execution_chunk_float32(raw);torch.cuda.synchronize();seconds=time.perf_counter()-started
        row=dict(slot,seconds=seconds,query=state.query,precise=details['precise'],retained_visual_tokens=details['retained_visual_tokens'])
        append('timing.jsonl',row);records['timing'].append(row)
    progress()
    for slot in c['episode_slots']:
        condition=slot['condition'];cfg.task_suite_name=condition['suite']
        cfg.unnorm_key=condition['suite'] if condition['suite'] in model.norm_stats else condition['suite']+'_no_noops'
        suite=benchmark.get_benchmark_dict()[condition['suite']]()
        matches=[i for i in range(suite.n_tasks) if suite.get_task(i).name==condition['task_id']]
        require(len(matches)==1,'task must resolve uniquely')
        initial=suite.get_task_init_states(matches[0])[condition['initial_state_id']].copy()
        require(hashlib.sha256(initial.tobytes()).hexdigest()==c['initial_state_sha256'][condition['condition_id']],'initial state changed')
        set_seed_everywhere(7);env,instruction=get_libero_env(suite.get_task(matches[0]),cfg.model_family,resolution=cfg.env_img_res)
        try:
            query=(lambda obs,text,previous,state:shadow(slot,obs,text,previous,state)) if slot['arm']=='native' else (
                  lambda obs,text,previous,state:call(slot['arm'],obs,text,previous,state))
            result=execute_measured_episode(evaluation,cfg,env,initial,instruction,query,224,
                episode_id=slot['slot_id'],controller=False,synchronize=torch.cuda.synchronize)
        finally:env.close()
        row=dict(result,slot_id=slot['slot_id'],condition_id=condition['condition_id'],suite=condition['suite'],arm=slot['arm'])
        append('episodes.jsonl',row);records['episodes'].append(row);progress()
    require(names=={x.name for x in checkpoint.iterdir()},'checkpoint inventory changed')
    verify(c);resources.close();base.write_once(p/'records.json',records)
    return records


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--preflight-only',action='store_true');args=parser.parse_args()
    require(Path.cwd().resolve()==ROOT and Path(sys.prefix).resolve().is_relative_to(ROOT/'envs/openvla-oft'),'project runtime required')
    require(args.config.resolve().is_relative_to(ROOT/'configs/openvla'),'project config required')
    c=json.loads(args.config.read_text());old,ref,manifest=verify(c)
    from savr.openvla.offline_camera_inputs import load_offline_inputs
    observations,audit=load_offline_inputs(ROOT,manifest)
    require(audit==json.loads(qualification.scoped(old['input_preflight_report']).read_text())['input_audit'],'actual input changed')
    if args.preflight_only:print(json.dumps(dict(preflight_passed=True,model_queries=0)));return
    require(os.environ.get('CUDA_VISIBLE_DEVICES')==str(c['gpu']['index']),'GPU selection differs')
    initial=base.snapshot(c['gpu']['index'])
    require(initial['uuid']==c['gpu']['uuid'] and initial['memory_mib']<=1024 and initial['utilization']<=5,'selected GPU not idle')
    p=qualification.scoped(c['output_root']);p.mkdir(exist_ok=False);base.environment(p)
    base.write_once(p/'launch.json',dict(config=c,config_sha256=base.sha(args.config),pid=os.getpid(),
        started_at=datetime.now(timezone.utc).isoformat(),initial_gpu=initial))
    base.write_once(p/'input_preflight.json',dict(complete=True,input_audit=audit,performed_before_model_loading=True))
    resources=base.Resources(dict(gpu=c['gpu'],caps=dict(queries=7000,memory_mib=23552)))
    resources.peak=initial['memory_mib'];resources.thread.start();started=time.monotonic()
    def timeout(*_):raise RuntimeError('frozen time cap reached')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(CAPS['seconds'])
    try:
        records=run(c,ref,observations,p,resources)
        s=dict(complete=True,model_queries=resources.queries,episodes=len(records['episodes']),training_performed=False,
            automatic_retry=False,positive_method_result=False,checkpoint_unchanged=True,authenticated_files_unchanged=True,
            worker_sha256=c['worker_sha256'],elapsed_seconds=time.monotonic()-started,
            peak_aggregate_gpu_memory_mib=resources.peak,completed_at=datetime.now(timezone.utc).isoformat())
        reconcile(c,ref,s,records,c['observation_ids'])
        s['artifacts_sha256']={x.name:base.sha(x) for x in p.iterdir() if x.is_file() and x.suffix in ('.json','.jsonl')}
        require(sum(x.stat().st_size for x in p.rglob('*') if x.is_file())<CAPS['artifact_bytes']-65536,'artifact cap reached')
        base.write_once(p/'worker_summary.json',s);print(json.dumps(dict(complete=True,episodes=128)))
    except BaseException as e:
        with (p/'technical_traceback.log').open('x') as f:traceback.print_exc(file=f)
        base.write_once(p/'technical_stop.json',dict(complete=False,error_type=type(e).__name__,error=str(e),
            model_queries=resources.queries,automatic_retry=False,elapsed_seconds=time.monotonic()-started,
            peak_aggregate_gpu_memory_mib=resources.peak))
        raise
    finally:
        signal.alarm(0);resources.stop.set();resources.thread.join(timeout=12)


if __name__=='__main__':main()
