#!/usr/bin/env python3
"""One frozen 112-call fixed-current-token qualification, no robot episodes."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
import traceback
import run_openvla_original_baseline as base
import run_contemporary_reference_recovery01 as prior

ROOT=Path('/home/ved/SAVR')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from savr.openvla.spatial_selection import stratified_visual_positions

CAPS=dict(model_calls=112,episodes=0,seconds=1800,aggregate_memory_mib=23552,artifact_bytes=268435456)
REFERENCE='configs/openvla/contemporary_reference_evaluation_v2.json'
REFERENCE_SHA='afa1bc948880b072b74c9f150e7dbab77fe90468040746c73b82af603f91ec67'
FILES=('scripts/run_fixed_compression_qualification.py','scripts/analyze_fixed_compression_qualification.py',
       'src/savr/openvla/fixed_compression.py','src/savr/openvla/spatial_selection.py',
       'src/savr/openvla/visual_compaction.py','scripts/run_contemporary_reference_recovery01.py',
       'docs/CURRENT_FRAME_COMPRESSION_SCREEN_V1.md',
       'tests/openvla/test_fixed_compression.py','tests/openvla/test_fixed_compression_qualification.py')


def require(value,message):
    if not value: raise ValueError(message)


def scoped(relative):
    p=(ROOT/relative).resolve()
    require(not Path(relative).is_absolute() and p.is_relative_to(ROOT),'path outside project')
    return p


def verify(c):
    require(c['schema_version']=='fixed-compression-qualification-v1' and c['launch_ready'] is True
            and c['caps']==CAPS and c['budgets']==[512,384,256] and c['tolerance']==1e-6
            and c['automatic_retry'] is False and c['output_root']=='results/fixed-compression-qualification-v01',
            'frozen qualification contract differs')
    require(set(FILES)<=set(c['authenticated_files']),'source manifest incomplete')
    for n,h in c['authenticated_files'].items(): require(base.sha(scoped(n))==h,'source hash differs: '+n)
    require(base.sha(ROOT/REFERENCE)==REFERENCE_SHA,'reference changed')
    old=json.loads((ROOT/REFERENCE).read_text());_,reference,manifest=prior.verify(old)
    require(c['gpu']==old['gpu'] and c['observation_ids']==old['observation_ids'],'GPU/input identities differ')
    require(c['worker_sha256']==base.sha(Path(__file__)) and c['analyzer_sha256']==base.sha(scoped(FILES[1])),
            'worker/analyzer identity differs')
    summary=scoped(c['reference_summary'])
    require(str(summary.relative_to(ROOT))=='results/contemporary-reference-v02/worker_summary.json'
            and base.sha(summary)==c['reference_summary_sha256'],'completed reference changed')
    s=json.loads(summary.read_text())
    require(s['complete'] is True and s['episodes']==88 and not (summary.parent/'technical_stop.json').exists(),
            'completed reference required')
    for n,h in s['artifacts_sha256'].items():
        p=(summary.parent/n).resolve();require(p.is_relative_to(summary.parent) and base.sha(p)==h,'reference artifact changed')
    return old,reference,manifest


def reconcile(c,s,rows):
    require(s.get('complete') is True and s.get('model_queries')==112 and s.get('episodes')==0
            and s.get('training_performed') is False and s.get('automatic_retry') is False
            and s.get('positive_method_result') is False and s.get('checkpoint_unchanged') is True
            and s.get('authenticated_files_unchanged') is True,'incomplete qualification')
    require(s['worker_sha256']==c['worker_sha256'],'worker identity differs')
    require(all(type(s[k]) in (int,float) and math.isfinite(s[k]) and 0<=s[k]<cap for k,cap in
        (('elapsed_seconds',1800),('peak_aggregate_gpu_memory_mib',23552))), 'resource cap exceeded')
    require([(r['observation_id'],r['frame']) for r in rows]==[(i,f) for i in c['observation_ids'] for f in (0,1)],
            'qualification record count/order differs')
    for r in rows:
        require(r['input_unchanged'] is True and r['dense_command_equal'] is True and
                set(r['dense_errors'])=={'hidden','normalized','raw'} and
                all(type(v) in (int,float) and math.isfinite(v) and 0<=v<=1e-6 for v in r['dense_errors'].values()),
                'dense parity failed')
        full=r['full_tokens'];require(type(full) is int and full>570,'invalid full sequence length')
        require([m['budget'] for m in r['modes']]==[512,384,256],'mode order differs')
        for m in r['modes']:
            expected=sorted([0,*stratified_visual_positions(m['budget']),*range(513,full)])
            require(m['finite'] is True and m['hooks_neutral'] is True and m['positions']==expected
                    and m['layer_lengths']==[len(expected)]*32 and m['original_positions_all_layers'] is True
                    and m['no_cache_all_layers'] is True and m['state_query']==1,'compressed execution contract failed')
    return True


def run(c,reference,observations,p,resources):
    import numpy as np
    import torch
    import transformers
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(1);tf.config.threading.set_inter_op_parallelism_threads(1)
    require(torch.__version__=='2.2.0+cu118' and transformers.__version__=='4.40.1'
            and torch.cuda.device_count()==1,'original one-GPU runtime required')
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(23000/24576,0)
    sys.path.insert(0,str(ROOT/'third_party/openvla-oft'))
    from experiments.robot import openvla_utils as utils
    from experiments.robot.libero import run_libero_eval as evaluation
    from experiments.robot.robot_utils import set_seed_everywhere
    from savr.cac.c1 import instruction_token_indices
    from savr.openvla.fixed_compression import FixedCompressionQuery
    from savr.openvla.specprune import EpisodeState
    from savr.openvla.specprune_episode import SpecPruneEpisodeQuery
    from savr.openvla.offline_camera_inputs import OfflineCameraPair
    from savr.openvla.official_semantics import OfficialActionHeadCapture
    from savr.openvla.attention_diagnostic import LlamaAttentionDiagnostic
    from savr.openvla.execution_precision import processed_command_chunk

    checkpoint=ROOT/reference['checkpoint']; names={x.name for x in checkpoint.iterdir()}
    cfg=evaluation.GenerateConfig(pretrained_checkpoint=str(checkpoint),task_suite_name='libero_spatial',
        num_trials_per_task=0,seed=7,local_log_dir=str(p/'logs'),use_wandb=False,center_crop=True,
        num_open_loop_steps=8,num_images_in_input=2,use_proprio=True,use_l1_regression=True,use_diffusion=False,use_film=False)
    evaluation.validate_config(cfg);set_seed_everywhere(7)
    update,sync=utils.update_auto_map,utils.check_model_logic_mismatch
    utils.update_auto_map=lambda _:None;utils.check_model_logic_mismatch=lambda _:None
    try:model,head,proprio,noisy,processor=evaluation.initialize_model(cfg)
    finally:utils.update_auto_map,utils.check_model_logic_mismatch=update,sync
    require(head is not None and proprio is not None and noisy is None and
            not any(m.training for m in (model,head,proprio)),'released inference components differ')
    decoder=model.language_model.model
    require(len(decoder.layers)==32,'32 layers required')
    base.write_once(p/'loaded_runtime.json',dict(torch=torch.__version__,transformers=transformers.__version__,
        configuration=vars(cfg),tensorflow_gpu_disabled=True,checkpoint_metadata_mutation_disabled=True))
    shared=dict(model=model,head=head,proprio=proprio,processor=processor,cfg=cfg,utils=utils,
                instruction_indexer=instruction_token_indices)
    old=SpecPruneEpisodeQuery(**shared,enabled=False)
    bridges={n:FixedCompressionQuery(**shared,budget=n) for n in (512,384,256)}
    rows=[]

    def fingerprint(obs):
        h=hashlib.sha256()
        for k in ('full_image','wrist_image','state'):
            a=np.asarray(obs[k]);h.update(k.encode());h.update(str((a.shape,a.dtype.str)).encode());h.update(a.tobytes())
        return h.hexdigest()

    @torch.inference_mode()
    def call(bridge,obs,text,previous,audited):
        resources.consume();state=EpisodeState();state.reset('fixed-qualification')
        layers=[];handles=[]
        def trace(module,args,kwargs):
            layers.append(dict(length=int(args[0].shape[1]),positions=kwargs['position_ids'][0].cpu().tolist(),
                               no_cache=kwargs.get('past_key_value') is None and kwargs.get('use_cache') is False))
        if audited:
            handles=[l.register_forward_pre_hook(trace,with_kwargs=True) for l in decoder.layers]
        try:
            if audited:
                with LlamaAttentionDiagnostic(torch,decoder.layers,'original') as attention:
                    with OfficialActionHeadCapture(head) as capture:raw=bridge(obs,text,previous,state)
                values=dict(hidden=capture.exact_hidden().float().cpu().numpy(),
                            normalized=capture.exact_output().float().cpu().numpy(),raw=raw)
                require(len(attention.records)==32 and all(np.isfinite(v).all() for v in values.values()),'invalid audited output')
            else:raw=bridge(obs,text,previous,state);values=None
        finally:
            for handle in handles:handle.remove()
        commands=processed_command_chunk(raw,evaluation.process_action,cfg.model_family)
        require(state.query==1,'query state did not advance once')
        return commands,values,layers,dict(bridge.last_query)

    for identity,frames in observations.values():
        cfg.unnorm_key=identity['normalization_statistics_key'];text=identity['language_instruction']
        for frame,obs in enumerate(frames):
            before=fingerprint(obs);previous=OfflineCameraPair.capture(frames[0],0)
            ref,ref_values,ref_layers,_=call(old,obs,text,previous,True)
            modes=[];errors={};equal=False
            for budget in (512,384,256):
                plain,_,_,plain_details=call(bridges[budget],obs,text,previous,False)
                audited,values,layers,details=call(bridges[budget],obs,text,previous,True)
                neutral=plain.tobytes()==audited.tobytes() and plain_details==details
                require(neutral,'diagnostic hooks changed commands')
                if budget==512:
                    require(all(values[k].shape==ref_values[k].shape for k in values),'parity shape differs')
                    errors={k:float(np.max(np.abs(values[k].astype(np.float64)-ref_values[k].astype(np.float64)))) for k in values}
                    equal=ref.tobytes()==audited.tobytes()
                    require(equal and all(v<=1e-6 for v in errors.values()),'all-token dense parity failed')
                modes.append(dict(budget=budget,finite=True,hooks_neutral=neutral,positions=layers[0]['positions'],
                    layer_lengths=[l['length'] for l in layers],original_positions_all_layers=all(l['positions']==layers[0]['positions'] for l in layers),
                    no_cache_all_layers=all(l['no_cache'] for l in layers),state_query=details['query']))
            require(before==fingerprint(obs),'input mutated')
            row=dict(observation_id=identity['trajectory_id'],frame=frame,input_unchanged=True,
                full_tokens=ref_layers[0]['length'],dense_errors=errors,dense_command_equal=equal,modes=modes)
            rows.append(row)
            # Validate the completed prefix without opening any scientific outcomes.
            require(all(m['original_positions_all_layers'] and m['no_cache_all_layers'] for m in modes),'position/cache contract failed')
            with (p/'checks.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
            with (p/'progress.jsonl').open('a') as f:f.write(json.dumps(dict(check_records=len(rows),model_queries=resources.queries))+'\n')
            require(sum(x.stat().st_size for x in p.rglob('*') if x.is_file())<CAPS['artifact_bytes']-65536,'artifact cap reached')
    require(names=={x.name for x in checkpoint.iterdir()},'checkpoint inventory changed')
    verify(c);resources.close();base.write_once(p/'records.json',rows)
    return rows


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--preflight-only',action='store_true');args=parser.parse_args()
    require(Path.cwd().resolve()==ROOT and Path(sys.prefix).resolve().is_relative_to(ROOT/'envs/openvla-oft'),'project runtime required')
    require(args.config.resolve().is_relative_to(ROOT/'configs/openvla'),'project config required')
    c=json.loads(args.config.read_text());old,reference,manifest=verify(c)
    from savr.openvla.offline_camera_inputs import load_offline_inputs
    observations,audit=load_offline_inputs(ROOT,manifest)
    require(audit==json.loads(scoped(old['input_preflight_report']).read_text())['input_audit'],'real input audit changed')
    if args.preflight_only:print(json.dumps(dict(preflight_passed=True,model_calls=0)));return
    require(os.environ.get('CUDA_VISIBLE_DEVICES')==str(c['gpu']['index']),'GPU selection differs')
    initial=base.snapshot(c['gpu']['index'])
    require(initial['uuid']==c['gpu']['uuid'] and initial['memory_mib']<=1024 and initial['utilization']<=5,'selected GPU not idle')
    p=scoped(c['output_root']);p.mkdir(exist_ok=False);base.environment(p)
    base.write_once(p/'launch.json',dict(config=c,config_sha256=base.sha(args.config),pid=os.getpid(),
        started_at=datetime.now(timezone.utc).isoformat(),initial_gpu=initial))
    base.write_once(p/'input_preflight.json',dict(complete=True,input_audit=audit,performed_before_model_loading=True))
    resources=base.Resources(dict(gpu=c['gpu'],caps=dict(queries=112,memory_mib=23552)))
    resources.peak=initial['memory_mib'];resources.thread.start();started=time.monotonic()
    def timeout(*_):raise RuntimeError('frozen time cap reached')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(1800)
    try:
        rows=run(c,reference,observations,p,resources)
        s=dict(complete=True,model_queries=resources.queries,episodes=0,training_performed=False,automatic_retry=False,
            positive_method_result=False,checkpoint_unchanged=True,authenticated_files_unchanged=True,
            worker_sha256=c['worker_sha256'],elapsed_seconds=time.monotonic()-started,
            peak_aggregate_gpu_memory_mib=resources.peak,completed_at=datetime.now(timezone.utc).isoformat())
        reconcile(c,s,rows)
        s['artifacts_sha256']={x.name:base.sha(x) for x in p.iterdir() if x.is_file() and x.suffix in ('.json','.jsonl')}
        require(sum(x.stat().st_size for x in p.rglob('*') if x.is_file())<CAPS['artifact_bytes']-65536,'artifact cap reached')
        base.write_once(p/'worker_summary.json',s);print(json.dumps(dict(complete=True,model_queries=resources.queries)))
    except BaseException as e:
        with (p/'technical_traceback.log').open('x') as f:traceback.print_exc(file=f)
        base.write_once(p/'technical_stop.json',dict(complete=False,error_type=type(e).__name__,error=str(e),
            model_queries=resources.queries,automatic_retry=False,elapsed_seconds=time.monotonic()-started,
            peak_aggregate_gpu_memory_mib=resources.peak))
        raise
    finally:
        signal.alarm(0);resources.stop.set();resources.thread.join(timeout=12)


if __name__=='__main__':main()
