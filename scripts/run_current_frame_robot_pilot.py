#!/usr/bin/env python3
"""Single frozen deployment qualification and 488-episode development evaluation."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time
import traceback
import run_fixed_compression_qualification as qualification
import run_fixed_compression_screen as screen
import run_openvla_original_baseline as base

ROOT=Path('/home/ved/SAVR')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from savr.openvla.current_frame_robot_contract import CAPS,GATES,ARMS,validate,reconcile,qualification_pass
from savr.openvla.contemporary_contract import require,HORIZONS

LEDGER='reports/cac_c0_recovery01/simulator_populations_v1.jsonl'
FIT='results/current-frame-adapter-fit-v01'
FILES=('scripts/run_current_frame_robot_pilot.py','scripts/analyze_current_frame_robot_pilot.py',
 'src/savr/openvla/current_frame_robot_contract.py','tests/openvla/test_current_frame_robot.py',
 'docs/CURRENT_FRAME_ROBOT_PILOT_V1.md','src/savr/openvla/current_frame_features.py',
 'src/savr/openvla/current_frame_corrector.py','src/savr/openvla/current_frame_records.py',
 'scripts/freeze_current_frame_robot_states.py',LEDGER)

def verify(c):
    require(base.sha(ROOT/'configs/openvla/fixed_compression_screen_v1.json')==
            'cc0100823806caac8770f4bb533dda181985b6224a3275997af0793b19590ac3','screen ancestor changed')
    old,ref,manifest=screen.verify(json.loads((ROOT/'configs/openvla/fixed_compression_screen_v1.json').read_text()))
    ledger=[json.loads(x) for x in (ROOT/LEDGER).read_text().splitlines()]
    validate(c,ledger,ref,c['observation_ids'])
    require(c['gpu']==old['gpu'] and c['observation_ids']==old['observation_ids'],'GPU/observations changed')
    require(set(c['authenticated_files'])==set(FILES),'source manifest missing')
    for name,digest in c['authenticated_files'].items():
        require(base.sha(qualification.scoped(name))==digest,'source changed: '+name)
    for name,digest in {
        'current_frame_corrector.py':'c17017eb13c0cd04468e109c0586b6e07458760a6a328508e59ea28db95206cc',
        'current_frame_features.py':'e63177ade228ed90bf1e836c598ed08ad4a23172a26fc09f3abad5198b0057a0',
        'current_frame_records.py':'c1d1d7f986c92278894da04c95e9da211dd1a2bab352526320c986e3b5368486',
    }.items():require(base.sha(ROOT/'src/savr/openvla'/name)==digest,'qualified adapter boundary changed')
    for name,digest in {'worker_summary.json':'c863b31bff5ea15f4c93bed21fda430734711642df56aa3e791e9a402c466627',
                        'analysis.json':'d9ed6108eef40d4c4466343e04baea42d58a88ba5e06a44368f89bfcbb64a2de'}.items():
        require(base.sha(ROOT/FIT/name)==digest,'completed fitting evidence changed')
    result=json.loads((ROOT/FIT/'analysis.json').read_text())
    require(result['technical_pass'] and result['decision']['eligible_for_development_evaluation'],'offline gate failed')
    for directory,summary_hash in ((ROOT/FIT,'c863b31bff5ea15f4c93bed21fda430734711642df56aa3e791e9a402c466627'),
            (ROOT/'results/current-frame-learning-qualification-v01','a0a1ea06d51883692057862cbaf38e81208486e4d717864a64e62ac5b48a2934')):
        require(base.sha(directory/'worker_summary.json')==summary_hash and not (directory/'technical_stop.json').exists(),'predecessor changed')
        for name,digest in json.loads((directory/'worker_summary.json').read_text())['artifacts_sha256'].items():
            p=(directory/name).resolve();require(p.is_relative_to(directory) and base.sha(p)==digest,'predecessor artifact changed')
    require(c['weights']=={a:dict(path=a+'-final.pt',sha256=h) for a,h in
            [('action_only','8a01608b9690bf2e62017968fbc2ab4afd86325503ccfd3322994ab1838b8640'),
             ('visual','15489080d2752d3cf90a7b9cb016cb4c95c9a395b335b400d63e74c33f6bb951')]},'weights changed')
    return old,ref,manifest


def cpu_adapter_preflight(c):
    """Load frozen deployment models and real feature shapes without a GPU or labels."""
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU preflight must hide CUDA')
    import torch
    from savr.openvla.current_frame_corrector import CurrentFrameCorrector
    from savr.openvla.current_frame_records import decode_record,arrays_to_tensors
    torch.set_num_threads(1)
    qdir=ROOT/'results/current-frame-learning-qualification-v01'
    approved=json.loads((qdir/'approved_samples.json').read_text())
    rows=json.loads((qdir/'records.json').read_text())
    require(len(rows)==16,'real deployment feature count')
    models={}
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        for arm in ('action_only','visual'):
            m=CurrentFrameCorrector(use_visual=arm=='visual').eval().requires_grad_(False)
            m.load_state_dict(torch.load(ROOT/FIT/c['weights'][arm]['path'],map_location='cpu',weights_only=True),strict=True)
            require(all(bool(torch.isfinite(v).all()) for v in m.parameters()),'nonfinite frozen weight')
            models[arm]=m
    with torch.inference_mode():
        for row in rows:
            _,arrays=decode_record((qdir/row['artifact']).read_bytes(),row['sha256'],approved)
            values=arrays_to_tensors(arrays);values.pop('teacher_actions')
            for m in models.values():
                actions=m(**{k:v.unsqueeze(0) for k,v in values.items()})
                require(tuple(actions.shape)==(1,8,7) and bool(torch.isfinite(actions).all()),'invalid deployment output')
    return dict(real_feature_records=16,adapters_loaded=2,model_queries=0,episodes=0,optimizer_updates=0)


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
        utils=utils,instruction_indexer=instruction_token_indices,budget=n) for n in (512,384)}
    from savr.openvla.current_frame_corrector import CurrentFrameCorrector
    from savr.openvla.current_frame_features import CurrentFrameFeatureQuery,CorrectedCurrentFrameQuery
    from savr.openvla.current_frame_records import decode_record,arrays_to_tensors
    for module in (model,head,proprio):module.requires_grad_(False)
    feature=CurrentFrameFeatureQuery(model=model,head=head,proprio=proprio,processor=processor,cfg=cfg,
        utils=utils,instruction_indexer=instruction_token_indices,budget=384)
    adapters={}
    for arm,visual in (('action_only',False),('visual',True)):
        adapter=CurrentFrameCorrector(use_visual=visual).cuda().eval()
        adapter.load_state_dict(torch.load(ROOT/'results/current-frame-adapter-fit-v01'/c['weights'][arm]['path'],
                                           map_location='cuda:0',weights_only=True),strict=True)
        adapter.requires_grad_(False);adapters[arm]=adapter
    zero=CurrentFrameCorrector().cuda().eval();zero.requires_grad_(False)
    bridges={'dense':bridges['512'],'compressed':bridges['384'],
             'zero':CorrectedCurrentFrameQuery(feature,zero),
             **{arm:CorrectedCurrentFrameQuery(feature,m) for arm,m in adapters.items()}}
    records=dict(episodes=[],timing=[],parity=[],qualification=[])

    def append(name,value):
        require(shutil.disk_usage(p).free>=10*1024**3,'free-space floor')
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
        other,_,b,nb=audited('dense',obs,text,previous,state)
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

    qdir=ROOT/'results/current-frame-learning-qualification-v01'
    approved=json.loads((qdir/'approved_samples.json').read_text())
    qualified={r['sample_id']:r for r in json.loads((qdir/'records.json').read_text())}
    for identity,frames in observations.values():
        cfg.unnorm_key=identity['normalization_statistics_key']
        for frame,obs in enumerate(frames):
            fingerprint=obs_hash(obs); previous=OfflineCameraPair.capture(frames[0],0)
            outputs={};dense_values={};features_equal=True
            sample=identity['trajectory_id']+':'+str(frame);record=qualified[sample]
            _,arrays=decode_record((qdir/record['artifact']).read_bytes(),record['sha256'],approved)
            stored=arrays_to_tensors(arrays);stored.pop('teacher_actions')
            for arm in ('native','dense','compressed','zero','action_only','visual'):
                state=EpisodeState();state.reset('integration:'+sample+':'+arm)
                if arm in ('native','dense'):
                    with OfficialActionHeadCapture(head) as capture:raw,details=call(arm,obs,identity['language_instruction'],previous,state)
                    dense_values[arm]=dict(hidden=capture.exact_hidden().detach().float().cpu().numpy(),
                        normalized=capture.exact_output().detach().float().cpu().numpy(),raw=raw)
                else:
                    raw,details=call(arm,obs,identity['language_instruction'],previous,state)
                    require(state.query==1,'integration query accounting')
                outputs[arm]=processed_command_chunk(raw,evaluation.process_action,cfg.model_family)
                if arm in ('zero','action_only','visual'):
                    features_equal &= all(torch.equal(feature.features[k].cpu(),v) for k,v in stored.items())
            errors={k:float(np.max(np.abs(dense_values['native'][k].astype(np.float64)-
                                        dense_values['dense'][k].astype(np.float64)))) for k in dense_values['native']}
            direct={}
            with torch.inference_mode():
                x={k:v.unsqueeze(0).cuda() for k,v in stored.items()}
                for arm,adapter in adapters.items():
                    norm=adapter(**x)[0].float().cpu().numpy()
                    raw=np.asarray(model._unnormalize_actions(norm,cfg.unnorm_key))
                    direct[arm]=processed_command_chunk(raw,evaluation.process_action,cfg.model_family)
            row=dict(observation_id=identity['trajectory_id'],frame=frame,
                input_unchanged=obs_hash(obs)==fingerprint,dense_command_equal=outputs['native'].tobytes()==outputs['dense'].tobytes(),
                zero_command_equal=outputs['zero'].tobytes()==outputs['compressed'].tobytes(),
                action_only_command_equal=outputs['action_only'].tobytes()==direct['action_only'].tobytes(),
                visual_command_equal=outputs['visual'].tobytes()==direct['visual'].tobytes(),
                features_equal=bool(features_equal),dense_errors=errors,
                no_gradients=all(not v.requires_grad and v.grad is None for m in (model,head,proprio,*adapters.values(),zero)
                                 for v in m.parameters()))
            append('qualification.jsonl',row);records['qualification'].append(row)
            require(all(row[k] is True for k in ('input_unchanged','dense_command_equal','zero_command_equal',
                'action_only_command_equal','visual_command_equal','features_equal','no_gradients')) and
                all(v<=1e-6 for v in errors.values()),'trained deployment qualification failed')
    qualification_pass(records['qualification'],c['observation_ids']);progress()
    del zero;del bridges['zero']
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
    require(args.config.resolve()==ROOT/'configs/openvla/current_frame_robot_pilot_v1.json','config path')
    c=json.loads(args.config.read_text());old,ref,manifest=verify(c)
    from savr.openvla.offline_camera_inputs import load_offline_inputs
    observations,audit=load_offline_inputs(ROOT,manifest)
    require(audit==json.loads(qualification.scoped(old['input_preflight_report']).read_text())['input_audit'],'offline inputs changed')
    from freeze_current_frame_robot_states import state_hashes
    require(state_hashes(c['episode_slots'])==c['initial_state_sha256'],'initial states changed')
    require(shutil.disk_usage(ROOT).free>=10*1024**3+CAPS['artifact_bytes'],'insufficient free space')
    if args.preflight_only:
        audit=cpu_adapter_preflight(c)
        print(json.dumps(dict(preflight_passed=True,conditions=124,adapter_preflight=audit)));return
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='0','GPU selection differs')
    initial=base.snapshot(0)
    require(initial['uuid']==c['gpu']['uuid'] and initial['memory_mib']<=1024 and initial['utilization']<=5,'GPU not idle')
    p=qualification.scoped(c['output_root']);p.mkdir(exist_ok=False);base.environment(p)
    base.write_once(p/'launch.json',dict(config=c,config_sha256=base.sha(args.config),pid=os.getpid(),
        started_at=datetime.now(timezone.utc).isoformat(),initial_gpu=initial))
    base.write_once(p/'input_preflight.json',dict(complete=True,input_audit=audit,initial_state_sha256=c['initial_state_sha256']))
    resources=base.Resources(dict(gpu=c['gpu'],caps=dict(queries=22000,memory_mib=23552)))
    resources.peak=initial['memory_mib'];resources.thread.start();started=time.monotonic()
    def timeout(*_):raise RuntimeError('pilot time cap')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(CAPS['seconds'])
    try:
        records=run(c,ref,observations,p,resources)
        s=dict(complete=True,model_queries=resources.queries,episodes=len(records['episodes']),training_performed=False,
            automatic_retry=False,positive_method_result=False,checkpoint_unchanged=True,authenticated_files_unchanged=True,
            elapsed_seconds=time.monotonic()-started,peak_aggregate_gpu_memory_mib=resources.peak,
            completed_at=datetime.now(timezone.utc).isoformat())
        reconcile(c,s,records)
        s['artifacts_sha256']={x.name:base.sha(x) for x in p.iterdir() if x.is_file() and x.suffix in ('.json','.jsonl')}
        s['artifact_bytes_before_summary']=sum(x.stat().st_size for x in p.rglob('*') if x.is_file())
        require(s['artifact_bytes_before_summary']<CAPS['artifact_bytes'],'artifact cap')
        base.write_once(p/'worker_summary.json',s);print(json.dumps(dict(complete=True,episodes=488)))
    except BaseException as error:
        with (p/'technical_traceback.log').open('x') as f:traceback.print_exc(file=f)
        base.write_once(p/'technical_stop.json',dict(complete=False,error_type=type(error).__name__,error=str(error),
            model_queries=resources.queries,automatic_retry=False,elapsed_seconds=time.monotonic()-started))
        raise
    finally:
        signal.alarm(0);resources.stop.set();resources.thread.join(timeout=12)

if __name__=='__main__':main()
