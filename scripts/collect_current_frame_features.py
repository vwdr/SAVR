#!/usr/bin/env python3
"""Frozen training-only collection. Never executes actions or fits an adapter."""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import signal
import sys
import time
import traceback
from types import SimpleNamespace
import run_openvla_original_baseline as base
import run_current_frame_learning_qualification as qualification
from freeze_current_frame_training_inputs import read_observation

ROOT = Path('/home/ved/SAVR')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from savr.openvla.current_frame_sampling import select_samples
from savr.openvla.offline_camera_inputs import observation_sha256
require, scoped = qualification.require, qualification.scoped
INPUTS = 'configs/openvla/current_frame_training_inputs_v1.json'
INPUT_SHA = '2fd13ced3e5a9a2b810eb299a5c1667f8c1de3ee2c5487a4a0683e61847d9ae8'
QUAL = 'configs/openvla/current_frame_learning_qualification_v1.json'
QUAL_SHA = 'b82b4ec88792d9253a7a90bbf7f4bdad90e83b2dcf384d5cdb91df1a84a3c8b5'
CAPS = dict(model_calls=8000, records=4000, seconds=28800, memory_mib=23552,
            artifact_bytes=20*1024**3, minimum_initial_free_bytes=30*1024**3,
            minimum_free_bytes=10*1024**3)
FILES = ('scripts/collect_current_frame_features.py', 'scripts/freeze_current_frame_training_inputs.py',
         'src/savr/openvla/current_frame_sampling.py', 'tests/openvla/test_feature_collection.py',
         'docs/CURRENT_FRAME_FEATURE_COLLECTION_V1.md')


def verify(c, *, raw_inputs):
    require(c['schema_version'] == 'current-frame-feature-collection-v1' and c['caps'] == CAPS
            and c['budget'] == 384 and c['automatic_retry'] is False
            and c['output_root'] == 'results/current-frame-feature-collection-v01', 'collection contract differs')
    require(set(c['authenticated_files']) == set(FILES), 'incomplete source manifest')
    for name, digest in c['authenticated_files'].items():
        require(base.sha(scoped(name)) == digest, 'collection source changed: '+name)
    require(base.sha(scoped(QUAL)) == QUAL_SHA and base.sha(scoped(INPUTS)) == INPUT_SHA, 'frozen inputs changed')
    q = json.loads(scoped(QUAL).read_text())
    _, reference, _ = qualification.verify(q)
    require(c['gpu'] == q['gpu'], 'GPU differs')
    directory = scoped('results/current-frame-learning-qualification-v01')
    require(not (directory/'technical_stop.json').exists(), 'qualification stopped')
    for name in ('worker_summary.json', 'analysis.json'):
        require(base.sha(directory/name) == c['qualification_sha256'][name], 'qualification evidence changed')
    s = json.loads((directory/'worker_summary.json').read_text())
    a = json.loads((directory/'analysis.json').read_text())
    require(a['qualification_passed'] is True and a['summary_sha256'] == base.sha(directory/'worker_summary.json'),
            'passed qualification analysis required')
    for name, digest in s['artifacts_sha256'].items():
        p = (directory/name).resolve()
        require(p.is_relative_to(directory) and base.sha(p) == digest, 'qualification artifact changed')
    qualification.reconcile(q, s, json.loads((directory/'records.json').read_text()),
                             json.loads((directory/'fit_checks.json').read_text()))
    m = json.loads(scoped(INPUTS).read_text())
    require(base.sha(scoped(m['training_index_path'])) == m['training_index_sha256'], 'index changed')
    selected = select_samples([json.loads(x) for x in scoped(m['training_index_path']).read_text().splitlines()])
    require([{k:v for k,v in r.items() if k!='observation_sha256'} for r in m['samples']] == selected,
            'selection or role differs')
    if raw_inputs:
        import h5py
        for source, digest in m['source_sha256'].items():
            path = scoped(m['data_root_relative']+'/'+source)
            require(base.sha(path) == digest, 'source data changed')
            with h5py.File(path, 'r') as h:
                for sample in (r for r in m['samples'] if r['identity']['source_path']==source):
                    obs = read_observation(h, sample['identity'], sample['frame'])
                    require(observation_sha256(obs) == sample['observation_sha256'], 'raw observation changed')
    return q, reference, m


def metadata(q, sample):
    r = sample['identity']; h = sample['observation_sha256']
    return dict(schema_version='current-frame-feature-v1', sample_id=sample['sample_id'],
        trajectory_id=r['trajectory_id'], source_sha256=r['source_sha256'], frame=sample['frame'],
        split='train', split_hash=r['split_hash'], student_observation_sha256=h, teacher_observation_sha256=h,
        compression_budget=384, selector_sha256=base.sha(scoped('src/savr/openvla/spatial_selection.py')),
        backbone_sha256=qualification.ANCESTOR_SHA,
        extractor_sha256=q['authenticated_files']['src/savr/openvla/current_frame_features.py'],
        normalization_statistics_key=r['normalization_statistics_key'],
        instruction_pooling='fp32_mean_current_instruction_input_embeddings',
        feature_origin='hard_compacted_current_frame_decoder')


def reconcile(c, m, s, rows):
    require(s.get('complete') is True and s.get('model_queries') == 8000 and s.get('records') == 4000
            and s.get('optimizer_steps') == 0 and s.get('episodes') == 0
            and s.get('positive_method_result') is False and s.get('automatic_retry') is False
            and s.get('authenticated_files_unchanged') is True, 'incomplete collection')
    require(len(rows) == len(m['samples']) == 4000, 'record count differs')
    for i, (r, expected) in enumerate(zip(rows, m['samples'])):
        require(r['sample_id'] == expected['sample_id'] and r['learning_role'] == expected['learning_role']
                and r['model_queries'] == 2*(i+1) and r['input_unchanged'] is True
                and r['record_roundtrip_equal'] is True and r['query_counts_equal_one'] is True
                and r['no_backbone_gradients'] is True, 'sample alignment failed')
        require(r['artifact'] == 'features/'+str(i).zfill(4)+'.npz', 'artifact order differs')
    require(sum(r['learning_role']=='fit' for r in rows)==3200 and
            sum(r['learning_role']=='validation' for r in rows)==800, 'role counts differ')
    for key, cap in (('elapsed_seconds', CAPS['seconds']), ('peak_aggregate_gpu_memory_mib', CAPS['memory_mib']),
                     ('artifact_bytes_before_summary', CAPS['artifact_bytes'])):
        require(type(s[key]) in (int,float) and math.isfinite(s[key]) and 0 <= s[key] < cap, 'resource cap exceeded')
    return True


def run(c, q, reference, m, p, resources):
    import h5py
    import numpy as np
    import torch
    import transformers
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(1); tf.config.threading.set_inter_op_parallelism_threads(1)
    require(torch.__version__=='2.2.0+cu118' and transformers.__version__=='4.40.1'
            and torch.cuda.device_count()==1, 'original single-GPU runtime required')
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(23000/24576,0)
    sys.path.insert(0,str(ROOT/'third_party/openvla-oft'))
    from experiments.robot import openvla_utils as utils
    from experiments.robot.libero import run_libero_eval as evaluation
    from experiments.robot.robot_utils import set_seed_everywhere
    from savr.cac.c1 import instruction_token_indices
    from savr.openvla.fixed_compression import FixedCompressionQuery
    from savr.openvla.current_frame_features import CurrentFrameFeatureQuery
    from savr.openvla.official_semantics import OfficialActionHeadCapture
    from savr.openvla.specprune import EpisodeState
    from savr.openvla.current_frame_records import encode_record, decode_record, tensors_to_arrays
    cfg=evaluation.GenerateConfig(pretrained_checkpoint=str(scoped(reference['checkpoint'])),
        task_suite_name='libero_spatial',num_trials_per_task=0,seed=7,local_log_dir=str(p/'logs'),
        use_wandb=False,center_crop=True,num_open_loop_steps=8,num_images_in_input=2,use_proprio=True,
        use_l1_regression=True,use_diffusion=False,use_film=False)
    evaluation.validate_config(cfg); set_seed_everywhere(7)
    update,sync=utils.update_auto_map,utils.check_model_logic_mismatch
    utils.update_auto_map=lambda _:None;utils.check_model_logic_mismatch=lambda _:None
    try:model,head,proprio,noisy,processor=evaluation.initialize_model(cfg)
    finally:utils.update_auto_map,utils.check_model_logic_mismatch=update,sync
    require(head is not None and proprio is not None and noisy is None and
            not any(x.training for x in (model,head,proprio)), 'released components differ')
    for x in (model,head,proprio):x.requires_grad_(False)
    shared=dict(model=model,head=head,proprio=proprio,processor=processor,cfg=cfg,utils=utils,
                instruction_indexer=instruction_token_indices)
    dense=FixedCompressionQuery(**shared,budget=512);student=CurrentFrameFeatureQuery(**shared,budget=384)
    approved={r['sample_id']:metadata(q,r) for r in m['samples']}
    base.write_once(p/'approved_samples.json',approved)
    base.write_once(p/'loaded_runtime.json',dict(torch=torch.__version__,transformers=transformers.__version__,
        configuration=vars(cfg),backbone_frozen=True,backbone_sha256_meaning='authenticated ancestry config, not weight shard'))
    rows=[];handle=None;open_source=None;feature_bytes=0
    (p/'features').mkdir()
    try:
        for i,sample in enumerate(m['samples']):
            r=sample['identity'];source=r['source_path']
            if source!=open_source:
                if handle is not None:handle.close()
                handle=h5py.File(scoped(m['data_root_relative']+'/'+source),'r');open_source=source
            obs=read_observation(handle,r,sample['frame']);before=observation_sha256(obs)
            require(before==sample['observation_sha256'],'frozen observation changed')
            cfg.unnorm_key=r['normalization_statistics_key'];previous=SimpleNamespace(step=sample['frame'])
            states=[]
            with torch.inference_mode():
                state=EpisodeState();state.reset(sample['sample_id']);resources.consume()
                with OfficialActionHeadCapture(head) as capture:
                    dense(obs,r['language_instruction'],previous,state)
                    target=capture.exact_output().reshape(8,7).float().detach().clone()
                states.append(state.query)
                state=EpisodeState();state.reset(sample['sample_id']);resources.consume()
                student(obs,r['language_instruction'],previous,state);states.append(state.query)
                arrays=tensors_to_arrays(dict(student.features,teacher_actions=target))
            payload,digest=encode_record(approved[sample['sample_id']],arrays,approved)
            name='features/'+str(i).zfill(4)+'.npz'
            with (p/name).open('xb') as f:f.write(payload)
            meta,again=decode_record((p/name).read_bytes(),digest,approved)
            roundtrip=meta==approved[sample['sample_id']] and all(np.array_equal(arrays[k],again[k]) for k in arrays)
            grad_free=all(not x.requires_grad and x.grad is None for module in (model,head,proprio) for x in module.parameters())
            unchanged=before==observation_sha256(obs)
            require(roundtrip and grad_free and unchanged and states==[1,1],'feature-generation contract failed')
            rows.append(dict(sample_id=sample['sample_id'],learning_role=sample['learning_role'],artifact=name,
                sha256=digest,model_queries=resources.queries,input_unchanged=unchanged,record_roundtrip_equal=roundtrip,
                no_backbone_gradients=grad_free,query_counts_equal_one=states==[1,1]))
            feature_bytes+=len(payload)
            require(feature_bytes<CAPS['artifact_bytes']-128*1024**2,'artifact budget exhausted')
            if i%25==0:require(shutil.disk_usage(p).free>=CAPS['minimum_free_bytes'],'free-space floor reached')
            with (p/'progress.jsonl').open('a') as f:
                f.write(json.dumps(dict(feature_records=i+1,model_queries=resources.queries,feature_bytes=feature_bytes))+'\n')
    finally:
        if handle is not None:handle.close()
    verify(c,raw_inputs=True);resources.close();base.write_once(p/'records.json',rows)
    return rows


def analyze(p):
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU-only analysis required')
    require(not (p/'technical_stop.json').exists(),'technical stop forbids acceptance')
    s=json.loads((p/'worker_summary.json').read_text());launch=json.loads((p/'launch.json').read_text());c=launch['config']
    require(base.sha(scoped('configs/openvla/current_frame_feature_collection_v1.json'))==launch['config_sha256'],
            'launch config changed')
    q,_,m=verify(c,raw_inputs=False)
    for name,digest in s['artifacts_sha256'].items():
        path=(p/name).resolve();require(path.is_relative_to(p) and base.sha(path)==digest,'artifact changed')
    rows=json.loads((p/'records.json').read_text());reconcile(c,m,s,rows)
    from savr.openvla.current_frame_records import decode_record
    approved={r['sample_id']:metadata(q,r) for r in m['samples']}
    require(approved==json.loads((p/'approved_samples.json').read_text()),'record allowlist changed')
    for r in rows:decode_record((p/r['artifact']).read_bytes(),r['sha256'],approved)
    a=dict(complete=True,collection_passed=True,records=4000,fit=3200,validation=800,model_queries=8000,
           optimizer_steps=0,episodes=0,positive_method_result=False,summary_sha256=base.sha(p/'worker_summary.json'))
    base.write_once(p/'analysis.json',a);print(json.dumps(a))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path)
    parser.add_argument('--preflight-only',action='store_true');parser.add_argument('--analyze',type=Path)
    args=parser.parse_args()
    require(Path.cwd().resolve()==ROOT and Path(sys.prefix).resolve().is_relative_to(ROOT/'envs/openvla-oft'),
            'original project runtime required')
    if args.analyze:
        p=args.analyze.resolve();require(p==scoped('results/current-frame-feature-collection-v01'),'run differs')
        analyze(p);return
    require(args.config and args.config.resolve().is_relative_to(ROOT/'configs/openvla'),'project config required')
    c=json.loads(args.config.read_text());q,reference,m=verify(c,raw_inputs=True)
    require(shutil.disk_usage(ROOT).free>=CAPS['minimum_initial_free_bytes'],'insufficient project disk space')
    if args.preflight_only:print(json.dumps(dict(preflight_passed=True,samples=4000,model_queries=0)));return
    require(os.environ.get('CUDA_VISIBLE_DEVICES')==str(c['gpu']['index']),'GPU selection differs')
    initial=base.snapshot(c['gpu']['index'])
    require(initial['uuid']==c['gpu']['uuid'] and initial['memory_mib']<=1024 and initial['utilization']<=5,'GPU not idle')
    p=scoped(c['output_root']);p.mkdir(exist_ok=False);base.environment(p)
    base.write_once(p/'launch.json',dict(config=c,config_sha256=base.sha(args.config),pid=os.getpid(),
        started_at=datetime.now(timezone.utc).isoformat(),initial_gpu=initial,input_manifest_sha256=INPUT_SHA))
    resources=base.Resources(dict(gpu=c['gpu'],caps=dict(queries=8000,memory_mib=23552)))
    resources.peak=initial['memory_mib'];resources.thread.start();started=time.monotonic()
    def timeout(*_):raise RuntimeError('collection time cap reached')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(CAPS['seconds'])
    try:
        rows=run(c,q,reference,m,p,resources)
        artifacts={str(x.relative_to(p)):base.sha(x) for x in p.rglob('*') if x.is_file() and 'runtime-cache' not in x.parts}
        s=dict(complete=True,model_queries=resources.queries,records=len(rows),optimizer_steps=0,episodes=0,
            positive_method_result=False,automatic_retry=False,authenticated_files_unchanged=True,
            elapsed_seconds=time.monotonic()-started,peak_aggregate_gpu_memory_mib=resources.peak,
            artifact_bytes_before_summary=sum(x.stat().st_size for x in p.rglob('*') if x.is_file()),
            artifacts_sha256=artifacts,completed_at=datetime.now(timezone.utc).isoformat())
        reconcile(c,m,s,rows);base.write_once(p/'worker_summary.json',s)
        print(json.dumps(dict(complete=True,records=4000,model_queries=8000)))
    except BaseException as e:
        with (p/'technical_traceback.log').open('x') as f:traceback.print_exc(file=f)
        base.write_once(p/'technical_stop.json',dict(complete=False,error_type=type(e).__name__,error=str(e),
            model_queries=resources.queries,automatic_retry=False,elapsed_seconds=time.monotonic()-started))
        raise
    finally:
        signal.alarm(0);resources.stop.set();resources.thread.join(timeout=12)


if __name__=='__main__':main()
