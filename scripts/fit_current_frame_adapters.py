#!/usr/bin/env python3
"""One prospective matched fit. No backbone loading, robot execution or tuning."""
import argparse
from collections import OrderedDict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import run_openvla_original_baseline as base
from savr.openvla.current_frame_training_plan import (
    ARMS, OPTIMIZER, CAPS, require, schedule, authorized_indices, offline_decision)

COLLECTION = ROOT/'results/current-frame-feature-collection-v01'
OUTPUT = ROOT/'results/current-frame-adapter-fit-v01'
INPUT = ROOT/'configs/openvla/current_frame_training_inputs_v1.json'
FILES = ('scripts/fit_current_frame_adapters.py', 'scripts/verify_current_frame_collection_copy.py',
         'scripts/run_openvla_original_baseline.py', 'src/savr/openvla/current_frame_training_plan.py',
         'src/savr/openvla/current_frame_corrector.py', 'src/savr/openvla/current_frame_records.py',
         'tests/openvla/test_current_frame_training.py', 'docs/CURRENT_FRAME_ADAPTER_FIT_V1.md')


def now(): return datetime.now(timezone.utc).isoformat()
def read(path): return json.loads(path.read_text())


def checked_config(path):
    c = read(path)
    require(c['schema_version']=='current-frame-adapter-fit-v1' and c['optimizer']==OPTIMIZER
            and c['caps']==CAPS and c['arms']==list(ARMS) and c['budget']==384
            and c['automatic_retry'] is False and c['output_root']==str(OUTPUT.relative_to(ROOT)), 'config contract')
    require(c['gpu']==dict(index=0,uuid='GPU-bb2451d6-2989-a112-5c18-8892943710e4'), 'GPU contract')
    require(set(c['authenticated_files'])==set(FILES), 'source manifest')
    for name,digest in c['authenticated_files'].items():
        require(base.sha(ROOT/name)==digest, 'source changed: '+name)
    for name,digest in c['collection_sha256'].items():
        require(name in ('worker_summary.json','analysis.json') and base.sha(COLLECTION/name)==digest,
                'collection identity')
    require(set(c['collection_sha256'])=={'worker_summary.json','analysis.json'}, 'collection identities missing')
    require(base.sha(INPUT)==c['input_sha256'], 'input identity')
    return c


def preflight(path):
    c=checked_config(path)
    from verify_current_frame_collection_copy import main as verify_copy
    verify_copy()
    samples=read(INPUT)['samples']; plan=schedule(samples)
    require(shutil.disk_usage(ROOT).free>=CAPS['minimum_free_bytes']+CAPS['artifact_bytes'], 'free disk')
    return c, samples, plan


class Reader:
    def __init__(self, plan):
        self.plan=plan; self.rows=read(COLLECTION/'records.json')
        self.approved=read(COLLECTION/'approved_samples.json')
        self.cache=OrderedDict(); self.opened={'fit':set(),'validation':set()}
        self.validation_ready=False

    def get(self, i, role):
        from savr.openvla.current_frame_records import decode_record, arrays_to_tensors
        authorized_indices(self.plan,[i],role)
        require(role!='validation' or self.validation_ready,'validation before checkpoint freeze')
        require(self.rows[i]['learning_role']==role, 'record role')
        self.opened[role].add(i)
        if i not in self.cache:
            r=self.rows[i]; path=(COLLECTION/r['artifact']).resolve()
            require(path.is_relative_to(COLLECTION), 'record path escape')
            meta,arrays=decode_record(path.read_bytes(),r['sha256'],self.approved)
            require(meta['sample_id']==r['sample_id'],'record identity')
            self.cache[i]=arrays_to_tensors(arrays)
            if len(self.cache)>32: self.cache.popitem(last=False)
        self.cache.move_to_end(i)
        return self.cache[i]

    def batch(self, indices, role, device, *, shuffled=False):
        import torch
        authorized_indices(self.plan,indices,role)
        require(not shuffled or role=='fit','shuffled validation forbidden')
        records=[self.get(i,role) for i in indices]
        target_indices=[self.plan['shuffled_targets'][str(i)] for i in indices] if shuffled else indices
        targets=[self.get(i,role)['teacher_actions'] for i in target_indices]
        x={k:torch.stack([r[k] for r in records]).to(device)
           for k in records[0] if k!='teacher_actions'}
        return x,torch.stack(targets).to(device)


def state_digest(state):
    h=hashlib.sha256()
    for name,value in sorted(state.items()):
        h.update(name.encode());h.update(str(tuple(value.shape)).encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def initial_states(**dimensions):
    import torch
    from savr.openvla.current_frame_corrector import CurrentFrameCorrector
    torch.manual_seed(7)
    visual=CurrentFrameCorrector(use_visual=True,**dimensions)
    torch.manual_seed(7)
    action=CurrentFrameCorrector(use_visual=False,**dimensions)
    vs=visual.state_dict(); state=action.state_dict()
    require(all(k in vs and vs[k].shape==v.shape for k,v in state.items()), 'unmatched shared parameter')
    action.load_state_dict({k:vs[k].clone() for k in state},strict=True)
    return {False:action.state_dict(), True:visual.state_dict()}


def update(model, x, target, optimizer):
    import torch
    from savr.openvla.current_frame_corrector import teacher_imitation_loss
    optimizer.zero_grad(set_to_none=True)
    loss=teacher_imitation_loss(model(**x),target)
    loss.backward()
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True)
    require(bool(torch.isfinite(norm)) and all(p.grad is None or bool(torch.isfinite(p.grad).all())
            for p in model.parameters()), 'nonfinite gradient')
    optimizer.step()
    require(all(bool(torch.isfinite(p).all()) for p in model.parameters()), 'nonfinite parameter')
    require(all(not v.requires_grad and v.grad is None for v in (*x.values(),target)), 'upstream gradients')
    return float(loss.detach())


def aggregate(rows):
    require(len(rows)==800 and len({r['sample_id'] for r in rows})==800, 'validation counts')
    metrics={}
    for arm in ('base',*ARMS):
        vals=[r[arm+'_l1'] for r in rows]
        require(all(math.isfinite(v) and v>=0 for v in vals),'validation finite')
        metrics[arm+'_l1']=sum(vals)/len(vals)
        metrics[arm+'_gripper_sign_disagreement']=sum(r[arm+'_gripper_sign_disagreement'] for r in rows)/len(rows)
    metrics['by_suite']={suite:{arm+'_l1':sum(r[arm+'_l1'] for r in rows if r['suite']==suite)/
        sum(r['suite']==suite for r in rows) for arm in ('base',*ARMS)} for suite in sorted({r['suite'] for r in rows})}
    return metrics


def validate_predictions(rows, samples, plan):
    """Recompute each metric from saved predictions, independently of Torch."""
    require(len(rows)==800,'prediction count')
    def chunk(value):
        require(isinstance(value,list) and len(value)==8 and all(isinstance(r,list) and len(r)==7 for r in value),
                'prediction chunk shape')
        require(all(type(v) in (int,float) and math.isfinite(v) for row in value for v in row),'prediction finite')
        return value
    for row,i in zip(rows,plan['validation']):
        sample=samples[i];identity=sample['identity']
        require((row['sample_id'],row['suite'],row['task_id'])==
                (sample['sample_id'],identity['suite'],identity['task_id']),'prediction identity')
        target=chunk(row['teacher'])
        for arm in ('base',*ARMS):
            prediction=chunk(row[arm+'_prediction'])
            l1=sum(abs(prediction[j][k]-target[j][k]) for j in range(8) for k in range(7))/56
            gripper=sum((prediction[j][6]>0)!=(target[j][6]>0) for j in range(8))/8
            require(math.isclose(l1,row[arm+'_l1'],rel_tol=1e-6,abs_tol=1e-7)
                    and gripper==row[arm+'_gripper_sign_disagreement'],'saved metric differs')


def fit(c, samples, plan, p, resources):
    import torch
    from savr.openvla.current_frame_corrector import CurrentFrameCorrector
    require(torch.__version__=='2.2.0+cu118' and torch.cuda.device_count()==1,'original single-GPU runtime')
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True);torch.use_deterministic_algorithms(True)
    torch.cuda.set_per_process_memory_fraction(CAPS['own_allocated_mib']/24576,0)
    torch.cuda.reset_peak_memory_stats()
    reader=Reader(plan); initial=initial_states(); checkpoints={}; fits=[]; steps=0
    base.write_once(p/'schedule.json',plan)
    base.write_once(p/'runtime.json',dict(torch=torch.__version__,device=torch.cuda.get_device_name(0),
        adapter_dtype='float32',backbone_loaded=False,tf32=False,attention='math',deterministic_algorithms=True,
        initial_state_sha256={str(k):state_digest(v) for k,v in initial.items()}))
    def guard():
        if resources.error: raise RuntimeError(resources.error)
        require(torch.cuda.max_memory_allocated()/1024**2<CAPS['own_allocated_mib'],'own memory cap')
        require(shutil.disk_usage(p).free>=CAPS['minimum_free_bytes'],'free disk floor')
    for arm in ARMS:
        torch.manual_seed(OPTIMIZER['seed']);visual=arm!='action_only'
        model=CurrentFrameCorrector(use_visual=visual).cuda().train()
        model.load_state_dict(initial[visual],strict=True)
        require(all(v.dtype==torch.float32 for v in model.parameters()),'FP32 parameters')
        optimizer=torch.optim.AdamW(model.parameters(),lr=OPTIMIZER['lr'],betas=tuple(OPTIMIZER['betas']),
            eps=OPTIMIZER['eps'],weight_decay=0.,foreach=False)
        losses=[]; own_steps=0
        for order in plan['epoch_orders']:
            epoch_total=0.
            for j in range(0,len(order),16):
                guard();ids=order[j:j+16]
                x,target=reader.batch(ids,'fit','cuda:0',shuffled=arm=='visual_shuffled')
                if own_steps==0:
                    with torch.no_grad():require(torch.equal(model(**x),x['base_actions']),'zero-init mismatch')
                epoch_total+=update(model,x,target,optimizer); own_steps+=1;steps+=1
                with (p/'progress.jsonl').open('a') as f:
                    f.write(json.dumps(dict(arm=arm,arm_steps=own_steps,optimizer_steps=steps))+'\n')
            losses.append(epoch_total/200)
        require(own_steps==2000 and not reader.opened['validation'],'fit accounting or leakage')
        model.eval();x,_=reader.batch(plan['fit'][:16],'fit','cuda:0')
        with torch.no_grad():pred=model(**x)
        name=arm+'-final.pt'
        with (p/name).open('xb') as handle:torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},handle)
        loaded=CurrentFrameCorrector(use_visual=visual).cuda().eval()
        loaded.load_state_dict(torch.load(p/name,map_location='cuda:0',weights_only=True),strict=True)
        with torch.no_grad():require(torch.equal(pred,loaded(**x)),'reload prediction mismatch')
        checkpoints[arm]=dict(path=name,sha256=base.sha(p/name),optimizer_steps=own_steps)
        fits.append(dict(arm=arm,steps=own_steps,epoch_mean_l1=losses,reload_equal=True,
                         parameters=sum(v.numel() for v in model.parameters()),
                         initial_state_sha256=state_digest(initial[visual])))
        del optimizer,model,loaded,pred,x,target
    require(steps==6000 and reader.opened['fit']==set(plan['fit']) and not reader.opened['validation'],'fit accounting')
    base.write_once(p/'checkpoints_frozen.json',dict(complete=True,created_at=now(),checkpoints=checkpoints,
        optimizer_steps=steps,validation_records_opened=0,checkpoint_rule='final_only'))
    seal_sha=base.sha(p/'checkpoints_frozen.json')
    reader.cache.clear();reader.validation_ready=True
    base.write_once(p/'validation_started.json',dict(started_at=now(),checkpoints_frozen_sha256=seal_sha,
                    optimizer_steps=steps))
    models={}
    for arm in ARMS:
        m=CurrentFrameCorrector(use_visual=arm!='action_only').cuda().eval()
        require(base.sha(p/checkpoints[arm]['path'])==checkpoints[arm]['sha256'],'checkpoint changed')
        m.load_state_dict(torch.load(p/checkpoints[arm]['path'],map_location='cuda:0',weights_only=True),strict=True)
        models[arm]=m
    rows=[]
    for j in range(0,800,16):
        guard();ids=plan['validation'][j:j+16];x,target=reader.batch(ids,'validation','cuda:0')
        with torch.no_grad():predictions={'base':x['base_actions'],**{arm:m(**x) for arm,m in models.items()}}
        for k,i in enumerate(ids):
            sample=samples[i];row=dict(sample_id=sample['sample_id'],suite=sample['identity']['suite'],
                                      task_id=sample['identity']['task_id'],teacher=target[k].cpu().tolist())
            for arm,pred in predictions.items():
                require(bool(torch.isfinite(pred).all()),'nonfinite validation prediction')
                row[arm+'_l1']=float((pred[k]-target[k]).abs().mean())
                row[arm+'_gripper_sign_disagreement']=float(((pred[k,:,6]>0)!=(target[k,:,6]>0)).float().mean())
                row[arm+'_prediction']=pred[k].cpu().tolist()
            rows.append(row)
    require(reader.opened['validation']==set(plan['validation']),'validation coverage')
    require(base.sha(p/'checkpoints_frozen.json')==seal_sha,'seal changed')
    for info in checkpoints.values():require(base.sha(p/info['path'])==info['sha256'],'weights changed after validation')
    base.write_once(p/'fit_details.json',fits);base.write_once(p/'validation.json',rows)
    base.write_once(p/'access_audit.json',dict(fit=sorted(reader.opened['fit']),validation=sorted(reader.opened['validation']),
                                             optimizer_steps_before_validation=steps,optimizer_steps_after_validation=steps))
    guard()
    return dict(optimizer_steps=steps,checkpoints=3,validation_records=len(rows),
                own_peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2)


def analyze(p):
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU-only analysis')
    require(not (p/'technical_stop.json').exists(),'technical stop')
    s=read(p/'worker_summary.json');launch=read(p/'launch.json')
    path=ROOT/'configs/openvla/current_frame_adapter_fit_v1.json';c=checked_config(path)
    require(launch['config_sha256']==base.sha(path) and launch['config']==c,'launch identity')
    require(s['complete'] and s['optimizer_steps']==6000 and s['checkpoints']==3
        and s['validation_records']==800 and s['model_calls']==s['episodes']==0
        and s['positive_method_result'] is False,'summary accounting')
    for name,digest in s['artifacts_sha256'].items():
        f=(p/name).resolve();require(f.is_relative_to(p) and base.sha(f)==digest,'artifact mismatch')
    for key,limit in [('elapsed_seconds',28800),('peak_aggregate_gpu_memory_mib',23552),
                       ('own_peak_allocated_mib',4096),('artifact_bytes_before_summary',2*1024**3)]:
        require(math.isfinite(s[key]) and 0<=s[key]<limit,'resource cap')
    plan=schedule(read(INPUT)['samples']);require(read(p/'schedule.json')==plan,'schedule changed')
    seal=read(p/'checkpoints_frozen.json');start=read(p/'validation_started.json');access=read(p/'access_audit.json')
    require(seal['validation_records_opened']==0 and seal['optimizer_steps']==6000
            and seal['checkpoint_rule']=='final_only' and set(seal['checkpoints'])==set(ARMS),'checkpoint freeze')
    require(start['checkpoints_frozen_sha256']==base.sha(p/'checkpoints_frozen.json') and
            start['optimizer_steps']==6000 and seal['created_at']<=start['started_at'],'validation order')
    require(access==dict(fit=sorted(plan['fit']),validation=sorted(plan['validation']),
                        optimizer_steps_before_validation=6000,optimizer_steps_after_validation=6000),'access audit')
    for info in seal['checkpoints'].values():
        require(info['optimizer_steps']==2000 and base.sha(p/info['path'])==info['sha256'],'checkpoint hash/steps')
    fits=read(p/'fit_details.json');require([x['arm'] for x in fits]==list(ARMS),'fit arms')
    require(all(x['steps']==2000 and x['reload_equal'] and len(x['epoch_mean_l1'])==10
            and all(math.isfinite(y) and y>=0 for y in x['epoch_mean_l1']) for x in fits),'fit contract')
    require(fits[1]['initial_state_sha256']==fits[2]['initial_state_sha256'],'control initialization')
    progress=[json.loads(line) for line in (p/'progress.jsonl').read_text().splitlines()]
    require(len(progress)==6000 and all(r==dict(arm=ARMS[i//2000],arm_steps=i%2000+1,
        optimizer_steps=i+1) for i,r in enumerate(progress)),'update progress accounting')
    samples=read(INPUT)['samples'];rows=read(p/'validation.json')
    require([r['sample_id'] for r in rows]==[samples[i]['sample_id'] for i in plan['validation']], 'validation membership')
    validate_predictions(rows,samples,plan)
    metrics=aggregate(rows)
    result=dict(complete=True,technical_pass=True,metrics=metrics,decision=offline_decision(metrics),
                summary_sha256=base.sha(p/'worker_summary.json'),positive_method_result=False)
    base.write_once(p/'analysis.json',result);print(json.dumps(result))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path)
    parser.add_argument('--preflight-only',action='store_true');parser.add_argument('--analyze',action='store_true')
    args=parser.parse_args()
    require(ROOT==Path('/home/ved/SAVR') and Path.cwd()==ROOT and
            Path(sys.prefix).resolve().is_relative_to(ROOT/'envs/openvla-oft'),'project runtime')
    if args.analyze:analyze(OUTPUT);return
    path=args.config.resolve();require(path==ROOT/'configs/openvla/current_frame_adapter_fit_v1.json','config path')
    c,samples,plan=preflight(path)
    if args.preflight_only:
        # No action values are printed or used for tuning; this is after config freeze.
        reader=Reader(plan);x,target=reader.batch(plan['fit'][:16],'fit','cpu')
        require(tuple(target.shape)==(16,8,7) and not reader.opened['validation'],'real batch contract')
        print(json.dumps(dict(preflight_passed=True,fit_records_decoded=16,
                              validation_records_decoded=0,optimizer_steps=0,episodes=0)))
        return
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='0' and os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8','GPU environment')
    initial=base.snapshot(0)
    require(initial['uuid']==c['gpu']['uuid'] and initial['memory_mib']<=1024 and initial['utilization']<=5,'GPU not idle')
    OUTPUT.mkdir(exist_ok=False);base.environment(OUTPUT)
    base.write_once(OUTPUT/'launch.json',dict(config=c,config_sha256=base.sha(path),started_at=now(),pid=os.getpid(),initial_gpu=initial))
    resources=base.Resources(dict(gpu=c['gpu'],caps=dict(memory_mib=23552,queries=0)))
    resources.peak=initial['memory_mib'];resources.thread.start();started=time.monotonic()
    def timeout(*_):raise RuntimeError('fit time cap')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(CAPS['seconds'])
    try:
        counts=fit(c,samples,plan,OUTPUT,resources)
        checked_config(path);resources.close()
        artifacts={str(x.relative_to(OUTPUT)):base.sha(x) for x in OUTPUT.rglob('*')
                   if x.is_file() and 'runtime-cache' not in x.parts}
        total=sum(x.stat().st_size for x in OUTPUT.rglob('*') if x.is_file())
        require(total<CAPS['artifact_bytes'],'artifact cap')
        base.write_once(OUTPUT/'worker_summary.json',dict(complete=True,**counts,model_calls=0,episodes=0,
            positive_method_result=False,automatic_retry=False,authenticated_files_unchanged=True,
            elapsed_seconds=time.monotonic()-started,peak_aggregate_gpu_memory_mib=resources.peak,
            artifact_bytes_before_summary=total,artifacts_sha256=artifacts,completed_at=now()))
        print(json.dumps(dict(complete=True,optimizer_steps=6000,checkpoints=3,validation_records=800)))
    except BaseException as error:
        with (OUTPUT/'technical_traceback.log').open('x') as handle:traceback.print_exc(file=handle)
        base.write_once(OUTPUT/'technical_stop.json',dict(complete=False,error_type=type(error).__name__,error=str(error),
                            automatic_retry=False,elapsed_seconds=time.monotonic()-started))
        raise
    finally:
        signal.alarm(0);resources.stop.set();resources.thread.join(timeout=12)


if __name__=='__main__':main()
