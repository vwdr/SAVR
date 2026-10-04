#!/usr/bin/env python3
"""CPU-only screen reconciliation and predeclared substrate selection."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
import run_fixed_compression_screen as worker
from savr.openvla.compression_screen_contract import reconcile
from savr.openvla.contemporary_contract import SUITES,require
from analyze_contemporary_reference_v2 import stats


def analyze(c,reference,s,r,*,draws=10000):
    reconcile(c,reference,s,r,c['observation_ids'])
    require(type(draws) is int and draws>=100,'too few descriptive draws')
    arms=('512','384','256');pairs=[];controls=[];suites=[]
    for suite in SUITES:
        episodes=[e for e in r['episodes'] if e['suite']==suite]
        native=[e for e in episodes if e['arm']=='native']
        ids=list(dict.fromkeys(e['condition_id'] for e in episodes if e['arm']!='native'))
        local=[]
        for identity in ids:
            entry=dict(condition_id=identity,suite=suite,arms={e['arm']:e for e in episodes if e['condition_id']==identity and e['arm']!='native'})
            pairs.append(entry);local.append(entry)
        flag=len({local[0]['arms']['512']['success'],*(e['success'] for e in native)})!=1
        traces=[[(p['observation_sha256'],p['command_sha256']) for p in r['parity'] if p['slot_id']==e['slot_id']] for e in native]
        controls.append(dict(suite=suite,before=native[0],after=native[1],repeatability_limited=flag,
                             observation_command_traces_equal=traces[0]==traces[1]))
        suites.append(dict(suite=suite,conditions=10,repeatability_limited=flag,
                           successes={a:sum(e['arms'][a]['success'] for e in local) for a in arms}))
    successes={a:sum(e['arms'][a]['success'] for e in pairs) for a in arms}
    baseline_concern=successes['512']<39 or any(x['repeatability_limited'] for x in controls)
    timing=[t for t in r['timing'] if not t['warmup']]
    timing_stats={a:{label:stats([t['seconds'] for t in timing if t['arm']==a and (frame is None or t['frame']==frame)])
        for label,frame in (('first_query',0),('second_query',1),('combined',None))} for a in arms}
    costs={a:{'episode_seconds':stats([p['arms'][a]['episode_seconds'] for p in pairs]),
        'policy_queries':stats([p['arms'][a]['policy_queries'] for p in pairs]),
        'executed_steps':stats([p['arms'][a]['executed_steps'] for p in pairs]),
        'query_seconds':stats([q['seconds'] for p in pairs for q in p['arms'][a]['queries']])} for a in arms}
    means={a:np.array([np.mean([t['seconds'] for t in timing if t['arm']==a and t['observation_id']==i])
        for i in c['observation_ids']]) for a in arms}
    rng=np.random.default_rng(7);task_indices=rng.integers(0,10,(4,draws,10));trace_indices=rng.integers(0,8,(draws,8))
    candidates=[]
    for a in ('384','256'):
        diff=np.array([[int(p['arms'][a]['success'])-int(p['arms']['512']['success']) for p in pairs if p['suite']==suite] for suite in SUITES])
        sampled=sum(diff[i][task_indices[i]].sum(1) for i in range(4))/40
        reduction=float(1-means[a].mean()/means['512'].mean())
        trace_sample=1-means[a][trace_indices].mean(1)/means['512'][trace_indices].mean(1)
        loss=successes['512']-successes[a]
        eligible=not baseline_concern and loss<=6 and reduction>=.10
        candidates.append(dict(budget=int(a),successes=successes[a],success_count_loss=loss,
            success_difference=float(diff.mean()),dense_only_successes=int((diff==-1).sum()),
            compressed_only_successes=int((diff==1).sum()),controlled_mean_query_reduction=reduction,
            descriptive_paired_success_interval_95=np.quantile(sampled,[.025,.975]).tolist(),
            descriptive_trace_reduction_interval_95=np.quantile(trace_sample,[.025,.975]).tolist(),
            success_triage_passed=loss<=6,time_triage_passed=reduction>=.10,eligible=eligible))
    selected=next((x['budget'] for x in candidates if x['eligible']),None)
    return dict(schema_version='fixed-compression-screen-analysis-v1',technical_reconciliation_passed=True,
        positive_method_result=False,training_performed=False,population='40 consumed development conditions; initial state 0; seed 7',
        successes=successes,baseline_concern=baseline_concern,suites=suites,native_controls=controls,pairs=pairs,
        timing_seconds=timing_stats,episode_costs=costs,candidates=candidates,selected_budget=selected,
        decision='eligible_for_separately_frozen_training_pilot' if selected else 'stop_before_training',
        bootstrap=dict(draws=draws,seed=7,success_unit='task, stratified by suite',timing_unit='paired trajectory cluster'),
        limitations=['Eligibility is not evidence that a corrector can recover success.',
            'One state/seed per consumed task; no confirmatory or noninferiority claim.',
            'Independent native rollout variation remains reported even with matching successes.',
            'Controlled timing is a short eight-trajectory corpus, not general deployment acceleration.',
            'No learned corrector or adapter overhead is included.'])


def load(run):
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CUDA must be hidden')
    run=run.resolve();require(run.is_relative_to(worker.ROOT/'results') and (run/'worker_summary.json').is_file()
        and not (run/'technical_stop.json').exists(),'completed immutable summary required')
    s=json.loads((run/'worker_summary.json').read_text());require(s['complete'] is True,'summary incomplete')
    require(sum(1 for _ in (run/'episodes.jsonl').open())==128 and sum(1 for _ in (run/'timing.jsonl').open())==204,'unblind counts differ')
    for n,h in s['artifacts_sha256'].items():
        p=(run/n).resolve();require(p.is_relative_to(run) and worker.base.sha(p)==h,'artifact hash differs')
    launch=json.loads((run/'launch.json').read_text());c=launch['config']
    require(worker.base.sha(worker.ROOT/'configs/openvla/fixed_compression_screen_v1.json')==launch['config_sha256'],'launch config differs')
    _,ref,_=worker.verify(c);r=json.loads((run/'records.json').read_text())
    for n in ('episodes','timing','parity'):
        require(r[n]==[json.loads(x) for x in (run/(n+'.jsonl')).read_text().splitlines()],'bundle and stream differ')
    return c,ref,s,r


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);args=p.parse_args()
    result=analyze(*load(args.run));worker.base.write_once(args.run/'analysis.json',result)
    print(json.dumps({k:result[k] for k in ('technical_reconciliation_passed','successes','baseline_concern','selected_budget','positive_method_result')}))
