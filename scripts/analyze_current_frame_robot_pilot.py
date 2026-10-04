#!/usr/bin/env python3
"""Post-completion CPU-only pilot analysis; never selects a new model or population."""
import json
import os
from pathlib import Path
import numpy as np
import run_current_frame_robot_pilot as worker
from savr.openvla.current_frame_robot_contract import ARMS,decision,reconcile
from savr.openvla.contemporary_contract import SUITES,require


def stats(values):
    a=np.asarray(values,dtype=float)
    require(a.size>0 and np.isfinite(a).all(),'invalid timing vector')
    return dict(n=len(a),mean=float(a.mean()),median=float(np.median(a)),p95=float(np.quantile(a,.95)))


def analyze(c,s,r):
    reconcile(c,s,r)
    by_id={}
    for slot,e in zip(c['episode_slots'],r['episodes']):
        if slot['role']=='primary':
            ident=slot['condition'];key=ident['condition_id']
            by_id.setdefault(key,dict(condition=ident,arms={} ))['arms'][e['arm']]=e
    pairs=list(by_id.values());require(len(pairs)==120 and all(set(p['arms'])==set(ARMS) for p in pairs),'primary pairs')
    successes={a:sum(p['arms'][a]['success'] for p in pairs) for a in ARMS}
    suites={su:{a:sum(p['arms'][a]['success'] for p in pairs if p['condition']['suite']==su) for a in ARMS} for su in SUITES}
    native=[e for e in r['episodes'] if e['arm']=='native']
    controls=[]
    for suite in SUITES:
        es=[e for e in native if e['suite']==suite]
        traces=[[(q['observation_sha256'],q['command_sha256']) for q in r['parity'] if q['slot_id']==e['slot_id']] for e in es]
        controls.append(dict(suite=suite,successes=[e['success'] for e in es],
                             observation_command_traces_equal=traces[0]==traces[1]))
    measured=[t for t in r['timing'] if not t['warmup']]
    times={a:stats([t['seconds'] for t in measured if t['arm']==a]) for a in ARMS}
    means={a:np.array([np.mean([t['seconds'] for t in measured if t['arm']==a and t['observation_id']==i])
                      for i in c['observation_ids']]) for a in ARMS}
    reductions={a:float(1-means[a].mean()/means['dense'].mean()) for a in ARMS if a!='dense'}
    rng=np.random.default_rng(7);taskdraws=rng.integers(0,10,(4,10000,10));tracedraws=rng.integers(0,8,(10000,8))
    comparisons=[]
    for candidate,reference in (('compressed','dense'),('action_only','dense'),('visual','dense'),
                                 ('visual','compressed'),('visual','action_only'),('action_only','compressed')):
        values=[]
        for suite in SUITES:
            ps=sorted([p for p in pairs if p['condition']['suite']==suite],key=lambda p:(p['condition']['task_id'],p['condition']['initial_state_id']))
            values.append(np.array([int(p['arms'][candidate]['success'])-int(p['arms'][reference]['success']) for p in ps]).reshape(10,3))
        diff=np.array(values);sampled=sum(diff[i][taskdraws[i]].sum(axis=(1,2)) for i in range(4))/120
        timed=1-means[candidate][tracedraws].mean(1)/means[reference][tracedraws].mean(1)
        comparisons.append(dict(candidate=candidate,reference=reference,net_successes=int(diff.sum()),
            candidate_only=int((diff==1).sum()),reference_only=int((diff==-1).sum()),success_difference=float(diff.mean()),
            descriptive_paired_success_interval_95=np.quantile(sampled,[.025,.975]).tolist(),
            controlled_mean_query_reduction=float(1-means[candidate].mean()/means[reference].mean()),
            descriptive_paired_time_interval_95=np.quantile(timed,[.025,.975]).tolist()))
    costs={a:{'episode_seconds':stats([p['arms'][a]['episode_seconds'] for p in pairs]),
              'query_seconds':stats([q['seconds'] for p in pairs for q in p['arms'][a]['queries']]),
              'policy_queries':stats([p['arms'][a]['policy_queries'] for p in pairs]),
              'executed_steps':stats([p['arms'][a]['executed_steps'] for p in pairs]),
              **{k:stats([p['arms'][a][k] for p in pairs]) for k in ('simulator_seconds','controller_seconds','nonquery_preparation_seconds')}} for a in ARMS}
    return dict(complete=True,technical_pass=True,successes=successes,per_suite=suites,native_controls=controls,
        controlled_policy_query_seconds=times,controlled_reductions=reductions,episode_costs=costs,comparisons=comparisons,
        decision=decision(successes,sum(e['success'] for e in native),suites,reductions['visual']),
        population='120 development conditions, 40 tasks, states 1/2/3, seed 7; not independent held-out data',
        old_state_zero_screen='historical context only, excluded from primary counts',
        bootstrap=dict(draws=10000,seed=7,success_unit='task cluster stratified by suite, three states kept together',
                       timing_unit='paired two-frame trajectory'),
        limitations=['Single training seed and rollout seed; development exposure, no confirmatory inference.',
          'Degenerate bootstrap intervals at ceiling do not imply zero uncertainty.',
          'Controlled timing excludes simulator observation resizing; episode query timing includes it.',
          'Different closed-loop trajectories confound causal comparisons of episode-average time.',
          'No outcome-selected checkpoint, state, arm or subset; no timing-only positive claim.'])


if __name__=='__main__':
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU-only analysis required')
    p=worker.ROOT/'results/current-frame-robot-pilot-v01'
    require((p/'worker_summary.json').exists() and not (p/'technical_stop.json').exists(),'completed run required')
    s=json.loads((p/'worker_summary.json').read_text());require(s['complete'],'summary incomplete')
    for name,n in [('episodes',488),('timing',272),('qualification',16)]:
        require(sum(1 for _ in (p/(name+'.jsonl')).open())==n,'terminal count')
    for name,h in s['artifacts_sha256'].items():
        path=(p/name).resolve();require(path.is_relative_to(p) and worker.base.sha(path)==h,'artifact hash')
    launch=json.loads((p/'launch.json').read_text());c=launch['config']
    require(worker.base.sha(worker.ROOT/'configs/openvla/current_frame_robot_pilot_v1.json')==launch['config_sha256'],'config changed')
    worker.verify(c);r=json.loads((p/'records.json').read_text())
    for name in ('episodes','timing','qualification','parity'):
        require(r[name]==[json.loads(x) for x in (p/(name+'.jsonl')).read_text().splitlines()],'record bundle/stream mismatch')
    require(s['artifact_bytes_before_summary']<worker.CAPS['artifact_bytes'],'artifact cap')
    a=analyze(c,s,r);a['summary_sha256']=worker.base.sha(p/'worker_summary.json')
    worker.base.write_once(p/'analysis.json',a)
    print(json.dumps({k:a[k] for k in ('technical_pass','successes','decision')}))
