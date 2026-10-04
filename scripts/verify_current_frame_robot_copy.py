#!/usr/bin/env python3
"""Independent, read-only, standard-library reconciliation of completed pilot copy.

Does not import the worker, its decision function, or its analyzer. No GPU use.
"""
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'results/current-frame-robot-pilot-v01'
ARMS=('dense','compressed','action_only','visual')
SUITES=('libero_spatial','libero_object','libero_goal','libero_10')
HORIZONS=dict(zip(SUITES,(220,280,300,520)))


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4194304),b''):h.update(b)
    return h.hexdigest()


def check(value,message):
    if not value:raise ValueError(message)


def main():
    s=json.loads((RUN/'worker_summary.json').read_text())
    check(s['complete'] is True and not (RUN/'technical_stop.json').exists(),'completion')
    check(sha(RUN/'worker_summary.json')=='699de8203703ea508ab5668584ed2d7d52452f53af6dc20a539809a50221588e','summary identity')
    cpath=ROOT/'configs/openvla/current_frame_robot_pilot_v1.json'
    check(sha(cpath)=='9e3f6728be56821d5da73185f310f1119adda32e2a4fc6e50672cc2e71612c8e','frozen config')
    c=json.loads(cpath.read_text())
    for name,h in c['authenticated_files'].items():
        path=(ROOT/name).resolve();check(path.is_relative_to(ROOT) and sha(path)==h,'source: '+name)
    for name,h in s['artifacts_sha256'].items():
        path=(RUN/name).resolve();check(path.is_relative_to(RUN) and sha(path)==h,'artifact: '+name)
    r=json.loads((RUN/'records.json').read_text())
    for name in ('episodes','timing','qualification','parity'):
        check(r[name]==[json.loads(x) for x in (RUN/(name+'.jsonl')).read_text().splitlines()],'bundle/stream: '+name)
    check((len(r['episodes']),len(r['timing']),len(r['qualification']))==(488,272,16),'counts')
    check(s['training_performed'] is False and s['automatic_retry'] is False and
          s['positive_method_result'] is False and s['checkpoint_unchanged'] is True and
          s['authenticated_files_unchanged'] is True,'worker invariants')
    check(0<s['elapsed_seconds']<57600 and 0<s['peak_aggregate_gpu_memory_mib']<23552 and
          0<s['artifact_bytes_before_summary']<536870912,'resource bounds')
    # Runtime caches may release temporary files on exit. Authenticate immutable
    # evidence by its manifest, and independently check the present storage bound.
    check(sum(x.stat().st_size for x in RUN.rglob('*') if x.is_file())<536870912,'current artifact byte cap')
    check([(x['observation_id'],x['frame']) for x in r['qualification']]==
          [(i,f) for i in c['observation_ids'] for f in (0,1)],'qualification identities')
    for q in r['qualification']:
        check(all(q[k] is True for k in ('input_unchanged','dense_command_equal','zero_command_equal',
              'action_only_command_equal','visual_command_equal','features_equal','no_gradients')),'qualification')
        check(set(q['dense_errors'])=={'hidden','normalized','raw'} and
              all(math.isfinite(v) and 0<=v<=1e-6 for v in q['dense_errors'].values()),'qualification parity')
    calls=96+272;native=[];pairs={};expected_parity=[]
    for slot,e in zip(c['episode_slots'],r['episodes']):
        ident=slot['condition']
        check((e['slot_id'],e['condition_id'],e['arm'],e['suite'])==
              (slot['slot_id'],ident['condition_id'],slot['arm'],ident['suite']),'episode identity')
        n=e['policy_queries'];steps=e['executed_steps']
        check(type(e['success']) is bool and 0<steps<=HORIZONS[e['suite']] and
              n==math.ceil(steps/8) and 8*n==steps+e['remaining_actions'] and
              e['replans']==e['discarded_actions']==0 and e['controller_enabled'] is False,'execution')
        check(len(e['queries'])==n and sum(q['seconds'] for q in e['queries'])<=e['episode_seconds'],'query times')
        for i,q in enumerate(e['queries'],1):
            check(q['query']==i and q['retained_visual_tokens']==(512 if e['arm'] in ('dense','native') else 384)
                  and q['precise'] is False and 0<q['seconds']<math.inf,'query boundary')
        calls+=n*(2 if e['arm']=='native' else 1)
        if slot['role']=='primary':
            check(ident['seed']==7 and ident['initial_state_id'] in (1,2,3),'development population')
            pair=pairs.setdefault(ident['condition_id'],dict(condition=ident,arms={}))
            check(e['arm'] not in pair['arms'],'duplicate arm');pair['arms'][e['arm']]=e
        else:
            check(e['arm']=='native' and ident['initial_state_id']==0,'native control')
            native.append(e);expected_parity.extend((e['slot_id'],i) for i in range(1,n+1))
    check(len(pairs)==120 and len(native)==8 and calls==s['model_queries']<=22000,'paired/call counts')
    check(all(set(p['arms'])==set(ARMS) for p in pairs.values()),'missing paired arm')
    check([(p['slot_id'],p['query']) for p in r['parity']]==expected_parity,'native shadows')
    for p in r['parity']:
        check(p['command_bytes_equal'] is True and p['layers']==[32,32] and
              set(p['errors'])=={'hidden','normalized','raw'} and
              all(math.isfinite(v) and 0<=v<=1e-6 for v in p['errors'].values()),'shadow parity')
    for slot,t in zip(c['timing_slots'],r['timing']):
        check(all(t[k]==v for k,v in slot.items()) and t['query']==t['frame']+1 and
              t['retained_visual_tokens']==(512 if t['arm']=='dense' else 384) and
              t['precise'] is False and 0<t['seconds']<math.inf,'timing membership')
    successes={arm:sum(p['arms'][arm]['success'] for p in pairs.values()) for arm in ARMS}
    suites={su:{arm:sum(p['arms'][arm]['success'] for p in pairs.values() if p['condition']['suite']==su)
                for arm in ARMS} for su in SUITES}
    for su in SUITES:
        ps=[p for p in pairs.values() if p['condition']['suite']==su]
        tasks={p['condition']['task_id'] for p in ps}
        check(len(ps)==30 and len(tasks)==10,'suite balance')
        check(all(sorted(p['condition']['initial_state_id'] for p in ps if p['condition']['task_id']==t)==[1,2,3]
                  for t in tasks),'state balance')
    times={arm:mean(t['seconds'] for t in r['timing'] if t['arm']==arm and not t['warmup']) for arm in ARMS}
    a=json.loads((RUN/'analysis.json').read_text())
    check(a['technical_pass'] and a['successes']==successes and a['per_suite']==suites,'analysis counts')
    check(a['summary_sha256']==sha(RUN/'worker_summary.json'),'analysis provenance')
    for arm in ARMS:check(math.isclose(a['controlled_policy_query_seconds'][arm]['mean'],times[arm],rel_tol=1e-12),'timing mean')
    checks=dict(baseline=successes['dense']>=108 and sum(e['success'] for e in native)==8,
        incremental_success=successes['visual']-successes['compressed']>=3,
        visual_over_action_only=successes['visual']-successes['action_only']>=1,
        dense_success_retained=successes['dense']-successes['visual']<=2,
        no_large_suite_loss=all(v['compressed']-v['visual']<=2 for v in suites.values()),
        useful_controlled_speed=1-times['visual']/times['dense']>=.1)
    check(a['decision']==dict(checks=checks,positive_development_signal=all(checks.values()),
          paper_ready=False,confirmatory_result=False),'all frozen gates')
    for comp in a['comparisons']:
        differences=[int(p['arms'][comp['candidate']]['success'])-int(p['arms'][comp['reference']]['success']) for p in pairs.values()]
        check(comp['net_successes']==sum(differences) and comp['candidate_only']==differences.count(1)
              and comp['reference_only']==differences.count(-1),'paired comparison')
    print(json.dumps(dict(independent_reconciliation_passed=True,artifacts_verified=len(s['artifacts_sha256']),
        successes=successes,per_suite=suites,controlled_mean_ms={a:1000*v for a,v in times.items()},
        frozen_checks=checks,native_successes=sum(e['success'] for e in native),model_calls=calls,
        summary_sha256=sha(RUN/'worker_summary.json'),analysis_sha256=sha(RUN/'analysis.json')),sort_keys=True))


if __name__=='__main__':main()
