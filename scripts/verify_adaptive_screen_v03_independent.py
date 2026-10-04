#!/usr/bin/env python3
"""Read-only, standard-library reconciliation independent of frozen worker imports."""
import collections
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / 'results/adaptive-screen-v03'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_text())
s = read(P / 'worker_summary.json')
launch = read(P / 'launch.json')
c = launch['config']
r = read(P / 'records.json')
assert s['complete'] is True and not (P / 'technical_stop.json').exists()
assert sha(ROOT / 'configs/openvla/adaptive_screen_v3.json') == launch['config_sha256'] == 'ae1cbd7d0dbaccb3ff14a335cae6b451baa2b6ea62f25968e52c5ea5e3c8c768'
for name, h in s['artifacts_sha256'].items():
    assert (P / name).resolve().is_relative_to(P) and sha(P / name) == h, name
for name, h in c['authenticated_files'].items():
    assert sha(ROOT / name) == h, name
for name in ('episodes', 'timing', 'parity'):
    assert r[name] == [json.loads(x) for x in (P / (name + '.jsonl')).read_text().splitlines()]
assert len(r['episodes']) == s['episodes'] == 168 and len(r['timing']) == 240
assert s['worker_sha256'] == c['worker_sha256']
assert all(s[k] is True for k in ('checkpoint_unchanged', 'authenticated_files_unchanged'))
assert all(s[k] is False for k in ('automatic_retry', 'training_performed', 'positive_method_result'))
assert 0 <= s['elapsed_seconds'] < 21600 and 0 <= s['peak_aggregate_gpu_memory_mib'] < 23552
artifact_bytes = sum(x.stat().st_size for x in P.rglob('*') if x.is_file())
assert artifact_bytes < 536870912
assert s['adaptive_zero_tolerance'] == 1e-6 and s['adaptive_zero_tolerance_max'] == .001
arms = ('512', '384', 'adaptive_75', 'adaptive_50')
horizons = dict(libero_spatial=220, libero_object=280, libero_goal=300, libero_10=520)
def kept(a, q):
    return 384 if a == '384' else (384 if a == 'adaptive_75' else 256) if a.startswith('adaptive') and q > 3 else 512
def finite(x):
    return type(x) in (int, float) and math.isfinite(x) and x >= 0
calls = 240
parity_order = []
for slot, e in zip(c['episode_slots'], r['episodes']):
    assert e['slot_id'] == slot['slot_id'] and e['arm'] == slot['arm']
    assert e['condition_id'] == slot['condition']['condition_id'] and e['suite'] == slot['condition']['suite']
    n = e['policy_queries']; steps = e['executed_steps']
    assert type(e['success']) is bool and type(n) is int and type(steps) is int
    assert 0 < n <= steps <= horizons[e['suite']]
    assert e['controller_enabled'] is False and e['replans'] == e['discarded_actions'] == 0
    assert 0 <= e['remaining_actions'] < 8 and 8*n == steps + e['remaining_actions']
    assert len(e['queries']) == n and finite(e['episode_seconds']) and e['episode_seconds'] > 0
    for k in ('simulator_seconds', 'controller_seconds', 'nonquery_preparation_seconds'): assert finite(e[k])
    for i, q in enumerate(e['queries'], 1):
        assert q['query'] == i and q['retained_visual_tokens'] == kept(e['arm'], i)
        assert q['precise'] is False and finite(q['seconds']) and q['seconds'] > 0
    assert sum(q['seconds'] for q in e['queries']) <= e['episode_seconds']
    calls += n * (3 if e['arm'] == 'native' else 1)
    if e['arm'] == 'native':
        parity_order.extend((e['slot_id'], i, pair) for i in range(1,n+1) for pair in ('native-vs-512','native-vs-adaptive-zero'))
assert calls == s['model_queries'] <= 7000
assert [(p['slot_id'],p['query'],p['pair']) for p in r['parity']] == parity_order
for p in r['parity']:
    assert p['command_bytes_equal'] and p['layers'] == [32,32]
    assert p['sdpa_calls'] >= 1 and p['shadow_sdpa_calls'] >= 1
    if p['pair'] == 'native-vs-adaptive-zero': assert p['shadow_sdpa_calls'] == 32
    assert set(p['errors']) == {'hidden','normalized','raw'}
    assert all(finite(v) and v <= 1e-6 for v in p['errors'].values())
    for k in ('observation_sha256', 'command_sha256'):
        assert len(p[k]) == 64 and all(x in '0123456789abcdef' for x in p[k])
for slot,t in zip(c['timing_slots'],r['timing']):
    assert all(t[k] == v for k,v in slot.items())
    assert t['query'] == t['position'] and t['retained_visual_tokens'] == kept(t['arm'],t['query'])
    assert t['precise'] is False and finite(t['seconds']) and t['seconds'] > 0
assert sum(t['warmup'] for t in r['timing']) == 16
out = {'verified': True, 'model_calls': calls, 'episode_count':168, 'timing_count':240,
       'parity_count':len(r['parity']), 'maximum_parity_error':max(v for p in r['parity'] for v in p['errors'].values()),
       'summary_sha256':sha(P/'worker_summary.json'), 'local_artifact_bytes':artifact_bytes,
       'arms':{}, 'controls':{}}
for a in arms:
    eps = [e for e in r['episodes'] if e['arm'] == a]; assert len(eps)==40
    ts = [t for t in r['timing'] if t['arm']==a and not t['warmup']]; assert len(ts)==56
    out['arms'][a] = dict(successes=sum(e['success'] for e in eps),
        mean_ms=1000*statistics.mean(t['seconds'] for t in ts),
        regime_mean_ms={g:1000*statistics.mean(t['seconds'] for t in ts if t['regime']==g) for g in set(t['regime'] for t in ts)},
        mean_episode_seconds=statistics.mean(e['episode_seconds'] for e in eps),
        per_suite={su:sum(e['success'] for e in eps if e['suite']==su) for su in horizons})
for su in horizons:
    ns=[e for e in r['episodes'] if e['suite']==su and e['arm']=='native']; assert len(ns)==2
    dense=[e for e in r['episodes'] if e['suite']==su and e['arm']=='512'][0]
    traces=[[(p['observation_sha256'],p['command_sha256']) for p in r['parity'] if p['slot_id']==e['slot_id']] for e in ns]
    out['controls'][su]=dict(successes=[e['success'] for e in ns], first_dense=dense['success'],
        repeatability_limited=len({dense['success'],*(e['success'] for e in ns)}) != 1,
        traces_equal=traces[0]==traces[1])
out['baseline_concern']=out['arms']['512']['successes'] < 39 or any(x['repeatability_limited'] for x in out['controls'].values())
for a in arms[1:]:
    x=out['arms'][a]; d=out['arms']['512']
    x['reduction']=1-x['mean_ms']/d['mean_ms']; x['count_loss']=d['successes']-x['successes']
    x['triage_eligible']=not out['baseline_concern'] and x['count_loss']<=6 and x['reduction']>=.10
analysis=read(P/'analysis.json')
assert analysis['technical_reconciliation_passed'] and analysis['selected_arm'] is None
assert analysis['positive_method_result'] is False and analysis['baseline_concern']==out['baseline_concern']
for a in arms:
    x=out['arms'][a]
    assert analysis['successes'][a]==x['successes']
    assert math.isclose(analysis['timing_seconds'][a]['all']['mean']*1000,x['mean_ms'],abs_tol=1e-9)
    assert math.isclose(analysis['episode_costs'][a]['mean'],x['mean_episode_seconds'],abs_tol=1e-9)
    if a!='512':
        ca=next(v for v in analysis['candidates'] if v['arm']==a)
        assert ca['eligible']==x['triage_eligible'] and ca['success_count_loss']==x['count_loss']
        assert math.isclose(ca['controlled_mean_query_reduction'],x['reduction'],abs_tol=1e-12)
        paired={e['condition_id']:e for e in r['episodes'] if e['arm']=='512'}
        es=[e for e in r['episodes'] if e['arm']==a]
        assert ca['dense_only_successes']==sum(paired[e['condition_id']]['success'] and not e['success'] for e in es)
        assert ca['compressed_only_successes']==sum(e['success'] and not paired[e['condition_id']]['success'] for e in es)
out['frozen_analyzer_reconciled']=True
print(json.dumps(out,indent=2))
