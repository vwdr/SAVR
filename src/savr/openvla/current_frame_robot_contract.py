"""Outcome-blind schedules and bounded pilot accounting."""
import math
from .contemporary_contract import SUITES,HORIZONS,require,finite

ARMS=('dense','compressed','action_only','visual')
CAPS=dict(model_calls=22000,episodes=488,seconds=57600,aggregate_memory_mib=23552,artifact_bytes=536870912)
GATES=dict(dense_minimum=108,native_successes=8,visual_gain_over_compressed=3,
           visual_gain_over_action_only=1,visual_loss_to_dense=2,maximum_suite_loss_to_compressed=2,
           minimum_controlled_mean_reduction=.10)


def sha256_string(value):
    return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)


def episode_slots(ledger,reference):
    selected=[r for r in ledger if r['seed']==7 and
              ((r['population']=='headroom_stage1' and r['initial_state_id'] in (1,2)) or
               (r['population']=='headroom_extension' and r['initial_state_id']==3))]
    require(len(selected)==120 and len({r['condition_id'] for r in selected})==120,'120 development identities required')
    slots=[];index=0
    def add(row,arm,role):
        row={k:v for k,v in row.items() if k!='arm_order'}
        slots.append(dict(slot=len(slots)+1,slot_id=f"{role}:{row['condition_id']}:{arm}",
                          condition=row,arm=arm,role=role))
    for suite in SUITES:
        local=sorted([r for r in selected if r['suite']==suite],key=lambda r:(r['task_id'],r['initial_state_id']))
        require(len(local)==30 and len({r['task_id'] for r in local})==10,'suite task balance')
        for task in {r['task_id'] for r in local}:
            require([r['initial_state_id'] for r in local if r['task_id']==task]==[1,2,3],'state balance')
        controls=[r for r in reference['episode_conditions'] if r['suite']==suite]
        require(len(controls)==10 and controls[0]['initial_state_id']==0,'old control population')
        add(controls[0],'native','native_before')
        for row in local:
            shift=index%4;index+=1
            for arm in ARMS[shift:]+ARMS[:shift]:add(row,arm,'primary')
        add(controls[0],'native','native_after')
    require(len(slots)==488,'episode count')
    return slots


def timing_slots(ids):
    require(len(ids)==len(set(ids))==8,'eight timing trajectories required')
    out=[]
    for warmup,rounds in ((True,2),(False,4)):
        for rnd in range(rounds):
            for i,identity in enumerate(ids[:1] if warmup else ids):
                shift=(i+rnd)%4
                for arm in ARMS[shift:]+ARMS[:shift]:
                    for frame in (0,1):out.append(dict(index=len(out)+1,warmup=warmup,round=rnd,
                                                     observation_id=identity,arm=arm,frame=frame))
    return out


def validate(c,ledger,reference,ids):
    require(c['schema_version']=='current-frame-robot-pilot-v1' and c['caps']==CAPS and c['gates']==GATES
            and c['automatic_retry'] is False and c['arms']==list(ARMS)
            and c['output_root']=='results/current-frame-robot-pilot-v01','pilot contract')
    require(c['episode_slots']==episode_slots(ledger,reference) and c['timing_slots']==timing_slots(ids),'schedule changed')
    identities={r['condition']['condition_id'] for r in c['episode_slots']}
    require(set(c['initial_state_sha256'])==identities and len(identities)==124,'initial-state manifest')
    require(all(sha256_string(h) for h in c['initial_state_sha256'].values()),'invalid state hash')


def qualification_pass(rows,ids):
    require([(r['observation_id'],r['frame']) for r in rows]==[(i,f) for i in ids for f in (0,1)],'qualification membership')
    for r in rows:
        require(all(r[k] is True for k in ('input_unchanged','dense_command_equal','zero_command_equal',
            'action_only_command_equal','visual_command_equal','features_equal','no_gradients')),'integration qualification failed')
        require(set(r['dense_errors'])=={'hidden','normalized','raw'} and
                all(finite(x) and x<=1e-6 for x in r['dense_errors'].values()),'dense parity')


def reconcile(c,s,r):
    require(s['complete'] is True and s['training_performed'] is False and s['automatic_retry'] is False
        and s['positive_method_result'] is False
        and s['checkpoint_unchanged'] is True and s['authenticated_files_unchanged'] is True,'completion contract')
    require(len(c['episode_slots'])==488 and len(c['timing_slots'])==272,'schedule counts')
    require(len(r['episodes'])==s['episodes']==488 and len(r['timing'])==272,'terminal/timing counts')
    qualification_pass(r['qualification'],c['observation_ids'])
    require(finite(s['elapsed_seconds'],upper=57600) and finite(s['peak_aggregate_gpu_memory_mib'],upper=23552),'resource cap')
    calls=96+272;parity=[]
    for slot,e in zip(c['episode_slots'],r['episodes']):
        require(e['slot_id']==slot['slot_id'] and e['arm']==slot['arm'] and
                e['condition_id']==slot['condition']['condition_id'] and e['suite']==slot['condition']['suite'],'episode identity')
        n,steps=e['policy_queries'],e['executed_steps']
        require(type(e['success']) is bool and type(n) is int and type(steps) is int and
                0<n<=steps<=HORIZONS[e['suite']] and n==math.ceil(steps/8),'episode counts')
        require(e['controller_enabled'] is False and e['replans']==e['discarded_actions']==0
                and all(type(e[k]) is int for k in ('replans','discarded_actions','remaining_actions'))
                and 0<=e['remaining_actions']<8 and 8*n==steps+e['remaining_actions'],'queue')
        require(len(e['queries'])==n and finite(e['episode_seconds'],strict_lower=True)
                and all(finite(e[k]) for k in ('simulator_seconds','controller_seconds','nonquery_preparation_seconds')),'episode costs')
        budget=512 if e['arm'] in ('native','dense') else 384
        require(all(q['query']==i and q['retained_visual_tokens']==budget and q['precise'] is False
                    and finite(q['seconds'],strict_lower=True) for i,q in enumerate(e['queries'],1)),'query contract')
        require(sum(q['seconds'] for q in e['queries'])<=e['episode_seconds'],'query timing accounting')
        calls+=n*(2 if e['arm']=='native' else 1)
        if e['arm']=='native':parity.extend((e['slot_id'],i) for i in range(1,n+1))
    require([(p['slot_id'],p['query']) for p in r['parity']]==parity,'native shadow counts')
    for p in r['parity']:
        require(p['command_bytes_equal'] is True and p['layers']==[32,32] and
                set(p['errors'])=={'hidden','normalized','raw'} and
                all(finite(v) and v<=1e-6 for v in p['errors'].values()),'native shadow parity')
        require(all(sha256_string(p[k]) for k in ('observation_sha256','command_sha256')),'invalid native trace hash')
    for slot,t in zip(c['timing_slots'],r['timing']):
        require(all(t[k]==v for k,v in slot.items()) and t['query']==t['frame']+1 and t['precise'] is False
                and t['retained_visual_tokens']==(512 if t['arm']=='dense' else 384)
                and finite(t['seconds'],strict_lower=True),'controlled timing')
    require(type(s['model_queries']) is int and s['model_queries']==calls<=CAPS['model_calls'],'model call accounting')
    return True


def decision(successes,native,by_suite,reduction):
    checks=dict(baseline=successes['dense']>=108 and native==8,
        incremental_success=successes['visual']-successes['compressed']>=3,
        visual_over_action_only=successes['visual']>successes['action_only'],
        dense_success_retained=successes['dense']-successes['visual']<=2,
        no_large_suite_loss=all(v['compressed']-v['visual']<=2 for v in by_suite.values()),
        useful_controlled_speed=reduction>=.10)
    return dict(checks=checks,positive_development_signal=all(checks.values()),
                paper_ready=False,confirmatory_result=False)
