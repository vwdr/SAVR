"""Frozen three-arm screen accounting; science triage never changes execution."""
import math
from .contemporary_contract import SUITES, HORIZONS, require, finite

BUDGETS=(512,384,256)
CAPS=dict(model_calls=7000,episodes=128,seconds=21600,aggregate_memory_mib=23552,artifact_bytes=536870912)


def episode_slots(reference):
    rows=reference['episode_conditions']
    require(len(rows)==40 and len({r['condition_id'] for r in rows})==40,'40 distinct conditions required')
    slots=[]
    def add(condition,arm,role):
        slots.append(dict(slot=len(slots)+1,slot_id=f"{role}:{condition['condition_id']}:{arm}",
                          condition={k:v for k,v in condition.items() if k!='arm_order'},arm=arm,role=role))
    for s,suite in enumerate(SUITES):
        local=rows[10*s:10*(s+1)]
        require(all(r['suite']==suite and r['initial_state_id']==0 and r['seed']==7 for r in local),'population differs')
        add(local[0],'native','native_before')
        for i,row in enumerate(local):
            rotate=(10*s+i)%3; order=BUDGETS[rotate:]+BUDGETS[:rotate]
            for budget in order:add(row,str(budget),'primary')
        add(local[0],'native','native_after')
    return slots


def timing_slots(ids):
    require(len(ids)==len(set(ids))==8,'eight unique trajectories required')
    out=[]
    for warmup,rounds in ((True,2),(False,4)):
        for rnd in range(rounds):
            for i,identity in enumerate(ids[:1] if warmup else ids):
                shift=(i+rnd)%3
                for budget in BUDGETS[shift:]+BUDGETS[:shift]:
                    for frame in (0,1):
                        out.append(dict(index=len(out)+1,warmup=warmup,round=rnd,
                            observation_id=identity,arm=str(budget),frame=frame))
    return out


def validate_config(c,reference,ids):
    require(c['schema_version']=='fixed-compression-screen-v1' and c['launch_ready'] is True
            and c['caps']==CAPS and c['automatic_retry'] is False and c['budgets']==list(BUDGETS)
            and c['tolerance']==1e-6 and c['output_root']=='results/fixed-compression-screen-v01', 'screen contract changed')
    require(c['episode_slots']==episode_slots(reference) and c['timing_slots']==timing_slots(ids),'schedule differs')
    require(c['triage']==dict(minimum_mean_time_reduction=.10,maximum_success_count_loss=6,
        dense_minimum_successes=39,preference=[384,256]),'triage differs')
    return True


def reconcile(c,reference,s,r,ids):
    validate_config(c,reference,ids)
    require(s.get('complete') is True and s.get('training_performed') is False
        and s.get('automatic_retry') is False and s.get('positive_method_result') is False
        and s.get('checkpoint_unchanged') is True and s.get('authenticated_files_unchanged') is True,'incomplete evidence')
    require(s['worker_sha256']==c['worker_sha256'],'worker differs')
    require(finite(s['elapsed_seconds'],upper=CAPS['seconds']) and finite(s['peak_aggregate_gpu_memory_mib'],upper=23552),'resource cap exceeded')
    require(len(r['episodes'])==s['episodes']==128 and len(r['timing'])==204,'counts differ')
    calls=204;parity=[]
    for slot,e in zip(c['episode_slots'],r['episodes']):
        require(e['slot_id']==slot['slot_id'] and e['arm']==slot['arm']
            and e['condition_id']==slot['condition']['condition_id'] and e['suite']==slot['condition']['suite'],'episode ordering differs')
        n,steps=e['policy_queries'],e['executed_steps']
        require(type(e['success']) is bool and type(n) is int and type(steps) is int
            and 0<n<=steps<=HORIZONS[e['suite']] and n==math.ceil(steps/8),'episode horizon/query mismatch')
        require(e['controller_enabled'] is False and e['replans']==e['discarded_actions']==0
            and type(e['remaining_actions']) is int and 0<=e['remaining_actions']<8
            and 8*n==steps+e['remaining_actions'],'queue/controller changed')
        require(len(e['queries'])==n and finite(e['episode_seconds'],strict_lower=True)
            and all(finite(e[k]) for k in ('simulator_seconds','controller_seconds','nonquery_preparation_seconds')),'episode cost missing')
        budget=512 if e['arm']=='native' else int(e['arm'])
        for i,q in enumerate(e['queries'],1):
            require(q['query']==i and q['retained_visual_tokens']==budget and q['precise'] is False
                and finite(q['seconds'],strict_lower=True),'query budget/cost differs')
        require(sum(q['seconds'] for q in e['queries'])<=e['episode_seconds'],'query cost exceeds episode')
        calls+=n*(2 if e['arm']=='native' else 1)
        if e['arm']=='native':parity.extend((e['slot_id'],i) for i in range(1,n+1))
    require([(p['slot_id'],p['query']) for p in r['parity']]==parity,'shadow accounting differs')
    for p in r['parity']:
        require(p['command_bytes_equal'] is True and p['layers']==[32,32]
            and set(p['errors'])=={'hidden','normalized','raw'} and all(finite(v) and v<=1e-6 for v in p['errors'].values()),'native parity failed')
        for name in ('observation_sha256','command_sha256'):
            h=p[name];require(isinstance(h,str) and len(h)==64 and all(x in '0123456789abcdef' for x in h),'invalid trace hash')
    for slot,t in zip(c['timing_slots'],r['timing']):
        require(all(t[k]==v for k,v in slot.items()) and t['retained_visual_tokens']==int(t['arm'])
            and t['query']==t['frame']+1 and t['precise'] is False and finite(t['seconds'],strict_lower=True),'timing order/cost differs')
    require(type(s['model_queries']) is int and s['model_queries']==calls<=CAPS['model_calls'],'model calls differ')
    return True
