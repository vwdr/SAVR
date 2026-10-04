#!/usr/bin/env python3
"""Resolve only prospectively selected simulator state arrays; no simulator/GPU."""
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path('/home/ved/SAVR')


def state_hashes(slots):
    os.environ['LIBERO_CONFIG_PATH']=str(ROOT/'configs/pair/libero_runtime')
    from libero.libero import benchmark,get_libero_path
    if not Path(get_libero_path('init_states')).resolve().is_relative_to(ROOT):
        raise ValueError('initial states outside project')
    conditions={s['condition']['condition_id']:s['condition'] for s in slots}
    out={}
    for suite_name in sorted({r['suite'] for r in conditions.values()}):
        suite=benchmark.get_benchmark_dict()[suite_name]()
        for i in range(suite.n_tasks):
            task=suite.get_task(i)
            selected=[r for r in conditions.values() if r['suite']==suite_name and r['task_id']==task.name]
            if not selected:continue
            states=suite.get_task_init_states(i)
            for row in selected:
                if row['initial_state_id'] not in (0,1,2,3) or row['seed']!=7:
                    raise ValueError('unapproved state/seed')
                out[row['condition_id']]=hashlib.sha256(states[row['initial_state_id']].copy().tobytes()).hexdigest()
    if len(out)!=124:raise ValueError('expected 124 state identities')
    return out


if __name__=='__main__':
    if Path.cwd()!=ROOT or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU-only project preparation')
    sys.path.insert(0,str(ROOT/'src'))
    from savr.openvla.current_frame_robot_contract import episode_slots
    ledger=[json.loads(x) for x in (ROOT/'reports/cac_c0_recovery01/simulator_populations_v1.jsonl').read_text().splitlines()]
    ref=json.loads((ROOT/'configs/openvla/original_baseline_40task_v1.json').read_text())
    print(json.dumps(state_hashes(episode_slots(ledger,ref)),sort_keys=True))
