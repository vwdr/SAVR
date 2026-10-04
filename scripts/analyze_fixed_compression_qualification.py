#!/usr/bin/env python3
"""CPU-only verification of a completed fixed-compression qualification."""
import argparse
import json
import os
from pathlib import Path
import run_fixed_compression_qualification as worker


def analyze(run):
    root=worker.ROOT;run=run.resolve()
    worker.require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','hide CUDA for analysis')
    worker.require(run.is_relative_to(root/'results') and (run/'worker_summary.json').exists()
                   and not (run/'technical_stop.json').exists(),'completed immutable summary required')
    s=json.loads((run/'worker_summary.json').read_text());worker.require(s['complete'] is True,'incomplete summary')
    for n,h in s['artifacts_sha256'].items():
        p=(run/n).resolve();worker.require(p.is_relative_to(run) and worker.base.sha(p)==h,'artifact hash differs')
    launch=json.loads((run/'launch.json').read_text());c=launch['config'];worker.verify(c)
    worker.require(worker.base.sha(root/'configs/openvla/fixed_compression_qualification_v1.json')==launch['config_sha256'],
                   'launch config identity differs')
    rows=json.loads((run/'records.json').read_text())
    worker.require(rows==[json.loads(x) for x in (run/'checks.jsonl').read_text().splitlines()],'record bundle differs')
    worker.reconcile(c,s,rows)
    return dict(complete=True,technical_qualification_passed=True,model_calls=112,observations=16,
        hook_neutrality_checks=48,dense_parity_checks=16,simulator_episodes=0,positive_method_result=False,
        maximum_dense_error=max(v for r in rows for v in r['dense_errors'].values()),
        worker_summary_sha256=worker.base.sha(run/'worker_summary.json'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);args=p.parse_args()
    result=analyze(args.run);worker.base.write_once(args.run/'analysis.json',result);print(json.dumps(result))
