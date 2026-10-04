#!/usr/bin/env python3
"""CPU-only verification of a completed adaptive (stage S0) qualification.

Independently recomputes the adaptive-zero tolerance from the recorded
zero-pruning parity rows (eager 4.40.1 manual attention vs original SDPA at
dense inputs only) and requires it to equal the worker's summary value.
Pruning-induced steady differences are reported separately as
``pruning_shift_observed_max`` and are never permitted to calibrate or gate
the numerical tolerance (S0 separation rule).
"""
import argparse
import json
import os
from pathlib import Path
import run_adaptive_qualification as worker


def analyze(run):
    root = worker.ROOT
    run = run.resolve()
    worker.require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'hide CUDA for analysis')
    worker.require(run.is_relative_to(root / 'results') and (run / 'worker_summary.json').exists()
                   and not (run / 'technical_stop.json').exists(), 'completed immutable summary required')
    s = json.loads((run / 'worker_summary.json').read_text())
    worker.require(s['complete'] is True, 'incomplete summary')
    for name, expected in s['artifacts_sha256'].items():
        p = (run / name).resolve()
        worker.require(p.is_relative_to(run) and worker.base.sha(p) == expected, 'artifact hash differs')
    launch = json.loads((run / 'launch.json').read_text())
    c = launch['config']
    worker.require(worker.base.sha(root / 'configs/openvla/adaptive_qualification_v3.json')
                   == launch['config_sha256'], 'launch config identity differs')
    rows = json.loads((run / 'records.json').read_text())
    worker.require(rows == [json.loads(x) for x in (run / 'checks.jsonl').read_text().splitlines()],
                   'record bundle differs')
    worker.verify(c)
    worker.reconcile(c, s, rows)
    # Independent recomputation: tolerance derives from zero-pruning parity
    # rows only; pruning-induced steady differences never enter it.
    zero_parity_observed = max([e for r in rows for e in r['zero_errors'].values()])
    pruning_shift_observed = max([e for r in rows for m in r['modes']
                                  for e in m['steady_errors'].values()])
    recomputed_tolerance = max(2.0 * zero_parity_observed, 1e-6)
    worker.require(recomputed_tolerance == s['adaptive_zero_tolerance'],
                   'tolerance does not independently recompute from zero-pruning parity')
    return dict(complete=True, technical_qualification_passed=True, model_calls=worker.MODEL_CALLS,
        observations=16, modes=2, simulator_episodes=0, positive_method_result=False,
        adaptive_zero_tolerance=recomputed_tolerance,
        adaptive_zero_tolerance_max=s['adaptive_zero_tolerance_max'],
        zero_parity_observed_max=zero_parity_observed,
        pruning_shift_observed_max=pruning_shift_observed,
        worker_summary_sha256=worker.base.sha(run / 'worker_summary.json'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    result = analyze(parser.parse_args().run)
    print(json.dumps(result, allow_nan=False))
