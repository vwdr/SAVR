#!/usr/bin/env python3
"""CPU-only four-arm screen analysis; evidence recording, no auto-selection."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
import run_adaptive_screen as worker
from savr.openvla.adaptive_screen_contract import ARMS, reconcile
from savr.openvla.contemporary_contract import SUITES, require
from analyze_contemporary_reference_v2 import stats

REGIMES = ('warm', 'steady')


def analyze(c, reference, s, r, *, draws=10000):
    reconcile(c, reference, s, r, c['observation_ids'])
    require(type(draws) is int and draws >= 100, 'too few descriptive draws')
    timing = [t for t in r['timing'] if not t['warmup']]
    pairs = []; controls = []; suites = []
    for suite in SUITES:
        episodes = [e for e in r['episodes'] if e['suite'] == suite]
        native = [e for e in episodes if e['arm'] == 'native']
        ids = list(dict.fromkeys(e['condition_id'] for e in episodes if e['arm'] != 'native'))
        local = []
        for identity in ids:
            entry = dict(condition_id=identity, suite=suite,
                         arms={e['arm']: e for e in episodes
                               if e['condition_id'] == identity and e['arm'] != 'native'})
            pairs.append(entry); local.append(entry)
        flag = len({local[0]['arms']['512']['success'], *(e['success'] for e in native)}) != 1
        traces = [[(p['observation_sha256'], p['command_sha256']) for p in r['parity']
                   if p['slot_id'] == e['slot_id']] for e in native]
        controls.append(dict(suite=suite, before=native[0], after=native[1], repeatability_limited=flag,
                             observation_command_traces_equal=traces[0] == traces[1]))
        suites.append(dict(suite=suite, conditions=10, repeatability_limited=flag,
                           successes={a: sum(e['arms'][a]['success'] for e in local) for a in ARMS}))
    successes = {a: sum(e['arms'][a]['success'] for e in pairs) for a in ARMS}
    baseline_concern = successes['512'] < 39 or any(x['repeatability_limited'] for x in controls)
    timing_stats = {}
    for a in ARMS:
        rows = [t for t in timing if t['arm'] == a]
        by = {'all': stats([t['seconds'] for t in rows])}
        for regime in REGIMES:
            sub = [t['seconds'] for t in rows if t['regime'] == regime]
            if sub:
                by[regime] = stats(sub)
        timing_stats[a] = by
    episode_costs = {a: stats([e['episode_seconds'] for e in r['episodes']
                              if e['arm'] == a]) for a in ARMS}
    means = {a: np.array([np.mean([t['seconds'] for t in timing
                                   if t['arm'] == a and t['observation_id'] == i])
                          for i in c['observation_ids']]) for a in ARMS}
    rng = np.random.default_rng(7)
    task_indices = rng.integers(0, 10, (4, draws, 10)); trace_indices = rng.integers(0, 8, (draws, 8))
    candidates = []
    for a in ('384', 'adaptive_75', 'adaptive_50'):
        diff = np.array([[int(p['arms'][a]['success']) - int(p['arms']['512']['success'])
                          for p in pairs if p['suite'] == suite] for suite in SUITES])
        sampled = sum(diff[i][task_indices[i]].sum(1) for i in range(4)) / 40
        reduction = float(1 - means[a].mean() / means['512'].mean())
        steady = [t['seconds'] for t in timing if t['arm'] == a and t['regime'] == 'steady']
        steady_rows = [t['seconds'] for t in timing if t['arm'] == '512']
        trace_sample = 1 - means[a][trace_indices].mean(1) / means['512'][trace_indices].mean(1)
        loss = successes['512'] - successes[a]
        eligible = not baseline_concern and loss <= 6 and reduction >= .10
        candidates.append(dict(arm=a, retained=dict(adaptive_75=384, adaptive_50=256).get(a, 384),
            successes=successes[a], success_count_loss=loss, success_difference=float(diff.mean()),
            dense_only_successes=int((diff == -1).sum()), compressed_only_successes=int((diff == 1).sum()),
            controlled_mean_query_reduction=reduction,
            descriptive_paired_success_interval_95=np.quantile(sampled, [.025, .975]).tolist(),
            descriptive_trace_reduction_interval_95=np.quantile(trace_sample, [.025, .975]).tolist(),
            steady_state_query_reduction=float(1 - np.mean(steady) / np.mean(steady_rows)) if steady else None,
            startup_queries_per_episode=0 if a == '384' else 3, success_triage_passed=loss <= 6,
            time_triage_passed=reduction >= .10, eligible=eligible))
    adaptive_zero = [p for p in r['parity'] if p['pair'] == 'native-vs-adaptive-zero']
    return dict(schema_version='adaptive-screen-analysis-v1', technical_reconciliation_passed=True,
        positive_method_result=False, training_performed=False,
        population='40 consumed development conditions; initial state 0; seed 7',
        successes=successes, baseline_concern=baseline_concern, suites=suites,
        native_controls=controls, pairs=pairs, timing_seconds=timing_stats, episode_costs=episode_costs,
        candidates=candidates, selected_arm=None,
        adaptive_zero=dict(maximum_error=max(v for p in adaptive_zero for v in p['errors'].values()),
            command_equal_queries=sum(p['command_bytes_equal'] for p in adaptive_zero),
            observed_queries=len(adaptive_zero)),
        decision='evidence_recorded_no_automatic_selection',
        bootstrap=dict(draws=draws, seed=7, success_unit='task, stratified by suite',
                       timing_unit='paired trajectory cluster'),
        limitations=['All-query timing includes the adaptive arms\' required 3-query dense startup per episode.',
            'Success difference intervals are descriptive, not confirmatory power statements.',
            'Independent native rollout variation remains reported even with matching successes.',
            'Controlled timing is a short eight-trajectory corpus, not general deployment acceleration.',
            'The adaptive-zero bound is measured at the released checkpoint before any outcome is scored.'])


def load(run):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CUDA must be hidden')
    run = run.resolve()
    require(run.is_relative_to(worker.ROOT / 'results') and (run / 'worker_summary.json').is_file()
            and not (run / 'technical_stop.json').exists(), 'completed immutable summary required')
    s = json.loads((run / 'worker_summary.json').read_text())
    require(s['complete'] is True, 'summary incomplete')
    require(sum(1 for _ in (run / 'episodes.jsonl').open()) == 168
            and sum(1 for _ in (run / 'timing.jsonl').open()) == 240, 'unblind counts differ')
    for name, expected in s['artifacts_sha256'].items():
        p = (run / name).resolve()
        require(p.is_relative_to(run) and worker.base.sha(p) == expected, 'artifact hash differs')
    launch = json.loads((run / 'launch.json').read_text()); c = launch['config']
    require(worker.base.sha(worker.ROOT / 'configs/openvla/adaptive_screen_v3.json') == launch['config_sha256'],
            'launch config differs')
    _, ref, _, _ = worker.verify(c)
    r = json.loads((run / 'records.json').read_text())
    for n in ('episodes', 'timing', 'parity'):
        require(r[n] == [json.loads(x) for x in (run / (n + '.jsonl')).read_text().splitlines()],
                'bundle and stream differ')
    return c, ref, s, r


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    result = analyze(*load(args.run))
    worker.base.write_once(args.run / 'analysis.json', result)
    print(json.dumps({k: result[k] for k in ('technical_reconciliation_passed', 'successes', 'baseline_concern',
                                             'selected_arm', 'positive_method_result', 'decision')}))
