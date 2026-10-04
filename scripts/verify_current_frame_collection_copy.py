#!/usr/bin/env python3
"""Read-only independent integrity/role check of the completed local collection."""
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'results/current-frame-feature-collection-v01'
SUMMARY_SHA = '7bde8cd59846d5f5b54c7b9a8302b30ebe74ef13606eb92aeb754465d5b1a8c3'
INPUT_SHA = '2fd13ced3e5a9a2b810eb299a5c1667f8c1de3ee2c5487a4a0683e61847d9ae8'
CONFIG_SHA = 'e810cb53f7a836d7a4d9dbb95ee406e919355f70a215e6788d5b760b7863a124'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    require(sha(RUN / 'worker_summary.json') == SUMMARY_SHA, 'summary hash')
    inputs = ROOT / 'configs/openvla/current_frame_training_inputs_v1.json'
    require(sha(inputs) == INPUT_SHA, 'input manifest hash')
    require(sha(ROOT / 'configs/openvla/current_frame_feature_collection_v1.json') == CONFIG_SHA,
            'configuration hash')
    s = json.loads((RUN / 'worker_summary.json').read_text())
    a = json.loads((RUN / 'analysis.json').read_text())
    samples = json.loads(inputs.read_text())['samples']
    rows = json.loads((RUN / 'records.json').read_text())
    approved = json.loads((RUN / 'approved_samples.json').read_text())
    require(not (RUN / 'technical_stop.json').exists(), 'technical stop present')
    require(s['complete'] and a['complete'] and a['collection_passed'], 'completion')
    require(a['summary_sha256'] == SUMMARY_SHA, 'analysis ancestry')
    for key, value in [('records', 4000), ('model_queries', 8000), ('optimizer_steps', 0),
                       ('episodes', 0), ('positive_method_result', False)]:
        require(s[key] == a[key] == value, key)
    require(a['fit'] == 3200 and a['validation'] == 800, 'analysis roles')
    require(s['authenticated_files_unchanged'] and not s['automatic_retry'], 'worker contract')
    require(0 < s['elapsed_seconds'] < 28800, 'elapsed cap')
    require(0 < s['peak_aggregate_gpu_memory_mib'] < 23552, 'memory cap')
    require(0 < s['artifact_bytes_before_summary'] < 20 * 1024**3, 'artifact cap')
    require(len(samples) == len(rows) == len(approved) == 4000, 'counts')
    require(len({r['sample_id'] for r in rows}) == 4000, 'unique samples')
    roles, tasks = Counter(), Counter()
    trajectories = {'fit': set(), 'validation': set()}
    for i, (row, sample) in enumerate(zip(rows, samples)):
        require(row['sample_id'] == sample['sample_id'], 'sample order')
        role = sample['learning_role']
        require(row['learning_role'] == role, 'role assignment')
        require(row['model_queries'] == 2 * (i + 1), 'call accounting')
        require(row['artifact'] == f'features/{i:04d}.npz', 'file order')
        require(all(row[k] is True for k in ('input_unchanged', 'record_roundtrip_equal',
                    'query_counts_equal_one', 'no_backbone_gradients')), 'record contract')
        meta = approved[row['sample_id']]
        require(meta['teacher_observation_sha256'] == meta['student_observation_sha256'] ==
                sample['observation_sha256'], 'observation identity')
        identity = sample['identity']
        require(meta['trajectory_id'] == identity['trajectory_id'] and
                meta['frame'] == sample['frame'] and meta['split'] == identity['split'] == 'train',
                'source identity')
        require(meta['compression_budget'] == 384, 'compression budget')
        require(row['sha256'] == s['artifacts_sha256'][row['artifact']], 'record digest')
        roles[role] += 1
        tasks[(identity['suite'], identity['task_id'], role)] += 1
        trajectories[role].add(identity['trajectory_id'])
    require(roles == {'fit': 3200, 'validation': 800}, 'role counts')
    require(len(tasks) == 80 and all(n == (80 if role == 'fit' else 20)
                for (_, _, role), n in tasks.items()), 'task balance')
    require(not trajectories['fit'] & trajectories['validation'], 'trajectory leakage')
    require(len(s['artifacts_sha256']) == 4005, 'artifact manifest count')
    for name, digest in s['artifacts_sha256'].items():
        path = (RUN / name).resolve()
        require(path.is_relative_to(RUN.resolve()) and sha(path) == digest, 'artifact hash: ' + name)
    print(json.dumps(dict(copy_verified=True, verified_root=str(RUN), hashed_artifacts=4005, records=4000,
        fit=3200, validation=800, tasks=40, fit_trajectories=len(trajectories['fit']),
        validation_trajectories=len(trajectories['validation']), model_queries=8000,
        optimizer_steps=0, episodes=0, positive_method_result=False,
        summary_sha256=SUMMARY_SHA, analysis_sha256=sha(RUN / 'analysis.json'))))


if __name__ == '__main__':
    main()
