"""Outcome-blind task-balanced sampling from the existing training split."""
import hashlib


def select_samples(trajectories):
    groups = {}
    for row in trajectories:
        if row['split'] == 'train':
            groups.setdefault((row['suite'], row['task_id']), []).append(row)
    if len(groups) != 40 or len({r['trajectory_id'] for rows in groups.values() for r in rows}) != 1400:
        raise ValueError('exactly 40 tasks and 1400 unique training trajectories required')
    selected = []
    for key in sorted(groups):
        rows = sorted(groups[key], key=lambda r: (r['split_hash'], r['trajectory_id']))
        if len(rows) != 35:
            raise ValueError('35 training trajectories per task required')
        for role, pool, count in (('fit', rows[:28], 80), ('validation', rows[28:], 20)):
            candidates = []
            for row in pool:
                frames = range(0, row['step_count']-7, 8)
                if len(frames) != row['eligible_query_count']:
                    raise ValueError('eight-action alignment differs')
                for frame in frames:
                    sample_id = row['trajectory_id'] + ':' + str(frame)
                    rank = hashlib.sha256(('current-frame-pilot-v1|' + sample_id).encode()).hexdigest()
                    candidates.append(dict(identity=row, frame=frame, sample_id=sample_id,
                                           learning_role=role, selection_hash=rank))
            if len(candidates) < count:
                raise ValueError('insufficient eligible training-only observations')
            selected.extend(sorted(candidates, key=lambda r: (r['selection_hash'], r['sample_id']))[:count])
    fit = {r['identity']['trajectory_id'] for r in selected if r['learning_role'] == 'fit'}
    validation = {r['identity']['trajectory_id'] for r in selected if r['learning_role'] == 'validation'}
    if len(selected) != 4000 or fit & validation:
        raise ValueError('sample count or trajectory separation failed')
    return selected
