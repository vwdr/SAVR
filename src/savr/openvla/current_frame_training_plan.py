"""Outcome-independent schedules and acceptance logic for one bounded fit."""
import hashlib
from collections import Counter

ARMS = ('action_only', 'visual', 'visual_shuffled')
OPTIMIZER = dict(name='AdamW', lr=0.0001, betas=[0.9, 0.999], eps=1e-8,
                 weight_decay=0.0, batch_size=16, epochs=10, steps_per_arm=2000,
                 seed=7, gradient_clip_norm=1.0, scheduler='none', checkpoint='final_only')
CAPS = dict(seconds=28800, memory_mib=23552, own_allocated_mib=4096,
            artifact_bytes=2*1024**3, minimum_free_bytes=10*1024**3,
            optimizer_steps=6000, model_calls=0, episodes=0)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def rank(text):
    return hashlib.sha256(text.encode()).hexdigest()


def schedule(samples, *, epochs=10, batch=16, seed=7):
    require(len(samples) == 4000, 'exactly 4000 frozen samples required')
    require(len({r['sample_id'] for r in samples}) == 4000, 'duplicate sample')
    require(Counter(r['learning_role'] for r in samples) == {'fit':3200, 'validation':800}, 'roles')
    groups, trajectories = {}, {'fit':set(), 'validation':set()}
    for i, r in enumerate(samples):
        ident = r['identity']; role = r['learning_role']
        require(ident['split'] == 'train', 'non-training source')
        trajectories[role].add(ident['trajectory_id'])
        groups.setdefault((ident['suite'], ident['task_id'], role), []).append(i)
    require(not trajectories['fit'] & trajectories['validation'], 'trajectory overlap')
    require(len(groups) == 80 and all(len(v)==(80 if k[2]=='fit' else 20)
            for k,v in groups.items()), 'task balance')
    fit = [i for i,r in enumerate(samples) if r['learning_role']=='fit']
    validation = [i for i,r in enumerate(samples) if r['learning_role']=='validation']
    require(len(fit)%batch==0 and epochs>0, 'batch/epoch alignment')
    orders = [sorted(fit,key=lambda i:(rank(f'current-frame-fit-v1|{seed}|{e}|'+samples[i]['sample_id']),i))
              for e in range(epochs)]
    # Fixed within-task cyclic derangement of entire teacher chunks. No validation target enters fitting.
    shuffled = {}
    for (suite,task,role), ids in sorted(groups.items()):
        if role!='fit': continue
        ids = sorted(ids,key=lambda i:(rank('current-frame-label-control-v1|'+samples[i]['sample_id']),i))
        shuffled.update({str(i):ids[(j+1)%len(ids)] for j,i in enumerate(ids)})
    require(all(int(i)!=j and j in fit for i,j in shuffled.items()), 'invalid shuffled target')
    return dict(fit=fit, validation=validation, epoch_orders=orders, shuffled_targets=shuffled,
                batch_size=batch, seed=seed)


def authorized_indices(plan, indices, role):
    require(role in ('fit','validation'), 'unknown role')
    require(bool(indices) and set(indices)<=set(plan[role]), 'unauthorized observation role')


def offline_decision(metrics):
    """Engineering triage, not a success claim or statistically calibrated test."""
    dense_gap = metrics['base_l1']
    if dense_gap <= 0:
        return dict(eligible_for_development_evaluation=False, reason='zero_teacher_discrepancy')
    visual = metrics['visual_l1']
    okay = visual < dense_gap and visual < metrics['visual_shuffled_l1']
    return dict(eligible_for_development_evaluation=okay,
                reason='offline_triage_pass' if okay else 'offline_triage_fail',
                visual_relative_l1_reduction=1-visual/dense_gap,
                visual_better_than_action_only=visual < metrics['action_only_l1'],
                positive_method_result=False)
