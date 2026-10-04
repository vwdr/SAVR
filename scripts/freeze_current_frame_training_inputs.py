#!/usr/bin/env python3
"""Freeze 3200 fit and 800 validation observations, no GPU or action-label access."""
import hashlib
import json
import os
from pathlib import Path
import sys
import run_openvla_original_baseline as base

ROOT = Path('/home/ved/SAVR')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from savr.openvla.current_frame_sampling import select_samples
from savr.openvla.offline_camera_inputs import observation_sha256
INDEX = 'data/pair/index/f13aa24a3da8c43c7225569f28c562979fa0e35a/trajectory_index.jsonl'
INDEX_SHA = '4d5c8829e3070db1ba96af1920a7bdb1e9581af295932fbc7d310cae15c40f8a'
DATA = 'data/pair/libero_hdf5/f13aa24a3da8c43c7225569f28c562979fa0e35a'


def read_observation(handle, row, frame):
    import numpy as np
    group = handle['data/' + row['original_trajectory_id'] + '/obs']
    for name, shape, dtype in (('agentview_rgb', (128, 128, 3), np.uint8),
        ('eye_in_hand_rgb', (128, 128, 3), np.uint8), ('ee_states', (6,), np.float64),
        ('gripper_states', (2,), np.float64)):
        if group[name].shape != (row['step_count'], *shape) or group[name].dtype != dtype:
            raise ValueError('stored camera/state schema changed: ' + name)
    if type(frame) is not int or frame < 0 or frame % 8 or frame + 7 >= row['step_count']:
        raise ValueError('invalid complete-chunk frame')
    obs = dict(full_image=np.asarray(group['agentview_rgb'][frame]).copy(),
               wrist_image=np.asarray(group['eye_in_hand_rgb'][frame]).copy(),
               state=np.concatenate((group['ee_states'][frame], group['gripper_states'][frame])).copy())
    if not np.isfinite(obs['state']).all():
        raise ValueError('nonfinite state')
    return obs


def main():
    if (Path.cwd().resolve() != ROOT or os.environ.get('CUDA_VISIBLE_DEVICES') != ''
            or not Path(sys.prefix).resolve().is_relative_to(ROOT/'envs/openvla-oft')):
        raise ValueError('project original runtime with GPU hidden required')
    output = ROOT/'configs/openvla/current_frame_training_inputs_v1.json'
    if output.exists(): raise ValueError('frozen manifest exists; no overwrite')
    if base.sha(ROOT/INDEX) != INDEX_SHA: raise ValueError('training index changed')
    trajectories = [json.loads(line) for line in (ROOT/INDEX).read_text().splitlines()]
    # Read index membership, not any calibration/locked trajectory observations or labels.
    selected = select_samples(trajectories)
    import h5py
    sources = {}; samples = []
    for source in sorted({r['identity']['source_path'] for r in selected}):
        subset = [r for r in selected if r['identity']['source_path'] == source]
        path = (ROOT/DATA/source).resolve()
        if not path.is_relative_to(ROOT/DATA): raise ValueError('source outside approved data root')
        digest = base.sha(path)
        if any(r['identity']['source_sha256'] != digest for r in subset): raise ValueError('source bytes changed')
        sources[source] = digest
        with h5py.File(path, 'r') as handle:
            for sample in subset:
                obs = read_observation(handle, sample['identity'], sample['frame'])
                samples.append(dict(sample, observation_sha256=observation_sha256(obs)))
    order = {r['sample_id']: i for i, r in enumerate(selected)}
    samples.sort(key=lambda r: order[r['sample_id']])
    result = dict(schema_version='current-frame-training-inputs-v1', budget=384, seed=7,
        training_index_path=INDEX, training_index_sha256=INDEX_SHA, data_root_relative=DATA,
        source_sha256=sources, samples=samples, count=4000, fit_count=3200, validation_count=800,
        sampling_source_sha256=base.sha(ROOT/'src/savr/openvla/current_frame_sampling.py'),
        freeze_script_sha256=base.sha(Path(__file__)),
        selection='28 lowest split-hash training trajectories fit; remaining 7 training trajectories validation; '
                  '80/20 smallest salted sample hashes per task, stride 8, complete chunks only',
        calibration_observations_opened=False, locked_observations_opened=False, action_fields_opened=False,
        model_calls=0, positive_method_result=False)
    base.write_once(output, result)
    print(json.dumps(dict(complete=True, samples=4000, fit=3200, validation=800,
                         manifest_sha256=base.sha(output), model_calls=0)))


if __name__ == '__main__': main()
