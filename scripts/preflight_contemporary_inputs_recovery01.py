#!/usr/bin/env python3
"""Actual stored-frame preprocessing on CPU only, before any model loading."""
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import run_openvla_original_baseline as base
import run_specprune_qualification as prior

ROOT = Path('/home/ved/SAVR')


def main():
    if Path.cwd().resolve() != ROOT or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise SystemExit('project root and explicitly hidden GPU required')
    if not Path(sys.prefix).resolve().is_relative_to(ROOT / 'envs/openvla-oft'):
        raise SystemExit('original project environment required')
    output = ROOT / 'reports/CONTEMPORARY_RECOVERY01_REAL_INPUT_PREFLIGHT_V1.json'
    if output.exists():
        raise SystemExit('preflight evidence exists; no overwrite')
    base.environment(ROOT / 'reports/contemporary-recovery01-cpu-runtime')
    _, manifest = prior.verify(json.loads((ROOT / 'configs/openvla/specprune_real_qualification_v1.json').read_text()))
    sys.path.insert(0, str(ROOT / 'src'))
    from savr.openvla.offline_camera_inputs import load_offline_inputs, audit_policy_preprocessing
    observations, audit = load_offline_inputs(ROOT, manifest)
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    import torch
    torch.set_num_threads(1)
    if torch.cuda.is_available():
        raise RuntimeError('GPU must remain hidden for input preflight')
    sys.path.insert(0, str(ROOT / 'third_party/openvla-oft'))
    from experiments.robot.openvla_utils import prepare_images_for_vla
    checks = audit_policy_preprocessing(observations, prepare_images_for_vla, SimpleNamespace(center_crop=True))
    if len(checks) != 16:
        raise RuntimeError('incomplete two-frame/eight-trajectory preprocessing checks')
    result = dict(complete=True, gpu_visible=False, model_queries=0, input_audit=audit, checks=checks,
                  offline_input_module_sha256=base.sha(ROOT / 'src/savr/openvla/offline_camera_inputs.py'),
                  preflight_script_sha256=base.sha(Path(__file__)),
                  published_preprocessing_source_sha256=base.sha(ROOT / 'third_party/openvla-oft/experiments/robot/openvla_utils.py'),
                  observation_manifest_sha256=base.sha(ROOT / 'configs/pair/p3_inputs_v1.json'))
    base.write_once(output, result)
    print(json.dumps(dict(complete=True, input_frames=len(audit), preprocessing_checks=len(checks), model_queries=0)))


if __name__ == '__main__': main()
