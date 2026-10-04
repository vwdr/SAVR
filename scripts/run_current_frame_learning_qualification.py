#!/usr/bin/env python3
"""Integrated real-feature/optimizer/reload qualification; no scientific rollout."""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
import traceback
import run_openvla_original_baseline as base
import run_fixed_compression_qualification as prior

ROOT = Path('/home/ved/SAVR')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
CAPS = dict(model_calls=80, optimizer_steps=128, episodes=0, seconds=1800,
            aggregate_memory_mib=23552, artifact_bytes=536870912)
FILES = ('scripts/run_current_frame_learning_qualification.py',
         'src/savr/openvla/current_frame_features.py', 'src/savr/openvla/current_frame_corrector.py',
         'src/savr/openvla/current_frame_records.py', 'tests/openvla/test_current_frame_features.py',
         'tests/openvla/test_learning_qualification.py', 'docs/CURRENT_FRAME_LEARNING_PILOT_V1.md')
ANCESTOR = 'configs/openvla/fixed_compression_qualification_v1.json'
ANCESTOR_SHA = 'd4cb82d508b16a923270d4e979eced5f243b4a902d851b71772bc9702aa862b4'
require, scoped = prior.require, prior.scoped


def verify(c):
    require(c['schema_version'] == 'current-frame-learning-qualification-v1'
            and c['caps'] == CAPS and c['budget'] == 384 and c['automatic_retry'] is False
            and c['optimizer'] == dict(name='AdamW', steps_per_arm=64, lr=0.0001, weight_decay=0, batch=2, seed=7)
            and c['output_root'] == 'results/current-frame-learning-qualification-v01', 'contract differs')
    require(set(c['authenticated_files']) == set(FILES), 'source manifest incomplete')
    for name, digest in c['authenticated_files'].items():
        require(base.sha(scoped(name)) == digest, 'source changed: ' + name)
    require(base.sha(scoped(ANCESTOR)) == ANCESTOR_SHA, 'ancestor changed')
    ancestor = json.loads(scoped(ANCESTOR).read_text())
    old, reference, manifest = prior.verify(ancestor)
    require(c['gpu'] == ancestor['gpu'], 'GPU identity differs')
    for run, digest in (
        ('fixed-compression-qualification-v01', 'ad197c96e75d8c6c753d4c786c924aa49415d5e5632873ff81fcbe9a197ca5a9'),
        ('fixed-compression-screen-v01', '56bfa17c0d7779275f35ad651a2bebd4e3b72ca9b40be3c255c616f92af977a3')):
        p = scoped('results/' + run)
        require(base.sha(p/'worker_summary.json') == digest and not (p/'technical_stop.json').exists(),
                'completed predecessor changed')
        s = json.loads((p/'worker_summary.json').read_text())
        require(s['complete'] is True, 'incomplete predecessor')
        for name, h in s['artifacts_sha256'].items():
            path = (p/name).resolve()
            require(path.is_relative_to(p) and base.sha(path) == h, 'predecessor artifact changed')
    return old, reference, manifest


def approved_samples(c, observations):
    from savr.openvla.offline_camera_inputs import observation_sha256
    result = {}
    for identity, frames in observations.values():
        for frame, obs in enumerate(frames):
            sample = identity['trajectory_id'] + ':' + str(frame)
            result[sample] = dict(schema_version='current-frame-feature-v1', sample_id=sample,
                trajectory_id=identity['trajectory_id'], source_sha256=identity['source_sha256'],
                frame=frame, split='train', split_hash=identity['split_hash'],
                student_observation_sha256=observation_sha256(obs), teacher_observation_sha256=observation_sha256(obs),
                compression_budget=384, selector_sha256=base.sha(scoped('src/savr/openvla/spatial_selection.py')),
                backbone_sha256=ANCESTOR_SHA,
                extractor_sha256=c['authenticated_files']['src/savr/openvla/current_frame_features.py'],
                normalization_statistics_key=identity['normalization_statistics_key'],
                instruction_pooling='fp32_mean_current_instruction_input_embeddings',
                feature_origin='hard_compacted_current_frame_decoder')
    return result


def reconcile(c, s, records, fits):
    require(s.get('complete') is True and s.get('model_queries') == 80 and s.get('optimizer_steps') == 128
            and s.get('episodes') == 0 and s.get('positive_method_result') is False
            and s.get('automatic_retry') is False and s.get('authenticated_files_unchanged') is True,
            'completion accounting differs')
    require(len(records) == 16 and [(r['sample_id'], r['frame']) for r in records] ==
            [(i + ':' + str(f), f) for i in c['observation_ids'] for f in (0, 1)], 'frame order/count differs')
    for r in records:
        require(all(r.get(k) is True for k in ('input_unchanged', 'feature_commands_equal',
            'zero_action_only_equal', 'zero_visual_equal', 'record_roundtrip_equal', 'no_backbone_gradients')),
            'feature or zero-init parity failed')
    require([f['use_visual'] for f in fits] == [False, True], 'fit arms differ')
    for f in fits:
        require(f['steps'] == 64 and f['reload_equal'] is True and f['no_backbone_gradients'] is True
                and f['nonzero_output_gradient'] is True and f['all_updates_finite'] is True
                and type(f['initial_l1']) in (int, float) and math.isfinite(f['initial_l1'])
                and type(f['final_l1']) in (int, float) and math.isfinite(f['final_l1'])
                and 0 <= f['final_l1'] < f['initial_l1'], 'optimizer qualification failed')
    for key, cap in (('elapsed_seconds', 1800), ('peak_aggregate_gpu_memory_mib', 23552),
                     ('artifact_bytes_before_summary', CAPS['artifact_bytes'])):
        require(type(s[key]) in (int, float) and math.isfinite(s[key]) and 0 <= s[key] < cap,
                'resource bound failed')
    return True


def run(c, reference, observations, p, resources, approved):
    import numpy as np
    import torch
    import transformers
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(1); tf.config.threading.set_inter_op_parallelism_threads(1)
    require(torch.__version__ == '2.2.0+cu118' and transformers.__version__ == '4.40.1'
            and torch.cuda.device_count() == 1, 'original one-GPU runtime required')
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(23000/24576, 0)
    sys.path.insert(0, str(ROOT/'third_party/openvla-oft'))
    from experiments.robot import openvla_utils as utils
    from experiments.robot.libero import run_libero_eval as evaluation
    from experiments.robot.robot_utils import set_seed_everywhere
    from savr.cac.c1 import instruction_token_indices
    from savr.openvla.fixed_compression import FixedCompressionQuery
    from savr.openvla.current_frame_features import CurrentFrameFeatureQuery, CorrectedCurrentFrameQuery
    from savr.openvla.current_frame_corrector import CurrentFrameCorrector, teacher_imitation_loss
    from savr.openvla.current_frame_records import tensors_to_arrays, arrays_to_tensors, encode_record, decode_record
    from savr.openvla.official_semantics import OfficialActionHeadCapture
    from savr.openvla.specprune import EpisodeState
    from savr.openvla.offline_camera_inputs import OfflineCameraPair, observation_sha256
    from savr.openvla.execution_precision import processed_command_chunk
    checkpoint = ROOT/reference['checkpoint']
    inventory = {x.name for x in checkpoint.iterdir()}
    cfg = evaluation.GenerateConfig(pretrained_checkpoint=str(checkpoint), task_suite_name='libero_spatial',
        num_trials_per_task=0, seed=7, local_log_dir=str(p/'logs'), use_wandb=False, center_crop=True,
        num_open_loop_steps=8, num_images_in_input=2, use_proprio=True, use_l1_regression=True,
        use_diffusion=False, use_film=False)
    evaluation.validate_config(cfg); set_seed_everywhere(7)
    update, sync = utils.update_auto_map, utils.check_model_logic_mismatch
    utils.update_auto_map = lambda _: None; utils.check_model_logic_mismatch = lambda _: None
    try: model, head, proprio, noisy, processor = evaluation.initialize_model(cfg)
    finally: utils.update_auto_map, utils.check_model_logic_mismatch = update, sync
    require(head is not None and proprio is not None and noisy is None and
            not any(m.training for m in (model, head, proprio)), 'released components differ')
    for m in (model, head, proprio): m.requires_grad_(False)
    def no_backbone_gradients():
        return all(not x.requires_grad and x.grad is None for m in (model, head, proprio) for x in m.parameters())
    shared = dict(model=model, head=head, proprio=proprio, processor=processor, cfg=cfg, utils=utils,
                  instruction_indexer=instruction_token_indices)
    dense = FixedCompressionQuery(**shared, budget=512)
    plain = FixedCompressionQuery(**shared, budget=384)
    extracted = CurrentFrameFeatureQuery(**shared, budget=384)
    zero = {v: CurrentFrameCorrector(use_visual=v).cuda().eval() for v in (False, True)}
    corrected = {v: CorrectedCurrentFrameQuery(extracted, zero[v]) for v in zero}
    base.write_once(p/'loaded_runtime.json', dict(torch=torch.__version__, transformers=transformers.__version__,
        configuration=vars(cfg), backbone_frozen=True, feature_precision='BF16 with FP32 normalized actions/state'))
    def call(query, obs, text, previous):
        resources.consume(); state = EpisodeState(); state.reset('learning-qualification')
        raw = query(obs, text, previous, state)
        require(state.query == 1, 'query count differs')
        return processed_command_chunk(raw, evaluation.process_action, cfg.model_family)
    rows = []; training = []
    (p/'features').mkdir()
    for identity, frames in observations.values():
        cfg.unnorm_key = identity['normalization_statistics_key']
        for frame, obs in enumerate(frames):
            sample = identity['trajectory_id'] + ':' + str(frame)
            text = identity['language_instruction']; previous = OfflineCameraPair.capture(frames[0], 0)
            before = observation_sha256(obs)
            commands = call(plain, obs, text, previous)
            with torch.inference_mode(), OfficialActionHeadCapture(head) as capture:
                call(dense, obs, text, previous)
                teacher = capture.exact_output().reshape(8, 7).float().detach().clone()
            extracted_commands = call(extracted, obs, text, previous)
            tensors = dict(extracted.features, teacher_actions=teacher)
            arrays = tensors_to_arrays(tensors)
            payload, digest = encode_record(approved[sample], arrays, approved)
            name = 'features/' + identity['trajectory_id'] + '-' + str(frame) + '.npz'
            with (p/name).open('xb') as f: f.write(payload)
            meta, reread = decode_record((p/name).read_bytes(), digest, approved)
            equal = meta == approved[sample] and all(np.array_equal(arrays[k], reread[k]) for k in arrays)
            if len(training) < 2: training.append(arrays_to_tensors(reread))
            zero_equal = {v: commands.tobytes() == call(corrected[v], obs, text, previous).tobytes() for v in zero}
            row = dict(sample_id=sample, frame=frame, artifact=name, sha256=digest,
                input_unchanged=before == observation_sha256(obs),
                feature_commands_equal=commands.tobytes() == extracted_commands.tobytes(),
                zero_action_only_equal=zero_equal[False], zero_visual_equal=zero_equal[True],
                record_roundtrip_equal=equal, no_backbone_gradients=no_backbone_gradients())
            require(all(row[k] is True for k in ('input_unchanged', 'feature_commands_equal',
                'zero_action_only_equal', 'zero_visual_equal', 'record_roundtrip_equal', 'no_backbone_gradients')),
                'actual feature parity failed')
            rows.append(row)
            with (p/'progress.jsonl').open('a') as f:
                f.write(json.dumps(dict(feature_records=len(rows), model_queries=resources.queries))+'\n')
    # Convert inference tensors outside inference mode; no large-model backward graph.
    batch = {k: torch.stack([r[k] for r in training]).cuda() for k in training[0]}
    target = batch.pop('teacher_actions')
    fits = []
    for visual in (False, True):
        torch.manual_seed(7)
        adapter = CurrentFrameCorrector(use_visual=visual).cuda().train()
        optimizer = torch.optim.AdamW(adapter.parameters(), lr=0.0001, weight_decay=0)
        with torch.no_grad(): initial = float(teacher_imitation_loss(adapter(**batch), target))
        require(initial > 0, 'zero teacher discrepancy: uninformative fit diagnostic')
        any_gradient = False
        started = time.monotonic()
        for step in range(64):
            if resources.error: raise RuntimeError(resources.error)
            optimizer.zero_grad(set_to_none=True)
            loss = teacher_imitation_loss(adapter(**batch), target)
            loss.backward()
            require(all(x.grad is None or bool(torch.isfinite(x.grad).all()) for x in adapter.parameters()),
                    'nonfinite optimizer gradient')
            any_gradient |= bool(adapter.output.weight.grad.abs().sum() > 0)
            optimizer.step()
            require(all(bool(torch.isfinite(x).all()) for x in adapter.parameters()), 'nonfinite optimizer weight')
        adapter.eval()
        with torch.no_grad():
            prediction = adapter(**batch); final = float(teacher_imitation_loss(prediction, target))
        name = 'diagnostic-visual.pt' if visual else 'diagnostic-action-only.pt'
        with (p/name).open('xb') as f: torch.save(adapter.state_dict(), f)
        loaded = CurrentFrameCorrector(use_visual=visual).cuda().eval()
        loaded.load_state_dict(torch.load(p/name, map_location='cuda:0', weights_only=True), strict=True)
        with torch.no_grad(): reload_equal = torch.equal(prediction, loaded(**batch))
        fits.append(dict(use_visual=visual, steps=64, initial_l1=initial, final_l1=final,
            reload_equal=reload_equal, no_backbone_gradients=no_backbone_gradients(),
            nonzero_output_gradient=any_gradient, all_updates_finite=True,
            parameters=sum(x.numel() for x in adapter.parameters()), artifact=name,
            elapsed_seconds=time.monotonic()-started, purpose='engineering_only_not_for_evaluation'))
        del optimizer, adapter, loaded, prediction
        with (p/'progress.jsonl').open('a') as f:
            f.write(json.dumps(dict(feature_records=16, model_queries=resources.queries, fitted_arms=len(fits)))+'\n')
    require(inventory == {x.name for x in checkpoint.iterdir()}, 'checkpoint inventory changed')
    verify(c); resources.close()
    base.write_once(p/'records.json', rows); base.write_once(p/'fit_checks.json', fits)
    return rows, fits


def analyze(p):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only analysis required')
    require(not (p/'technical_stop.json').exists(), 'stopped run cannot pass')
    s = json.loads((p/'worker_summary.json').read_text())
    launch = json.loads((p/'launch.json').read_text()); c = launch['config']
    verify(c)
    require(base.sha(scoped('configs/openvla/current_frame_learning_qualification_v1.json')) == launch['config_sha256'],
            'launch config changed')
    for name, digest in s['artifacts_sha256'].items():
        path = (p/name).resolve()
        require(path.is_relative_to(p) and base.sha(path) == digest, 'artifact changed: '+name)
    rows = json.loads((p/'records.json').read_text()); fits = json.loads((p/'fit_checks.json').read_text())
    reconcile(c, s, rows, fits)
    from savr.openvla.current_frame_records import decode_record
    approved = json.loads((p/'approved_samples.json').read_text())
    for row in rows: decode_record((p/row['artifact']).read_bytes(), row['sha256'], approved)
    result = dict(complete=True, qualification_passed=True, model_queries=80, feature_records=16,
                  optimizer_steps=128, episodes=0, positive_method_result=False, fits=fits,
                  summary_sha256=base.sha(p/'worker_summary.json'))
    base.write_once(p/'analysis.json', result)
    print(json.dumps(result))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--config', type=Path)
    parser.add_argument('--preflight-only', action='store_true'); parser.add_argument('--analyze', type=Path)
    args = parser.parse_args()
    require(Path.cwd().resolve() == ROOT and Path(sys.prefix).resolve().is_relative_to(ROOT/'envs/openvla-oft'),
            'project runtime required')
    if args.analyze:
        p = args.analyze.resolve(); require(p == scoped('results/current-frame-learning-qualification-v01'), 'run differs')
        analyze(p); return
    require(args.config and args.config.resolve().is_relative_to(ROOT/'configs/openvla'), 'project config required')
    c = json.loads(args.config.read_text()); old, reference, manifest = verify(c)
    from savr.openvla.offline_camera_inputs import load_offline_inputs
    observations, audit = load_offline_inputs(ROOT, manifest)
    require(list(observations) == c['observation_ids'] and
            audit == json.loads(scoped(old['input_preflight_report']).read_text())['input_audit'], 'input audit differs')
    approved = approved_samples(c, observations)
    if args.preflight_only:
        print(json.dumps(dict(preflight_passed=True, approved_frames=len(approved), model_calls=0))); return
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == str(c['gpu']['index']), 'GPU selection differs')
    initial = base.snapshot(c['gpu']['index'])
    require(initial['uuid'] == c['gpu']['uuid'] and initial['memory_mib'] <= 1024
            and initial['utilization'] <= 5, 'selected GPU not idle')
    p = scoped(c['output_root']); p.mkdir(exist_ok=False); base.environment(p)
    base.write_once(p/'launch.json', dict(config=c, config_sha256=base.sha(args.config), pid=os.getpid(),
        started_at=datetime.now(timezone.utc).isoformat(), initial_gpu=initial))
    base.write_once(p/'approved_samples.json', approved)
    base.write_once(p/'input_preflight.json', dict(complete=True, input_audit=audit))
    resources = base.Resources(dict(gpu=c['gpu'], caps=dict(queries=80, memory_mib=23552)))
    resources.peak = initial['memory_mib']; resources.thread.start(); started = time.monotonic()
    def timeout(*_): raise RuntimeError('frozen time cap reached')
    signal.signal(signal.SIGALRM, timeout); signal.alarm(1800)
    try:
        rows, fits = run(c, reference, observations, p, resources, approved)
        artifacts = {str(x.relative_to(p)): base.sha(x) for x in p.rglob('*')
                     if x.is_file() and 'runtime-cache' not in x.parts}
        s = dict(complete=True, model_queries=resources.queries, optimizer_steps=128, episodes=0,
            positive_method_result=False, automatic_retry=False, authenticated_files_unchanged=True,
            elapsed_seconds=time.monotonic()-started, peak_aggregate_gpu_memory_mib=resources.peak,
            artifact_bytes_before_summary=sum(x.stat().st_size for x in p.rglob('*') if x.is_file()),
            artifacts_sha256=artifacts, completed_at=datetime.now(timezone.utc).isoformat())
        reconcile(c, s, rows, fits); base.write_once(p/'worker_summary.json', s)
        print(json.dumps(dict(complete=True, model_queries=80, feature_records=16, optimizer_steps=128)))
    except BaseException as e:
        with (p/'technical_traceback.log').open('x') as f: traceback.print_exc(file=f)
        base.write_once(p/'technical_stop.json', dict(complete=False, error_type=type(e).__name__, error=str(e),
            model_queries=resources.queries, automatic_retry=False, elapsed_seconds=time.monotonic()-started))
        raise
    finally:
        signal.alarm(0); resources.stop.set(); resources.thread.join(timeout=12)


if __name__ == '__main__': main()
