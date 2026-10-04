#!/usr/bin/env python3
"""Bounded technical diagnosis only: 16 consumed inputs, four calls each, no episodes."""
import contextlib
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback
import run_openvla_original_baseline as base
import run_adaptive_qualification as qual

ROOT = Path('/home/ved/SAVR')
OUT = ROOT / 'results/adaptive-zero-diagnosis-v01'
FILES = tuple(dict.fromkeys((
    'scripts/diagnose_adaptive_zero_v1.py', *qual.FILES,
    'src/savr/openvla/specprune_episode.py', 'src/savr/openvla/specprune.py',
    'src/savr/openvla/official_semantics.py', 'src/savr/openvla/execution_precision.py',
    'configs/openvla/contemporary_reference_evaluation_v2.json',
    'docs/ADAPTIVE_ZERO_RECOVERY_PLAN_2026-09-25.md')))


def main():
    assert Path.cwd().resolve() == ROOT
    assert Path(sys.prefix).resolve().is_relative_to(ROOT / 'envs/openvla-oft')
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '0'
    assert base.sha(ROOT / qual.REFERENCE) == qual.REFERENCE_SHA
    c = json.loads((ROOT / qual.REFERENCE).read_text())
    initial = base.snapshot(0)
    assert initial['uuid'] == c['gpu']['uuid']
    assert initial['memory_mib'] <= 1024 and initial['utilization'] <= 5
    pins = {f: base.sha(ROOT / f) for f in FILES}
    OUT.mkdir(exist_ok=False)
    base.environment(OUT)
    base.write_once(OUT / 'launch.json', dict(pid=os.getpid(), sources=pins,
        initial_gpu=initial, caps=dict(calls=64, seconds=1800, memory_mib=23552,
        artifact_bytes=67108864), episodes=0, automatic_retry=False))
    resources = base.Resources(dict(gpu=c['gpu'], caps=dict(queries=64, memory_mib=23552)))
    resources.peak = initial['memory_mib']
    resources.thread.start()
    started = time.monotonic()

    def timeout(*_):
        raise RuntimeError('diagnostic time cap reached')
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(1800)
    try:
        _, reference, manifest = qual.prior.verify(c)
        from savr.openvla.offline_camera_inputs import load_offline_inputs, OfflineCameraPair
        observations, _ = load_offline_inputs(ROOT, manifest)
        import numpy as np
        import torch
        import transformers
        import tensorflow as tf
        tf.config.set_visible_devices([], 'GPU')
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
        assert torch.__version__ == '2.2.0+cu118' and transformers.__version__ == '4.40.1'
        assert torch.cuda.device_count() == 1
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.cuda.set_per_process_memory_fraction(23000 / 24576, 0)
        sys.path.insert(0, str(ROOT / 'third_party/openvla-oft'))
        from experiments.robot import openvla_utils as utils
        from experiments.robot.libero import run_libero_eval as evaluation
        from experiments.robot.robot_utils import set_seed_everywhere
        from savr.cac.c1 import instruction_token_indices
        from savr.openvla.specprune import EpisodeState
        from savr.openvla.specprune_episode import SpecPruneEpisodeQuery
        from savr.openvla.adaptive_query import AdaptivePruneQuery
        from savr.openvla.official_semantics import OfficialActionHeadCapture
        from savr.openvla.execution_precision import processed_command_chunk
        checkpoint = ROOT / reference['checkpoint']
        names = {p.name for p in checkpoint.iterdir()}
        cfg = evaluation.GenerateConfig(pretrained_checkpoint=str(checkpoint),
            task_suite_name='libero_spatial', num_trials_per_task=0, seed=7,
            local_log_dir=str(OUT / 'logs'), use_wandb=False, center_crop=True,
            num_open_loop_steps=8, num_images_in_input=2, use_proprio=True,
            use_l1_regression=True, use_diffusion=False, use_film=False)
        evaluation.validate_config(cfg)
        set_seed_everywhere(7)
        update, sync = utils.update_auto_map, utils.check_model_logic_mismatch
        utils.update_auto_map = lambda _: None
        utils.check_model_logic_mismatch = lambda _: None
        try:
            model, head, proprio, noisy, processor = evaluation.initialize_model(cfg)
        finally:
            utils.update_auto_map, utils.check_model_logic_mismatch = update, sync
        assert head is not None and proprio is not None and noisy is None
        assert not any(m.training for m in (model, head, proprio))
        decoder = model.language_model.model
        assert len(decoder.layers) == 32
        shared = dict(model=model, head=head, proprio=proprio, processor=processor,
            cfg=cfg, utils=utils, instruction_indexer=instruction_token_indices)
        native = SpecPruneEpisodeQuery(**shared, enabled=False)
        adaptive = AdaptivePruneQuery(**shared, fastv_r=0.)

        @contextlib.contextmanager
        def eager_reference(enabled):
            # Change attention implementation only, not preparation/readout/mask/positions.
            def hook(module, args, kwargs):
                kwargs['output_attentions'] = True
                return args, kwargs
            handles = [l.register_forward_pre_hook(hook, with_kwargs=True)
                       for l in decoder.layers] if enabled else []
            try:
                yield
            finally:
                for h in handles:
                    h.remove()

        @torch.inference_mode()
        def call(bridge, obs, text, previous, eager=False):
            resources.consume()
            state = EpisodeState()
            state.reset('technical-diagnostic')
            with eager_reference(eager), OfficialActionHeadCapture(head) as capture:
                raw = bridge(obs, text, previous, state)
                vals = dict(hidden=capture.exact_hidden().float().cpu().numpy(),
                            normalized=capture.exact_output().float().cpu().numpy(),
                            raw=raw, commands=processed_command_chunk(raw, evaluation.process_action, cfg.model_family))
            assert all(np.isfinite(v).all() for v in vals.values())
            return vals

        def compare(a, b):
            assert all(a[k].shape == b[k].shape for k in a)
            return dict(max_abs={k: float(np.max(np.abs(a[k].astype('float64') - b[k].astype('float64')))) for k in a},
                byte_equal={k: a[k].tobytes() == b[k].tobytes() for k in a},
                gripper_disagreements=int(np.count_nonzero(a['commands'][:, 6] != b['commands'][:, 6])),
                motion_max_abs=float(np.max(np.abs(a['commands'][:, :6] - b['commands'][:, :6]))))

        rows = []
        for oid in c['observation_ids']:
            identity, frames = observations[oid]
            cfg.unnorm_key = identity['normalization_statistics_key']
            previous = OfflineCameraPair.capture(frames[0], 0)
            for frame, obs in enumerate(frames):
                args = (obs, identity['language_instruction'], previous)
                a = call(native, *args)
                repeat = call(native, *args)
                b = call(adaptive, *args)
                eager = call(native, *args, eager=True)
                rows.append(dict(observation_id=oid, frame=frame,
                    repeat=compare(a, repeat), adaptive_vs_native=compare(a, b),
                    eager_vs_native=compare(a, eager), adaptive_vs_eager=compare(eager, b),
                    values={name: {k: v.tolist() for k, v in d.items() if k != 'hidden'}
                            for name, d in [('native', a), ('adaptive', b), ('eager', eager)]}))
                with (OUT / 'diagnostics.jsonl').open('a') as f:
                    f.write(json.dumps(rows[-1], allow_nan=False) + '\n')
                with (OUT / 'progress.jsonl').open('a') as f:
                    f.write(json.dumps(dict(records=len(rows), calls=resources.queries)) + '\n')
                assert sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file()) < 67108864 - 65536
        assert resources.queries == 64 and len(rows) == 16
        assert names == {p.name for p in checkpoint.iterdir()}
        assert pins == {f: base.sha(ROOT / f) for f in FILES}
        resources.close()
        base.write_once(OUT / 'worker_summary.json', dict(complete=True, episodes=0,
            records=16, model_calls=64, elapsed_seconds=time.monotonic()-started,
            peak_aggregate_gpu_memory_mib=resources.peak, automatic_retry=False,
            positive_method_result=False, artifacts_sha256={p.name: base.sha(p)
                for p in OUT.iterdir() if p.is_file()}))
    except BaseException as exc:
        base.write_once(OUT / 'technical_stop.json', dict(complete=False, error=str(exc),
            model_calls=resources.queries, automatic_retry=False))
        with (OUT / 'technical_traceback.log').open('x') as f:
            traceback.print_exc(file=f)
        raise
    finally:
        signal.alarm(0)
        resources.stop.set()
        resources.thread.join(timeout=12)


if __name__ == '__main__':
    main()
