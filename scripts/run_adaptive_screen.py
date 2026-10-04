#!/usr/bin/env python3
"""Stage S1: one frozen four-arm adaptive screen on GPU 0, gated by stage S0.

Runs only after explicit bounded-launch approval and only after the S0 real-
checkpoint adaptive qualification completed in the same session (its summary is
validated structurally in-session and bound to the frozen S0 config through the
launch.json config sha). Arms: dense 512, fixed 384, adaptive retention 75%
(fastv_r 0.25) and 50% (fastv_r 0.50). Timing-corrected protocol: fixed-384 and
dense-512 run compressed/dense from query one; only adaptive arms run the
release's required dense startup (first 3 queries after each episode reset),
and every per-arm all-query mean counts that startup. Each native episode
shadows against dense-512 (<=1e-6, unchanged) and against the adaptive path
with fastv_r=0 (native SDPA; parity bound pinned by S0), carrying SDPA
call witnesses for the output_attentions audit. Science triage never changes
execution.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback
import run_openvla_original_baseline as base
import run_contemporary_reference_recovery01 as prior
import run_fixed_compression_qualification as qualification
import run_adaptive_qualification as adaptive_qualification

ROOT = Path('/home/ved/SAVR')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from savr.openvla.adaptive_screen_contract import ARMS, CAPS, reconcile, timing_slots, validate_config
from savr.openvla.contemporary_contract import HORIZONS, require

REFERENCE = 'configs/openvla/contemporary_reference_evaluation_v2.json'
REFERENCE_SHA = 'afa1bc948880b072b74c9f150e7dbab77fe90468040746c73b82af603f91ec67'
QUAL_CONFIG = 'configs/openvla/fixed_compression_qualification_v1.json'
QUAL_CONFIG_SHA = 'd4cb82d508b16a923270d4e979eced5f243b4a902d851b71772bc9702aa862b4'
ADAPTIVE_QUAL_CONFIG = 'configs/openvla/adaptive_qualification_v3.json'
FILES = ('scripts/run_adaptive_screen.py', 'scripts/analyze_adaptive_screen.py',
         'src/savr/openvla/adaptive_screen_contract.py', 'src/savr/openvla/vlapruner_adaptive.py',
         'src/savr/openvla/adaptive_query.py', 'tests/openvla/test_adaptive_screen.py',
         'docs/ADAPTIVE_SCREEN_PROTOCOL_V2.md', 'src/savr/openvla/adaptive_sdpa.py',
         'tests/openvla/test_adaptive_sdpa.py')


def scoped(relative):
    p = (ROOT / relative).resolve()
    require(not Path(relative).is_absolute() and p.is_relative_to(ROOT), 'path outside project')
    return p


def verify(c):
    require(c['schema_version'] == 'adaptive-screen-v1' and c['launch_ready'] is True
            and c['caps'] == CAPS and c['arms'] == list(ARMS) and c['tolerance'] == 1e-6
            and c['automatic_retry'] is False and c['output_root'] == 'results/adaptive-screen-v03',
            'frozen screen contract differs')
    require(set(FILES) <= set(c['authenticated_files']), 'source manifest incomplete')
    for name, expected in c['authenticated_files'].items():
        require(base.sha(scoped(name)) == expected, 'source hash differs: ' + name)
    require(base.sha(ROOT / REFERENCE) == REFERENCE_SHA, 'reference changed')
    require(base.sha(ROOT / QUAL_CONFIG) == QUAL_CONFIG_SHA, 'fixed qualification config changed')
    q = json.loads((ROOT / QUAL_CONFIG).read_text())
    old, ref, manifest = qualification.verify(q)
    validate_config(c, ref, q['observation_ids'])
    require(c['gpu'] == q['gpu'] and c['observation_ids'] == q['observation_ids']
            and c['initial_state_sha256'] == old['initial_state_sha256'], 'GPU/population/initial states differ')
    require(c['worker_sha256'] == base.sha(Path(__file__))
            and c['analyzer_sha256'] == base.sha(scoped(FILES[1])), 'worker/analyzer identity differs')
    summary = scoped(c['qualification_summary'])
    require(str(summary.relative_to(ROOT)) == 'results/fixed-compression-qualification-v01/worker_summary.json'
            and base.sha(summary) == c['qualification_summary_sha256']
            and not (summary.parent / 'technical_stop.json').exists(), 'fixed qualification changed/stopped')
    s = json.loads(summary.read_text())
    require(s['complete'] is True, 'fixed qualification incomplete')
    for name, expected in s['artifacts_sha256'].items():
        x = (summary.parent / name).resolve()
        require(x.is_relative_to(summary.parent) and base.sha(x) == expected, 'fixed qualification artifact differs')
    qualification.reconcile(q, s, json.loads((summary.parent / 'records.json').read_text()))
    adaptive = scoped(c['adaptive_qualification_summary'])
    require(str(adaptive.relative_to(ROOT)) == 'results/adaptive-qualification-v03/worker_summary.json',
            'S0 summary path differs')
    require(adaptive.is_file(), 'S0 qualification not present — run S0 (adaptive-qualification-v1) first')
    require(not (adaptive.parent / 'technical_stop.json').exists(), 'S0 qualification stopped')
    s0 = json.loads(adaptive.read_text())
    require(s0['complete'] is True, 'S0 qualification incomplete')
    require((adaptive.parent / 'launch.json').is_file(), 'S0 launch.json missing')
    launch0 = json.loads((adaptive.parent / 'launch.json').read_text())
    require(base.sha(ROOT / ADAPTIVE_QUAL_CONFIG) == launch0['config_sha256']
            and base.sha(ROOT / ADAPTIVE_QUAL_CONFIG) == c['adaptive_qualification_config_sha256'],
            'S0 did not run the frozen adaptive-qualification config')
    for name, expected in s0['artifacts_sha256'].items():
        x = (adaptive.parent / name).resolve()
        require(x.is_relative_to(adaptive.parent) and base.sha(x) == expected, 'S0 artifact differs')
    a0 = json.loads((ROOT / ADAPTIVE_QUAL_CONFIG).read_text())
    adaptive_qualification.reconcile(a0, s0, json.loads((adaptive.parent / 'records.json').read_text()))
    zero_tolerance = float(s0['adaptive_zero_tolerance'])
    require(zero_tolerance <= c['adaptive_zero_tolerance_max']
            and c['adaptive_zero_tolerance_max'] <= 1e-3, 'S0 tolerance exceeds the frozen ceiling')
    return old, ref, manifest, zero_tolerance


def run(c, ref, observations, p, resources, zero_tolerance):
    import numpy as np
    import torch
    import transformers
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(1); tf.config.threading.set_inter_op_parallelism_threads(1)
    require(torch.__version__ == '2.2.0+cu118' and transformers.__version__ == '4.40.1'
            and torch.cuda.device_count() == 1, 'original one-GPU runtime required')
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(23000 / 24576, 0)
    sys.path.insert(0, str(ROOT / 'third_party/openvla-oft'))
    from experiments.robot import openvla_utils as utils
    from experiments.robot.libero import run_libero_eval as evaluation
    from experiments.robot.robot_utils import set_seed_everywhere, get_image_resize_size
    from experiments.robot.libero.libero_utils import get_libero_env
    from libero.libero import benchmark
    from savr.cac.c1 import instruction_token_indices
    from savr.openvla.fixed_compression import FixedCompressionQuery
    from savr.openvla.adaptive_query import AdaptivePruneQuery, count_sdpa
    from savr.openvla.specprune import EpisodeState
    from savr.openvla.offline_camera_inputs import OfflineCameraPair
    from savr.openvla.official_semantics import OfficialActionHeadCapture
    from savr.openvla.attention_diagnostic import LlamaAttentionDiagnostic
    from savr.openvla.execution_precision import processed_command_chunk, execution_chunk_float32
    from savr.openvla.contemporary_episode import execute_measured_episode
    checkpoint = ROOT / ref['checkpoint']
    names = {x.name for x in checkpoint.iterdir()}
    cfg = evaluation.GenerateConfig(pretrained_checkpoint=str(checkpoint), task_suite_name='libero_spatial',
        num_trials_per_task=0, seed=7, local_log_dir=str(p / 'logs'), use_wandb=False, center_crop=True,
        num_open_loop_steps=8, num_images_in_input=2, use_proprio=True, use_l1_regression=True,
        use_diffusion=False, use_film=False)
    evaluation.validate_config(cfg); set_seed_everywhere(7)
    update, sync = utils.update_auto_map, utils.check_model_logic_mismatch
    utils.update_auto_map = lambda _: None; utils.check_model_logic_mismatch = lambda _: None
    try:
        model, head, proprio, noisy, processor = evaluation.initialize_model(cfg)
    finally:
        utils.update_auto_map, utils.check_model_logic_mismatch = update, sync
    require(head is not None and proprio is not None and noisy is None and
            not any(m.training for m in (model, head, proprio)), 'released components differ')
    decoder = model.language_model.model
    require(len(decoder.layers) == 32 and all(type(l.self_attn).__name__ == 'LlamaSdpaAttention'
            for l in decoder.layers), 'original SDPA decoder required')
    require(all(evaluation.TASK_MAX_STEPS[s] == h for s, h in HORIZONS.items())
            and get_image_resize_size(cfg) == 224 and cfg.num_steps_wait == 10, 'episode contract changed')
    base.write_once(p / 'loaded_runtime.json', dict(torch=torch.__version__, transformers=transformers.__version__,
        configuration=vars(cfg), tensorflow_gpu_disabled=True, checkpoint_metadata_mutation_disabled=True))
    shared = dict(model=model, head=head, proprio=proprio, processor=processor, cfg=cfg, utils=utils,
                  instruction_indexer=instruction_token_indices)
    bridges = {str(n): FixedCompressionQuery(**shared, budget=n) for n in (512, 384)}
    bridges.update({name: AdaptivePruneQuery(**shared, fastv_r=ratio, attention_backend='native_sdpa')
                    for name, ratio in (('adaptive_75', 0.25), ('adaptive_50', 0.5))})
    bridges['zero'] = AdaptivePruneQuery(**shared, fastv_r=0.0, attention_backend='native_sdpa')
    records = dict(episodes=[], timing=[], parity=[])

    def append(name, value):
        require(sum(x.stat().st_size for x in p.rglob('*') if x.is_file())
                < CAPS['artifact_bytes'] - 65536, 'artifact cap reached')
        with (p / name).open('a') as f:
            f.write(json.dumps(value, allow_nan=False) + '\n')

    def progress():
        append('progress.jsonl', dict(episodes=len(records['episodes']),
                                      timing_records=len(records['timing']), model_queries=resources.queries))

    def obs_hash(obs):
        h = hashlib.sha256()
        for k in ('full_image', 'wrist_image', 'state'):
            a = np.asarray(obs[k]); h.update(k.encode())
            h.update(str((a.shape, a.dtype.str)).encode()); h.update(a.tobytes())
        return h.hexdigest()

    @torch.inference_mode()
    def call(arm, obs, text, previous, state):
        resources.consume()
        if arm == 'native':
            raw = evaluation.get_action(cfg, model, {k: np.asarray(v).copy() for k, v in obs.items()}, text,
                processor=processor, action_head=head, proprio_projector=proprio,
                noisy_action_projector=noisy, use_film=False)
            return np.asarray(raw), dict(retained_visual_tokens=512, precise=False)
        raw = bridges[arm](obs, text, previous, state)
        return raw, dict(bridges[arm].last_query)

    def audited(arm, obs, text, previous, state):
        # Every arm uses native outputs; count decoder layers rather than vision calls.
        with count_sdpa() as sdpas:
            with LlamaAttentionDiagnostic(torch, decoder.layers, 'original') as audit:
                with OfficialActionHeadCapture(head) as capture:
                    raw, details = call(arm, obs, text, previous, state)
            observed = len(audit.records)
        values = dict(hidden=capture.exact_hidden().detach().float().cpu().numpy(),
                      normalized=capture.exact_output().detach().float().cpu().numpy(), raw=raw)
        require(observed == 32, 'released attention audit incomplete')
        return raw, details, values, observed, observed

    def shadow(slot, obs, text, previous, state):
        fingerprint = obs_hash(obs)
        raw, details, a, na, native_sdpa = audited('native', obs, text, previous, state)
        other, _, b, nb, sdpa_512 = audited('512', obs, text, previous, state)
        zstate = EpisodeState(); zstate.reset(slot['slot_id'])
        zero_raw, _, z, nz, sdpa_zero = audited('zero', obs, text, previous, zstate)
        require(obs_hash(obs) == fingerprint and all(a[k].shape == b[k].shape == z[k].shape
                and np.isfinite(a[k]).all() and np.isfinite(b[k]).all() and np.isfinite(z[k]).all()
                for k in a), 'shadow observation/output changed')
        errors_512 = {k: float(np.max(np.abs(a[k].astype(np.float64) - b[k].astype(np.float64)))) for k in a}
        errors_zero = {k: float(np.max(np.abs(a[k].astype(np.float64) - z[k].astype(np.float64)))) for k in a}
        commands = processed_command_chunk(raw, evaluation.process_action, cfg.model_family)
        other_commands = processed_command_chunk(other, evaluation.process_action, cfg.model_family)
        zero_commands = processed_command_chunk(zero_raw, evaluation.process_action, cfg.model_family)
        row1 = dict(slot_id=slot['slot_id'], query=state.query, pair='native-vs-512', errors=errors_512,
            layers=[na, nb], sdpa_calls=native_sdpa, shadow_sdpa_calls=sdpa_512,
            observation_sha256=fingerprint, command_sha256=hashlib.sha256(commands.tobytes()).hexdigest(),
            command_bytes_equal=commands.tobytes() == other_commands.tobytes())
        row2 = dict(slot_id=slot['slot_id'], query=state.query, pair='native-vs-adaptive-zero',
            errors=errors_zero, layers=[na, nz], sdpa_calls=native_sdpa, shadow_sdpa_calls=sdpa_zero,
            observation_sha256=fingerprint, command_sha256=hashlib.sha256(zero_commands.tobytes()).hexdigest(),
            command_bytes_equal=commands.tobytes() == zero_commands.tobytes())
        append('parity.jsonl', row1); append('parity.jsonl', row2)
        require(row1['command_bytes_equal'] and [na, nb] == [32, 32]
                and all(v <= 1e-6 for v in errors_512.values()), 'native/512 parity failed')
        require(row2['command_bytes_equal'] and [na, nz] == [32, 32] and sdpa_zero == 32
                and all(v <= zero_tolerance for v in errors_zero.values()),
                'adaptive-zero parity exceeded the S0-pinned bound')
        records['parity'].append(row1); records['parity'].append(row2)
        return raw, details

    current_trace = None
    for slot in c['timing_slots']:
        identity, frames = observations[slot['observation_id']]
        cfg.unnorm_key = identity['normalization_statistics_key']
        trace = (slot['warmup'], slot['round'], slot['observation_id'], slot['arm'])
        if trace != current_trace:
            require(slot['frame'] == 0, 'trace must start at frame zero')
            state = EpisodeState(); state.reset(trace); current_trace = trace
        torch.cuda.synchronize(); started = time.perf_counter()
        previous = OfflineCameraPair.capture(frames[0], 0)
        raw, details = call(slot['arm'], frames[slot['frame']], identity['language_instruction'], previous, state)
        execution_chunk_float32(raw); torch.cuda.synchronize(); seconds = time.perf_counter() - started
        row = dict(slot, seconds=seconds, query=state.query, precise=details['precise'],
                   retained_visual_tokens=details['retained_visual_tokens'])
        append('timing.jsonl', row); records['timing'].append(row)
    progress()
    for slot in c['episode_slots']:
        condition = slot['condition']; cfg.task_suite_name = condition['suite']
        cfg.unnorm_key = condition['suite'] if condition['suite'] in model.norm_stats else condition['suite'] + '_no_noops'
        suite = benchmark.get_benchmark_dict()[condition['suite']]()
        matches = [i for i in range(suite.n_tasks) if suite.get_task(i).name == condition['task_id']]
        require(len(matches) == 1, 'task must resolve uniquely')
        initial = suite.get_task_init_states(matches[0])[condition['initial_state_id']].copy()
        require(hashlib.sha256(initial.tobytes()).hexdigest() == c['initial_state_sha256'][condition['condition_id']],
                'initial state changed')
        set_seed_everywhere(7)
        env, instruction = get_libero_env(suite.get_task(matches[0]), cfg.model_family, resolution=cfg.env_img_res)
        try:
            query = (lambda obs, text, previous, state: shadow(slot, obs, text, previous, state)) \
                if slot['arm'] == 'native' else \
                (lambda obs, text, previous, state: call(slot['arm'], obs, text, previous, state))
            result = execute_measured_episode(evaluation, cfg, env, initial, instruction, query, 224,
                episode_id=slot['slot_id'], controller=False, synchronize=torch.cuda.synchronize)
        finally:
            env.close()
        row = dict(result, slot_id=slot['slot_id'], condition_id=condition['condition_id'],
                   suite=condition['suite'], arm=slot['arm'])
        append('episodes.jsonl', row); records['episodes'].append(row); progress()
    require(names == {x.name for x in checkpoint.iterdir()}, 'checkpoint inventory changed')
    verify(c)
    resources.close(); base.write_once(p / 'records.json', records)
    return records


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--preflight-only', action='store_true'); args = parser.parse_args()
    require(Path.cwd().resolve() == ROOT and Path(sys.prefix).resolve().is_relative_to(ROOT / 'envs/openvla-oft'),
            'project runtime required')
    require(args.config.resolve().is_relative_to(ROOT / 'configs/openvla'), 'project config required')
    c = json.loads(args.config.read_text())
    old, ref, manifest, zero_tolerance = verify(c)
    from savr.openvla.offline_camera_inputs import load_offline_inputs
    observations, audit = load_offline_inputs(ROOT, manifest)
    require(audit == json.loads(qualification.scoped(old['input_preflight_report']).read_text())['input_audit'],
            'actual input changed')
    if args.preflight_only:
        print(json.dumps(dict(preflight_passed=True, model_queries=0))); return
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == str(c['gpu']['index']), 'GPU selection differs')
    initial = base.snapshot(c['gpu']['index'])
    require(initial['uuid'] == c['gpu']['uuid'] and initial['memory_mib'] <= 1024
            and initial['utilization'] <= 5, 'selected GPU not idle')
    p = scoped(c['output_root']); p.mkdir(exist_ok=False); base.environment(p)
    base.write_once(p / 'launch.json', dict(config=c, config_sha256=base.sha(args.config), pid=os.getpid(),
        started_at=datetime.now(timezone.utc).isoformat(), initial_gpu=initial))
    base.write_once(p / 'input_preflight.json',
                    dict(complete=True, input_audit=audit, performed_before_model_loading=True))
    resources = base.Resources(dict(gpu=c['gpu'], caps=dict(queries=7000, memory_mib=23552)))
    resources.peak = initial['memory_mib']; resources.thread.start(); started = time.monotonic()

    def timeout(*_):
        raise RuntimeError('frozen time cap reached')
    signal.signal(signal.SIGALRM, timeout); signal.alarm(CAPS['seconds'])
    try:
        records = run(c, ref, observations, p, resources, zero_tolerance)
        s = dict(complete=True, model_queries=resources.queries, episodes=len(records['episodes']),
            training_performed=False, automatic_retry=False, positive_method_result=False,
            checkpoint_unchanged=True, authenticated_files_unchanged=True, worker_sha256=c['worker_sha256'],
            adaptive_zero_tolerance=zero_tolerance, adaptive_zero_tolerance_max=c['adaptive_zero_tolerance_max'],
            elapsed_seconds=time.monotonic() - started, peak_aggregate_gpu_memory_mib=resources.peak,
            completed_at=datetime.now(timezone.utc).isoformat())
        reconcile(c, ref, s, records, c['observation_ids'])
        s['artifacts_sha256'] = {x.name: base.sha(x) for x in p.iterdir()
                                 if x.is_file() and x.suffix in ('.json', '.jsonl')}
        require(sum(x.stat().st_size for x in p.rglob('*') if x.is_file())
                < CAPS['artifact_bytes'] - 65536, 'artifact cap reached')
        base.write_once(p / 'worker_summary.json', s)
        print(json.dumps(dict(complete=True, episodes=168, model_queries=resources.queries,
                              adaptive_zero_tolerance=zero_tolerance)))
    except BaseException as e:
        with (p / 'technical_traceback.log').open('x') as f:
            traceback.print_exc(file=f)
        base.write_once(p / 'technical_stop.json', dict(complete=False, error_type=type(e).__name__,
            error=str(e), model_queries=resources.queries, automatic_retry=False,
            elapsed_seconds=time.monotonic() - started, peak_aggregate_gpu_memory_mib=resources.peak))
        raise
    finally:
        signal.alarm(0); resources.stop.set(); resources.thread.join(timeout=12)


if __name__ == '__main__':
    main()
