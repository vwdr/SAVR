#!/usr/bin/env python3
"""S0 v03: qualify backend-aligned adaptive selection before robot evaluation.

All arms retain native SDPA outputs. Zero pruning must preserve executed commands
exactly. Test 16 consumed inputs, parallel audited/plain four-query histories,
retention, absolute positions, nonvisual tokens and all 32 decoder SDPA calls.
The numerical ceiling remains 1e-3, calibrated only from zero-pruning comparisons;
pruning-induced differences never enter that bound. See protocol V2.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
import traceback
import run_openvla_original_baseline as base
import run_contemporary_reference_recovery01 as prior
import run_fixed_compression_qualification as qualification

ROOT = Path('/home/ved/SAVR')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

CAPS = dict(model_calls=256, episodes=0, seconds=3600,
            aggregate_memory_mib=23552, artifact_bytes=268435456)
# Exact frozen per-input schedule: 16 observations × (native + zero plain + zero
# audited) + 8 frame-1 observations × 2 modes × 8 calls (4 plain + 4 audited
# parallel traces) = 48 + 128 = 176 model calls. CAPS['model_calls'] remains the
# ceiling; MODEL_CALLS is the exact schedule count the summary must report.
MODEL_CALLS = 16 * 3 + 8 * 2 * 8
REFERENCE = 'configs/openvla/contemporary_reference_evaluation_v2.json'
REFERENCE_SHA = 'afa1bc948880b072b74c9f150e7dbab77fe90468040746c73b82af603f91ec67'
QUAL_CONFIG = 'configs/openvla/fixed_compression_qualification_v1.json'
QUAL_CONFIG_SHA = 'd4cb82d508b16a923270d4e979eced5f243b4a902d851b71772bc9702aa862b4'
MODES = dict(adaptive_75=0.25, adaptive_50=0.5)
FILES = ('scripts/run_adaptive_qualification.py', 'scripts/analyze_adaptive_qualification.py',
         'src/savr/openvla/vlapruner_adaptive.py', 'src/savr/openvla/adaptive_query.py',
         'tests/openvla/test_vlapruner_adaptive.py', 'tests/openvla/test_adaptive_qualification.py',
         'docs/ADAPTIVE_SCREEN_PROTOCOL_V2.md', 'src/savr/openvla/adaptive_sdpa.py',
         'tests/openvla/test_adaptive_sdpa.py')


def require(value, message):
    if not value:
        raise ValueError(message)


def scoped(relative):
    p = (ROOT / relative).resolve()
    require(not Path(relative).is_absolute() and p.is_relative_to(ROOT), 'path outside project')
    return p


def verify(c):
    require(c['schema_version'] == 'adaptive-qualification-v1' and c['launch_ready'] is True
            and c['caps'] == CAPS and c['automatic_retry'] is False
            and c['arms'] == list(MODES) and c['fastv_ratios'] == MODES
            and c['tolerance'] == 1e-6 and c['adaptive_zero_tolerance_max'] == 1e-3
            and c['output_root'] == 'results/adaptive-qualification-v03',
            'frozen adaptive qualification contract differs')
    require(set(FILES) <= set(c['authenticated_files']), 'source manifest incomplete')
    for name, expected in c['authenticated_files'].items():
        require(base.sha(scoped(name)) == expected, 'source hash differs: ' + name)
    require(base.sha(ROOT / REFERENCE) == REFERENCE_SHA, 'reference changed')
    reference_config = json.loads((ROOT / REFERENCE).read_text())
    _, reference, manifest = prior.verify(reference_config)
    require(c['gpu'] == reference_config['gpu'] and c['observation_ids'] == reference_config['observation_ids'],
            'GPU/input identities differ')
    require(base.sha(ROOT / QUAL_CONFIG) == QUAL_CONFIG_SHA, 'fixed qualification config changed')
    qualification.verify(json.loads((ROOT / QUAL_CONFIG).read_text()))
    require(c['worker_sha256'] == base.sha(Path(__file__))
            and c['analyzer_sha256'] == base.sha(scoped(FILES[1])), 'worker/analyzer identity differs')
    summary = scoped(c['reference_summary'])
    require(str(summary.relative_to(ROOT)) == 'results/contemporary-reference-v02/worker_summary.json'
            and base.sha(summary) == c['reference_summary_sha256'], 'completed reference changed')
    s = json.loads(summary.read_text())
    require(s['complete'] is True and s['episodes'] == 88 and not (summary.parent / 'technical_stop.json').exists(),
            'completed reference required')
    for name, expected in s['artifacts_sha256'].items():
        p = (summary.parent / name).resolve()
        require(p.is_relative_to(summary.parent) and base.sha(p) == expected, 'reference artifact changed')
    return reference, manifest


def reconcile(c, s, rows):
    require(s.get('complete') is True and s.get('model_queries') == MODEL_CALLS
            and s.get('episodes') == 0 and s.get('training_performed') is False
            and s.get('automatic_retry') is False and s.get('positive_method_result') is False
            and s.get('checkpoint_unchanged') is True and s.get('authenticated_files_unchanged') is True,
            'incomplete qualification')
    require(s['worker_sha256'] == c['worker_sha256'], 'worker identity differs')
    require(all(type(s[k]) in (int, float) and math.isfinite(s[k]) and 0 <= s[k] < cap for k, cap in
                (('elapsed_seconds', 3600), ('peak_aggregate_gpu_memory_mib', 23552))), 'resource cap exceeded')
    achieved = s['adaptive_zero_tolerance']
    zero_parity = s['zero_parity_observed_max']
    pruning_shift = s['pruning_shift_observed_max']
    require(type(achieved) in (int, float) and math.isfinite(achieved)
            and 1e-6 <= achieved <= c['adaptive_zero_tolerance_max'], 'S0 tolerance outside frozen ceiling')
    # Tolerance calibration is separated from pruning-induced differences:
    # the eager-vs-SDPA gap is measured at dense inputs only (zero pruning).
    expect = max(2.0 * zero_parity, 1e-6)
    require(isinstance(zero_parity, float) and math.isfinite(zero_parity) and 0 <= zero_parity <= 5e-4
            and achieved == expect, 'S0 tolerance must derive from zero-pruning parity only')
    require(isinstance(pruning_shift, float) and math.isfinite(pruning_shift) and pruning_shift >= 0,
            'pruning shift recorded separately must be finite and non-negative')
    require([(r['observation_id'], r['frame']) for r in rows]
            == [(i, f) for i in c['observation_ids'] for f in (0, 1)], 'record order/count differs')
    for r in rows:
        require(r['input_unchanged'] is True and r['full_tokens'] > 570 and r['sdpa_native'] >= 1
                and r['sdpa_zero'] == 32 and r['zero_hooks_neutral'] is True
                and r['zero_command_equal'] is True, 'zero parity witness failed')
        require(set(r['zero_errors']) == {'hidden', 'normalized', 'raw'}
                and all(type(v) in (int, float) and math.isfinite(v) and 0 <= v
                        for v in r['zero_errors'].values()), 'zero errors invalid')
        require(r['frame'] == 1 and [m['mode'] for m in r['modes']] == list(MODES)
                or r['frame'] == 0 and r['modes'] == [], 'mode records misordered')
        for m in r['modes']:
            expected = MODES[m['mode']]
            require(m['retained_expected'] == int(round(512 * (1.0 - expected)))
                    and m['retained_queries'] == [512, 512, 512, m['retained_expected']]
                    and m['steady_query'] == 4 and m['state_query'] == 4 and m['pruned'] is True
                    and m['effective_fastv_r'] == expected and m['sdpa_calls'] == 32
                    and m['hooks_neutral'] is True and m['lengths_before_prune'] is True
                    and m['lengths_after_prune'] is True and m['original_positions_all_layers'] is True
                    and m['no_cache_all_layers'] is True, 'compressed execution contract failed')
            # Numeric prune-boundary evidence: layers 0-3 run full; layers 4-31
            # run on the post-prune sequence length full - (512 - retained_visual)
            # (the port keeps [0] + top_visual + [513..full), so the trailing
            # non-visual tokens remain). Recomputing it here keeps the frozen
            # boundary CPU-verifiable instead of trusting a boolean flag.
            require(type(m['full_tokens']) is int and 570 < m['full_tokens'] < 1024
                    and type(m['post_prune_length']) is int
                    and m['post_prune_length'] == m['full_tokens'] - 512 + m['retained_expected']
                    and m['layer_lengths'] == [m['full_tokens']] * 4 + [m['post_prune_length']] * 28,
                    'post-prune layer lengths do not match the frozen boundary')
            require(set(m['steady_errors']) == {'hidden', 'normalized', 'raw'}
                    and all(type(v) in (int, float) and math.isfinite(v) and 0 <= v
                            for v in m['steady_errors'].values()), 'steady errors invalid')
            # Steady (pruned) differences vs dense are evidence only; they are
            # never bounded against the numerical tolerance or the dense output.
    require(max([e for r in rows for e in r['zero_errors'].values()]) == zero_parity,
            'zero-parity max must equal the recorded calibration source')
    require(max([e for r in rows for m in r['modes'] for e in m['steady_errors'].values()])
            == pruning_shift, 'pruning-shift max must equal the recorded evidence')
    return True


def run(c, reference, observations, p, resources):
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
    from experiments.robot.robot_utils import set_seed_everywhere
    from savr.cac.c1 import instruction_token_indices
    from savr.openvla.specprune import EpisodeState
    from savr.openvla.specprune_episode import SpecPruneEpisodeQuery
    from savr.openvla.adaptive_query import AdaptivePruneQuery, count_sdpa
    from savr.openvla.offline_camera_inputs import OfflineCameraPair
    from savr.openvla.official_semantics import OfficialActionHeadCapture
    from savr.openvla.attention_diagnostic import LlamaAttentionDiagnostic
    from savr.openvla.execution_precision import processed_command_chunk

    checkpoint = ROOT / reference['checkpoint']
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
            not any(m.training for m in (model, head, proprio)), 'released inference components differ')
    decoder = model.language_model.model
    require(len(decoder.layers) == 32 and all(type(l.self_attn).__name__ == 'LlamaSdpaAttention'
            for l in decoder.layers), 'original 32-layer SDPA decoder required')
    base.write_once(p / 'loaded_runtime.json', dict(torch=torch.__version__, transformers=transformers.__version__,
        configuration=vars(cfg), tensorflow_gpu_disabled=True, checkpoint_metadata_mutation_disabled=True))
    shared = dict(model=model, head=head, proprio=proprio, processor=processor, cfg=cfg, utils=utils,
                  instruction_indexer=instruction_token_indices)
    old = SpecPruneEpisodeQuery(**shared, enabled=False)
    zero = AdaptivePruneQuery(**shared, fastv_r=0.0, attention_backend='native_sdpa')
    bridges = {name: AdaptivePruneQuery(**shared, fastv_r=ratio, attention_backend='native_sdpa')
               for name, ratio in MODES.items()}
    rows = []
    # Numerical parity: backend-aligned adaptive-zero vs native at dense inputs.
    # Pruning-induced steady differences are recorded separately
    # (pruning_shift_max) and never gate or inflate the tolerance.
    zero_parity_max = 0.0
    pruning_shift_max = 0.0

    def fingerprint(obs):
        h = hashlib.sha256()
        for key in ('full_image', 'wrist_image', 'state'):
            a = np.asarray(obs[key]); h.update(key.encode())
            h.update(str((a.shape, a.dtype.str)).encode()); h.update(a.tobytes())
        return h.hexdigest()

    @torch.inference_mode()
    def call(bridge, obs, text, previous, audited_, state=None):
        resources.consume()
        if state is None:
            state = EpisodeState(); state.reset('adaptive-qualification')
        # Every arm retains native SDPA outputs; witness decoder layers only.
        layers = []; handles = []

        def trace(module, args, kwargs):
            layers.append(dict(length=int(args[0].shape[1]),
                               positions=kwargs['position_ids'][0].cpu().tolist(),
                               no_cache=kwargs.get('past_key_value') is None
                                        and kwargs.get('use_cache') is False))
        if audited_:
            handles = [l.register_forward_pre_hook(trace, with_kwargs=True) for l in decoder.layers]
        try:
            if audited_:
                with OfficialActionHeadCapture(head) as capture:
                    with count_sdpa() as sdpas:
                        with LlamaAttentionDiagnostic(torch, decoder.layers, 'original') as attention:
                            raw = bridge(obs, text, previous, state)
                        observed = len(attention.records)
                    values = dict(hidden=capture.exact_hidden().float().cpu().numpy(),
                                  normalized=capture.exact_output().float().cpu().numpy(), raw=raw)
                require(all(np.isfinite(v).all() for v in values.values()), 'invalid audited output')
                require(observed == 32, 'released attention audit incomplete')
            else:
                with count_sdpa() as sdpas:
                    raw = bridge(obs, text, previous, state)
                values = None
        finally:
            for handle in handles:
                handle.remove()
        commands = processed_command_chunk(raw, evaluation.process_action, cfg.model_family)
        require(1 <= state.query <= 4, 'query state advanced unexpectedly')
        decoder_calls = (bridge.last_query['sdpa_calls'] if isinstance(bridge, AdaptivePruneQuery)
                         else observed if audited_ else sdpas[0])
        return commands, values, layers, dict(bridge.last_query), decoder_calls

    for identity, frames in observations.values():
        cfg.unnorm_key = identity['normalization_statistics_key']
        text = identity['language_instruction']
        previous = OfflineCameraPair.capture(frames[0], 0)
        for frame, obs in enumerate(frames):
            before = fingerprint(obs)
            ref, ref_values, ref_layers, _, ref_sdpa = call(old, obs, text, previous, True)
            zplain, _, _, z_plain_details, z_plain_sdpa = call(zero, obs, text, previous, False)
            zaud, z_values, z_layers, z_details, z_sdpa = call(zero, obs, text, previous, True)
            neutral = zplain.tobytes() == zaud.tobytes() and z_plain_details == z_details
            require(neutral, 'audit hooks changed adaptive-zero commands')
            require(all(z_values[k].shape == ref_values[k].shape for k in z_values), 'parity shape differs')
            errors = {k: float(np.max(np.abs(z_values[k].astype(np.float64)
                                            - ref_values[k].astype(np.float64)))) for k in z_values}
            equal = ref.tobytes() == zaud.tobytes()
            require(equal, 'backend-aligned adaptive-zero commands differ from native')
            # Exact executed commands remain a hard gate, alongside numerical parity.
            zero_parity_max = max([zero_parity_max, *errors.values()])
            row = dict(observation_id=identity['trajectory_id'], frame=frame, input_unchanged=True,
                full_tokens=ref_layers[0]['length'], sdpa_native=ref_sdpa, sdpa_zero=z_sdpa,
                zero_hooks_neutral=neutral, zero_command_equal=equal, zero_errors=errors, modes=[])
            if frame == 1:
                for mode, bridge in bridges.items():
                    expected = int(round(512 * (1.0 - MODES[mode])))

                    def run_trace(audited_):
                        # 4-query episode on its own reset state: queries 1-3 are
                        # the required dense startup (frame 0), query 4 the steady
                        # pruned state (frame 1). Parallel plain/audited traces
                        # keep query numbering identical (a shared second call
                        # would advance past query 4 and invalidate the neutral
                        # comparison).
                        st = EpisodeState(); st.reset('adaptive-' + mode)
                        retained = []; last_raw = last_d = None
                        values = layers = sdpa_n = None
                        for qi in (1, 2, 3, 4):
                            frame_obs = frames[0] if qi <= 3 else frames[1]
                            if audited_ and qi == 4:
                                last_raw, values, layers, last_d, sdpa_n = call(
                                    bridge, frame_obs, text, previous, True, st)
                            else:
                                last_raw, _, _, last_d, _ = call(
                                    bridge, frame_obs, text, previous, False, st)
                            retained.append(last_d['retained_visual_tokens'])
                        return retained, last_raw, last_d, values, layers, sdpa_n

                    retained_p, q4_plain, d_plain, _, _, _ = run_trace(False)
                    retained_a, q4_audited, d, values, layers, sdpa_n = run_trace(True)
                    require(q4_plain.tobytes() == q4_audited.tobytes() and d_plain == d
                            and retained_p == retained_a, 'audit hooks changed adaptive commands')
                    require(d['query'] == 4 and d['pruned'] is True
                            and retained_a == [512, 512, 512, expected],
                            'adaptive retention contract failed')
                    steady_errors = {k: float(np.max(np.abs(values[k].astype(np.float64)
                                                - ref_values[k].astype(np.float64)))) for k in values}
                    # Pruning-induced steady differences vs dense are recorded
                    # as evidence (pruning_shift) but are NOT numerical-parity
                    # measurements: the pruned decoder runs a different, shorter
                    # computation, so these gaps must not calibrate or gate the
                    # eager-vs-SDPA tolerance.
                    pruning_shift_max = max([pruning_shift_max, *steady_errors.values()])
                    lengths = [l['length'] for l in layers]
                    full = ref_layers[0]['length']
                    kept = d['retained_visual_tokens']
                    # Post-prune layer length = full - (512 - kept_visual): the
                    # port keeps [0] + top_visual + [513..full), i.e. one leading
                    # token, the kept visual tokens, and all trailing non-visual
                    # tokens. comparing against 'kept' alone would never hold.
                    post_prune = full - (512 - kept)
                    require(kept == expected and len(lengths) == 32
                            and lengths[:4] == [full] * 4 and lengths[4:] == [post_prune] * 28,
                            'prune boundary after layer 3 violated (layers 0-3 full, 4-31 short)')
                    row['modes'].append(dict(mode=mode, fastv_r=MODES[mode], retained_expected=expected,
                        retained_queries=retained_a, steady_query=d['query'], state_query=d['query'],
                        pruned=d['pruned'], effective_fastv_r=d['effective_fastv_r'],
                        full_tokens=full, post_prune_length=post_prune, layer_lengths=lengths,
                        lengths_before_prune=True, lengths_after_prune=True,
                        original_positions_all_layers=all(
                            l['positions'] == layers[4]['positions'] for l in layers[4:]),
                        no_cache_all_layers=all(l['no_cache'] for l in layers),
                        hooks_neutral=True, sdpa_calls=sdpa_n, steady_errors=steady_errors))
            require(before == fingerprint(obs), 'input mutated')
            rows.append(row)
            with (p / 'checks.jsonl').open('a') as f:
                f.write(json.dumps(row, allow_nan=False) + '\n')
            with (p / 'progress.jsonl').open('a') as f:
                f.write(json.dumps(dict(check_records=len(rows), model_queries=resources.queries)) + '\n')
            require(sum(x.stat().st_size for x in p.rglob('*') if x.is_file())
                    < CAPS['artifact_bytes'] - 65536, 'artifact cap reached')
    achieved = max(2.0 * zero_parity_max, 1e-6)
    require(achieved <= c['adaptive_zero_tolerance_max'],
            f'measured eager-vs-SDPA gap {achieved:.3e} exceeds the frozen ceiling')
    require(names == {x.name for x in checkpoint.iterdir()}, 'checkpoint inventory changed')
    verify(c)
    resources.close()
    base.write_once(p / 'records.json', rows)
    return rows, achieved, zero_parity_max, pruning_shift_max


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--preflight-only', action='store_true'); args = parser.parse_args()
    require(Path.cwd().resolve() == ROOT and Path(sys.prefix).resolve().is_relative_to(ROOT / 'envs/openvla-oft'),
            'project runtime required')
    require(args.config.resolve().is_relative_to(ROOT / 'configs/openvla'), 'project config required')
    c = json.loads(args.config.read_text())
    reference, manifest = verify(c)
    from savr.openvla.offline_camera_inputs import load_offline_inputs
    observations, audit = load_offline_inputs(ROOT, manifest)
    require(audit == json.loads(scoped(c['input_preflight_report']).read_text())['input_audit'],
            'real input audit changed')
    if args.preflight_only:
        print(json.dumps(dict(preflight_passed=True, model_calls=0))); return
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == str(c['gpu']['index']), 'GPU selection differs')
    initial = base.snapshot(c['gpu']['index'])
    require(initial['uuid'] == c['gpu']['uuid'] and initial['memory_mib'] <= 1024
            and initial['utilization'] <= 5, 'selected GPU not idle')
    p = scoped(c['output_root']); p.mkdir(exist_ok=False); base.environment(p)
    base.write_once(p / 'launch.json', dict(config=c, config_sha256=base.sha(args.config), pid=os.getpid(),
        started_at=datetime.now(timezone.utc).isoformat(), initial_gpu=initial))
    base.write_once(p / 'input_preflight.json',
                    dict(complete=True, input_audit=audit, performed_before_model_loading=True))
    resources = base.Resources(dict(gpu=c['gpu'], caps=dict(queries=CAPS['model_calls'], memory_mib=23552)))
    resources.peak = initial['memory_mib']; resources.thread.start(); started = time.monotonic()

    def timeout(*_):
        raise RuntimeError('frozen time cap reached')
    signal.signal(signal.SIGALRM, timeout); signal.alarm(CAPS['seconds'])
    try:
        rows, achieved, zero_parity_max, pruning_shift_max = run(c, reference, observations, p, resources)
        s = dict(complete=True, model_queries=resources.queries, episodes=0, training_performed=False,
            automatic_retry=False, positive_method_result=False, checkpoint_unchanged=True,
            authenticated_files_unchanged=True, worker_sha256=c['worker_sha256'],
            adaptive_zero_tolerance=achieved, adaptive_zero_tolerance_max=c['adaptive_zero_tolerance_max'],
            zero_parity_observed_max=zero_parity_max, pruning_shift_observed_max=pruning_shift_max,
            elapsed_seconds=time.monotonic() - started,
            peak_aggregate_gpu_memory_mib=resources.peak,
            completed_at=datetime.now(timezone.utc).isoformat())
        reconcile(c, s, rows)
        s['artifacts_sha256'] = {x.name: base.sha(x) for x in p.iterdir()
                                 if x.is_file() and x.suffix in ('.json', '.jsonl')}
        require(sum(x.stat().st_size for x in p.rglob('*') if x.is_file())
                < CAPS['artifact_bytes'] - 65536, 'artifact cap reached')
        base.write_once(p / 'worker_summary.json', s)
        print(json.dumps(dict(complete=True, model_queries=resources.queries,
                              adaptive_zero_tolerance=achieved)))
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
