# Frozen SpecPrune same-checkpoint qualification

Date: 2026-09-11. User requested continuation of the approved comparator work.
This authorizes one bounded implementation check, not training or task evaluation.

## Frozen configuration

`configs/openvla/specprune_real_qualification_v1.json` records exact source hashes,
eight previously consumed training-observation IDs, original reference config,
GPU identity, mode order, tolerance, limits and output root. The original baseline
configuration authenticates the checkpoint shards, runtime and input sources.
Worker: `scripts/run_specprune_qualification.py`.

Exactly five complete policy calls per observation, eight observations (40 calls):

1. Native released evaluator, with captured action-head input/output.
2. Adapted decoder with all pruning disabled.
3. Adapted decoder with attention scoring enabled but all tokens retained.
4. Source-default coarse pruning on a fresh first query. Both previous images
   equal the current policy-prepared images, as in the source's first-query case.
5. Native released evaluator again, to verify restoration.

The state is reset independently for each adapted call. This is an offline
integration test. It does not test a full episode's controller/frame history,
evaluate a trained corrector, or establish a success–latency improvement.

## Gates fixed before model execution

- Native, disabled, keep-all and restored action boundaries must be finite.
  Maximum absolute error against native must be at most 1e-6 separately for
  the 1x56x4096 action-head input, normalized 8x7 output and unnormalized 8x7
  policy action chunk. This is the existing original-runtime parity tolerance.
  These action chunks are not simulator-executed actions in this experiment.
- Exactly 32 original bidirectional SDPA layers per call. No manual-attention
  fallback, installed-runtime modification, or stale K/V reuse.
- Every compressed output must be finite, actually shorter than dense, and
  retain BOS, all language/proprioception/action/stop states with correct absolute
  action positions. Disabled and keep-all paths must retain every token.
- Compression is not required to reproduce dense actions; that is the later
  empirical question. No action-error efficacy threshold is introduced here.
- Exact observation/mode/call accounting and unchanged authenticated sources,
  checkpoint bytes and checkpoint file inventory, checked before and after.
- Maximum 40 model calls, zero simulator episodes, 1,800 seconds, aggregate
  selected-GPU memory strictly below 23,552 MiB, artifacts below 256 MiB.
- One GPU: TITAN GPU 0, UUID `GPU-bb2451d6-2989-a112-5c18-8892943710e4`.
  Require a fresh idle check before loading. Never inspect unrelated processes.
- Immutable output: `results/specprune-real-qualification-v01`. Refuse to replace
  it if present. No automatic retry, tolerance adjustment, or alternate preset.

## Runtime and monitoring

Use only `ssh titan` and `/home/ved/SAVR`. Use existing `envs/openvla-oft`.
Disable TensorFlow GPU access, gradients, online model loading and checkpoint
metadata updates. Keep every cache/log/temp path inside the result directory.
No installations, checkpoint/data downloads, or university configuration changes.

Monitor progress counts, owned-process health and aggregate GPU telemetry while
running. Read detailed qualification records only after a completed summary.
On early stop inspect technical error/traceback, preserve evidence and stop
without retry. No scientific negative result follows from a technical stop.

On completion verify artifact hashes, run the CPU reconciliation routine in the
worker independently, sync evidence and status locally, and report. Do not start
a simulator benchmark or adapter training automatically from this qualification.

## Interpretation and later fair timing

Passing establishes correctness of these integration paths on eight observations,
not GPU speedup, closed-loop success or publication readiness. This qualification
includes extra diagnostics and repeated image preparation, so its duration is
not an inference-latency measurement. A later timed dense control must share any
decoder-only wrapper optimization that omits unused logits/hidden-state history.
