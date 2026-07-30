# Phase 4 Correctness and Instrumentation Report

Date: 2026-07-29

Status: **TECHNICALLY COMPLETE — CHECKPOINT REVIEW PENDING**

## Scope

Phase 4 expanded dependency-light controller/cache tests, audited synchronized
timing and immutable recovery records, and ran the approved six-query
real-model correctness matrix. It used one pinned OpenVLA-OFT model load, one
LIBERO-Spatial task-0/state-0 reset, and zero rollout episodes.

This is a correctness gate. It is not a latency, task-success, calibration, or
SAVR performance experiment.

## Frozen inputs

- SAVR runner revision:
  `28d5eb3dd0874279d04f2c0f51e337b27efdeb09`
- OpenVLA-OFT:
  `e4287e94541f459edc4feabc4e181f537cd569a8`
- LIBERO:
  `8f1084e3132a39270c3a13ebe37270a43ece2a01`
- combined checkpoint:
  `638918f3d1c2e43a39a8a20772bdb8b91835e4b7`
- run ID: `phase4-correctness-v1`
- task/state/seed: `0/0/0`
- model path: two images, current proprioception, L1 continuous action head,
  eight-action chunks, center crop enabled, FiLM and diffusion disabled

All source trees were clean and exact before execution. The checkpoint
inventory contained all 25 accepted files with their declared sizes, and the
three accepted metadata hashes matched before loading.

## CPU gates

The expanded suite covers:

- FR and PR periods `1`, `2`, and `3`
- both VOR cameras and state/action independence
- SAVR image-only, state-only, action-only, no-trigger, and overlap cases
- warm-up, exact horizon, invalid image/state/action data, and every context
  identity field
- cache age, tensor metadata, exception restoration, immutable records, and
  interrupted-run visibility
- CUDA-event boundary order and zero visual-hook calls through a fake backend
- deterministic state-B construction and exact action-parity rejection
- runtime validation of episode, query, and run schemas

Results:

- local: `44` tests discovered; `43` passed and the runtime-schema test was
  skipped because the local lightweight Python lacks `jsonschema`
- local dependency-free simulation: `44` tests discovered; `40` passed and
  four optional runtime-dependency tests were skipped
- TITAN pinned environment: `44/44` tests passed, including runtime schemas
- Ruff, mypy, byte compilation, bootstrap validation, and diff checks passed

The independent TITAN schema preflight initially found a validator-interface
bug before GPU inspection or use. It was fixed, recommitted, and all CPU gates
were repeated before the runner was frozen. This preserved the fail-closed
sequence.

## Real-model correctness result

All six planned records completed:

| Query | Path | Visual calls | Result |
|---:|---|---:|---|
| 1 | unmodified, state A | `1` | reference action chunk A |
| 2 | wrapped FR, state A | `1` | exact parity with query 1 |
| 3 | wrapped FR, state A | `1` | exact parity with query 1 |
| 4 | unmodified, state B | `1` | reference action chunk B |
| 5 | wrapped VOR refresh, state A | `1` | real projected tensor cached at age `0` |
| 6 | wrapped VOR reuse, state B | `0` | exact parity with query 4; cache age `1` |

For every required comparison:

- action shape was exactly `[8, 7]`
- `numpy.array_equal` was true
- maximum absolute action difference was exactly `0.0`
- wrapped FR refreshed on both invocations

The cache contained a `[1, 512, 4096]` `torch.bfloat16` tensor on `cuda:0`.
On query 6, both vision-backbone and visual-projector hook counts were zero;
the language model and action head each executed once. The proprioception
projector received normalized state B exactly, not state A. State B changed
only dimension 0 from `-0.2108148624` to `0.0332788052`, within its
`q01=-0.2727657300` and `q99=0.1352936503` bounds, and was never sent to the
simulator.

Timing records used synchronized CUDA events and explicit query boundaries.
The observed reuse query had `0.0 ms` recorded visual CUDA time, but the six
queries are correctness instrumentation only and must not be used for a
latency or efficiency claim.

## Records and integrity

- manifest status: `completed`
- immutable query records: `6/6`, indices `1-6`
- immutable run events: `RUNNING`, then `COMPLETED`
- simulator resets: `1`
- rollout episodes: `0`
- logical result-directory size: `27,760` bytes, below `256 MiB`
- model-load time: `5.87 s`
- peak GPU memory allocated: `16,093,921,792` bytes
- peak GPU memory reserved: `16,200,499,200` bytes
- checkpoint metadata: restored byte-for-byte
- expected upstream backup files removed after restoration: `2`
- unexpected new checkpoint files: `0`
- OpenVLA-OFT and LIBERO worktrees after execution: clean

All actual manifest and query records validate against their schemas. Raw
immutable evidence remains at:

`/home/ved/SAVR/results/phase4-correctness-v1`

## GPU and server safety

Physical GPU `0`
(`GPU-bb2451d6-2989-a112-5c18-8892943710e4`) was selected using only three
aggregate samples two seconds apart. Every pre-run sample showed `0%`
utilization, `6 MiB` used, and `24,018 MiB` free. Three post-run samples
showed the same values.

- only physical GPU `0` was exposed
- no process identities were inspected
- no download, installation, training, or weight change occurred
- no unrelated university file, process, environment, service, permission, or
  allocation was inspected or changed
- all server writes remained inside `/home/ved/SAVR`

## Interpretation and next gate

Phase 4 establishes that the project-owned adapter can cache and reuse the real
projected visual tensor while preserving current proprioception and exact
deterministic action output for the controlled query. It also establishes that
wrapped FR is an exact correctness oracle for this path.

It does not establish trajectory correctness, task success under reuse,
threshold quality, latency benefit, or a paper-level SAVR claim. Phase 5
remains unauthorized until this report and PR #13 are reviewed and merged,
followed by a separately bounded proposal.
