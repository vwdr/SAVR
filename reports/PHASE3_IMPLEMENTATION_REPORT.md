# Phase 3 Controller and Cache Implementation Report

Date: 2026-07-29

Status: **COMPLETE — ACCEPTED IN PR #10**

## Scope completed

Phase 3 produced project-owned, dependency-light implementations of:

- a common controller interface for FR, PR, VOR, and SAVR
- deterministic two-camera image, robot-state, and action-history signals
- a context- and tensor-compatible projected visual-feature cache
- an exception-safe OpenVLA-OFT projected-feature adapter
- immutable per-query and per-episode JSON record storage
- CPU unit tests for the controller/cache boundary

No GPU, simulator, threshold calibration, or performance experiment ran.

## Verified integration boundary

Pinned OpenVLA-OFT commit:
`e4287e94541f459edc4feabc4e181f537cd569a8`

The inspected `predict_action` path computes:

1. visual backbone output
2. visual projector output
3. current proprioception projection and append
4. language-model execution
5. unchanged L1 action head

SAVR caches only item 2. The temporary model-instance interceptor is restored
in `finally`, including when downstream inference raises. No upstream source,
model weight, or action-head code was changed.

## Implemented behavior

- FR refreshes on every policy query.
- PR refreshes at exact query-index multiples of its configured period.
- VOR uses only two-camera image change and maximum reuse horizon.
- SAVR combines image, previous-query state, two-chunk action history, and
  maximum reuse horizon.
- SAVR forces conservative refresh until two prior action chunks exist.
- Scores exceeding thresholds refresh; reaching the configured horizon
  refreshes.
- Empty/incompatible cache, invalid signal data, or changed cache context
  forces refresh.
- Image references update only after an actual successful refresh.
- State always compares against the preceding successful query.
- Current proprioception stays outside the cache and remains fresh on reuse.
- Cache compatibility covers context, batch, patch count, language dimension,
  dtype, and device.
- Cache age counts consecutive reuse queries and resets to zero on store.
- Query records include both policy-query and environment-step indices.
- Immutable records use exclusive atomic publication and reject overwrite.

Thresholds remain injected parameters; none were selected or tuned.

## Validation evidence

Local validation:

- bootstrap validator: passed, `52` required files
- Python unit tests: `29/29` passed
- Ruff: passed
- mypy: passed with no issues across seven source files
- Python byte compilation: passed
- Git diff whitespace check: passed

TITAN validation:

- system-Python tests: `29/29` passed
- pinned `envs/openvla-oft` tests: `29/29` passed
- package wheel built without dependency download:
  `savr-0.1.0-py3-none-any.whl`
- wheel SHA-256:
  `3a05578b39b9aced9c69b4f9e1c96a46cf3d0800253ad51c9bc3d8e466b1749b`
- pinned OpenVLA-OFT worktree: clean at the expected commit
- SAVR worktree: clean after removing only generated `build/` and
  `src/savr.egg-info/` directories created by the wheel check

## Safety and resource audit

- GPU use: none; `CUDA_VISIBLE_DEVICES` was empty for pinned-environment tests
- model/checkpoint loading: none
- simulator execution: none
- network or asset download: none
- upstream changes: none
- server writes: only inside `/home/ved/SAVR`
- unrelated university files/processes/services/permissions: untouched

## Limitations and next gate

Phase 3 tests use deterministic fake tensor/model objects for the interception
boundary. They establish project-code behavior, not numerical parity with the
loaded VLA.

Phase 4 must separately verify:

- wrapped FR versus unmodified OpenVLA-OFT actions
- real cached tensor shape/dtype/device
- full controller truth tables and reset cases
- synchronized timing and logging-schema audit
- interrupted-run recovery

The evidence was explicitly approved and merged in PR #10. All technical
Phase 3 exit criteria are satisfied. Phase 3 remains administratively
`IN_PROGRESS` until an explicit Phase 4 go/no-go decision. Phase 4 and all
GPU/simulator work remain unauthorized.
