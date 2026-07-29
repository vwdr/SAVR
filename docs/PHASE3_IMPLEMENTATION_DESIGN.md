# Phase 3 Controller and Cache Implementation Design

Date: 2026-07-29

Status: **APPROVED — IMPLEMENTATION IN PROGRESS**

## Scope

Phase 3 implements the project-owned controller, signal, projected-feature
cache, immutable record, and OpenVLA-OFT adapter components. It includes
CPU-only unit tests and source-level integration verification.

This phase does not authorize:

- GPU execution
- simulator evaluation
- threshold calibration
- success or latency claims
- model, checkpoint, or dataset downloads
- changes to upstream source, model weights, or the action head
- Phase 4 numerical/behavioral parity claims

## Verified upstream boundary

The boundary was inspected in pinned OpenVLA-OFT commit
`e4287e94541f459edc4feabc4e181f537cd569a8`.

In `prismatic/extern/hf/modeling_prismatic.py`:

1. `_process_vision_features` runs `vision_backbone` and then `projector`.
2. `predict_action` receives that projected visual tensor.
3. Only afterward, `_process_proprio_features` appends the current
   proprioception token.
4. The combined sequence then enters the language model and unchanged L1
   action head.

Therefore, the safe cache target is the output of
`_process_vision_features`: shape
`[batch, patches_per_image × image_count, language_dimension]`. The cache must
not include the proprioception token, language-model states, or action-head
output.

The official evaluation helper builds the eight-dimensional LIBERO state from
end-effector position, axis-angle orientation, and gripper joints, then
normalizes it with the checkpoint's `q01`/`q99` proprioception statistics.
This matches the execution protocol.

## Adapter strategy

The project adapter wraps an otherwise unmodified upstream policy query:

1. Snapshot the two signal images and raw eight-dimensional state.
2. Ask the selected controller for a refresh decision.
3. Temporarily intercept only the model instance's
   `_process_vision_features` method under a re-entrant lock.
4. On refresh, call the original method and store its detached output.
5. On reuse, return the compatible cached projected tensor without calling the
   backbone or projector.
6. Let upstream `predict_action` append current proprioception and execute its
   original language-model/action-head logic.
7. Restore the original instance state in `finally`.
8. Record the returned action chunk for the next action-change decision.

No upstream file is edited. The adapter rejects FiLM in this initial boundary
because language-conditioned visual features are outside the frozen
configuration. Diffusion remains outside the initial configuration.

## Controller contract

Every controller receives the same query observation and returns a structured
decision containing:

- refresh or reuse
- cache age before the decision
- image, state, and action scores when applicable
- per-camera image scores
- ordered trigger reasons
- active thresholds and horizon

Policies:

- **FR:** refresh every query.
- **PR(k):** refresh at query indices `0, k, 2k, ...`.
- **VOR:** image threshold or maximum reuse horizon.
- **SAVR:** image, state, action, or maximum reuse horizon.

Forced refresh takes precedence for an empty/incompatible cache, changed
context, invalid data, insufficient action history, or reached horizon.

## Signal contract

- Images: both cameras, normalized to `[0,1]`, deterministically sampled to
  `32 × 32`, per-camera mean absolute difference, then camera mean. The
  reference is the most recent successful refresh.
- State: current versus previous query, checkpoint `q01`/`q99`
  normalization, then RMS difference.
- Action: the two most recent completed action chunks before the current
  prediction, checkpoint `q01`/`q99` normalization, then RMS difference.
- Missing, non-finite, or shape-incompatible data raises a validation signal
  and forces refresh.

Threshold values are injected configuration. Phase 3 does not select or tune
them.

## Cache lifetime and compatibility

A cache context identifies the episode, task, checkpoint, and configuration.
Changing any context field resets the controller and cache. A cached tensor is
reused only when its batch size, patch count, language dimension, dtype, and
device match the current model call.

Cache age counts consecutive reused policy queries:

- store/refresh sets age to `0`
- successful reuse increments age by `1`
- a decision at age `H_max` must refresh

## Immutable records

Each query and episode is written to a unique JSON file using exclusive
creation. Existing records cannot be overwritten. Records include both
`query_index` and `environment_step`.

## Phase 3 acceptance checks

- common controller interface exists for FR, PR, VOR, and SAVR
- exact signal definitions have deterministic unit tests
- projected-feature cache validates context and tensor metadata
- adapter skips fake visual computation on reuse while preserving fresh
  proprioception behavior
- upstream method state is restored after success and failure
- record writes reject overwrite
- all repository and CPU unit tests pass
- no upstream source is changed

GPU parity, CUDA timing, full truth-table coverage, interrupted-run recovery,
and simulator behavior remain Phase 4 work.
