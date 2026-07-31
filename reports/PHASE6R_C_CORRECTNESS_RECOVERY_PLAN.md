# Phase 6R-C Correctness Recovery Plan

Status: FROZEN BEFORE RECOVERY EXECUTION

Plan date: 2026-07-31

## First-run observation

The immutable `phase6r-c-correctness-v1` run used LIBERO-Spatial task `0`,
initial state `0`, directly after environment initialization. It executed ten
model queries and zero rollout episodes. The following checks passed:

- wrapped FR and unmodified OpenVLA-OFT actions were exactly equal;
- all nine published query records had the expected component counts;
- the SAVR 2.0 controller correctly forced six fresh queries;
- checkpoint restoration and new-file audits passed.

The planned seventh SAVR 2.0 query refreshed instead of reusing because the
real predicted action chunk contained both open and closed gripper commands.
The controller correctly applied the frozen `gripper_transition.mixed_latest`
veto. The run stopped and remains recorded as failed because the selected
correctness fixture could not exercise the reuse path.

This is a fixture mismatch, not permission to weaken or bypass the veto. No
controller, threshold, temporal rule, or cache rule will change.

## Frozen recovery

Use the already-existing Phase 6 FR calibration trace:

- path: `results/phase6-fr-signals-v1/queries/query_00000000.json`;
- identity: `fr_task_00_state_00`, query environment step `10`;
- SHA-256:
  `ff9f4bfc004b861260e36d61c5eab641356a9c27c25f7ceccf511e04dd687a63`;
- observed saved gripper commands are transition-free;
- the trace predates SAVR 2.0 and was not selected from a new SAVR 2.0 outcome.

Reconstruct the saved `32 x 32 x 3` images as deterministic `uint8` inputs,
reuse the saved state, and use the unchanged task-0 language instruction. Run:

1. one unmodified real-model reference query;
2. six SAVR 2.0 fresh queries;
3. one required SAVR 2.0 reuse query.

The recovery therefore executes eight additional model queries, zero simulator
resets, and zero rollout episodes. Together with the first run, Phase 6R-C uses
at most `18/20` permitted model queries. The original one-GPU, 45-minute, and
256-MiB bounds remain unchanged.

## Required recovery gates

- the reuse decision has no transition, signal, temporal, or budget veto;
- reuse executes zero vision-backbone and zero projector calls;
- language model, action head, and current proprioception each execute once;
- reuse actions exactly match the unmodified reference for the same input;
- all local camera, grouped state/action, counter, cache, schema, artifact, and
  checkpoint-restoration invariants reconcile;
- the failed first run remains immutable and visible.

If any recovery gate fails, Phase 6R-C stops and Phase 6R-D does not begin.
