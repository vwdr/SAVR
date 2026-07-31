# SAVR3 Phase 6S Protocol

Status: FROZEN BEFORE SAVR3 IMPLEMENTATION OR OUTCOME COLLECTION

Protocol version: 1.0

Freeze date: 2026-07-31

Parent protocol: `docs/PHASE6R_PROTOCOL_V1.md`

## 1. Purpose and claim boundary

Phase 6S evaluates one final training-free redesign derived from the disclosed
Phase 6R-D Stage 1 forensic analysis. It is development validation, not final
holdout evidence. All prior negative results remain immutable and visible.

The final holdout remains untouched: initial-state IDs `10-49` and seeds `7`,
`17`, and `27`.

## 2. Frozen SAVR3 controller

Configuration ID: `savr3-rv-w375-b15`.

Use the same OpenVLA-OFT checkpoint, projected-visual-feature cache boundary,
current proprioception path, action head, preprocessing, eight-action chunk,
and two cameras as SAVR2. No training, weight, checkpoint, prompt, upstream,
dataset, or model change is allowed.

Start from `savr2-b15` with these exact thresholds:

- image: `full_image = 0.3203492696658941`, `wrist_image = 0.375`;
- state: `translation = 1.0480906581712492`,
  `orientation = 0.42270480698181334`, `gripper = 1.8`;
- action: `translation = 0.9280092130749463`,
  `rotation = 0.7812834808536592`, `gripper = 1.797821144380998`;
- hard episode-prefix skip cap: `0.15`.

Retain all SAVR2 rules: minimum query index `5`, two stable fresh queries,
maximum one consecutive reuse, independent local camera vetoes, grouped
state/action vetoes, gripper-transition vetoes, cache identity, and fail-closed
invalid-input behavior.

Add exactly one action rule. For each translation dimension `0-2`, take its
mean raw command across the newest and preceding `8 x 7` action chunks. Force
refresh if the product of either pair of means is strictly below zero. A zero
mean is not a reversal. Log all three reversal flags.

## 3. Development split and non-reuse of evidence

- Design split already used: tasks `0-9`, initial states `0-2`, seed `0`.
- Fresh SAVR3 validation: tasks `0-9`, initial states `3-9`, seed `0` — exactly
  70 fixed episodes and seven episodes per task.
- States `0-2` may not be rerun for SAVR3 outcome validation.
- States `3-9` have appeared under earlier policies, so this is
  policy-specific fresh validation rather than a statistically pristine final
  test.
- No threshold, rule, pairing, success gate, or skip gate may change after the
  first SAVR3 outcome is observed.

## 4. Implementation and correctness gate

SAVR2 behavior and records must remain unchanged. Before rollout:

- add a distinct `SAVR3` policy identity;
- unit-test each reversal dimension, exact zero, veto behavior, no-veto reuse,
  configuration validation, logging, and runner invariants;
- retain exact cache/component accounting and immutable query records;
- pass the complete CPU test, Ruff, mypy, byte-compilation, and diff suites;
- perform at most ten real-model queries and zero simulator episodes only if
  needed to reconfirm the shared adapter boundary.

Optional real-model correctness work is limited to one responsibly selected
GPU, 45 minutes, and 256 MiB. Failure stops before rollout.

## 5. Frozen positive gate

Run `savr3-rv-w375-b15` once on all 70 fixed validation episodes. A positive
method result requires every condition:

- 70/70 terminal episode records reconcile;
- 70/70 episodes succeed and each task succeeds 7/7;
- aggregate online visual-feature skip rate is at least 5%;
- every reuse reduces both vision-backbone and projector calls by exactly one;
- no technical, cache, component, counter, schema, or instrumentation error;
- all temporal and episode-prefix-budget invariants reconcile.

If every condition passes, stop immediately before matched baselines,
additional confirmation, or final-holdout work and request user approval.

If any condition fails, SAVR3 is a frozen negative result. Do not rerun failed
episodes, tune the threshold, relax the gate, search another local variant, or
inspect the final holdout.

## 6. Execution limits and records

- one responsibly selected idle GPU and one model process;
- at most 12 GPU-hours and 1 GiB of new result artifacts;
- no downloads, training, upstream edits, or final-holdout access;
- save every query decision, signal, counter, action hash, component count,
  terminal episode record, attempt record, progress record, manifest, and
  summary;
- restore protected checkpoint metadata and confirm aggregate GPU state after
  execution.
