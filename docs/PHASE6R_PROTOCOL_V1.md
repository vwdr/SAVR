# SAVR 2.0 Phase 6R Protocol

Status: FROZEN BEFORE SAVR 2.0 IMPLEMENTATION AND OUTCOME COLLECTION

Protocol version: 1.0

Freeze date: 2026-07-31

Parent protocol: `docs/SAVR_EXECUTION_PROTOCOL.md`

## 1. Purpose and claim boundary

This protocol implements and calibrates a safety-constrained SAVR redesign
after the visible negative Phase 6 result. It governs Phase 6R-C through
Phase 6R-E.

Phase 6R remains development/calibration work. It cannot establish final
non-inferiority, speedup, superiority, or cross-suite generalization. The
original Phase 6 artifacts remain immutable and are not replaced or relabeled.

The final holdout remains untouched:

- initial-state IDs `10-49`;
- seeds `7`, `17`, and `27`;
- every outcome reserved by the future Phase 7 protocol.

## 2. Frozen assets and exclusions

- base policy: OpenVLA-OFT;
- checkpoint: `openvla-7b-oft-libero-four-suite`;
- checkpoint revision: `638918f3d1c2e43a39a8a20772bdb8b91835e4b7`;
- OpenVLA-OFT revision: `e4287e94541f459edc4feabc4e181f537cd569a8`;
- LIBERO revision: `8f1084e3132a39270c3a13ebe37270a43ece2a01`;
- cache boundary: projected visual patch embeddings before current
  proprioception;
- action chunk: eight continuous actions;
- environment: validated project-local `envs/openvla-oft`.

No training, weight change, new model/checkpoint, dataset download, action-head
change, prompt change, preprocessing change, upstream source edit,
token-level KV cache, multi-GPU execution, or final-holdout access is allowed.
SAVR 1.0 remains available unchanged.

The official VLA-Cache external run remains technically excluded at pinned
revision `a4909880573868dee2769343d52e793c0341678b`. A corrected reimplementation
would not be labeled an official result and is outside this protocol.

## 3. SAVR 2.0 visual signal

Use both `full_image` and `wrist_image`.

1. Normalize pixels to `[0,1]` and produce the existing deterministic
   `32 x 32` representation.
2. Compare each camera with its corresponding image from the most recent
   actual refresh.
3. Divide each representation into an `8 x 8` grid of non-overlapping
   `4 x 4` spatial patches.
4. Compute mean absolute pixel change for every patch.
5. Define each camera's local score as the mean of its four largest patch
   scores.
6. Compare each camera with its own threshold. Either camera exceeding its
   threshold forces refresh.

Log the 64 patch scores, top-four local score, global mean score, and threshold
for each camera. Camera scores are never averaged to approve reuse.

## 4. Grouped state signal

Normalize the current and previous-query eight-dimensional LIBERO state with
the checkpoint's `q01/q99` statistics. Compute independent RMS changes for:

- translation: indices `0-2`;
- orientation: indices `3-5`;
- gripper joints: indices `6-7`.

Each group has an independent threshold. Any group above its threshold forces
refresh. Missing, non-finite, clipped-shape-incompatible, or invalid state data
forces refresh.

## 5. Grouped action signal and transition veto

Use the two most recent completed `8 x 7` predicted action chunks. Normalize
with checkpoint action `q01/q99` statistics and compute independent RMS change
for:

- translation action dimensions `0-2` across all eight actions;
- rotation action dimensions `3-5` across all eight actions;
- gripper action dimension `6` across all eight actions.

Any group above its threshold forces refresh.

Additionally binarize raw gripper commands at `0.5`. Force refresh when:

- the newest chunk contains both open and closed commands; or
- the final gripper command differs between the two latest chunks.

Log, but do not use as a primary gate in Version 1, per-axis translation
direction reversals. This prevents an uncalibrated fixed reversal magnitude
from entering the primary controller.

Insufficient or invalid action history forces refresh.

## 6. Temporal safety rules

All rules fail toward refresh.

- minimum eligible query index: `5` (zero-based);
- maximum consecutive reuses: `1`;
- required stable fresh queries before reuse: `2`;
- after any reuse, reset the stable-fresh counter to zero;
- a fresh query counts as stable only when every visual/state/action score is
  valid and below its reuse threshold and no gripper-transition veto is active;
- episode/task/checkpoint/configuration change clears every history and cache;
- empty or incompatible cache forces refresh.

These rules guarantee at least two completed fresh queries between reused
queries. They do not modify the eight-action execution chunk.

## 7. Online skip-budget cap

Each candidate has an episode-local maximum skip fraction `b`. Before approving
the current query as a reuse, require:

`(completed_reuses + 1) / (query_index + 1) <= b`.

The denominator includes the current query and uses one-based query count.
This is a hard prefix cap, not a desired average. It prevents closed-loop
trajectories from overshooting the declared budget.

Frozen candidate caps:

- `savr2-b05`: `b = 0.05`;
- `savr2-b10`: `b = 0.10`;
- `savr2-b15`: `b = 0.15`.

## 8. Threshold derivation

Use only the existing 100 Phase 6 FR traces from LIBERO-Spatial tasks `0-9`,
initial states `0-9`, seed `0`.

For each score family, construct adjacent-query empirical distributions. For
the shared quantile grid `q = 0.000, 0.001, ..., 1.000`, set every visual,
state, and action threshold to `0.90` times its score family's linear empirical
quantile at `q`. The `0.90` multiplier is the frozen safety margin.

Replay the exact SAVR 2.0 temporal and budget semantics. For each budget cap,
choose the quantile with simulated skip rate closest to but never above the
cap. Ties prefer:

1. lower simulated skip;
2. lower quantile;
3. lexicographically smaller configuration identifier.

Offline replay selects candidates only. It is not treated as an online budget,
success, or safety estimate.

## 9. Phase 6R-C implementation and correctness

Implement SAVR 2.0 as a separate controller and signal path. Required checks:

- all SAVR 1.0 tests continue to pass unchanged;
- independent camera veto and patch aggregation are exact;
- state/action group indexing and normalization are exact;
- gripper-transition veto is exact;
- query warm-up, stable-fresh counter, isolated reuse, and budget prefix cap
  are exact at boundary values;
- invalid input and incompatible cache fail toward refresh;
- immutable query records contain all new scores/counters;
- recovery/resume preserves configuration and counter identity;
- wrapped FR actions remain identical to unmodified OpenVLA-OFT;
- a real SAVR 2.0 reuse invokes zero visual-backbone/projector calls and uses
  current proprioception.

Real-model bound:

- one responsibly selected GPU;
- at most 20 model queries and zero rollout episodes;
- at most 45 minutes and 256 MiB of new artifacts;
- no downloads or upstream edits.

Phase 6R-C stops on any failed correctness invariant.

## 10. Phase 6R-D staged online calibration

Population remains the already-used development split:

- suite: `libero_spatial`;
- tasks: `0-9`;
- initial states: `0-9`;
- seed: `0`;
- pairing key: `(suite, task, initial_state, seed)`.

### Stage 1 safety screen

Run every candidate on initial states `0-2`: 30 episodes per candidate, 90
maximum episodes total.

A candidate advances only if:

- 30/30 terminal records reconcile;
- success is 30/30;
- no infrastructure/instrumentation error occurs;
- observed skip rate is at least 2%;
- every cache/component/counter invariant reconciles.

All failures remain visible. Non-advancing candidates are never rerun with
changed thresholds.

### Stage 2 full development calibration

For every advancing candidate, run initial states `3-9`, adding 70 episodes
and producing the full fixed 100-pair result. At most three candidates advance,
so Stage 2 contains at most 210 new episodes.

A candidate is eligible only if:

- all 100 terminal records reconcile;
- at least 98/100 episodes succeed, preserving the frozen `-2` percentage-point
  observed calibration margin relative to 100/100 FR;
- every task succeeds on at least 8/10 states;
- observed aggregate skip rate is at least 5%;
- no infrastructure/instrumentation error occurs.

Among eligible candidates select, in order:

1. highest observed skip rate;
2. highest success count;
3. lowest mean synchronized policy-query wall time;
4. lower budget cap;
5. lexicographically smaller identifier.

If none is eligible, Phase 6R stops. Thresholds, margins, stage populations,
and promotion rules are not relaxed.

Phase 6R-D resource bound:

- one responsibly selected GPU;
- 16 GPU-hours;
- 1 GiB new result artifacts;
- one model process;
- no downloads, training, upstream edits, or holdout access.

Save raw downsampled observations/state/actions for Stage 1. Stage 2 saves all
scores/counters and at most ten bounded diagnostic failure videos.

## 11. Phase 6R-E comparisons and ablations

Run Phase 6R-E only if Phase 6R-D selects an eligible SAVR 2.0 candidate.

### Matched visual-only baseline

VOR2 uses the same local per-camera visual signal, minimum query index,
isolated-reuse rule, stable-fresh rule, and online budget cap, but ignores all
state/action signals and the gripper-transition veto. Derive its camera
thresholds from the existing FR traces. Permit at most three predeclared
threshold attempts to achieve observed skip within 2 absolute percentage
points of selected SAVR 2.0. Preserve every attempt.

### Periodic baseline

Keep the manuscript-defined PR policy. Evaluate integer periods `1-32` against
the FR query lengths and select the nearest expected skip budget; ties favor
more refresh. Because low skip rates may be unreachable with integer periodic
refresh, label the result `nearest-budget` unless observed skip is within 2
percentage points.

### Required ablations

Run 100 paired episodes for:

- image + action, without state;
- image + state, without action/gripper veto;
- global two-camera mean image + full state/action safety;

All ablations retain the selected temporal rules and online budget cap. They
are development evidence, not final confirmation.

### Practical-efficiency gate

Before Phase 7, selected SAVR 2.0 must show:

- at least 5% skipped visual refreshes;
- exact corresponding reduction in visual-backbone/projector calls;
- at least 0.5% reduction in mean synchronized policy-query wall time versus
  FR, confirmed by fixed-trace timing that removes trajectory-length effects;
- non-negative lower bound for the paired bootstrap sensitivity of the timing
  difference, or an explicit reason the latency claim is narrowed to compute
  reduction only.

SAVR 2.0 must be Pareto-nondominated by matched VOR2 on calibration success and
skip rate. Support for the state/action claim additionally requires SAVR 2.0
to outperform at least one matched ablation in success at comparable skip.
Otherwise H1/H2 may proceed but H3 is marked unsupported and the manuscript
claim/title must be reconsidered.

## 12. Power planning

Use the selected 100 paired FR/SAVR outcomes and the original one-sided paired
non-inferiority design:

- alpha: `0.025`;
- power: `90%`;
- margin: `0.02` absolute success probability;
- planning alternative: zero difference;
- discordance planning value: maximum of observed discordance, its two-sided
  95% Wilson upper bound, and `0.01`.

Use the original paired risk-difference approximation and report 80%/90%
sensitivity over discordance `0.01-0.10`. If required final resources exceed
the parent scope, stop before Phase 7 rather than weakening the margin or
power.

## 13. Phase 6R-E resources

- one responsibly selected GPU;
- 20 GPU-hours;
- 1 GiB new result artifacts;
- one model process;
- no downloads, training, upstream edits, or holdout access.

## 14. Integrity and stop rules

Every run records code/config/protocol hashes, checkpoint and upstream
revisions, GPU identity, timestamps, pairing keys, component counts, timings,
and terminal status. Resume skips only validated terminal records with an
identical configuration hash.

Stop immediately for:

- a write outside `/home/ved/SAVR`;
- more than one selected GPU;
- dirty or changed protected assets;
- configuration/schema/counter/component mismatch;
- non-finite decision data;
- three infrastructure failures in one setting;
- resource-cap breach;
- final-holdout execution or inspection.

Ordinary unsuccessful task episodes are scientific outcomes and never trigger
silent deletion or favorable reruns.

## 15. Phase 6R exit

Phase 6R exits positive-development only if one SAVR 2.0 configuration passes
the success and practical-efficiency gates, matched comparisons reconcile, and
power/resources are feasible. This permits preparation of Phase 7; it does not
authorize or imply a final positive paper result.

Otherwise Phase 6R stops with all evidence visible.
