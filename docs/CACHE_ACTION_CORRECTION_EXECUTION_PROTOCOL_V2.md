# Cache Action Correction Execution Protocol V2

**Protocol version:** 2.0  
**Date:** 2026-08-30  
**Status:** ACCEPTED; Phase C0 complete; stop before C1  
**Accepted by user:** 2026-08-30  
**Phase C0 authorized:** 2026-08-30  
**Phase C0 Recovery 01 authorized:** 2026-08-31  
**Method:** Cache Action Correction (CAC)  
**Supersedes:** `docs/CACHE_ACTION_CORRECTION_EXECUTION_PROTOCOL_V1.md`  
**Companion audit:** `docs/CACHE_ACTION_CORRECTION_FINAL_PREEXECUTION_AUDIT_2026-08-30.md`

## 1. Purpose and authorization

This protocol tests whether the action-relevant effect of recursively mixed-age
visual K/V reuse can be repaired by a small frozen-backbone residual corrector,
without reconstructing the fresh K/V cache.

It freezes the question, method family, data ledger, comparator family,
statistics, resource limits, phase sequence, recovery rules, and claim boundary.
Its purpose is to give CAC a fair test while preventing outcome-driven redesign,
split leakage, underpowered confirmation, hidden dense fallback, and efficiency
claims based only on theoretical compute.

Creating or approving this document authorizes **no experiment**. Each phase
requires separate user approval. No phase approval authorizes a later phase.

## 2. Authority and immutable workflow

Precedence is:

1. system, developer, and current user instructions;
2. repository `AGENTS.md`;
3. this protocol;
4. the phase configuration and immutable manifest;
5. authenticated project evidence; and
6. conversation summaries.

At every phase start:

1. read this protocol, `AGENTS.md`, `PROJECT_STATUS.md`, `docs/DECISIONS.md`,
   the companion audit, and the latest phase report;
2. reconcile local and TITAN revisions without overwriting unrelated changes;
3. verify the preceding exit gate and the exact user authorization;
4. authenticate all code, configuration, data, model, and evidence hashes;
5. verify that later protected values remain inaccessible;
6. record only the authorized phase as `IN_PROGRESS`; and
7. stop on ambiguity.

Only reconciled, hashed evidence establishes completion.

## 3. TITAN and repository boundary

- Connect only through `ssh titan`.
- The only writable server path is `/home/ved/SAVR`.
- Never inspect, modify, move, delete, reprioritize, or attach to unrelated
  university files, jobs, users, services, processes, or allocations.
- Never use `sudo` or make system-wide changes.
- Use at most one user-coordinated GPU and one live VLA process.
- GPU selection may use only aggregate utilization/memory telemetry; do not
  inspect process identities.
- Store all environments, records, checkpoints, and logs under
  `/home/ved/SAVR`.
- No new checkpoint, dataset, model, or large dependency may be downloaded
  without a separate estimate and approval.
- Never commit credentials, datasets, raw images, large feature stores, model
  weights, or raw caches.
- Every report must confirm whether anything outside `/home/ved/SAVR` changed.
  The expected answer is no.

## 4. Scientific question and estimand

### 4.1 Primary question

For the pinned two-camera OpenVLA-OFT checkpoint on the 40-task LIBERO
evaluation, can CAC recover the task reliability lost by the fixed
`D62_BAL_PT1` mixed-age cache while retaining a real complete-cycle latency
benefit?

### 4.2 Mechanistic hypothesis

Mixed-age K/V corruption is high-dimensional, but its effect on the served
`8 x 7` action chunk may be predictable in a lower-dimensional action space
from:

- current projected visual evidence;
- exact layer/camera/tile source deltas and ages;
- the corrupted action-head representation and cached base action;
- current proprioception and instruction identity; and
- the fixed reuse/reset state.

CAC predicts the dense-policy action difference. It does not reconstruct K/V
tensors, choose tokens, change the cache mask, or learn a fallback policy.

### 4.3 Confirmatory estimand

The primary reliability estimand is the equally task-weighted paired terminal
success difference over all 40 tasks and official initial-state IDs `10-49`,
using the identical fixed simulator seed and initialization for each policy.
The finite evaluated population contains exactly 1,600 paired conditions.

The primary efficiency estimand is the complete model-query cycle from prepared
model input through normalized action-chunk availability, including cache
selection, source resolution, VLA execution, CAC, synchronization, and required
data movement. Simulator stepping and one-time model loading are reported
separately, not included in query latency.

### 4.4 Permitted final claim

Only if every confirmatory gate passes:

> On one pinned OpenVLA-OFT checkpoint and the evaluated 40-task LIBERO
> population, a compact cache-aware action corrector repaired the reliability
> loss of a fixed recursively mixed-age visual K/V reuse policy without fresh
> K/V reconstruction, while preserving a measured complete-cycle latency
> reduction.

### 4.5 Prohibited claims

Do not claim formal robot safety, universal VLA generalization, state or cache
reconstruction, superiority to unreproduced methods, end-to-end acceleration
from FLOPs alone, or closed-loop reliability from offline action error.

## 5. Literature and novelty boundary

Phase C0 must repeat a primary-source search through its execution date. It must
compare mechanism, inputs, target, training distribution, cache semantics,
timing boundary, and closed-loop evaluation for at least:

- OpenVLA and OpenVLA-OFT;
- VLA-Cache, LAC, AC2-VLA, Action-JND, and Gated VLA-Cache;
- Latent Bridge and CloudEdgeVLA;
- A2C2 / Leave No Observation Behind and Action ControlNet;
- ActionCache, ViTaR, and FiberTune; and
- newer cache repair, VLA residual correction, action distillation, or
  frozen-backbone adaptation work.

CAC's narrow intended contribution is the combination of:

1. correction after same-query selective layer/tile mixed-age K/V reuse;
2. exact recursive source provenance for every reused group;
3. direct prediction of the dense action effect rather than a reuse decision or
   full K/V delta;
4. use of current visual tokens already computed by the accelerated path; and
5. one controlled demonstration-to-on-policy dense-teacher aggregation round.

Residual correction itself is not novel. A2C2 corrects stale chunks, Latent
Bridge predicts feature/K/V deltas, and Action-JND/LAC learn safer compression.
If a source found in C0 contains all five CAC elements on a materially identical
problem, stop before implementation for advisor review.

## 6. Authenticated inherited evidence

C0 must re-authenticate rather than merely quote:

- repository, OpenVLA-OFT, LIBERO, environment, checkpoint, tokenizer,
  normalization-statistics, and action-processing revisions;
- the P1 demonstration index and split manifest;
- the exact `D62_BAL_PT1` schedule, dynamic mask logic, source tracker, age-four
  reset, and P3R physical-timing evidence;
- all historical simulator population use after the earlier ACR ledger; and
- the fact that no state ID `10-49` or P1 locked-test demonstration has been
  opened by later work.

Known evidence at protocol creation:

- projected vision/proprioception input: `1 x 513 x 4096`;
- cached action hidden states: `1 x 56 x 4096`;
- action chunk: `8 x 7` normalized coordinates;
- action queue: all eight actions are executed before the next query;
- source provenance: onset layer, camera, tile, source query, and age;
- onset layers: `(2, 6, 9, 11)`;
- P3R `D62_BAL_PT1`, horizon four: 24.59% raw complete-cycle saving,
  24.40% one-sided lower bound, 17.05% conservative net lower bound,
  0.78% scorer overhead, and 18,261 MiB peak aggregate GPU memory;
- P1: 2,000 demonstrations, with 1,400 train, 320 development-calibration,
  and 280 locked-test trajectories;
- the 320 P1 calibration trajectories were already consumed by PAIR P4/P4B
  and are **development evidence**, not untouched confirmation;
- the 280 locked-test trajectories remain sealed; and
- every LIBERO task has exactly 50 official initial states.

Any identity conflict stops C0. Earlier labels such as “calibration” do not
override the actual access ledger.

## 7. Fixed cache substrate

The only permitted substrate is the exact `D62_BAL_PT1` dynamic profile with:

- original two-camera ordering;
- exact vectorized actual-source scoring;
- onset layers `(2, 6, 9, 11)`;
- the authenticated dynamic tile mask and previous-salience rule;
- current proprioception at every query;
- a complete dense reset after at most four cached query intervals;
- unchanged prompt, preprocessing, positional semantics, action head,
  unnormalization, gripper processing, and eight-action queue; and
- no learned selector, outcome-aware fallback, threshold retuning, or hidden
  dense call.

“Fixed profile” means the selection algorithm and budgets are fixed; the actual
mask remains observation-dependent. A different profile requires a new
protocol version.

## 8. Formal CAC representation

Let query `q`, onset layer `l`, camera `c`, and 4x4 spatial tile `g` identify a
reuse group. Each tile contains 16 projected patch tokens.

### 8.1 Current and source visual tokens

For projected patch features `P` of width 4096:

```text
v[q,c,g]       = mean of the 16 current projected patch tokens in tile g
vsrc[q,l,c,g]  = mean of the 16 projected patch tokens from the exact physical
                  source query used by group (l,c,g)
delta[q,l,c,g] = v[q,c,g] - vsrc[q,l,c,g]
```

The representation therefore contains 32 current tile tokens and 128
layer-resolved source-delta tokens. A single source per tile is prohibited
because physical sources differ across onset layers.

Mean tile pooling is frozen because reuse itself is decided at the same tile
granularity and because retaining all five sets of 512 full-width patch tokens
would exceed the bounded C3 storage design. C1 must nevertheless test whether
tile construction, camera ordering, and source alignment are exact.

`delta` is an input-level projected-feature difference tied to the true source
identity; it is not the actual downstream K/V error. CAC relies on `Z_C` to
summarize the corrupted downstream computation and makes no K/V reconstruction
claim.

### 8.2 Corrupted action representation

The 56 cached-path hidden states are **not pooled**. They are reshaped exactly as
the frozen L1 action head does:

```text
H_C: 1 x 56 x 4096 -> 1 x 8 x (7*4096)
Z_C = layer_norm2(residual_MLP(relu(fc1(layer_norm1(H_C)))))
A_C = fc2(Z_C)
```

`Z_C` is the existing frozen action-head penultimate representation of shape
`1 x 8 x 4096`. It is captured before `fc2`, without CPU transfer or duplicate
action-head execution. C1 must prove `fc2(Z_C)` reproduces `A_C` within the
frozen numerical tolerance. Pooling the seven coordinate-token states is
prohibited because the action head preserves their ordered concatenation.

### 8.3 Other deployment inputs

- `A_C`: normalized cached base action, `8 x 7`;
- `s_q`: current normalized proprioception;
- `e_l`: mean frozen input embedding of instruction-only tokenizer positions,
  excluding template, special, padding, and action-placeholder tokens;
- exact reused/fresh bit, source age `0-4`, onset-layer ID, camera ID, tile row,
  tile column, and reset-phase ID for every source-delta token; and
- action-step index `0-7`.

Future observations, rewards, success labels, expert outcomes, dense hidden
states, dense actions, and later actions are prohibited deployment inputs.

### 8.4 Full adapter

The full candidate uses width 256 and at most 8 million trainable parameters:

1. shared linear projection for current and source-delta 4096D tokens;
2. a separate shared projection for the eight `Z_C` tokens;
3. learned type, layer, camera, tile, age, reset-phase, and action-step
   embeddings;
4. eight action-step query tokens formed from projected `Z_C`, `A_C`, step ID,
   proprioception, and instruction embedding;
5. two pre-normalized cross-attention blocks in which only the eight action
   queries attend to the visual/provenance context; and
6. a zero-initialized `256 -> 7` residual output per action step.

No quadratic self-attention over all context tokens is permitted. C0 must
compute the exact parameter count and operation estimate from the final module
definition before C1.

Ridge and MLP candidates encode the same permitted information through frozen
action-step/layer/camera-stratified mean, maximum, and norm summaries rather than token
attention. If either is selected under Section 11, it becomes the final CAC
architecture and every comparator/ablation is instantiated in that same class.
“Full CAC” always means the selected class with all permitted inputs, not
necessarily the cross-attention class.

### 8.5 Output and support guard

For coordinate `j` and step `k`:

```text
r[k,j]     = b[k,j] * tanh(raw_residual[k,j])
A_pre[k,j] = A_C[k,j] + r[k,j]
A_hat[k,j] = project_to_training_support(A_pre[k,j])
```

`b` is the training-only 99th percentile of absolute dense-minus-cache
residual, computed separately by step and coordinate with a prespecified floor.
The final support interval is the training-only `[0.1%, 99.9%]` dense normalized
action interval per step/coordinate, expanded by 10% of its interquantile width.
Both are frozen before development-calibration opens.

Every residual saturation and support projection is logged. Development
calibration fails if more than 1% of corrected motion coordinates or 2% of
gripper coordinates activate the support projection. No NaN/Inf is tolerated.

All-fresh/reset queries bypass CAC before residual computation and must be
bitwise identical to the dense baseline.

### 8.6 Motion and gripper objective

The label is always:

```text
DeltaA* = A_D - A_C
```

where `A_D` is the isolated dense policy on the identical current observation.
Motion loss is Smooth L1 after training-only coordinate scaling. The gripper
head uses both continuous Smooth L1 and binary cross-entropy for the exact
downstream open/close threshold in normalized coordinates. The gripper loss
weight and threshold are frozen in C0. The eight action steps receive equal
weight because all are executed; first-step error is additionally gated and
reported.

Expert actions are diagnostic only and never replace or mix with the dense
preservation target.

## 9. Required comparators and mechanism controls

Every learned comparator is trained independently using identical split,
schedule, optimizer budget, checkpoint rule, residual bounds, and random seeds.
Missing inputs are replaced by learned constant tokens so parameter count and
network depth remain matched. Inference-time masking of a full-model checkpoint
does not count as an ablation.

Required policies/models:

1. dense OpenVLA-OFT;
2. uncorrected `D62_BAL_PT1`;
3. **generic current-observation residual:** `A_C`, `Z_C`, current tiles,
   proprioception, instruction, and action-step ID, but no source deltas or
   exact cache provenance;
4. **cache-context residual:** `A_C`, `Z_C`, proprioception, instruction, and
   provenance, but no current/source visual delta;
5. full CAC;
6. full CAC without current tile tokens;
7. full CAC without `Z_C` (base `A_C` remains the residual anchor);
8. full CAC without exact provenance; and
9. full CAC without layer-resolved source deltas.

The generic comparator is A2C2-inspired but is not presented as a reproduction:
it operates once per VLA query on the same current observation, whereas A2C2
performs within-chunk real-time correction.

C0 must document whether an official, compatible implementation of Gated
VLA-Cache, Action-JND, or LAC can be executed within the one-GPU/no-new-training
boundary. If not, compare conceptually and report their published results; do
not claim empirical superiority.

## 10. True data and population ledger

### 10.1 Demonstrations

Per task, the P1 split contains 35 train, 8 previously consumed
development-calibration, and 7 sealed locked-test demonstrations.

The 35 train trajectories are deterministically subdivided per task:

- `adapter_fit`: 24 trajectories;
- `architecture_selection`: 5 trajectories; and
- `checkpoint_validation`: 6 trajectories.

Complete trajectories are the split and resampling unit. No query, horizon,
tile, or action coordinate crosses trajectory partitions.

The 8 previously consumed calibration trajectories per task are renamed
`development_calibration`. They may gate development after the model is frozen,
but cannot support an “unseen” or independent final claim.

The 7 locked-test trajectories per task remain sealed until C7 and provide the
independent offline/mechanism confirmation.

### 10.2 Simulator identities

All state IDs `0-9` are treated as project-exposed development identities,
regardless of whether CAC itself used them. No later claim calls them untouched.

- `headroom_stage1`: IDs `0-2`, 3 conditions/task, 120 total;
- `headroom_extension`: IDs `3-5`, used only by the prespecified ambiguity
  rule, another 120 conditions;
- `R0_development`: IDs `0-5`, 6 conditions/task, 240 total;
- `R1_holdout`: IDs `6-9`, 4 conditions/task, 160 total; and
- `final_confirmation`: IDs `10-49`, seed 7, 40 conditions/task, 1,600 total.

Seeds 17 and 27 on the same initial states are reserves, not new independent
conditions. They may be used only under a future protocol that models repeated
seed/initial-state clustering. V2 does not use them.

### 10.3 C2 contract schedule

C2 contains exactly 3,600 cache contracts: 30 per task at each source horizon
1, 2, and 4. At every task/horizon, 24 contracts come from `adapter_fit` and 6
from `architecture_selection`, yielding 2,880 fit and 720 selection contracts.

It additionally contains:

- 240 all-fresh controls, two per task/horizon; and
- 360 exact repeats balanced by suite, horizon, and gripper-transition stratum.

At least eight primary contracts per task must be centered on or immediately
precede a demonstrated gripper transition where available; shortfalls are
reported before model execution. Selection uses only identity hashes and the
prespecified transition stratum, never residual magnitude.

Each contract begins from a clean authenticated anchor and reproduces the exact
recursive D62 chronology. Overlapping source windows are permitted only within
one split; all intervals and uncertainty remain trajectory-clustered.

C0 must derive the exact branch/model-call count from the frozen schedule. C2
may not exceed 24,000 VLA calls or 10 GiB of compact records.

### 10.4 C3 schedules

- Training: up to 26,000 balanced cache contracts from `adapter_fit` plus
  `architecture_selection`, with no `checkpoint_validation` trajectory.
- Checkpoint selection: only `checkpoint_validation`.
- Development calibration: a frozen 4,800-contract schedule, exactly 120 per
  task and balanced across horizons and transition status as availability
  permits, drawn from the 320 consumed calibration trajectories.
- Final locked offline test: a frozen 2,880-contract schedule, exactly 72 per
  task and 24 per horizon, drawn from all 280 locked-test trajectories.

C3 feature storage is capped at 50 GiB. C0 must provide exact byte estimates
from the schema and a streaming/memory-map training plan. The 7B model is
unloaded during adapter optimization.

## 11. Model screen and freezing

C2 evaluates exactly three prespecified candidates:

1. ridge on frozen pooled summaries;
2. a two-layer MLP with at most 2 million parameters; and
3. the full cross-attention adapter in Section 8.4.

All three are trained before `architecture_selection` values open. Each has one
fixed hyperparameter configuration and three fixed screening seeds declared in
C0. C2 uses the mean across those seeds to select the model class; it does not
select the final R0 seed. After C2 chooses a class, C3 retrains its three frozen
seeds on the expanded fit population and selects exactly one seed/checkpoint
using only `checkpoint_validation`. Development calibration is unavailable to
both selections.

Eligible models must first pass every C2 minimum. Among eligible models, select
the greatest composite repair score

```text
S = 0.35*R_mean + 0.25*R_first + 0.20*R_p95 + 0.20*R_gripper,
```

where every `R` is the held-out relative repair defined in Section 12.1. Apply
a one-standard-error rule favoring fewer parameters when scores are
statistically indistinguishable under the frozen trajectory bootstrap. The old
smallest-first rule is prohibited.

After selection, architecture, inputs, objective, bounds procedure,
initialization family, optimizer, schedule, seed count, and stopping rule freeze.
C3 may increase training data but may not redesign the model.

## 12. Metrics and statistical rules

### 12.1 Offline error

Training-only coordinate scales are used throughout.

- primary mean absolute standardized error over 56 coordinates;
- first-action standardized error;
- per-query 95th-percentile error;
- translation, rotation, and step-index error;
- gripper continuous error and downstream threshold disagreement;
- residual singular spectrum and 80/90/95/99% variance dimensions; and
- support-guard and residual-saturation frequency.

Relative repair is `(cache_error - corrected_error) / cache_error`. If the
reference error is at most `1e-8`, relative repair is undefined: report the
absolute difference, require non-worsening, and assign a zero contribution to
the model-selection score. The same rule applies to a zero baseline gripper-
disagreement rate. Queries are never treated as independent for uncertainty;
resample complete trajectories, stratified by task.

### 12.2 Closed-loop reliability

- paired terminal success difference CAC minus dense;
- paired terminal success difference CAC minus uncorrected cache;
- per-suite and per-task paired differences;
- invalid action, simulator exception, timeout, and forced reset counts; and
- episode length as a diagnostic, never a replacement for success.

The task is equally weighted: compute each task estimate first, then average 40
task estimates. Primary confidence intervals use a hierarchical paired
bootstrap: resample tasks, then paired initial states within task, with 20,000
replicates and a frozen seed. Report paired discordance counts and a paired
score/Newcombe sensitivity interval.

The point estimate exactly describes the enumerated 1,600-condition benchmark
population. Bootstrap intervals express sensitivity to task/state resampling
and support only the explicitly limited benchmark-family claim; they are not
presented as uncertainty about which outcomes occurred in the enumerated set.

### 12.3 Efficiency

- complete-cycle wall time;
- synchronized CUDA time;
- cache scorer/source-resolution time;
- adapter-only time;
- realized reuse/reset rate;
- peak allocated and reserved memory; and
- full episode wall time as a secondary systems measure.

Warmups, randomized/Latin-square arm ordering, synchronized boundaries, clock
source, repetitions, and no-outlier-deletion rule freeze in C0.

### 12.4 Multiplicity and test order

Final claims use a fixed-sequence gate at one-sided alpha 0.05:

1. dense non-inferiority;
2. improvement over uncorrected cache;
3. complete-cycle latency benefit; and
4. cache-specific mechanism evidence.

A later claim is tested only if all earlier claims pass. This fixed order
controls the familywise error for the ordered positive claim. All descriptive
intervals are still reported. G6's multiple ablations use Holm correction.

No ordinary optional stopping is allowed. The headroom screen is a declared
development feasibility rule, not confirmatory inference.

### 12.5 Power boundary

The final population cannot exceed 1,600 independent paired initial-state
conditions. C0 must publish a paired non-inferiority power table over discordance
rates 0.05, 0.10, 0.15, and 0.20, true CAC-dense differences 0, -0.01, and
-0.02, alpha 0.05, and target power 0.80.

C0 must also simulate the frozen hierarchical bootstrap under the historical
task-level dense-success distribution. If the analytic and hierarchical
calculations disagree, the lower power estimate governs feasibility reporting.

At true equality, 1,600 pairs provide approximately 80% power for a 2-point
non-inferiority margin only when paired discordance is about 10% or lower. This
limitation cannot be solved by pretending repeated simulator seeds are
independent. C5/C6 must report observed discordance to refine the feasibility
forecast without changing the final margin or population.

## 13. Development and confirmatory gates

### H — repair-opportunity gate

Before adapter training, run dense and uncorrected D62 on `headroom_stage1`.

- Proceed directly if dense success is at least 75%, D62 success is at least
  50%, dense minus D62 is 8-35 points, and at least two suites favor dense.
- Stop if dense success is below 75%, D62 success is below 50%, dense minus D62
  is at most 2 points, or the gap exceeds 35 points.
- Otherwise open the prespecified `headroom_extension`. On all 240 cumulative
  conditions, proceed only if dense is at least 75%, D62 is at least 50%, the
  gap is 5-35 points, and at least two suites favor dense.

This gate confirms that there is enough loss to repair but not an obviously
collapsed substrate. No CAC model or feature is tuned using terminal outcomes.

### P — C2 predictability gate

On `architecture_selection`, an eligible candidate must show:

- at least 15% mean-error repair;
- at least 10% first-action repair;
- at least 5% 95th-percentile repair;
- non-worsening gripper disagreement;
- no suite mean-error worsening greater than 5%;
- at least 5% mean-error advantage over both learned generic/cache-context
  controls; and
- predicted adapter overhead within the C4 budget.

The point estimate and trajectory-bootstrap interval are reported; selection is
development, not final inference.

### G1 — frozen R0 development calibration

On the 320 previously consumed development-calibration trajectories, frozen R0
must achieve:

- mean repair at least 25%, one-sided 95% lower bound at least 15%;
- first-action repair at least 20%, lower bound at least 10%;
- 95th-percentile repair at least 15%, lower bound above zero;
- gripper disagreement repair at least 20%, lower bound above zero;
- at least 10% mean-error advantage over generic and cache-context controls,
  with lower bounds above zero;
- no suite mean or first-action worsening greater than 2%;
- favorable mean direction in at least three suites; and
- support projection within Section 8.5 limits.

G1 is a development continuation gate, not independent paper evidence.

### G2 — physical integration

- observed complete-cycle latency reduction versus dense at least 5%;
- one-sided 95% lower bound above zero;
- adapter overhead no more than 35% of gross D62 saving;
- peak aggregate memory strictly below 23,552 MiB;
- exact dense/reset bypass;
- no hidden dense call, second encoder, provenance mismatch, numerical failure,
  or changed reuse/reset behavior.

### D — development rollout gate

On a frozen R0 or R1 development population:

- observed success improvement over D62 at least 5 points;
- observed success loss versus dense no more than 2 points;
- no suite loss versus dense greater than 5 points;
- no invalid action or catastrophic task pattern; and
- G2 efficiency retained.

### G3 — final dense reliability

On all 1,600 final conditions:

- CAC minus dense observed success is at least -2 points; and
- the one-sided 95% hierarchical paired lower bound is above -2 points.

No suite may lose more than 5 observed points. No task may have zero CAC
successes when dense succeeds in at least half its conditions.

### G4 — final repair versus cache

- CAC minus D62 observed success is at least +5 points; and
- the one-sided 95% hierarchical paired lower bound is above zero.

### G5 — final physical benefit

- complete-cycle latency reduction versus dense at least 5%;
- one-sided 95% episode-clustered lower bound above zero;
- unchanged D62 reuse/reset semantics; and
- no hidden fallback.

### G6 — final mechanism support

On the sealed 280-trajectory locked offline population:

- full CAC has at least 10% lower mean error than both generic and
  cache-context controls, with Holm-adjusted favorable bounds;
- at least three of four independently trained ablations have at least 5%
  greater mean error than full CAC with Holm-adjusted favorable bounds;
- no ablation is more than 5% better than full CAC; and
- the full model retains the prespecified favorable direction in every suite.

Failure of G6 does not erase a reliability result, but prohibits a
cache-specific mechanism claim and therefore fails the full CAC positive-paper
claim under this protocol.

## 14. Phase C0 — final freeze and feasibility accounting

### Work

- perform the literature collision audit;
- authenticate every identity and historical population;
- verify the sealed locked-test and state `10-49` boundaries;
- freeze exact D62/reset semantics;
- generate all trajectory, contract, headroom, development, holdout, final,
  calibration, and locked-test manifests without opening outcomes;
- freeze the exact adapter module, parameter count, feature schema, gripper
  threshold, output bounds, loss, hyperparameters, seeds, and comparator models;
- publish the power table and fixed-sequence analysis code;
- derive exact model-call, episode, storage, wall-time, and memory budgets from
  prior measured rates;
- freeze technical-resume rules and output-sealing checks; and
- produce human- and machine-readable C0 reports.

### Exit

Every field above is explicit and hashed; C1/C2 compact storage is at most
10 GiB; C3 storage at most 50 GiB; the locked populations remain sealed; no GPU,
model, simulator, or protected outcome was accessed. Stop for C1 approval.

## 15. Phase C1 — tensor, chronology, and systems feasibility

### Work

- implement read-only hooks for current tiles, four-layer physical sources,
  `Z_C`, `A_C`, proprioception, instruction embedding, and provenance;
- implement isolated dense shadow and deployed cached branches from clean
  cache clones;
- prove the dense shadow cannot mutate cache, tracker, salience, RNG, or action
  queue state;
- verify source chronology, maximum age, reset phase, camera/tile ordering,
  normalization, dtypes, devices, and eight-action cadence;
- verify `fc2(Z_C)` reproduces the normal cached action;
- implement streaming schemas and asynchronous bounded host writes;
- add synthetic recursion, all-fresh, exact-repeat, cache-clone, exception,
  restart, and record-hash tests;
- run a bounded real-tensor smoke and gross-headroom measurement; and
- measure feature bytes, hook cost, peak memory, and predicted adapter cost.

### Caps and exit

- one GPU/process, at most 160 VLA calls, 4 GiB temporary evidence;
- no training or simulator outcomes;
- bitwise dense/reset bypass and exact masks/source IDs;
- numerical action tolerance at most `1e-6` where bitwise equality is not
  architecturally expected;
- zero nonfinite values;
- D62 gross complete-cycle headroom at least 8%;
- peak aggregate memory below 23,552 MiB with the adapter estimate included;
- C2/C3 storage and call estimates within Section 10.

Any failure requires a new protocol version or stops CAC. Stop for C1H
approval.

## 16. Phase C1H — terminal repair-opportunity screen

Run only dense and D62 using Section 10.2. Seal partial outcomes until each
stage is complete. Apply gate H mechanically. No CAC implementation choice may
use terminal outcomes. At most 480 episodes are permitted across both stages.

Failure means D62 is not an appropriate positive-solution substrate under V2.
Passing stops for C2 approval.

## 17. Phase C2 — train-only predictability and architecture selection

Execute the 4,200 frozen contracts, verify controls/repeats, compute the
residual spectrum, train all three candidates and matched controls, then open
only `architecture_selection`. Apply gate P and the one-standard-error selector.

During GPU collection, monitor only process health, record counts, bytes,
elapsed time, and aggregate selected-GPU telemetry. No partial residual or
ranking output may open before exact count/hash reconciliation.

Failure stops CAC without a fourth architecture. Passing freezes one model and
stops for C3 approval.

## 18. Phase C3 — scaled R0 training and development calibration

- collect the capped C3 fit records;
- unload the 7B model and train the selected full model and every required
  matched comparator/ablation;
- select checkpoints only on `checkpoint_validation`;
- freeze all checkpoints and hashes;
- open the previously consumed `development_calibration` schedule once;
- apply G1 and publish all suite/task/transition diagnostics.

No architecture, input, loss, bound, seed count, profile, or optimizer change
may follow calibration access. Failure stops. Passing stops for C4 approval.

## 19. Phase C4 — integration and physical timing

Integrate frozen R0 entirely on GPU after `Z_C/A_C` production and before
unnormalization. Validate output support, gripper processing, queue semantics,
reset bypass, tracker state, exception cleanup, and no extra encoder/dense call.

Use at least 12 randomized Latin-square paired timing blocks and at most 600 VLA
queries, with frozen warmups and no outlier deletion. Apply G2 once. Failure
stops; passing stops for C5 approval.

## 20. Phase C5 — R0 development rollouts and replay buffer

Use state IDs `0-5`. Reuse immutable dense/D62 headroom episodes only after a
frozen deterministic sentinel confirms identical environment behavior; run any
missing dense/D62 conditions and R0 on all 240 conditions. New C5 work is capped
at 480 episodes.

For each R0 query, store a server-only replay record containing exact current
camera observations or a lossless replayable equivalent, proprioception,
instruction, `A_C`, `Z_C`, compact visual/source/provenance inputs, reset state,
and hashes. Terminal outcomes are stored separately and never become features.

If R0 passes gate D, freeze it for C7 and skip C6. Otherwise R1 is eligible only
when R0 is no more than 10 points worse than D62, produces no invalid/catastrophic
pattern, retains G2, and the evidence is compatible with distribution shift.
Stop for C6 or C7 approval.

## 21. Phase C6 — one on-policy dense-teacher round

R1 is a single DAgger-style repair round, not a redesign.

- sample at most 20 query identities per C5 condition by a C0-frozen hash rule;
- use state IDs `0-4` for R1 fit and ID `5` only for action-level checkpoint
  validation;
- replay at most 4,800 records through an isolated dense teacher;
- label every record with `A_D - A_C`, never `A_D - A_R0`;
- initialize the same adapter from R0 and fine-tune with the C0-frozen 1:1
  task/horizon-matched mixture of demonstration and on-policy records;
- retain the same architecture, inputs, bounds, objective, profile, and reset;
- freeze one R1 checkpoint before state IDs `6-9` open; and
- evaluate dense, D62, and R1 once on the 160 R1-holdout conditions, capped at
  480 episodes.

R1 training is permitted only if sampled R0-state dense-action error degrades
at least 15% relative to task/horizon-matched development calibration in mean,
first-action, or gripper disagreement. Otherwise the proposed distribution gap
is unsupported and CAC stops.

R1 must pass gate D. If it fails, do not revert to R0 or train R2. If it passes,
R1 is final. Stop for C7 approval.

## 22. Phase C7 — independent confirmation

Before execution, authenticate the final checkpoint, code, profile, manifests,
analysis, power table, storage/time budget, and zero state-ID `10-49` or locked-
test access.

Run exactly dense, D62, and final CAC on all 1,600 final conditions: 4,800
primary episodes. No partial success values open before complete immutable
summaries, timing records, counts, exclusions, and hashes reconcile.

Then open the 2,880 locked offline contracts and evaluate the already frozen
full/comparator/ablation checkpoints. Apply G3-G6 in the fixed sequence. There
is no final rerun, task deletion, model reselection, threshold change, or second
aggregation round.

Outcomes are classified as:

- `positive_cac`: G3, G4, G5, and G6 pass;
- `reliability_only`: G3/G4 pass but G5 fails;
- `mechanism_ambiguous`: G3-G5 pass but G6 fails;
- `efficiency_only_negative`: G5 passes but G3 or G4 fails; or
- `negative`: required reliability gates fail.

## 23. Phase C8 — evidence and paper decision

Reconcile every gate, report every task/suite/exclusion/failure/interval, update
status and decisions, publish reviewable code/config/manifests, and sync approved
content to GitHub and the local review clone. Raw data and large records remain
off GitHub with checksums and regeneration instructions.

The manuscript claim follows the evidence classification. No rebranding or new
method begins inside V2.

## 24. Technical-stop, resume, and recovery rules

All schedules are append-only and unit-addressed. Each completed atomic unit has
a semantic hash and completion marker; outcome summaries remain sealed.

- A process restart with unchanged code/config/data may resume only missing
  units after verifying every completed unit. It is a resume, not a rerun.
- A code or environment change requires a versioned recovery report, expanded
  failing-path test, new hashes, and explicit approval.
- Completed units are never deleted, overwritten, or selectively repeated.
- A recovery is permitted only when the cause is technical, outcomes were not
  inspected, the scientific method/gates are unchanged, and the fix cannot
  depend on outcomes.
- After protected outcomes are opened, no repair/rerun is permitted under V2.
- Unfavorable scientific evidence is never labeled technical failure.

Technical failures include import/dependency failure, OOM, process termination,
schema/hash/count mismatch, nonfinite tensor, simulator crash, or exhausted
storage. No automatic retry is allowed.

## 25. Resource envelope

- one selected GPU and one VLA process;
- peak aggregate GPU memory strictly below 23,552 MiB;
- C1: 160 VLA calls, 4 GiB;
- C1H: at most 480 episodes;
- C2: 24,000 VLA calls, 10 GiB;
- C3 feature generation: 60,000 VLA calls, 50 GiB;
- C4: 600 VLA calls;
- C5: at most 480 new episodes;
- C6: 4,800 teacher calls and 480 episodes;
- C7: 4,800 primary episodes, 2,880 offline contracts, and at most 18,000 VLA
  calls for locked offline contract construction;
- no new large download; and
- no run begins without a measured wall-time forecast and sufficient project
  disk margin.

C0 may lower a cap based on exact manifests. It may not raise one without a new
protocol version and approval.

## 26. Evidence schema

Every phase records:

- protocol/config/code/data/model/environment hashes;
- exact population identities and prior-access state;
- selected GPU ID and non-identifying aggregate telemetry;
- command, start/end time, random seeds, counts, bytes, and peak memory;
- record and summary semantic hashes;
- every exclusion, technical stop, resume, and recovery;
- protected populations opened and still sealed;
- gate inputs and mechanical decision; and
- confirmation that nothing outside `/home/ved/SAVR` changed.

Every completed phase produces a Markdown report, machine-readable summary,
tests, and immutable evidence before any next-phase request.

## 27. Final readiness statement

CAC is technically feasible but not guaranteed to produce a positive result.
V2 deliberately moves the first terminal question—the existence of a repairable
D62 reliability gap—before expensive adapter training. It then separates
train-only prediction, previously consumed development calibration, physical
timing, exposed development rollouts, and genuinely locked confirmation.

Execution should begin only with C0. No GPU, simulator, training, protected
data, manuscript, or external publication is authorized by this document.
