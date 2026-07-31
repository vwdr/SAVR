# Phase 6 Calibration and Power Protocol

Status: FROZEN BEFORE PHASE 6 OUTCOME COLLECTION

Protocol date: 2026-07-29

Parent protocol: `docs/SAVR_EXECUTION_PROTOCOL.md`

## 1. Purpose and claim boundary

Phase 6 selects one primary configuration for FR, PR, VOR, and SAVR using only
the declared LIBERO-Spatial calibration split. It also estimates paired
discordance for the final power calculation.

Phase 6 is configuration selection, not final evaluation. Its outcomes must not
be described as paper-level confirmation of non-inferiority, speedup, or
superiority. No outcome from the final holdout may be inspected in this phase.

## 2. Frozen assets and integration

- base policy: OpenVLA-OFT
- checkpoint: `openvla-7b-oft-libero-four-suite`
- checkpoint revision: `638918f3d1c2e43a39a8a20772bdb8b91835e4b7`
- OpenVLA-OFT revision: `e4287e94541f459edc4feabc4e181f537cd569a8`
- LIBERO revision: `8f1084e3132a39270c3a13ebe37270a43ece2a01`
- action chunks: 8 actions
- images: third-person and wrist
- proprioception: current 8-dimensional LIBERO state
- visual cache boundary: projected visual patch embeddings before the current
  proprioception token
- environment: the validated project-local `envs/openvla-oft`

No dependency, checkpoint, dataset, policy-weight, action-head, prompt,
preprocessing, or simulator-setting change is permitted after collection
begins.

## 3. Calibration population and pairing

Every complete candidate setting uses exactly the same 100 paired episodes:

- suite: `libero_spatial`
- tasks: all task IDs `0-9`
- initial-state IDs: `0-9`
- seed: `0`
- one episode for every task/state pair
- unchanged official task limits and success definition

The pairing key is `(suite, task_id, initial_state_id, seed)`. Candidate order
is deterministically rotated across pairing keys to reduce order and thermal
bias. Episode identities, errors, and unsuccessful outcomes are retained.

The final holdout remains untouched:

- initial-state IDs `10-49`
- seeds `7`, `17`, and `27`
- every suite other than the declared Spatial calibration run

Loading task definitions is allowed; executing, aggregating, or inspecting
holdout outcomes is not.

## 4. Non-inferiority and calibration success rules

The frozen non-inferiority margin is an absolute success-rate decrease of
`2 percentage points` relative to paired FR:

`success(SAVR) - success(FR) > -0.02`.

This is the margin recommended by the parent protocol. The user's blanket
Phase 6 approval freezes it for calibration and power planning. It may not be
enlarged after observing calibration or final outcomes.

A SAVR candidate is calibration-eligible only when all conditions hold:

1. all 100 paired episodes have terminal, reconciled records;
2. its observed paired success-rate difference from FR is at least `-0.02`;
3. it has no runtime or instrumentation error;
4. no task has zero SAVR successes when paired FR has at least one success.

The 100-episode observed rule is a configuration-selection constraint, not a
confidence-bound non-inferiority conclusion. Formal inference is reserved for
the final holdout.

If no SAVR candidate is eligible, Phase 6 stops with a negative calibration
result. Thresholds, the margin, and the split are not relaxed.

## 5. Full Refresh signal collection

FR is the correctness oracle and refreshes on every policy query. The Phase 6
FR run records, for every query:

- pairing key and both policy-query/environment-step indices;
- 32-by-32 normalized representations for each camera;
- raw finite 8-dimensional robot state;
- predicted 8-by-7 action chunk;
- task success and terminal status;
- synchronized timing and visual-component counts;
- immutable record identity and checksums.

These traces initialize thresholds without executing the final holdout. Offline
replay must reproduce controller warm-up, most-recent-refresh image references,
state history, action history, cache age, and horizon behavior exactly.

FR collection must complete and reconcile before thresholds are derived.

## 6. SAVR candidate grid

The predeclared online grid contains nine settings:

- target skipped-refresh rates: `25%`, `50%`, `75%`;
- maximum reuse horizons in policy queries: `2`, `4`, `8`;
- 100 paired episodes per setting.

For each `(target, horizon)`, thresholds are derived only from the Phase 6 FR
traces:

1. Evaluate the fixed quantile grid `q = 0.000, 0.001, ..., 1.000`.
2. At each `q`, set the image, state, and action thresholds to their respective
   empirical linear-interpolation quantiles at `q`.
3. Replay each episode with the exact SAVR controller semantics. Image change
   is recomputed against the most recent simulated refresh; state change uses
   the previous query; action change uses the two preceding FR action chunks.
4. Select the `q` whose aggregate simulated skipped-refresh rate is closest to
   the target.
5. Resolve ties conservatively: prefer the lower skipped-refresh rate, then the
   lower `q`.

The resulting three thresholds are frozen before the corresponding online
setting runs. Online outcomes do not alter them.

The nine settings are executed in a deterministic rotated order over pairing
keys. A completed setting always contains all 100 pairings.

## 7. Primary SAVR selection rule

Among calibration-eligible SAVR settings, select in this order:

1. lowest observed refresh rate;
2. highest observed paired success difference from FR;
3. lowest median synchronized policy-query wall time;
4. smallest maximum reuse horizon;
5. lexicographically smaller frozen configuration identifier.

The rule is applied once after all nine settings reconcile. No post-hoc
threshold interpolation, favorable task removal, or manual trajectory choice
is allowed.

## 8. Matched-budget VOR and PR

The target budget is the selected SAVR setting's observed aggregate refresh
rate.

### VOR

- use the same maximum reuse horizon as selected SAVR;
- derive its image threshold from the FR traces with the same fixed quantile
  grid and exact most-recent-refresh replay;
- select the threshold whose replayed refresh rate is closest to the SAVR
  target, with conservative ties favoring more refresh;
- run 100 paired online episodes;
- budget matching succeeds when the observed VOR refresh rate is within
  `2 absolute percentage points` of SAVR.

If the first online VOR result misses tolerance, at most two additional
predeclared matching iterations may be run. Each iteration selects the unused
grid threshold whose replayed rate is next closest in the direction needed.
All attempts remain visible. The first attempt within tolerance is frozen. If
none match, freeze the closest attempt and label it `nearest-budget`, not
matched-budget.

### PR

PR remains the manuscript-defined fixed integer-period policy. Evaluate periods
`k = 1, ..., 8` against the reconciled FR query lengths and select the period
whose expected refresh rate is closest to SAVR. Ties favor more refresh and
then smaller `k`. Run that one period on the same 100 pairings.

Budget matching succeeds only if its observed refresh rate is within
`2 absolute percentage points` of SAVR. Because fixed integer periods have
discrete budgets, an unmatched result is retained and labeled
`nearest-budget`; the PR definition must not be changed to manufacture a
match.

FR's primary configuration is refresh on every query.

## 9. Paired power calculation

Let:

- `p10` be the proportion of pairs where FR succeeds and SAVR fails;
- `p01` be the proportion where FR fails and SAVR succeeds;
- `d = p01 - p10` be SAVR minus FR success;
- `pD = p10 + p01` be total discordance.

The final primary test is one-sided paired non-inferiority at `alpha = 0.025`
with margin `0.02`. Phase 6 computes the sample size needed for `90%` power
under the planning alternative `d = 0`, using the paired risk-difference normal
approximation:

`n = ceil(pD * (z_0.975 + z_0.90)^2 / 0.02^2)`.

To avoid an unstable zero-discordance estimate, the planning value of `pD` is
the larger of:

- observed paired discordance for selected SAVR;
- the two-sided 95% Wilson upper bound for discordance;
- `0.01`.

The calculation also reports 80% and 90% sensitivity over discordance values
from `0.01` through `0.10`, and checks the parent's planned `1,200` paired
episodes per policy per suite. The final protocol uses the larger of `1,200`
and the computed 90%-power requirement, rounded up to a complete balanced
task/seed block.

If the resulting final study exceeds the parent protocol's resource scope,
Phase 6 reports the estimate and stops for Phase 7 review; it does not weaken
the margin or power target.

Formal final analysis must additionally report task-stratified uncertainty and
a cluster/bootstrap sensitivity analysis. Calibration does not make a final
inferential claim.

## 10. Execution, recovery, and integrity

- one responsibly selected GPU only;
- aggregate utilization/memory checks only; no process or user inspection;
- all writes confined to `/home/ved/SAVR`;
- no `sudo`, training, downloads, system changes, or upstream edits;
- at most `48 GPU-hours` for Phase 6;
- at most `2 GiB` of new result artifacts;
- one model process at a time;
- atomic manifest updates and immutable terminal episode/query records;
- resume by skipping only validated terminal records with the same frozen
  configuration and code revision;
- preserve interrupted/error records and the original attempt;
- save no rollout video by default; bounded diagnostic video requires a
  documented reason;
- reject dirty source, checkpoint mismatch, revision drift, schema failure,
  non-finite data, or configuration drift before execution.

The runner stops immediately for:

- any write outside the project boundary;
- checkpoint or upstream integrity change;
- more than one visible/selected GPU;
- an artifact or GPU-time cap breach;
- three terminal infrastructure errors within one setting;
- any outcome from the final holdout being executed or inspected;
- an unreconciled record, counter, pairing, or controller invariant.

Ordinary task failures are scientific outcomes and do not trigger early
stopping.

## 11. Required checkpoints and status controls

The following checkpoints must be recorded:

1. protocol frozen and code revision recorded;
2. CPU tests and synthetic replay tests pass;
3. FR 100/100 records and signal traces reconcile;
4. nine SAVR settings each reconcile 100/100 pairings;
5. the primary SAVR rule is applied mechanically;
6. VOR and PR matching status is explicit;
7. paired power and resource projections are complete;
8. one configuration per method is frozen;
9. GitHub, TITAN, and the local review copy are synchronized;
10. Phase 6 stops before Phase 7.

Every long run must expose a machine-readable progress summary with expected,
complete, failed, remaining, elapsed, and estimated-remaining counts. A status
claim is valid only when checked against manifests and immutable records.

## 12. Required Phase 6 outputs

- frozen FR trace-collection configuration;
- frozen nine-setting SAVR grid;
- threshold-derivation artifact with input hashes;
- immutable 100-episode records for every completed setting;
- frozen VOR and PR configurations with budget-match status;
- paired outcome and discordance tables;
- reproducible power calculation and sensitivity table;
- resource, timing, integrity, and safety audit;
- `reports/PHASE6_CALIBRATION_REPORT.md`;
- updated `PROJECT_STATUS.md`, `docs/MILESTONES.md`, and
  `docs/DECISIONS.md`.

The report must explicitly state that calibration results are not final
evaluation and whether anything outside `/home/ved/SAVR` was modified.
