# PAIR P4B Independent Confirmatory Reliability Protocol V1

## 1. Decision and purpose

P4B is a proposed one-shot, independent development confirmation of the two
uncertainty gates that stopped P4. It is not an automatic retry, router
redesign, or post-hoc relaxation. P4B asks whether the already-frozen P4 router
reliably ranks and avoids high-regret physical cache contracts on calibration
trajectories that were never sampled by P4, across the same 40 fixed LIBERO
tasks. The P1 `locked_test` split remains sealed for P5.

P4B exists because P4 produced favorable point estimates but insufficient
one-sided uncertainty:

- structured calibration Spearman was 0.3452, with a one-sided 90% lower bound
  of 0.1260 versus the required value greater than 0.15;
- matched-service positive-regret CVaR90 improvement was 21.48%, with a
  one-sided 90% lower bound of exactly 0 versus the required value greater
  than 0; and
- the other eight frozen P4 gates passed.

The method therefore remains scientifically plausible but statistically
fragile. P4B is intentionally larger and independent. It uses a stricter
one-sided 95% confirmatory confidence level, preserves the P4 point thresholds,
and consumes the remaining calibration confirmation population only once.

This document freezes the proposed design. It does **not** authorize schedule
generation, expert-label access, model/GPU execution, or P5. Execution requires
separate explicit user approval after implementation and preflight review.

## 2. Claim boundary

If P4B passes, it supports only this conditional claim:

> On the pinned OpenVLA-OFT checkpoint and the fixed 40-task LIBERO benchmark,
> the frozen PAIR risk router predicts structured cache regret and improves
> the positive-regret tail at matched 70% service relative to the frozen
> strongest deployment-available proxy.

It does not establish formal robot safety, unseen-task generalization,
closed-loop task success, another backbone, or superiority over unavailable
methods. Simulator success remains sealed until a later separately authorized
phase.

## 3. Authenticated inherited evidence

P4B inherits without modification:

- the P0 method, feature, label, source-provenance, and router architecture;
- the P3R physical profile `D62_BAL_PT1` and its measured timing evidence;
- the P4 expert normalization, signed-regret definition, contract reduction,
  source-aware features, 70% service target, and strongest-proxy comparison;
- P4 router artifact
  `76a28416f57e94a86511f003772c5887417a3914218fd416e7bfb580b6f14519`;
- P4 router checkpoint SHA-256
  `ec1f83ea76c6977e2401db212e8e5373966ee491eea660d5017ec7ff030b435f`;
- training seed 43; and
- strongest proxy `projected_change`, orientation `-1`.

No P4B result may select a new seed, retrain or calibrate the router, change
feature standardization, reverse the proxy, alter service, choose a different
physical profile, or redefine regret.

## 4. Why a larger confirmation is justified

The design audit used only completed P4 evidence, the authenticated P4 schedule,
and P1 split metadata. It did not open any unused calibration or locked expert
action, observation, regret, or model output.

An empirical suite-by-horizon-stratified resampling audit of the 40 P4
structured calibration contracts estimated the following plug-in one-sided
95% lower bounds under P4B's balanced selection rule:

| Structured contracts | Spearman lower | CVaR90-improvement lower |
|---:|---:|---:|
| 48 | 0.1224 | 0.36% |
| 84 | 0.1803 | 3.79% |
| 120 | 0.2092 | 8.52% |
| 156 | 0.2238 | 8.07% |
| 204 | 0.2425 | 12.31% |
| 240 | 0.2503 | 13.41% |

At 240 contracts, none of the 20,000 empirical CVaR resamples were
nonpositive. The design-aligned P4 plug-in CVaR90 improvement was 20.43%,
close to the original P4 estimator's 21.48% despite the stricter balancing.
A Fisher-z planning approximation gives approximately 200 contracts for 90%
power to place a one-sided 95% Spearman lower bound above 0.15 if the true
correlation remains 0.345. If the true correlation is only 0.30, approximately
250 contracts would be needed for 80% power. Thus 240 is near the maximum
useful design supported by the untouched data; it is well powered under the P4
point estimate but not guaranteed under a materially smaller effect.

These calculations are planning evidence, not confirmatory results. They are
optimistic if the unused calibration trajectories shift or within-task
dependence is stronger than expected. P4B therefore uses all 240 untouched
calibration trajectories, reports a task-cluster sensitivity analysis, and
permits no second development confirmation.

## 5. Independent population

The authenticated P1 index contains exactly 2,000 trajectories across 40
tasks. Every task has 35 train, 8 calibration, and 7 `locked_test`
trajectories. P4 used exactly 8 train and 2 calibration trajectories per task.
After excluding all 400 P4 trajectory identities, every task has exactly 27
unused train, 6 unused calibration, and 7 untouched locked trajectories.

P4B uses:

- all 240 unused calibration trajectories as structured `D62_BAL_PT1`
  contracts, six per task;
- 40 unused training trajectories as horizon-4 all-fresh controls, one per
  task; and
- no P4-used or `locked_test` trajectory.

The six structured contracts per task contain exactly two at each horizon
1, 2, and 4. This gives 80 structured contracts per horizon and 20 per
suite-by-horizon cell. Each trajectory contributes only one anchor, preventing
overlap between base contracts.

The 240-trajectory confirmation population is consumed on the first
label-bearing P4B attempt. Any incomplete or technically invalid attempt stops
fail-closed and cannot be restarted without a new versioned protocol that
explicitly treats this population as opened. There is no replacement
development population. All 280 locked trajectories remain unopened for the
single P5 offline-test evaluation.

## 6. Outcome-blind schedule construction

Schedule seed is `20260901`. Canonical task order is inherited from P4. The
scheduler first authenticates and excludes every trajectory ID in the completed
P4 schedule. Within each task, the remaining six calibration trajectories are
SHA-256 ordered from dataset revision, suite, task, original trajectory
identity, and seed. The unused training control is the first SHA-256-ranked
eligible training trajectory after the same exclusion.

Slots are fixed before any action value is opened:

| Slot | Branch | Horizon |
|---:|---|---:|
| 0 | all-fresh control from unused train | 4 |
| 1 | structured D62 | 1 |
| 2 | structured D62 | 1 |
| 3 | structured D62 | 2 |
| 4 | structured D62 | 2 |
| 5 | structured D62 | 4 |
| 6 | structured D62 | 4 |

For each assigned trajectory, eligible eight-step-aligned starts are those
with an anchor observation, every future query, and every complete eight-action
expert chunk required by the slot horizon. Eligible starts are SHA-256 ordered
using only identities, chronology, and the seed. The first is selected.

The scheduler may inspect HDF5 dataset names and shapes but must not read action
values, gripper transitions, observations, rewards, success, or regret.
Gripper transition is not a sampling stratum in P4B. It is a prespecified
secondary subgroup computed only after the completed worker summary exists.
The scheduler must verify zero overlap with P4 and zero `locked_test` use.

## 7. Branch execution and accounting

Every anchor is prepared once per query. One dense anchor query initializes a
clean cache. At each future query, one same-stack dense action and one base
branch action are generated. Structured masks use the exact P3R vectorized
actual-source scorer and frozen anchor salience. Controls use an empty mask.

Exactly 24 structured contracts are exact repeats from clean anchor-cache
clones. Repeat selection is label-free and balanced as two contracts per
suite-by-horizon cell, giving eight repeats per horizon. Repeats replay the
identical realized group sequence. P4B has no complementary partitions because
P4 already passed the interaction gate and the confirmatory estimands concern
the frozen full contract.

Exact accounting is:

- 280 unique anchors: 240 unused calibration plus 40 unused train;
- 40 horizon-4 all-fresh controls;
- 240 structured bases: 80 each at horizons 1, 2, and 4;
- 24 exact repeats: 8 each at horizons 1, 2, and 4;
- 776 compact intervention-query records;
- 560 deployment-feature records;
- 304 contract summaries;
- 1,776 scheduled model calls;
- 4 dense warm-ups and 4 protected technical controls; and
- 1,784 planned calls under a hard cap of 1,900.

No call, anchor, mask, or record may be added, deleted, replaced, or retried
after the P4B expert labels open.

## 8. Protected execution and unblinding order

Before the worker opens any P4B expert action, it must pass:

1. exact protocol/config/schedule/input hashes;
2. all 280 unique trajectory identities, complete windows, zero P4 overlap,
   and zero locked-test use;
3. one GPU/process and compatibility-runtime checks;
4. sidecar off/on cache parity;
5. dense versus empty-mask all-fresh parity; and
6. exact legacy/vectorized D62 mask equivalence.

During execution, only process health, record counts, bytes, elapsed time, and
aggregate selected-GPU telemetry may be monitored. Partial regrets, actions,
contract summaries, calibration values, or aggregate scientific results may
not be inspected.

Scientific analysis is permitted only after an immutable completed worker
summary verifies exactly 776 intervention records, 560 feature records, 304
contract records, and 1,784 model calls. The analyzer then verifies the P4
router and proxy hashes before opening any P4B regret record. Locked-test
actions and labels remain inaccessible throughout P4B.

Raw expert, dense, and cached action arrays remain in memory only. Persisted
records contain only hashes and scalar diagnostics. No terminal-success or
reward label, simulator output, or locked-test value is accessed.

## 9. Primary confirmatory estimands

The confirmatory population is the 240 structured base contracts only.

### 9.1 Predictive ranking

For each contract, average the frozen router's per-query signed-regret
prediction to match the arithmetic-mean contract label. The first primary
estimand is pooled Spearman correlation between predicted and observed signed
contract regret across all 240 contracts.

### 9.2 Matched-service tail reduction

Within each of the 12 suite-by-horizon cells, serve the 14/20 contracts with
lowest predicted router risk. Apply the frozen proxy independently to select
14/20 in the same cell. Both methods therefore serve exactly 168/240 contracts
(70%) with identical suite and horizon composition.

The second primary estimand is:

`1 - CVaR90(router-served positive regret) / CVaR90(proxy-served positive regret)`.

Positive regret is `max(0, signed contract regret)`. CVaR90 is the arithmetic
mean of the largest ceiling-10% values. No threshold is tuned on P4B.

## 10. Confirmatory inference and multiplicity

Primary uncertainty uses 20,000 bootstrap replicates and seed `20260902`.
The resampling unit is the complete trajectory, stratified within each of the
12 suite-by-horizon cells so every replicate preserves 20 contracts per cell.
All queries and groups belonging to a selected trajectory remain together.

The confirmatory confidence level is one-sided 95%. Degenerate Spearman or
zero-proxy-tail replicates receive the conservative value `-1`.

The two primary null hypotheses form an intersection-union decision: P4B
passes only when both primary gates pass. Because both must be rejected, no
alpha splitting is used. P4B-only results are primary. Any P4/P4B combined
estimate is secondary and cannot rescue a failed P4B gate.

A task-cluster bootstrap that resamples ten tasks within each suite and keeps
all six structured contracts of each selected task is reported as a
sensitivity analysis. It is not substituted for the frozen primary result.

## 11. Frozen gates

P4B passes only if every gate below passes:

1. exact-repeat absolute signed-regret difference has median at most `1e-5`
   and p95 at most `1e-4`;
2. all-fresh maximum absolute action distortion is at most `2e-4` and absolute
   signed contract regret is at most `1e-5`;
3. all 12 suite-by-horizon cells contain exactly 20 structured contracts and
   both router/proxy serve exactly 14 per cell;
4. no one record contributes more than 20% of served positive-regret CVaR90;
5. pooled structured Spearman point estimate is at least 0.30 and its
   one-sided 95% trajectory-bootstrap lower bound is greater than 0.15;
6. matched-service CVaR90 improvement point estimate is at least 15% and its
   one-sided 95% trajectory-bootstrap lower bound is greater than 0;
7. horizon-2/4 matched-service CVaR90 improvement is at least 10%; and
8. every technical, identity, count, memory, protection, and artifact gate
   passes.

Per-suite correlations, task-cluster bounds, gripper-transition subgroups,
P4/P4B meta-analysis, q50/q90 calibration, and horizon-specific results are
secondary diagnostics. They must be reported but cannot change eligibility.

## 12. Resource and server boundary

- one aggregate-idle GPU and one model process;
- strict selected-GPU memory below 23,552 MiB;
- 1,784 planned calls, hard cap 1,900;
- eight wall hours and 4 GiB compact artifacts;
- zero downloads, simulator outcomes, terminal success fields, network use,
  sudo, system changes, unrelated process inspection, or writes outside
  `/home/ved/SAVR`;
- no automatic retry.

GPU selection may use only aggregate index, memory, and utilization telemetry.
Any mismatch, OOM, nonfinite value, incomplete record, or resource breach
preserves evidence and stops.

## 13. Decision rule and terminal outcomes

- **Pass:** all gates pass. P4B becomes independent reliability evidence.
  P5 remains blocked until separately approved.
- **Scientific stop:** any primary or scientific gate fails. PAIR stops; there
  is no P4C, router redesign on P4B data, population substitution, or gate
  change.
- **Technical stop before P4B labels open:** preserve evidence; a correction
  may be proposed only if it changes no scientific setting.
- **Technical stop after P4B labels open:** the remaining calibration
  confirmation population is spent; preserve evidence and stop for advisor
  review. No automatic rerun. The locked-test split remains sealed.

Even a P4B pass is not yet a positive-results paper. It would justify P5, which
must scale/freeze the primary router and evaluate once on the still-unopened
locked offline test before any closed-loop stage. Task success and
complete-cycle efficiency must then pass their own independently frozen gates.
