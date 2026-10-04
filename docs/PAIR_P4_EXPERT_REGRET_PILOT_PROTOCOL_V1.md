# PAIR P4 Expert-Regret Pilot Protocol V1

## 1. Purpose and authorization

P4 is the bounded offline reliability-risk pilot specified by the accepted
PAIR protocol. It asks whether regret caused by the exact P3R cache
intervention is numerically identifiable, stable, cumulative, and predictable
from deployment-available provenance features. It is not a simulator-success
experiment and does not establish formal robot safety.

The user authorized P4 on 2026-08-29. This authorization permits one frozen
pilot, one GPU/process, original sequential LIBERO demonstration actions, and
the training/calibration splits only. It does not permit locked-test labels,
simulator outcomes, P5, retries, downloads, population changes, or relaxed
gates.

## 2. Authenticated preconditions

- P0-P2 passed and their revisions, schemas, splits, features, router, labels,
  and gates remain authoritative.
- P3R passed all four timing points. The selected physical mechanism is
  `D62_BAL_PT1`, with horizon 4 as its best measured point.
- P3R analysis SHA-256:
  `6f1bad7752403884a880438a24ea77af6702bac48bb75a74c99ceb105907ebda`.
- Original sequential demonstrations and P1 query/split indexes must match
  their authenticated SHA-256 identifiers.
- The compatibility runtime, checkpoint, repository revisions, project-local
  LIBERO configuration, and strict 23,552-MiB memory boundary are unchanged.

## 3. Protected populations

P4 may open only demonstration expert actions from trajectories already marked
`train` or `calibration`. It must not open:

- `locked_test` expert actions or regret labels;
- terminal success, rewards, simulator states used as outcomes, or episodes;
- development or confirmatory success populations; or
- future observations/actions as router inputs.

Expert chunks are labels only. They are never router features. Calibration
labels may be analyzed only after all expected compact records and a completed,
immutable worker summary exist. Router fitting and seed/proxy selection use
training trajectories only; the selected pilot artifact is frozen before
calibration labels are read by the analyzer.

The sole pre-run exception is the expert gripper coordinate needed to apply the
already-frozen transition stratum. The CPU scheduler may derive only the
boolean transition status defined below; it may not expose or summarize expert
action values, dense/cache actions, or regret.

## 4. Deterministic population and schedule

Sampling seed is `20260829`. The schedule contains exactly 400 unique anchors:
10 per task across all 40 tasks and four suites.

For canonical task index `t`, slots 0--9 have this fixed design:

| Slot | Base branch | Horizon |
|---:|---|---:|
| 0 | all-fresh control | 1 |
| 1 | all-fresh control | 4 |
| 2 | atomic intervention | 1 |
| 3 | atomic intervention | 2 |
| 4 | atomic intervention | 4 |
| 5 | D62 profile contract | 1 |
| 6 | D62 profile contract | 2 |
| 7 | D62 profile contract | 4 |
| 8 | D62 profile contract | 2 |
| 9 | D62 profile contract | 4 |

The two calibration slots for task `t` are `t mod 10` and `(t+5) mod 10`;
the other eight slots use training trajectories. This gives exactly 320 train
and 80 calibration anchors while preserving the 20% control, 30% atomic, and
50% structured-contract allocation within each split in aggregate.

Each slot uses a different trajectory. Candidate trajectories and aligned query
starts are SHA-256 ordered from the dataset revision, suite, task, split, slot,
and seed. A window is eligible only if the anchor plus every future query and
all eight expert actions per query exist at exact eight-action spacing.

Desired gripper-transition status alternates by `(t+slot) mod 2`. Transition is
defined from the P1-normalized expert gripper coordinate: sign-binarize each of
the eight coordinates and mark a contract transition if the sign changes
within any of the `h` evaluated future chunks or relative to the immediately
preceding action when one exists. The scheduler first selects an eligible query
matching the desired status; a deterministic opposite-status fill is allowed
only when no match exists and must be reported.

Atomic identity is SHA-256 selected from the 128 frozen
camera/tile/onset-layer groups, with a balance audit over camera and onset
layer. Structured contracts use only `D62_BAL_PT1` and the exact P3R vectorized
actual-source scorer.

## 5. Executed branches and exact accounting

Every anchor is prepared once per chronological query. One dense anchor query
initializes the cache. For each of the `h` future queries, one same-stack dense
action and one base-branch action are generated.

- Control anchors execute an empty/all-fresh mask.
- Atomic anchors execute their frozen single group recursively.
- Structured anchors execute the exact P3R D62 vectorized profile recursively.
- Every structured anchor additionally executes two complementary,
  hash-determined partitions of the full branch's realized group schedule from
  a clean clone of the anchor cache. These two diagnostics measure interaction
  without selecting on regret.
- Exactly 12 atomic and 20 structured base branches are rerun from the clean
  anchor cache with the identical realized masks. These 32 repeats equal 10%
  of the 320 non-control anchors. Repeats are fixed by canonical task/slot, not
  by observed labels.

The schedule implies:

- 400 anchors;
- 80 controls, 120 atomic bases, and 200 structured bases;
- 400 complementary structured diagnostics;
- 32 exact repeats;
- 2,120 compact intervention-query records;
- 832 branch/contract summaries;
- 800 base non-control feature/label rows;
- 3,520 scheduled action-prediction calls;
- 4 dense warm-ups and 4 technical controls;
- exactly 3,528 planned calls, with a hard cap of 4,000.

No call may be replaced, added, deleted, or retried after labels open.

## 6. Intervention semantics

Each query is prepared from its current primary image, wrist image,
proprioception, and instruction. The anchor dense cache is cloned for every
branch. Masks are suffix-safe, tile-aligned, nested over onset layers 2, 6, 9,
and 11, and resolved against each branch's exact recursive source ledger.

For a fixed arbitrary group set, ordered token positions are canonical by
onset layer, camera, and tile. The cumulative active token count at each onset
layer divided by the final active count forms the cache schedule. An empty set
is all-fresh. The full D62 branch uses the P3R vectorized scorer; its exact
legacy equivalence is rechecked on bounded technical controls before labels.

Complementary diagnostics partition the full branch's realized groups by a
SHA-256 ordering. The same per-query group sequences are replayed from separate
clean anchor-cache clones. Exact repeats replay the identical group sequence.
No diagnostic or repeat mask depends on regret or action values.

## 7. Labels and compact records

The expert target is the exact P1-normalized 8x7 action chunk. For every future
query, compute:

`signed regret = L1(expert, branch) - L1(expert, dense)`.

Also compute positive regret, dense/branch action distortion, maximum absolute
distortion, and continuous gripper-coordinate distortion. Contract labels are
the arithmetic mean across the `h` future queries (`beta=1/h`). Incomplete
chunks are impossible by construction. Raw expert, dense, and branch action
arrays are used only in memory and are not persisted; compact hashes and scalar
diagnostics are retained.

Outcome-protected intervention records continue to satisfy
`schemas/pair/intervention_record.schema.json`. Separate authenticated feature
and contract schemas join through record/contract identifiers and may not
contain terminal outcomes or raw action arrays.

## 8. Deployment-available features

Feature construction uses only values available before the cache decision:

- exact source query/age, camera, tile, onset layer, horizon, and D62 identity;
- current-versus-actual-source raw and projected tile changes;
- current/source normalized proprioception;
- previous branch-produced action summaries and their source differences;
- retained anchor salience from the P3R contract start;
- previous gripper transition, remaining horizon, source-mixture count; and
- a fixed 16-dimensional Rademacher projection of the mean instruction
  embedding, generated with seed `20260828` and scale `1/sqrt(4096)`.

The retained anchor salience matches the timed P3R mechanism; P4 does not add a
new per-query attention sidecar. Expert/future actions, task outcomes, current
dense outputs, and hidden dense passes are forbidden features.

Zero-cost proxies are frozen as source age, raw-image change, projected-token
change, proprioception change, previous-action change, retained salience, and a
seeded random ranking. Proxy orientation and the strongest proxy are selected
using training trajectories only.

## 9. Pilot router and unblinding order

The P0 DeepSets router, optimizer, loss weights, seeds 17/29/43, parameter cap,
and deterministic settings remain unchanged. Training labels are standardized
from training trajectories only. Atomic group loss is applied only where an
atomic label exists; structured rows use contract, quantile, and distortion
losses without inventing per-group labels.

Training trajectories receive an internal SHA-256 80/20 fit/validation split.
The seed with lowest frozen validation composite is selected; within 1%, the
lower seed wins. Proxy identity/orientation is selected on the same internal
validation population. The chosen router, standardization, support envelope,
and proxy definition are written immutably before the P4 analyzer reads any
calibration label.

At deployment-like evaluation, per-query router risk is averaged across a
contract to match the frozen `beta=1/h` label. Risk serving uses the lowest-risk
70% within each horizon, so router and proxy have identical horizon counts and
physical budgets.

## 10. Frozen P4 gates

P4 passes only if every global technical/resource/protection gate passes and:

1. exact-repeat absolute regret noise has median at most `1e-5` and p95 at most
   `1e-4`;
2. all-fresh action maximum absolute difference is at most `2e-4` and absolute
   regret is at most `1e-5`;
3. non-control signed-regret IQR exceeds twice repeat-noise p95 and `1e-5`;
4. positive-regret prevalence is between 5% and 80%;
5. no record contributes more than 20% of positive CVaR90 mass;
6. on calibration structured contracts, source-aware predicted versus observed
   signed contract regret has Spearman point estimate at least 0.30 and a
   one-sided 90% trajectory-bootstrap lower bound above 0.15;
7. at matched 70% within-horizon service, the source-aware router reduces
   calibration positive-regret CVaR90 by at least 15% versus the training-locked
   strongest proxy, with a positive one-sided 90% bootstrap lower bound; and
8. on horizon-2/4 structured contracts, source-aware CVaR90 improvement is at
   least 10%, while the median absolute complementary-partition interaction
   residual is at least twice repeat-noise median.

Spearman and CVaR use 10,000 bootstrap replicates and seed `20260830`, resampling
whole calibration trajectories. A degenerate resample is conservatively
scored as failure. Gates are conjunctive screening rules, not paper claims.

If only cumulative structure fails while all other gates pass, stop as
`blocked_authority_single_query`; otherwise any failed gate is
`scientific_stop`. No label redefinition or post-hoc threshold change is
allowed.

## 11. Resource and safety boundary

- one aggregate-idle GPU and one model process;
- strict selected-GPU memory below 23,552 MiB;
- at most 4,000 action-prediction calls and 8 wall hours;
- at most 4 GiB compact artifacts;
- no network, download, simulator, terminal outcome, locked-test label, sudo,
  system change, unrelated process inspection, or write outside
  `/home/ved/SAVR`;
- no automatic retry.

GPU selection uses only aggregate index/memory/utilization telemetry. A
technical failure preserves all evidence and stops. Completion requires exact
record/call counts, an immutable worker summary, router training, frozen
analysis, hash reconciliation, local synchronization, and a stop before P5.
