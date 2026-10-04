# One independent empirical validation — design V1

2026-09-26. **PROPOSED DESIGN, NOT EXECUTABLE/FROZEN.** No GPU launch from this
document. User approved preparing the empirical-paper route. This specifies
the proposed scope and concrete remaining readiness work, not another method.

## 1. Purpose and endpoints

Estimate the success–latency tradeoff of the existing four policies on new
initial conditions, without training, changing selectors or tuning budgets.
Primary comparisons: each compressed arm versus dense. Matched-retention
fixed384 versus adaptive384 and fixed384 versus adaptive256 are secondary.
Every arm stays in the report regardless of ranking.

Primary robot endpoint: paired terminal-success difference, averaged equally
over the forty benchmark tasks. Report each success count and all paired
discordances, not only aggregate percentages. Primary deployment-cost endpoint:
mean complete-query latency within each episode, then equally averaged over
conditions/tasks, with algorithm startup INCLUDED. Also report query-weighted
latency, episode wall time, query count and per-suite results.

These live policies encounter different observations. Their rollout costs
measure deployment behavior, not isolated same-input kernel speed. Retain a
separately labelled controlled matched-input timing test to measure that latter
quantity. Do not combine its synthetic weighting with natural-rollout results.

No noninferiority margin is introduced merely to accommodate previous losses.
This is an estimation study, not a promise to pass an unchanged-success gate.
Report the uncertainty and observed quality cost. Do not equate a confidence
interval containing zero with equality, or select the winning comparison after
the run. If simultaneous significance claims are made across three primary
contrasts, the final analyzer must prespecify multiplicity control.

## 2. Population and exposure audit

Candidate pool: official state IDs 10–49 for each of forty tasks from the four
existing suites. All IDs 0–9 are development, even where a particular method did
not use them. New IDs mean new initial conditions, NOT unseen training tasks.
Locked offline demonstration labels must remain closed.

Read-only inventory completed locally and on TITAN on 2026-09-26:

- All 95 JSON configuration files match by SHA-256 across hosts.
- 65 local result directories and 114 remote result directories were inventoried
  by NAME ONLY. Many additional remote directories are historical/preflight
  runs; a directory count is not an exposure count.
- Explicit numeric references to IDs >=10 occur in three ACR configurations,
  all beneath `protected_populations`, not an executed population specification.
- 77 configurations have no explicit numeric state IDs in the recognized keys.
  Some inherit a manifest or use older runner defaults. Therefore this scan
  does NOT certify that the reserved states have never been executed.

Audit evidence: `reports/CONFIRMATION_METADATA_INVENTORY_V1.json` and
`scripts/audit_confirmation_metadata.py`. No outcome, image, feature, prediction
or locked-demonstration file was read by this inventory. It cannot detect an
unlogged external run, and that limitation must remain explicit.

Before release: resolve actual episode-producing historical launch records to
their state manifests or runner defaults, including remote-only run families.
Classify offline replay/tensor tests separately. Record per-family metadata paths
and hashes. Flag any unresolved simulator run; do not silently assume it used
0–9. If the pool is contaminated, redesign BEFORE opening any new outcomes.

## 3. Proposed fixed size and sampling

Propose **400 matched conditions: ten state IDs per task, forty tasks**. Four
arms give 1600 primary episodes. This is an effort/precision compromise, not a
universal publication requirement or guaranteed smallest sufficient sample.

After exposure clearance, sample ten IDs without replacement independently
within each task from 10–49 using a pinned deterministic sampling implementation
and seed 20260926. Publish the entire selected manifest before running; no
selection by observed images, dense success, trajectory length or pilot failures.
Use simulator seed 7 as before, paired across arms; record unavoidable numerical
variation. One evaluation seed does not establish seed robustness.

Planning sensitivity (two-sided 95% normal approximation, independent paired
conditions, difference near zero, ignoring finite-population correction):

| Matched conditions | Discordance 5% | Discordance 10% | Discordance 20% |
|---:|---:|---:|---:|
| 200 | 3.10 pp half-width | 4.38 pp | 6.20 pp |
| 400 | 2.19 pp | 3.10 pp | 4.38 pp |
| 800 | 1.55 pp | 2.19 pp | 3.10 pp |

Calculation: D=success_compressed−success_dense ∈{−1,0,1}, paired discordance q;
Var(D)=q−E[D]^2, approximate half-width 1.96 sqrt(q/N). This is planning
sensitivity, NOT the final interval or a power calculation for noninferiority.
The forty-task structure and heterogeneous failures can widen uncertainty.
400 conditions are intended to distinguish large tradeoffs, not reliably prove
one-percentage-point equivalence. 200 gives coarse uncertainty even under the
moderate-discordance assumption; 800 doubles the primary work for smaller gains.
Do not expand N after viewing the results to obtain significance.

Final analysis must distinguish inference over the fixed forty-task benchmark
from inference over a hypothetical new-task population. Implement and test a
paired, stratified analysis for the sampled states, with task-level sensitivity.
Specify the exact CI algorithm before launch and test zero discordance, all
successes, all failures, suite imbalance and small strata. Bootstrap [0,0] in a
zero-discordance sample must NOT be reported as proof of population equality.
If the intended precision cannot be supported by the chosen analysis, revise
this design prospectively; do not describe this draft as statistically frozen.

## 4. Execution design and existing-code reuse

Keep weights, precision, crop, proprioception, action normalization/readout,
positions, eight-step queue and four policies identical to S1 v03. Preserve all
old frozen files. Implement a separate confirmation contract/config/output path;
do not patch old hard-coded forty-condition schedules in place.

Plan two predefined sequential sessions on the same selected GPU, each with
five sampled states per task, 800 primary episodes and eight native controls
(one before/after per suite). Total: 1616 episodes, including sixteen controls.
Both sessions are one fixed study, not outcome-dependent follow-up attempts.
Balance arm positions within task/session and distribute suites throughout the
schedule. Reset simulator seed/state and adaptive history for every arm/episode.
No online parameter changes or selective restart of failed episodes.

At each session, repeat the existing controlled 240-call timing schedule
(16 excluded hardware-warmup calls +224 measured calls) using the same already
exposed timing inputs. Total 480 timing records. This checks session repeatability
but remains a seven-query, two-frame microbenchmark, not natural replay. Do not
call the repeated inputs independently sampled observations or use 480 as the
uncertainty sample size. Native controls verify dense/adaptive-zero command
parity in-session; keep their extra shadow work out of primary timing estimates.

Record per-step executed chunks, retained-token counts, observation hashes,
query count and timing boundaries; preserve the current queue checks. Store no
new raw imagery by default. Record per-arm allocator memory only if measurement
is qualified; otherwise make no method-specific memory claim. FLOPs need an
explicit accounting model, not a token-count percentage presented as measured
compute. Neither extra metric is a reason to modify a completed run retrospectively.

## 5. Time and resource estimate — not yet launch caps

Observed S1 elapsed time was 5688.63 s for 168 episodes, 3728 calls and controls.
Straight tenfold scaling is about 15.8 GPU-hours, not another two-hour run.
Allow roughly **16–20 hours** as an initial planning estimate, dependent on
trajectory lengths and shared-machine conditions; not an execution guarantee.

Worst-case primary calls from suite horizons (220/280/300/520) and chunk length8:
100 conditions/suite ×4 arms ×(28+35+38+65) =66400. Sixteen native controls with
three calls per query contribute at most1992; timing contributes480: subtotal
68872 calls. This excludes any extra qualification calls, which must be separately
enumerated before freeze. A provisional 70000-call ceiling has only1128 spare
calls; never silently spend them on extra scientific evaluation.

Candidate aggregate bounds: two sessions of at most24h each, one GPU0 with the
previous UUID, <=23552 MiB aggregate memory, <=2 GiB new artifacts. Validate these
against the new exact schedule and measured record sizes before freezing. Obtain
a fresh scoped GPU identity/idle check for each authorized launch; no alternate
GPU, unrelated process inspection, downloads, installations or cleanup.

The two-session design makes thermal/session drift visible. It does not provide
cross-hardware replication or guarantee resource availability. If the resource
caps or elapsed time expire, preserve partial evidence and stop. No automatic retry.

## 6. Readiness, blinding and stop conditions

Before dispatch, require all of:

1. Exposure ledger resolved, exact state manifest generated and hashes pinned.
2. New runner/config/analyzer/independent verifier reviewed and CPU-tested;
   old evidence hashes unchanged. Exact total counts and controls reconciled.
3. Statistical estimator/intervals and multiplicity policy fixed, synthetic
   boundary cases tested, claim scope checked against achievable precision.
4. Complete source/model/config hashes, accounting, resource caps and scheduler
   order frozen. No use of old S1 development gates as confirmatory tests.
5. Coordinated GPU0 availability checked; no overlapping owned worker.

While either session is active, inspect only count-only progress, bytes,
owned process health and aggregate selected-GPU telemetry. Do not read success
or timing outcomes between sessions. Use completed immutable summaries and
technical health only to advance the already fixed schedule.

Any technical stop: preserve evidence, inspect only technical diagnostics,
stop without retry and report. Poor task success is an outcome, not a technical
excuse to rerun. After both sessions complete, verify hashes/counts/caps first,
then analyze CPU-only and independently reconcile every arm. No outcome-dependent
sample expansion, hidden comparator removal or automatic new training.

## 7. Deliverable and decision

Produce one validation report with all estimates, uncertainty and limitations,
then populate the paper blueprint. If independent acceleration persists, report
it with its actual success cost; if not, say so. The empirical paper's contribution
is the comparative evidence, not ownership of an established algorithm.

Current status: evidence blueprint and candidate validation design completed;
exposure certification, final statistical implementation and executable freeze
remain. No confirmation launched, no new model trained, no email or GitHub push.
