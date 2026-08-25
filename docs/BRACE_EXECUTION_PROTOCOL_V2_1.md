# BRACE Execution Protocol V2.1

**Method:** Branch-Rollout Adaptive Cache Execution (BRACE)

**Working title:** *Branch Before You Cache: Outcome-Supervised Cache Contracts for Reliable VLA Acceleration*

**Date frozen:** 2026-08-24

**Formal-method amendment:** 2026-08-25

**Status:** Red-team protocol and formal method specification prepared; no
BRACE implementation or result exists

**Supersedes:** `docs/BRACE_EXECUTION_PROTOCOL_V2.md` before implementation

**Formal implementation definition:**
`docs/BRACE_FORMAL_METHOD_SPECIFICATION_V1.md`

**Authority:** This document authorizes no download, model/checkpoint load, GPU
operation, policy outcome, protected-population access, or server change. B1 is
the only next eligible phase.

## 1. Research question and positive-result standard

Test whether a lightweight pre-inference controller can select a bounded,
clean-provenance visual-KV cache contract that improves the closed-loop
reliability--efficiency frontier of the pinned OpenVLA-OFT checkpoint on
LIBERO.

A competitive positive result requires all of the following:

1. paired task-success non-inferiority to optimized full refresh (FR) under a
   predeclared two-percentage-point margin;
2. at least 10% net synchronized critical-path reduction, including anchor,
   router, provenance, abort, and transfer costs;
3. a statistically supported Pareto improvement over the faithfully corrected
   VLA-Cache baseline and the matched Gated VLA-Cache baseline;
4. no strict domination by the strongest executable exact-stack alternative,
   including VLA-ADP and actuation-slack refresh;
5. evidence that terminal-effect supervision materially improves risk--coverage
   over metadata, image-change, action-motion, and action-disagreement gates;
6. complete intent-to-treat, service, abort, total-work, and provenance
   accounting on untouched confirmation groups.

Latency alone, an offline router metric, or improvement over FR alone is not a
positive BRACE result. The work makes no formal robot-safety claim.

## 2. Corrections incorporated after the V2 red-team audit

V2.1 corrects every identified V1/V2 defect:

1. bounded treatments cover one, two, and four accelerated queries;
2. deployable semantic gates originate only from dense anchors;
3. mixed per-layer/per-token cache ages and actual sources are observable;
4. actuation-slack refresh is a mandatory measured baseline;
5. the invalid released VLA-Cache evaluator is replaced by a pinned,
   minimally corrected, explicitly named faithful reproduction;
6. a small outcome-blind cache-profile grid replaces a single brittle profile;
7. VLA-ADP is a mandatory exact-stack executable competitor, while SP-VLA,
   LAC, AC2-VLA, and other architecture-mismatched works receive a documented
   compatibility disposition;
8. comparator language now uses one consistent positive-result rule;
9. empirical selective-risk control replaces an unsupported pointwise risk
   guarantee;
10. multi-contract assignment probabilities and overlap are mandatory;
11. representative and enriched strata are separated and inclusion
    probabilities are logged;
12. duplicate-arm terminal discordance must be zero, not merely “negligible”;
13. collection expands through predeclared learning-curve checkpoints rather
    than directly targeting a large harmful-class count;
14. paired non-inferiority power uses observed discordance, episode clustering,
    suite multiplicity, and attrition;
15. the actuation-slack implementation is the exact one-chunk-old clean-pass
    pipeline and includes spillover blocking plus total GPU work;
16. development outcomes may select among the predeclared profile grid, but no
    confirmation population may tune profiles, thresholds, or claims.
17. layerwise reuse sets are nested because removed hidden states cannot be
    reintroduced at deeper decoder layers;
18. all visual/text/proprio/action spans are derived from the runtime sequence
    map rather than fixed token indices;
19. dense semantic attention uses a parity-verified sidecar path, never an
    unaccounted SDPA-to-eager fallback;
20. outcome-blind proprio/action drift envelopes address the fact that
    bidirectional OpenVLA-OFT visual K/V can change while pixels remain static;
21. calibration covers the complete fastest-accepted-contract router, not
    independent contract thresholds alone; and
22. paired reconstruction recreates and executes the historical anchor before
    branching at the following query;
23. every reused K/V points to its complete immutable multimodal source record,
    not image provenance alone; and
24. an experimental contract abort fills the remainder of its assigned horizon
    with FR rather than silently re-routing to a different treatment.

## 3. Claim and novelty boundary

The provisional contribution is:

> pre-inference selection of bounded clean-provenance VLA cache contracts,
> trained from replay-verified paired terminal intervention effects on
> cache-induced trajectories and evaluated with empirical selective-risk
> control.

Do not claim novelty for KV caching, token reuse, adaptive compute, action-aware
gating, camera asymmetry, periodic refresh, clean gate provenance,
counterfactual simulation, DAgger-style aggregation, provenance tracking,
selective classification, or OOD rejection individually.

Any final claim is limited to the evaluated checkpoint, cache substrate,
hardware, LIBERO suites, contract horizons, and simulator setting. Physical
reliability, energy, general robot safety, and cross-architecture transfer
require corresponding evidence.

## 4. Frozen system and server boundary

Preserve the accepted OpenVLA-OFT checkpoint, prompt, preprocessing,
normalization, continuous L1 action head, proprio projector, two cameras,
center crop, eight-action chunks, and LIBERO task semantics.

- Optimized FR is the correctness and timing oracle.
- One routing decision occurs only at a policy-query boundary.
- BRACE may select computation; it may not blend, repair, rerank, or
  post-process predicted actions.
- TITAN writes remain restricted to `/home/ved/SAVR`.
- Use at most one separately coordinated aggregate-idle GPU.
- Never use `sudo`, change services or permissions, inspect other users'
  processes, or write outside the project.
- Preserve the proven SAVR environment. New dependencies use an isolated,
  project-local runtime.
- CUDA graphs, model sharding, multi-GPU execution, and system-wide installs
  are outside scope.

## 5. Cache-contract state machine

### 5.1 Dense anchor

A dense anchor is an FR query that skips no visual computation. It creates:

- the base action chunk;
- complete per-layer visual K/V state;
- dense text-to-vision attention and layer statistics;
- scene/wrist images and patch identities;
- a semantic-gate identity; and
- a per-layer/per-token source ledger.

Only a dense anchor may define the semantic mask or layer schedule of a
deployable contract. Accelerated forwards may update recomputed K/V entries and
their source identities, but never replace the dense semantic gate.

The pinned OpenVLA-OFT decoder uses bidirectional multimodal attention, so
deeper visual K/V may depend on current proprio/action context even for
unchanged pixels. Anchor semantic attention must be captured without changing
the accepted dense backend or action; otherwise the affected profile is
removed before outcomes.

### 5.2 Contract

A contract is `c = (p, h)`, where `p` is a predeclared clean-provenance profile
and `h` is a maximum of 1, 2, or 4 accelerated policy queries.

At the first policy query after an anchor, the controller selects FR or one
eligible contract. A selected contract cannot become more aggressive, switch
profile, or exceed `h`. Expiry or any hard abort forces FR and creates a new
anchor. The maximum nominal accelerated-query fractions are therefore 1/2,
2/3, and 4/5 for horizons 1, 2, and 4 before early aborts.

### 5.3 Hard aborts

Force FR on:

- task, episode, initial-state, seed, prompt, camera, model, or checkpoint
  identity change;
- missing or inconsistent K/V, gate, layer, source, or contract metadata;
- age, horizon, or source-image ring-buffer violation;
- absent or invalid current proprioception or image input;
- outcome-blind profile-envelope violation in proprioceptive drift, previous
  action motion, or gripper transition;
- nonfinite router output, empirical-risk ineligibility, or explicit reject;
- unauthorized profile switch;
- exception, timeout, memory-cap breach, or failed invariant.

Abort rules and envelopes are frozen before their development outcomes are
opened. Aborted assignments remain in intent-to-treat latency and outcome
analysis.

## 6. Outcome-blind cache-profile grid

### 6.1 Profile families

| Family | Definition |
|---|---|
| P0 | Optimized FR oracle/fallback and dense-anchor creator. |
| P1 | Scene-only K/V reuse; wrist tokens always refresh; dense-anchor gate/schedule only. |
| P2 | Scene and wrist K/V reuse; wrist has stricter age and source-difference limits; dense-anchor gate/schedule only. |
| P3 | Faithfully corrected VLA-Cache algorithm with self-harvested attention, for baseline evaluation only. |
| P4 | Exact actuation-slack refresh: dense pass on chunk `t`'s frame during execution, discarded action, one-chunk-old clean gate/KV for chunk `t+1`. |

P1/P2 may refresh selected entries, but every entry retains its actual source
query. P3 is not BRACE-selectable. P4 is not a BRACE contract.

For P1/P2, reused-token sets must be nested across decoder depth and reuse
budgets nondecreasing after reuse begins. Once an implementation removes a
token's current hidden state, no deeper layer may reintroduce it. Sequence
positions and camera/text/proprio/action spans come from a validated runtime
map; fixed prompt-length slices are prohibited.

### 6.2 Grid construction and selection

After B3 timing, but before terminal outcomes, freeze at most six base profiles:

- at most three speed-eligible scene-only ratios/schedules;
- at most three speed-eligible dual-view ratios/schedules;
- horizons 1, 2, and 4 as separate contract conditions.

Choose grid points only from synchronized B3 latency, memory, reuse, and parity
evidence. Do not use task success. Preserve every grid point and rejection
reason.

B4 development outcomes may eliminate and select profiles through a frozen
rule. B5 freezes the deployable set. B7/B8 may not introduce, revive, or tune a
profile. A static clean-profile baseline uses the same selected grid, preventing
BRACE from receiving a more favorable cache substrate than its comparator.

## 7. Exact provenance and deployable features

### 7.1 Per-token ledger

For every layer and visual token, record:

- camera, crop, patch, and token position;
- source query and source-image digest;
- token age and dense-anchor age;
- recomputed/reused status;
- K/V position, dtype, shape, and digest;
- semantic-gate identity and dense provenance;
- contract/profile/horizon, remaining queries, and abort history.

Each source query also records normalized proprioception, both preprocessed
camera inputs, prompt/instruction and sequence-map digests, previously
executed-action summary, RNG/configuration identity, and counters. Mixed-source
context drift is measured against every live source, not the anchor alone.

Retain a bounded source-image ring buffer covering the maximum contract horizon
so current-to-actual-source differences are reproducible. A single scalar cache
age is never sufficient.

### 7.2 Router allowlist

The router may use only information available before the current VLA forward:

- current low-resolution scene and wrist images;
- last-dense-anchor thumbnails;
- current-to-actual-source patch differences for the candidate profile;
- per-camera/per-layer token-age and reuse summaries;
- dense semantic mask identity and age;
- current proprioception and its historical deltas;
- previously executed action chunk, motion summaries, query index, and queue
  position;
- frozen instruction/task encoding computed once per episode;
- candidate profile/horizon and prior abort history.

All images, maps, encodings, transfers, routing, and checks count in timing.
The task encoding implementation, weights, license, download, and latency are
frozen before B5. A task identifier or instruction encoding may not leak final
group identity.

Forbidden inputs include reward/success, privileged simulator state, future
data, the current FR action, current dense-VLA hidden states/logits/attention,
and protected-population statistics.

## 8. Replay-verified interventional supervision

### 8.1 Branch reconstruction

Branch states come from versioned behavior-mixture rollouts. Reconstruct each
arm by a fresh reset to the same task, initial state, and seed, followed by the
exact recorded low-level action prefix through normal `env.step()` calls.
Log the behavior-policy version, mixture probability, branch-time sampling
probability, and every eligibility decision needed to reconstruct the sampled
state distribution.

Direct MuJoCo `set_state()` alone is prohibited because it omits controller,
observable, wrapper, timing, queue, and cache state. Recreate and validate the
dense anchor separately for each arm, execute its identical action chunk, and
branch at the following policy query. Re-running an anchor at the treatment
observation changes the cache source and is invalid.

### 8.2 Assignment and overlap

At each accepted branch point, preassign one candidate contract `c=(p,h)` with
known nonzero probability among all eligible contracts. Then randomize arm
execution order between:

- `T=0`: FR for the next `h` policy queries; and
- `T=c`: the assigned contract, including all frozen abort behavior.

Log contract-assignment and arm-order probabilities. Every contract intended
for learned selection must have adequate overlap across its eligible feature
region; deterministic assignment by task, phase, or apparent difficulty is
prohibited.

If the contract arm aborts, execute FR for that query and every remaining query
in the assigned `h`-query window. Do not re-route inside the experimental
window. Normal deployment may make a new decision after the abort-created
anchor; B6/B7 evaluate that repeated-decision policy.

After `h`, both arms use the same versioned continuation policy. Round 0 uses
FR. The single allowed aggregation round uses frozen BRACE-v0. Treatment is the
complete bounded burst, not the first query alone.

### 8.3 Outcomes and controls

Primary paired categories are harmful (FR succeeds/contract fails), beneficial
(contract succeeds/FR fails), neutral-success, and neutral-failure. Also record
time/steps to success, termination, served queries, aborts, critical-path time,
and total compute.

Primary analysis is intent to treat. Conditioning on service or non-abort is
secondary.

Before outcomes, reserve at least 10% of branch executions for controls:

- FR versus identical FR;
- selected P1/P2 contract versus the identical contract;
- randomized arm order;
- deliberately corrupted prefix, source map, cache, and gate metadata that
  validators must reject.

Any unexplained duplicate-arm terminal discordance is a technical failure and
stops outcome collection. It is never subtracted, averaged away, or relabeled.

## 9. Router, empirical risk control, and deployment

Let `s_theta(x,c)` be a learned ranking score for paired terminal harm from
assigning contract `c` instead of FR. It is not called a true probability or a
pointwise upper bound without calibration evidence.

Use a deliberately small profile-conditioned model and freeze its architecture,
parameter count, training recipe, licenses, and latency budget before B5 model
selection. Predeclare a bounded hyperparameter grid, use validation groups only
for selection, and train at least five fixed optimization seeds. Freeze a
validation-only selection or ensemble rule before held-out results. Required
baselines use identical grouped splits:

- prevalence/contract prior;
- metadata only;
- image motion only;
- current-to-source difference threshold;
- previous-action motion/dynamics, including a VLA-ADP-style signal;
- action-disagreement supervision with matched capacity;
- BRACE without token provenance;
- BRACE trained only on FR-continuation states.

On model-selection validation, construct at most 20 complete joint policies
from a finite grid of score thresholds and service coverages, apply simultaneous
diagnostics, and freeze exactly one. A disjoint prospective calibration split
can only accept or reject that frozen policy using the exact one-sided local-harm
rule in B5. This is an empirical local population statement, not a per-state,
episode-level, or physical-safety guarantee.

At deployment, choose the lowest-latency eligible contract whose frozen
threshold accepts; otherwise choose FR. Any extrapolation outside calibration
support rejects to FR. Report risk, coverage, abort rate, and latency together.

Calibrate this complete joint fastest-accepted-contract rule on representative
groups. Separately valid contract thresholds do not by themselves validate the
argmin selection performed during deployment.

The action-disagreement comparator uses paired FR/cache action disagreement
only to create its development labels. Its deployed predictor receives the
same pre-inference allowlist and timing treatment as BRACE; it may not run FR to
make the current routing decision.

## 10. Baseline validity and comparison hierarchy

### 10.1 Mandatory same-stack executable methods

1. optimized FR;
2. faithfully corrected VLA-Cache at pinned revision
   `a4909880573868dee2769343d52e793c0341678b`;
3. matched Gated VLA-Cache reproduction;
4. official released VLA-ADP on OpenVLA-OFT;
5. official released VLA-Pruner on OpenVLA-OFT;
6. official released SpecPrune-VLA on OpenVLA-OFT;
7. best static clean-provenance contract from the same grid;
8. periodic FR matched to BRACE service coverage;
9. P4 actuation-slack refresh;
10. action-disagreement router with matched capacity;
11. BRACE-v1;
12. BRACE without provenance and without aggregation.

The VLA-Cache evaluator correction is limited to true previous-frame semantics
and explicit episode-error propagation. Preserve a patch, tests, before/after
source hashes, and algorithm-parity audit. Name results “faithfully corrected
VLA-Cache,” never untouched official output. The invalid released evaluator is
retained as source evidence and is not run as scientific data.

VLA-ADP, VLA-Pruner, and SpecPrune-VLA must use their official code and matched
checkpoint/hardware/settings or receive individual technical exclusions
supported by exact source/runtime evidence. A failed integration is not
evidence that BRACE beats the excluded method.

### 10.2 Compatibility-disposition methods

Audit LAC, SP-VLA, AC2-VLA, Action-JND, FlashVLA, EfficientVLA, ActionCache,
VLA-IAP, BFA++, VLA-Corrector, SAFE-Pruner, Selective Perception, LoopVLA, and
DySL-VLA for released code, checkpoint, action-head, benchmark, and training
compatibility. Execute only methodologically faithful same-stack comparisons.
Otherwise report published results as context and the exact reason a direct
comparison would be invalid.

### 10.3 Dominance rule

BRACE must improve the paired success--critical-path frontier over both
faithfully corrected VLA-Cache and matched Gated VLA-Cache. Across all mandatory
same-stack methods, no baseline may have statistically non-worse success,
non-worse critical-path latency, and non-worse total GPU work with at least one
strict improvement. If one does, the competitive BRACE claim fails.

## 11. Population, leakage, and sampling policy

Historical SAVR/ACR outcomes remain development evidence. Preserve untouched
LIBERO initial states 10--49 and seeds 7/17/27 until a final map is frozen.

- Group every split by complete task/initial-state/seed episode.
- Never split branch rows from one episode across train/validation/test.
- Freeze behavior-mixture weights and eligible branch-time strata before
  collection.
- Keep a deployment-representative stratum separate from enrichment.
- Enrichment may use only pre-outcome features and must log inclusion
  probabilities.
- Use inverse-probability weighting or an equally predeclared design-consistent
  estimator for natural-distribution risk; unweighted enriched prevalence is
  never a deployment claim.
- Reserve at least 40% of held-out branch evaluation for representative
  sampling.
- Final confirmation uses untouched state/seed groups and, if feasible, at
  least one task group absent from router training.

No protected outcome may alter features, profiles, horizons, thresholds,
sample size, operating point, or claim.

## 12. Phase B1 — Replay-verified branch harness

**Resource:** project-local CPU/OSMesa simulator only; no VLA, CUDA, or policy
outcome collection.

Create complete transcripts containing low-level actions, chunk/query
boundaries, observation/state/counter digests, reward/success, termination, and
queue state. For at least three tasks spanning free motion, contact, and gripper
transition:

1. record a deterministic scripted trajectory;
2. replay early/middle/late prefixes twice in fresh environments;
3. compare state, counters, proprioception, cameras, reward/success, and the
   next transition under an identical probe action;
4. reject one modified prefix action;
5. reject direct-state-only restoration;
6. prove transcript version/hash and incomplete-record rejection.

Acceptance requires 100% discrete and next-transition equality and justified
pre-outcome tolerances for continuous/image fields where bitwise equality is
unavailable.

## 13. Phase B2 — Baseline, cache, and provenance correctness

**Resource:** CPU and synthetic tensors only.

Freeze the isolated integration plan for the project's customized Transformers
4.40.1 stack and VLA-Cache's customized 4.47.0 stack. Implement and test:

- the two-line-class VLA-Cache evaluator correction and error propagation;
- faithful algorithm/configuration parity;
- exact DynamicCache clone/restore and transactional model configuration;
- per-layer/per-token source ledger and source-image ring buffer;
- complete immutable multimodal source records and exact-source drift checks;
- dense-gate identity and age;
- runtime-derived multimodal sequence spans with no fixed prompt indices;
- sidecar semantic-attention implementation and synthetic invariants; real
  dense backend/action parity is deferred to B3;
- P0--P4 state/reset semantics and the six-profile maximum grid;
- nested layerwise reuse sets, nondecreasing budgets, and nonvisual-drift
  envelope aborts;
- 1/2/4-query expiry, every abort, and randomized order;
- candidate/arm propensities and immutable intent-to-treat records;
- adversarial camera/token/layer permutations;
- zero cross-arm mutation and exact P0-disabled equivalence;
- official VLA-ADP, VLA-Pruner, SpecPrune-VLA, and Gated VLA-Cache
  compatibility preflights.

No method proceeds to B3 without source/license/config hashes, passing parity
tests, and a bounded resource command.

## 14. Phase B3 — Physical timing, parity, and slack feasibility

**Resource:** one separately authorized TITAN GPU; no simulator outcome. Freeze
the exact query cap after B2; it may not exceed 500 balanced real-model queries.

Interleave randomized, synchronized measurements of P0, the corrected cache
baseline, Gated VLA-Cache overhead, the outcome-blind P1/P2 grid, VLA-ADP,
VLA-Pruner, SpecPrune-VLA, and P4 components. Record warm-up, p50/p95/p99
critical-path wall time, GPU busy and component CUDA time, contract-cycle time
for horizons 1/2/4, total GPU work, router/provenance/transfer overhead, peak
memory, token counts/ages, and action/tensor parity.

For P4, run the dense pass on chunk `t`'s frame during its measured action
execution window, discard its action, and supply its one-chunk-old clean
gate/KV to chunk `t+1`. Measure completion, spillover blocking, queueing, and
total work. Artificial sleep is not evidence.

Advance only if:

- P0 matches the proven FR path;
- at least one clean profile has at least 10% accelerated-query and 8%
  amortized contract-cycle reduction after all overhead;
- memory remains below 23 GiB;
- every cache/provenance/reset invariant passes;
- corrected VLA-Cache, VLA-ADP, VLA-Pruner, and SpecPrune-VLA receive valid
  timing or individual reviewed technical exclusions; and
- P4 receives a measured disposition.

If only self-harvested caching is fast, or a mandatory exact-stack baseline
cannot be evaluated fairly, stop before outcomes.

## 15. Phase B4 — Technical branches and profile/prevalence screen

Requires B1--B3 acceptance and a separately frozen outcome authorization.

### 15.1 B4a technical pilot

- 12 opened development episodes;
- at most two preassigned branch points per episode;
- one clean profile at horizons 1 and 2;
- at most 48 primary paired contracts plus the frozen 10% controls;
- cap derived from B3 throughput, not a historical estimate.

Stop on any unexplained control discordance, cache mismatch, order effect,
attrition, or cost overrun.

### 15.2 B4b profile and prevalence screen

If B4a passes, freeze a balanced sequential screen over the B3 grid and
horizons 1/2/4. Use at most 300 primary paired contracts plus controls across at
least 30 grouped development episodes. Before outcomes, freeze assignment
probabilities, profile elimination rules, task balance, prevalence bands,
minimum terminal completion, and cost scaling.

Advance a profile only if it is speed-eligible, produces enough representative
harm to study or is already demonstrably low-risk, has zero control discordance,
and projects within the 40-GPU-hour development cap. Preserve horizon-specific
results.

Stop BRACE if useful clean profiles produce neither learnable harm nor a static
positive frontier; if harm appears only after unpredictable future motion; or
if a static threshold/profile already dominates the proposed learning premise.

## 16. Phase B5 — Sequential predictability and empirical risk control

Collect conditionally through predeclared checkpoints rather than one large
campaign:

| Checkpoint | Minimum cumulative harmful | Minimum cumulative neutral-success | Purpose |
|---|---:|---:|---|
| B5-20 | 20 | 40 | Sanity learning curve and leakage audit |
| B5-40 | 40 | 80 | Baseline separation and calibration feasibility |
| B5-80 | 80 | 160 | Held-out risk--coverage stability |
| B5-final | 100 train, 30 model-selection validation, 60 held-out; calibration is not harm-enriched | power-selected | Contract-specific evidence plus prospective calibration |

At each checkpoint use episode-grouped splits, untouched rows, and frozen
metrics. Stop if AUPRC/AUROC, calibration, and risk--coverage remain
indistinguishable from metadata/image/action baselines; if confidence intervals
exclude useful service coverage; or if the next checkpoint exceeds the
40-GPU-hour cap.

After the B4 screen, retain at most three contracts for learned deployment.
Evaluate at most 20 joint fastest-accepted-contract candidates on model-selection
validation, freeze exactly one, then calibrate only that policy prospectively.
Training, model-selection validation, calibration, and held-out episode groups
are disjoint; if the calibration population breaks the 40-GPU-hour cap, stop
rather than reuse model-selection or held-out outcomes.

The 1% representative branch-harm bound is a local development filter, not an
allocation of the final 2-point episode-success margin. Before labels, power
the calibration population so its exact one-sided bound can resolve 1%. Use one
preselected query boundary per independent episode group, directly execute the
frozen joint policy's selected contract against FR, and accept only if the 95%
Clopper--Pearson upper harm limit is at most 1%. Even zero observed harm requires
299 groups. Its exact coverage lower limit must also exceed the outcome-blind
B3 coverage needed to project 12% net speed. Calibration can only pass or reject;
it cannot choose another validation policy. B7/B8 remain the only episode-level
non-inferiority test.

Report by task, contract, horizon, camera motion, cache age, representative
versus enriched stratum, and abort status. Each deployed contract requires its
own evidence or a predeclared hierarchical-sharing model validated on held-out
groups. Dense current-query VLA features invalidate the latency premise.

## 17. Phase B6 — One on-policy aggregation round

Only after B5 passes:

1. freeze BRACE-v0 profiles, score model, calibration, and threshold;
2. collect separate BRACE-v0 development rollouts;
3. sample branch states and contracts with logged nonzero propensities;
4. use BRACE-v0 as the common continuation after treatment horizon;
5. train BRACE-v1 once;
6. compare v1/v0 on untouched grouped branches;
7. quantify label and coverage drift.

Proceed only if v1 preserves or improves calibrated risk--coverage and drift
passes a frozen bound. A second aggregation round requires a new protocol.

## 18. Phase B7 — Paired closed-loop development

Run all mandatory same-stack methods in Section 10 with identical population,
checkpoint, hardware, timing, task, and error semantics. Freeze the primary
BRACE operating point before comparative outcomes.

Advance only when FR non-inferiority, 10% net speed, same-mechanism Pareto
improvement, no exact-stack domination, no task-level catastrophic regression,
and mechanism ablations all pass under the frozen statistical plan.

Development comparisons choose the minimal confirmatory method set but never
remove the strongest corrected cache, strongest exact-stack competitor, FR, or
P4 solely because they are difficult to beat.

## 19. Phase B8 — Sealed confirmation

Before opening B8, freeze code, dependencies, source patches, checkpoint,
profiles, router, thresholds, populations, hypotheses, sample size, and
analysis. Determine sample size from paired discordance, two-point margin,
cluster design effect, suite multiplicity, planned attrition, and the expected
true difference. Five hundred episodes is neither a default nor a guarantee of
adequate power.

Use a hierarchical family-wise alpha of 0.05:

1. paired success non-inferiority to FR;
2. net critical-path improvement;
3. same-mechanism cache Pareto improvement;
4. exact-stack non-domination and mechanism claims.

Secondary per-suite/task claims use predeclared multiplicity adjustment. Report
paired success differences and intervals, discordant counts, binomial success
intervals, episode-grouped latency/work intervals, aborts, attrition, all four
branch categories, and the complete frontier. No B8 observation may change a
method or claim. Non-domination requires predeclared componentwise intervals;
a nonsignificant difference alone is not evidence of equivalence.

Freeze the analysis program before B8, validate it on synthetic known-answer
records covering every outcome and attrition category, and independently
recompute all primary counts and estimates from the immutable terminal records.
Any unexplained disagreement blocks claims and publication.

## 20. Resource, evidence, and monitoring policy

- Every phase has a run identity, source/config hashes, query/episode/time/
  storage cap, terminal summary, and fail-closed recovery rule.
- Outcome-blind monitors may inspect only process health, terminal-record count,
  bytes, elapsed time, and aggregate selected-GPU telemetry until the frozen
  completion boundary.
- Historical episode throughput is planning context only. B4a measures branch
  pair cost, including two arms, anchor reconstruction, controls, prefix replay,
  and attrition.
- Preserve technical stops and scientific failures. Never retry or exclude them
  without a predeclared recovery.
- Record source commits, licenses, dependency lock, checkpoint and artifact
  hashes, GPU ID/UUID, timing definitions, commands, and population manifests.
- Simulation-only evidence remains simulation-only.

## 21. Global stop conditions

Stop BRACE without reinterpretation if:

- exact replay branches cannot be established;
- the faithful baseline correction or cache integration lacks P0 parity;
- no clean profile produces meaningful net speed;
- controls show any unexplained terminal discordance;
- assignment overlap or representative sampling is inadequate;
- harm is too rare, too common, or too expensive for the frozen budget;
- inexpensive pre-inference features cannot predict residual harm at useful
  coverage;
- preprocessing/routing removes the speed advantage;
- VLA-ADP, corrected VLA-Cache, Gated VLA-Cache, P4, or another valid baseline
  dominates BRACE;
- sealed FR non-inferiority, speed, suite, or multiplicity gates fail;
- novelty is independently preempted.

Passing early phases increases conditional plausibility; it never guarantees a
positive result.

## 22. Required checkpoint audit

Every phase report must answer:

1. What was authorized and what ran?
2. Which revisions/configurations/populations were used?
3. What remained untouched?
4. Did records, propensities, controls, caps, and hashes reconcile?
5. Did any technical, scientific, statistical, or novelty gate fail?
6. Were any rules changed after outcomes?
7. Was anything outside `/home/ved/SAVR` modified?
8. What exact next action, if any, is eligible?

## 23. Next eligible task

Implement and verify **B1 only**. Do not begin B2, install or patch external
methods, download assets, load the VLA/checkpoint, inspect/select a GPU, or
collect policy outcomes without a separate checkpoint decision.

## 24. Primary literature governing V2.1

1. VLA-Cache: <https://openreview.net/forum?id=QZYZ0Xm58q>
2. The Gate, Not the Cache: <https://arxiv.org/abs/2608.00391>
3. Gated VLA-Cache: <https://arxiv.org/abs/2608.10824>
4. VLA-ADP: <https://arxiv.org/abs/2509.22093>
5. LAC: <https://arxiv.org/abs/2602.00686>
6. SP-VLA: <https://arxiv.org/abs/2506.12723>
7. AC2-VLA: <https://arxiv.org/abs/2601.19634>
8. Action-JND: <https://arxiv.org/abs/2608.21247>
9. FlashVLA: <https://arxiv.org/abs/2505.21200>
10. EfficientVLA: <https://arxiv.org/abs/2506.10100>
11. ActionCache: <https://arxiv.org/abs/2607.06370>
12. LoopVLA: <https://arxiv.org/abs/2605.09948>
13. DySL-VLA: <https://arxiv.org/abs/2602.22896>
14. DAgger: <https://proceedings.mlr.press/v15/ross11a.html>
15. VLA-Pruner: <https://arxiv.org/abs/2511.16449>
16. SpecPrune-VLA: <https://arxiv.org/abs/2509.05614>
17. BFA++: <https://arxiv.org/abs/2602.20566>
18. OpenVLA-OFT: <https://www.roboticsproceedings.org/rss21/p017.html>
19. Selective classification: <https://papers.neurips.cc/paper_files/paper/2017/hash/4a8423d5e91fda00bb7e46540e2b0cf1-Abstract.html>
20. Off-policy evaluation: <https://proceedings.mlr.press/v70/wang17a.html>
21. High-confidence off-policy evaluation:
    <https://ojs.aaai.org/index.php/AAAI/article/view/9541>

Revalidate recent papers, released code, revisions, licenses, and publication
status before B2, before B7, and immediately before submission.
