# Positive-Results Direction Reset Audit

**Project:** SAVR  
**Date:** 2026-08-26  
**Status:** Historical decision audit under the earlier, stricter criteria; no method is authorized for implementation  
**Evidence cutoff:** 2026-08-26

> **Supersession note (2026-08-26):** The user subsequently relaxed the target
> from a highly novel/competitive method to an organically novel positive-result
> direction. A new literature and feasibility analysis also introduced a
> distinct supervision signal that this audit's generic-router rejection did
> not consider: controlled substitutions with the exact provenance-resolved
> stale K/V values. The current recommendation is documented in
> [`PAIR_VLA_RESEARCH_DIRECTION_AND_EXECUTION_PLAN_2026-08-26.md`](PAIR_VLA_RESEARCH_DIRECTION_AND_EXECUTION_PLAN_2026-08-26.md).
> This audit remains preserved to show why generic proxy routing and the online
> certificate direction were not selected.

## 1. Executive verdict

The project does **not** currently have a method that can honestly be described
as both highly novel and highly likely to produce a competitive positive-result
paper.

This is not a statement that no solution exists. It is the result of applying
one fixed decision rule to the complete local evidence and the current
literature:

- the original SAVR family fails once useful visual reuse becomes aggressive;
- ACR can preserve success at moderate scene-camera reuse, but the measured
  end-to-end speed benefit disappears;
- BRACE finds one physically faster cache profile, but every tested cache
  profile changes the action output, including the no-reuse cache-stack control;
- the current 2026 literature already contains strong methods for learned cache
  routing, action-aware token tolerance, stale-feature compensation, feature/KV
  delta prediction, clean asynchronous refresh, and action correction.

Therefore, another full method protocol would repeat the earlier mistake: it
would turn an interesting hypothesis into a preferred solution before novelty,
identifiability, and physical headroom were established.

One direction survives as a **conditional novelty probe**, not as an approved
paper method:

> **Randomized temporal-cache attribution:** randomly refresh a probability
> sample of otherwise reused visual tokens and use the known sampling design to
> estimate cache-induced representation error and allocate refresh compute.

This direction is related to SAVR, directly responds to the observed failure of
deterministic self-signals, and appears not to have been evaluated specifically
for temporal VLA KV reuse. However, its closest theoretical precedent explicitly
finds that randomization supports attribution and scheduling—not general failure
prediction. Its integration into a deep temporal cache is nontrivial, and the
local BRACE speed margin is only 12.23%. It is therefore **not ready for a full
implementation or positive-paper claim**.

The correct next action is one bounded, outcome-blind feasibility study. If that
study fails any frozen gate, the cache-method search should stop rather than be
redesigned again.

## 2. Why the previous checks kept finding new problems

The previous process had four structural defects:

1. **The candidate was chosen before the audit was complete.** Each check asked
   how to rescue the current idea, so newly discovered facts became patches
   rather than evidence against the idea.
2. **The decision criteria moved.** Novelty, action fidelity, physical latency,
   on-policy validity, and comparator strength were evaluated at different
   times instead of conjunctively.
3. **Literature search and feasibility analysis were interleaved with design.**
   A later paper or system constraint could invalidate an architecture already
   treated as settled.
4. **Uncertainty was expressed as informal success probabilities.** Those
   numbers looked more precise than the evidence justified and changed whenever
   a missing dependency was found.

This audit removes those defects by freezing the evidence, rubric, candidate
families, and verdict categories before selecting a direction. A failed
criterion is not repaired in this document. It changes the verdict.

## 3. Frozen project evidence

The following facts are treated as authoritative for this decision. They are
not hypotheses and are not reinterpreted to favor a new method.

### 3.1 SAVR outcome evidence

The primary outcome ledger is
[`docs/evidence/negative_results_summary.csv`](evidence/negative_results_summary.csv).

- Full Refresh (FR) succeeded in 100/100 LIBERO-Spatial episodes.
- The best SAVR1 point succeeded in 52/100 episodes at 34.69% online visual
  skipping; more aggressive points fell to 0--23% success.
- SAVR2-b10 succeeded in 29/30 episodes at only 6.72% skipping, and SAVR2-b15
  succeeded in 27/30 at 10.57% skipping.
- SAVR3 succeeded in 69/70 episodes but reused only 0.9534% of online visual
  queries.

**Implication:** input/state heuristics can preserve reliability only by becoming
too conservative to deliver the desired efficiency.

### 3.2 ACR evidence

- A conservative asymmetric-camera configuration reached 29/30 success with
  26.06% scene-camera reuse.
- More aggressive configurations fell to 24/30 and 23/30 at approximately
  47--49% scene-camera reuse.
- V2-C reduced measured visual CUDA work by 50.12% but was 41% slower in
  weighted end-to-end wall time.
- V3-C was a positive microbenchmark: approximately 31.41% visual CUDA
  reduction and near-parity weighted wall time.
- V3-D then tied the batched full-refresh comparator at 67/70 success with
  25.24% scene reuse, but achieved only 8.46% visual reduction and a wall-time
  ratio of 1.00226. It failed both the predeclared speed gates.

**Implication:** preserving camera-level behavior is possible, but the pinned
stack does not convert that reuse into a competitive end-to-end benefit.

### 3.3 BRACE evidence

The current terminal status is recorded in [`PROJECT_STATUS.md`](../PROJECT_STATUS.md).

- BRACE-B3 V05 completed 356 real model queries without a technical stop.
- P2-D50 reduced accelerated-query time by 18.71% and complete-cycle time by
  12.23%.
- Every profile failed action parity: 0/42 planned parity checks passed, and all
  168 timed cached actions differed from their dense counterparts.
- The no-reuse dense cache-stack control also differed from optimized core-FR;
  maximum normalized action differences were 0.222--0.368.
- Corrected VLA-Cache parity was 0/10.
- Dense cache-stack latency was approximately 1.220 seconds at the median,
  longer than the approximately 0.4-second eight-action execution window.
- Aggregate peak model memory was approximately 18.4 GiB, inside the 23 GiB
  project ceiling.

**Implications:**

1. the downstream cache path has real but narrow physical speed headroom;
2. exact action agreement is not available even for the cache-stack control;
3. action disagreement is not by itself a valid label for terminal harm; and
4. a dense pass cannot be hidden inside the current action-chunk window on the
   available stack.

### 3.4 Resource boundary

The available validated environment is one OpenVLA-OFT/LIBERO stack on one
24-GB-class TITAN GPU at a time. A direction that requires full foundation-model
training, a new multi-GPU timing methodology, or a different robot platform is
not currently feasible evidence for this project.

## 4. Frozen decision rubric

Each candidate receives a 0--5 score on the same five criteria. Scores are
ordinal judgments used for comparison; they are **not probabilities**.

| Criterion | 0 | 3 | 5 |
|---|---|---|---|
| Novelty | Directly published | Distinct combination or untested domain transfer | Clear new mechanism and research question |
| Fit to measured problem | Does not address local failure | Addresses part of the observed bottleneck | Directly addresses reliability, speed, and provenance findings |
| Positive-result mechanism | No credible causal mechanism | Plausible but unvalidated | Strong theory or repeated directly comparable evidence |
| Local feasibility | Incompatible with resources/stack | Bounded implementation appears possible | Demonstrated on the current stack with adequate margin |
| Impact if successful | Engineering-only improvement | Useful new frontier point | General result that changes VLA acceleration practice |

Verdict rules were fixed before scoring:

- **Ready:** total at least 20/25, no criterion below 3, no unresolved
  identifiability or comparator problem.
- **Conditional probe:** total 15--19, at least one clearly novel research
  question, and all primary uncertainties can be tested without terminal
  outcomes or a large implementation.
- **Reject:** total at most 14, direct novelty collision, zero local feasibility,
  or a primary signal that is not identifiable from the proposed inputs.

## 5. Current literature map

The literature search covered temporal visual/KV reuse, token selection,
adaptive computation, stale representations, action-aware compression,
asynchronous inference, delta prediction, correction, and randomized cache
inference. These sources are used because they bear directly on a candidate;
they are not included to inflate a citation count.

Several of the August 2026 sources are very recent preprints. Their reported
numbers are not treated as settled facts or directly compared across hardware.
Their existence and method definitions still matter for novelty, and their
claims identify comparators that a later paper would have to test faithfully.

### 5.1 Temporal reuse and learned allocation

- [VLA-Cache](https://arxiv.org/abs/2502.02175) already performs training-free,
  visually adaptive temporal KV reuse with task-relevance and layer-adaptive
  selection.
- [Learning Adaptive Visual Token Caching](https://arxiv.org/abs/2602.00686)
  learns both which tokens to cache and the reuse ratio, reporting 1.76x
  wall-clock acceleration and positive success changes.
- [AC2-VLA](https://arxiv.org/abs/2601.19634) jointly routes temporal reuse,
  token pruning, and depth from visual, language, and action context, with
  action-guided self-distillation.
- [DySta](https://arxiv.org/abs/2602.03983) separates static and dynamic visual
  tokens and uses a recache gate, reporting simultaneous speed and success
  improvements.
- [Gated VLA-Cache](https://arxiv.org/abs/2608.10824) adds output-confidence
  invalidation to visual caching.

**Consequence:** another learned selector, threshold predictor, state/action
gate, or static/dynamic decomposition has weak novelty unless it introduces a
different observable or identifiability principle.

### 5.2 Action-aware and finer-grained compression

- [Action-JND](https://arxiv.org/html/2608.21247) learns deep-token action
  tolerance. On OpenVLA-OFT it reports 96.20% average success at its soft
  adaptive point versus 95.35% for VLA-Cache, and large recovery at aggressive
  reuse.
- [RoleSub](https://arxiv.org/abs/2608.18410) routes sub-token value dimensions
  conditioned on semantic role and language, outperforming token-only controls
  in 33 of 36 matched settings.

**Consequence:** “action-aware caching,” token-wise residual importance, and
finer-grained KV compression are already occupied claims.

### 5.3 Staleness compensation and delta prediction

- [Latent Bridge](https://arxiv.org/abs/2605.02739) predicts feature-space or
  KV-space deltas, uses a task-adaptive training pipeline including DAgger, and
  reports 1.65--1.73x net episode speedup with 95--100% performance retention.
- [CloudEdgeVLA](https://arxiv.org/abs/2608.00569) trains an action system to
  combine stale semantic features with fresh local vision and target current
  actions.
- [Think at 5 Hz, Act at 20 Hz](https://arxiv.org/abs/2607.15621) combines a
  slow frozen backbone with a fast action expert that consumes stale semantic
  state and fresh frames under randomized staleness.
- [TIC-VLA](https://arxiv.org/abs/2602.02459) explicitly trains latency-aware
  action prediction from delayed semantics and current observations.

**Consequence:** a feature/KV delta predictor, cache-conditioned action head,
or generic stale-aware adapter is a direct or near-direct novelty collision.

### 5.4 Clean refresh, streaming, and correction

- [The Gate, Not the Cache](https://arxiv.org/abs/2608.00391) finds that
  self-harvested accelerated gates can collapse while tested action detectors
  miss the failure; unconditional clean refresh during actuation slack restores
  performance and provides 18--22% critical-path speedup on its systems.
- [Reflex](https://arxiv.org/abs/2607.14695) uses mathematically exact streaming
  cache updates and asynchronous pipelining for flow-matching VLAs.
- [VLA-Corrector](https://arxiv.org/abs/2607.01804) detects visual-dynamics
  deviation and triggers corrective replanning and adaptive action horizons.

**Consequence:** triggered action correction and asynchronous clean refresh are
already active directions, and the local dense pass is too slow to hide inside
the current action chunk.

### 5.5 Randomized cache inference

- [Error Certificates for KV-Cache Eviction via Randomized Design](https://arxiv.org/html/2607.21475)
  proves that deterministic eviction cannot consistently estimate the
  attention error caused by information it discarded. Known-probability
  sampling restores an estimable variance signal. Critically, its real-workload
  study concludes that randomization buys **attribution and scheduling, not
  general task-failure prediction**.

The searches in this audit did not find a paper that applies a known-probability
refresh design and design-based difference estimator to **temporal stale-value
reuse in a closed-loop VLA**. This is a provisional novelty finding, not a
guarantee: unpublished or unindexed work may exist, and the cited eviction
paper is a close conceptual precedent.

## 6. Candidate comparison

| Candidate family | Novelty | Problem fit | Positive mechanism | Local feasibility | Impact | Total | Verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| A. CDRR action repair + next-step refresh | 2 | 3 | 2 | 2 | 2 | 11 | Reject |
| B. Learned/on-policy cache router | 1 | 3 | 2 | 2 | 2 | 10 | Reject |
| C. Feature or KV delta predictor | 1 | 4 | 4 | 1 | 2 | 12 | Reject |
| D. Stale-aware action head/adapter | 1 | 4 | 4 | 2 | 2 | 13 | Reject |
| E. Asynchronous clean refresh | 1 | 5 | 5 | 0 | 2 | 13 | Reject locally |
| F. Flow/dual-system architecture pivot | 1 | 2 | 5 | 0 | 3 | 11 | Reject for this project |
| G. Randomized temporal-cache attribution | 4 | 4 | 2 | 2 | 4 | 16 | Conditional probe |
| H. Generic cache-robust fine-tuning | 1 | 3 | 3 | 2 | 2 | 11 | Reject |

No candidate qualifies as **Ready**.

### 6.1 Why CDRR is rejected

CDRR proposes predicting the dense-minus-cached action residual, applying a
bounded correction, and rebuilding the cache after a large correction. It is
not selected because:

- reused tokens are chosen precisely where visible deltas are small, so the
  proposed inputs may not identify the missing action information;
- repairing the current action does not repair the internal stale state;
- an eight-action chunk delays the opportunity to rebuild;
- the dense cache stack is not an exact oracle for optimized core-FR locally;
- Action-JND, Latent Bridge, CloudEdgeVLA, and VLA-Corrector collectively occupy
  much of the claimed mechanism and comparison space; and
- success would require on-policy aggregation and full closed-loop validation
  before the primary signal has passed a small predictability test.

The fourth check therefore does not add another repair to CDRR. It rejects the
direction under the fixed rubric.

### 6.2 Why routing is rejected

BRACE already showed that the action-disagreement labels available cheaply on
the current stack are not equivalent to terminal harm. Closed-loop branch
labels are expensive and require exact state reconstruction. Meanwhile,
VLA-Cache, adaptive learned caching, AC2-VLA, Gated VLA-Cache, and Action-JND
provide strong direct comparators. A new router would need a new identifiable
supervision signal and a clearly different mechanism; neither is currently
available.

### 6.3 Why delta/stale-aware methods are rejected

These methods have the strongest external evidence of possible positive
performance, but that evidence also removes their novelty. Latent Bridge is a
direct feature/KV-delta method, while CloudEdgeVLA and latency-tolerant dual
systems directly train around stale semantic features. Reimplementing them on
OpenVLA-OFT would be reproduction or adaptation, not a new central method.

### 6.4 Why asynchronous refresh is rejected locally

The mechanism is externally convincing: a clean refresh can prevent corrupted
gate provenance. But BRACE measured approximately 1.220 seconds for the dense
cache-stack query, versus roughly 0.4 seconds of actuation slack for the current
eight-action chunk. Two GPUs would require model sharding or pipeline execution
and a new synchronized timing methodology. That is a material architecture and
resource pivot, not a feasible next phase of SAVR.

## 7. The sole surviving conditional probe

### 7.1 Working research question

**Can known-probability partial refresh make temporal cache damage attributable
and improve refresh allocation at matched compute in a closed-loop VLA?**

This asks a different question from deterministic gate learning. It does not
claim to infer unobserved damage from a deterministic cache. It deliberately
introduces randomized observation of the otherwise stale channel.

### 7.2 Conceptual mechanism

At a chosen layer and query, split the visual tokens into:

- a certainty set that is always recomputed; and
- a candidate-reuse set in which each token is refreshed with a known,
  nonzero inclusion probability.

For a sampled token, the system observes both the stored stale contribution and
its newly recomputed contribution. A design-based difference estimator can use
these sampled deltas to estimate the aggregate change that a dense refresh
would have produced at that layer. The estimator and its variance become a
cache-damage attribution signal. That signal may then allocate a fixed refresh
budget or force a later full refresh.

At a fixed query state, a candidate ratio estimator would start from stale
numerator and denominator totals, \(N^0\) and \(Z^0\), and add
inverse-probability sampled current-minus-stale corrections:

\[
\widehat{N}=N^0+\sum_i\frac{I_i}{\pi_i}
  \left(a_i^t v_i^t-a_i^0 v_i^0\right),\qquad
\widehat{Z}=Z^0+\sum_i\frac{I_i}{\pi_i}
  \left(a_i^t-a_i^0\right),\qquad
\widehat{y}=\widehat{N}/\widehat{Z}.
\]

Here \(I_i\) is the randomized refresh indicator and \(\pi_i>0\) is its known
inclusion probability. This is only a candidate object, not a claimed theorem:
in a deep temporal cache, the query and upstream representation are themselves
approximate. Gate R0 exists specifically to determine whether a useful target
remains identifiable under that dependence.

The intended contribution is **not** “random caching is more accurate.” It is:

1. a temporal-reuse formulation that exposes otherwise unobserved stale-value
   error through a known sampling design;
2. an empirical test of whether layer-level attribution survives through a VLA
   action decoder; and
3. matched-compute refresh scheduling based on that attribution.

### 7.3 Why it fits the observed problem

- SAVR showed that deterministic observation/state thresholds do not identify
  harmful reuse.
- The Gate, Not the Cache reports that tested action-level detectors can miss
  compounding cache collapse.
- BRACE showed that action differences are ubiquitous on the local cache stack,
  so disagreement alone is not a useful harm label.
- Known-probability refresh obtains new information instead of attempting to
  infer all missing information from the deterministic retained state.

### 7.4 Why it is not yet a method

Five unresolved problems prevent promotion:

1. **Layer dependence.** The current query and later token states already depend
   on approximate earlier layers. A per-layer estimator may quantify only
   conditional local error, not the full dense-policy error.
2. **Softmax difference estimation.** Temporal refresh changes keys, values, and
   attention normalization. A stable ratio/difference estimator needs to be
   derived and tested; the eviction theorem cannot simply be copied.
3. **Attribution is not success prediction.** The closest precedent explicitly
   finds that its certificate does not generally predict task failure.
4. **Narrow speed margin.** The best local cache profile saves 12.23% of the
   complete cycle. Sampling, bookkeeping, and any escalation must leave at
   least 10%; only about 2.23 percentage points of measured margin remain.
5. **Comparator strength.** A positive paper would still need matched-compute
   comparisons to VLA-Cache, Action-JND or a faithful action-aware control,
   confidence gating, visual-difference gating, and random refresh.

## 8. Frozen feasibility study before any protocol

This is a bounded research probe, not Phase 1 of an assumed method. It must not
read terminal episode outcomes.

### Gate R0 — mathematical object

Before model integration, write the exact estimand for one attention head:

- which dense quantity is being estimated;
- what is known before sampling;
- the inclusion law and probability floor;
- the difference/ratio estimator;
- its variance estimate;
- the conditioning introduced by approximate earlier layers; and
- which claims are theorem-backed versus empirically calibrated.

**Stop condition:** if the estimator requires unsampled current keys/values or a
dense query state, the proposed online certificate is not identifiable and the
direction ends.

### Gate R1 — captured-tensor replay

Use frozen captured tensors from the current stack, not simulator outcomes.
Compare randomized partial refresh estimates to the fully recomputed
attention-output difference across layers, heads, token budgets, and temporal
ages.

Predeclare:

- at least 0.85 empirical coverage for the nominal interval;
- at least 0.30 Spearman correlation with true layer error;
- no worse mean attention error than deterministic reuse at matched refreshed
  token count; and
- all results reported, including failed layer/budget cells.

These thresholds deliberately mirror the closest certificate work rather than
being selected after observing this project’s data.

### Gate R2 — action-error transfer

On paired, outcome-blind model queries, test whether the certificate ranks the
cache-induced action deviation from a same-stack dense comparator.

Required comparisons:

- certificate versus random score;
- visual change;
- action/logit confidence where defined;
- cache age and reuse ratio; and
- the best available deterministic cache-side signal.

**Stop condition:** if the certificate cannot exceed the strongest zero-cost
baseline by a predeclared practically meaningful margin on group-held-out tasks
and episode prefixes, it is not useful for refresh scheduling.

### Gate R3 — physical overhead

Measure synchronized complete-cycle latency, not FLOPs or isolated kernel time.

**Stop condition:** if randomized refresh plus bookkeeping leaves less than 10%
complete-cycle acceleration over same-stack dense inference, the direction
ends regardless of diagnostic quality.

### Gate R4 — decision checkpoint

Only if R0--R3 all pass should a separate method protocol be written. That
protocol must freeze terminal success evaluation, matched-compute comparators,
confidence intervals, seeds, task suites, and a no-redesign rule before any
outcomes are read.

No terminal pilot, on-policy training, manuscript rewrite, or positive-result
claim is authorized by this audit.

## 9. Required positive-paper standard if the probe passes

A later method would need to satisfy all of the following, not a selected
subset:

1. statistically supported terminal success non-inferiority to full refresh;
2. at least 10% synchronized complete-cycle latency reduction;
3. a better success--latency frontier than VLA-Cache and the strongest feasible
   action-aware/gated comparator;
4. ablations showing that known-probability sampling and the estimator, rather
   than extra refresh compute, cause the improvement;
5. results across all four LIBERO suites, not one task suite and seed;
6. uncertainty intervals and paired analyses where the design permits;
7. a clear separation between attention-error attribution, action-error
   ranking, and terminal task success; and
8. no claims about physical robot safety without collision or constraint data.

The official OpenVLA-OFT study reports 500 trials per suite and 8-step action
chunks; this is the scale and evaluation discipline a competitive follow-up
must approach, even if development gates are smaller.

## 10. Anti-loop governance

The following rules apply after this audit:

1. Do not assign a success probability to an untested method.
2. Do not rename a rejected method and preserve its architecture.
3. Do not add a new module to repair a failed feasibility gate.
4. Do not read terminal outcomes before R0--R3 pass.
5. Do not change thresholds after observing results.
6. Do not treat a positive microbenchmark as a positive paper result.
7. Do not treat an external paper’s speedup as evidence that the local stack has
   the same headroom.
8. Do not claim novelty from a combination unless the combination creates a new
   mechanism or identifiable research question.
9. If the sole conditional probe fails, stop the present cache-acceleration
   search and consult the advisors before selecting a broader architecture or
   problem pivot.
10. Reopen this audit only for a genuinely new fact: new measured local
    evidence, a new primary-source paper, a resource change, or advisor guidance.

## 11. Final decision

| Decision | Status |
|---|---|
| Start another full positive-method protocol | **No** |
| Continue CDRR/CDAR | **No** |
| Continue BRACE routing | **No** |
| Implement asynchronous dense refresh on TITAN | **No** |
| Run terminal outcome experiments | **No** |
| Conduct R0--R3 randomized-attribution feasibility study | **Recommended, but requires a separate explicit decision** |

The honest outcome of the reset is narrower than another proposed solution, but
more useful: the project now has one fixed, falsifiable research question and a
defined point at which to stop. That prevents the repeated cycle of declaring a
method promising and discovering its foundational problems one check later.
