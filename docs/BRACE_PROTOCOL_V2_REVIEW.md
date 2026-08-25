# Independent Review of BRACE Execution Protocol V2

**Date:** 2026-08-24

**Reviewed artifact:** `docs/BRACE_EXECUTION_PROTOCOL_V2.md`

**Review type:** Adversarial feasibility, validity, novelty, resource, and
positive-result audit

**Evidence boundary:** No BRACE implementation, model run, GPU operation, or
outcome exists

**Status:** Historical V2 review, superseded by
`docs/BRACE_PROTOCOL_V2_1_RED_TEAM_REVIEW.md`. Its probability ranges are not
the active assessment.

## 1. Verdict

Protocol V2 is internally coherent enough to begin the CPU-only B1 replay
harness. It corrects the four critical defects in V1: one-step treatment
mismatch, self-harvested semantic gates, incomplete staleness features, and the
missing actuation-slack baseline.

It is **not** yet evidence that BRACE will work. The highest-risk scientific
unknown remains whether terminal harm from bounded clean-provenance cache
contracts is both common enough to learn and predictable from inexpensive
pre-inference information at useful cache coverage.

Current judgment:

| Event | Plausibility after V2 |
|---|---:|
| B1 exact replay acceptance | 80--90% |
| B2 clean-cache/provenance implementation with P0 parity | 60--75% |
| B3 at least 8% net contract-cycle acceleration | 50--65% |
| B4 usable horizon-sensitive label population within budget | 35--50% |
| B5 useful held-out harm prediction at speed-eligible coverage | 30--45% |
| Competitive positive paper from the complete program | **25--40%** |
| Competitive result conditional on B1--B5 all passing | **50--60%** |

These ranges are engineering/research judgments, not statistical probabilities.
The protocol is worth stop-fast testing, but a positive result is not highly
likely before the physical-speed, prevalence, and predictability gates pass.

## 2. What V2 now gets right

### 2.1 Treatment matches deployment

BRACE selects a bounded `(profile, horizon)` contract and the intervention tests
that same contract for 1, 2, or 4 queries. This can reveal cumulative failures
that single-step action or outcome labels miss. Contract expiry also prevents
unbounded self-reinforcing reuse.

### 2.2 Gate provenance is explicit

Official VLA-Cache uses previous attention to filter task-relevant tokens. When
that attention comes from an accelerated forward, it is self-harvested. V2
allows self-harvested behavior only as diagnostic P3 and requires deployable
P1/P2 semantic masks and layer schedules to originate from P0.

### 2.3 Mixed-age cache state is observable

The per-layer/per-token source ledger and current-to-source difference maps
address the fact that a partially updated cache does not correspond to one
previous frame. The proposed features remain pre-inference and can therefore
preserve the latency premise if their measured overhead is small.

### 2.4 Causal analysis is defensible in simulation

Fresh reset plus exact low-level prefix replay is stronger than direct MuJoCo
state restoration. Bounded randomized contracts, common continuation,
duplicate-arm controls, randomized arm order, and intention-to-treat analysis
provide a defensible simulator intervention if B1 equivalence passes.

### 2.5 The comparison standard is appropriately difficult

V2 requires official VLA-Cache, a static clean contract, matched periodic FR, a
confidence gate, action-disagreement supervision, and actuation-slack refresh.
It also requires total GPU work when off-path refresh wins critical-path latency.
This prevents a trivial positive claim against FR alone.

## 3. Residual scientific risks

| Risk | Why it still matters | Protocol disposition |
|---|---|---|
| Clean semantic gate becomes stale | Fixed anchor attention may miss an object/region that becomes important during a contract | Horizons capped at four; source-change hard aborts; horizon-specific evaluation. Residual risk remains. |
| Harm is a sequence interaction | Even four queries may not capture longer accumulation | Four is a bounded first claim; longer contracts require a new protocol. Do not generalize beyond evaluated horizons. |
| Future hard abort is unknowable at entry | The router cannot see future camera motion | Intention-to-treat includes later aborts; speed/coverage must include them. |
| Terminal label is discontinuous | Tiny action differences can change binary success after a long delay | Paired branches and time/progress diagnostics help, but terminal success remains primary. |
| FR-continuation labels drift | Contract effects may change under BRACE continuation | One versioned aggregation round and explicit drift gate. One round may still be insufficient. |
| Rare harmful class | Reliable contracts may generate too few positive labels | B4 prevalence and 40-hour scaling stop. Stress-only labels cannot support deployable claims. |
| Router cannot see semantic risk cheaply | Cheap image/provenance features may not reveal task-critical change | B5 predictability stop; current dense-VLA features are prohibited as a rescue. |
| OOD rejection is overconfident or rejects everything | Empirical OOD detectors do not guarantee reliability | Treat as a measured ablation, not a formal guarantee; full-refresh fallback and service-coverage reporting. |
| Profile/horizon pooling hides failures | A model may perform well overall but fail on one contract | Contract-specific evidence rule and task/horizon reporting. |
| Simulator-specific labels | Real contact, camera noise, and timing differ | Claims remain checkpoint/LIBERO-specific without physical validation. |

## 4. Engineering feasibility

### 4.1 Replay harness

High feasibility. The project already has a pinned LIBERO/robosuite environment
and a successful CPU-only OSMesa smoke test. The main work is transcript
completeness and strict equivalence, not a new simulator installation.

### 4.2 Cache integration

Moderate feasibility. The proven environment uses a customized Transformers
4.40.1 fork, while VLA-Cache uses a customized 4.47.0 fork. The cache and LLaMA
files differ substantially, so installation alone is not proof of compatibility.
The isolated runtime, P0 parity, randomized-order tests, and transactional
configuration restoration are mandatory.

### 4.3 Provenance maps

Feasible but nontrivial. Exact source identity can be updated beside each
in-place K/V update. The larger risk is semantic correctness: per-layer token
positions must remain aligned across scene/wrist tokens, action/proprio tokens,
cropping, and profile switching. Synthetic tests must intentionally permute
camera and position mappings.

### 4.4 Memory

Likely feasible. Historical eager inference peaked near 15.38 GiB under a 23 GiB
cap, and rough visual K/V storage is approximately 0.25 GiB before auxiliary
attention/provenance. Sequential branch arms avoid holding multiple models, but
the B3 measured peak is authoritative.

### 4.5 Timing

Uncertain. Historical FR query time on TITAN was approximately 1.267 seconds,
far slower than the modern GPUs used by recent caching papers. This makes a
small router relatively cheap but may prevent a dense pass from fitting inside
roughly one action chunk of real-time slack. B3 must measure amortized
anchor/contract cycles rather than only accelerated queries.

## 5. Data, statistics, and total cost

Historical throughput was about 62.1 complete episodes per GPU-hour.

To obtain the recommended 190 harmful examples across training, validation,
and held-out test:

| Natural harmful prevalence | Required paired contracts | Lower-bound equivalent episode-hours before overhead |
|---|---:|---:|
| 5% | 3,800 | 61.2 h |
| 10% | 1,900 | 30.6 h |
| 15% | 1,267 | 20.4 h |

The 40-hour development cap is therefore credible only when harmful prevalence
is roughly 10% or higher, or targeted outcome-blind enrichment is efficient. B4 must
measure actual partial-rollout throughput; the table is only a conservative
planning conversion.

Sixty harmful held-out cases provide only a modest detector claim. With zero
misses, the simple 95% upper bound is approximately 4.9%. Any stronger risk
claim requires more positives and group-aware analysis.

Final confirmation remains expensive. At historical throughput, 500 episodes
for three methods across four suites is approximately 96.6 GPU-hours. V2
correctly requires a power-selected final comparison set instead of repeating
all ten development methods.

## 6. Novelty and competitive position

The field already contains:

- VLA-Cache for adaptive temporal visual-KV reuse;
- LAC for learned token selection/cache ratios;
- AC2-VLA for action-context adaptive caching/pruning/layer execution;
- Gated VLA-Cache for confidence invalidation;
- Action-JND for action-tolerance supervision;
- The Gate, Not the Cache for clean gate provenance and off-path refresh;
- LoopVLA and DySL-VLA for learned adaptive computation.

BRACE therefore cannot claim learned routing, action awareness, cache
provenance, bounded refresh, or adaptive compute individually. Its defensible
provisional novelty is:

> pre-inference selection of bounded clean-provenance cache contracts trained
> from replay-verified paired terminal intervention effects on cache-induced
> trajectories.

That contribution is meaningful if BRACE beats simpler baselines. It is weak if
the terminal labels do not outperform source-difference thresholds, if P4 hides
all critical-path cost, or if the router refreshes so often that a static clean
contract matches it.

## 7. Likely reviewer objections

1. **Why not always refresh off path?** Measure whether FR fits the actual TITAN/control slack and compare total GPU
   work. BRACE must win at least one meaningful frontier.

2. **Why are simulator rollouts a scalable training signal?** Present this as simulator interventional supervision for one checkpoint, not
   a universal real-robot data recipe. Report label cost explicitly.

3. **Why should a cheap router predict long-horizon task harm?** This is the central B5 hypothesis, not an assumption. Failure ends BRACE.

4. **Does clean-anchor attention remain valid?** Only within evaluated horizons and hard-abort envelopes. Include gate-age and
   no-provenance ablations.

5. **Are positive operating points selected post hoc?** Freeze contracts and thresholds on development data and use untouched sealed
   confirmation groups.

6. **Does the result transfer beyond TITAN/LIBERO?** Not without additional hardware, checkpoints, suites, or physical tests.
   Limit the claim accordingly.

## 8. Final go/no-go recommendation

- **GO:** B1 CPU replay harness. It is low-cost, independently useful, and does
  not assume the cache method works.
- **CONDITIONAL GO:** B2 only after V2 is accepted as the active protocol and a
  reviewed Transformers/cache integration plan is frozen.
- **NO-GO WITHOUT NEW AUTHORIZATION:** B3 or later, any model/checkpoint load,
  GPU use, download, simulator outcome collection, or protected-state access.

Protocol V2 is a materially stronger research plan than V1. It makes a positive
paper plausible, not probable. The correct next decision is whether to invest in
B1 and let the early gates determine whether the project deserves the much more
expensive B3--B8 path.

## 9. Primary sources rechecked

1. VLA-Cache: <https://openreview.net/forum?id=QZYZ0Xm58q>
2. The Gate, Not the Cache: <https://arxiv.org/abs/2608.00391>
3. Gated VLA-Cache: <https://arxiv.org/abs/2608.10824>
4. Action-JND: <https://arxiv.org/abs/2608.21247>
5. LAC: <https://arxiv.org/abs/2602.00686>
6. AC2-VLA: <https://arxiv.org/abs/2601.19634>
7. ActionCache: <https://arxiv.org/abs/2607.06370>
8. LoopVLA: <https://arxiv.org/abs/2605.09948>
9. DySL-VLA: <https://arxiv.org/abs/2602.22896>

Recent preprints remain provisional and must be rechecked before experimental
freeze and manuscript submission.
