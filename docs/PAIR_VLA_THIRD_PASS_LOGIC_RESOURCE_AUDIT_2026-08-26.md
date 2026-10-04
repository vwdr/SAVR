# PAIR-VLA Third-Pass Logic, Resource, and Plausibility Audit

**Date:** 2026-08-26

**Reviewed plan:** PAIR_VLA_RESEARCH_DIRECTION_AND_EXECUTION_PLAN_2026-08-26.md

**Reviewed version:** 1.1

**Corrected version:** 1.2
**Authorization:** Research and documentation only; no implementation,
download, GPU experiment, simulator run, or publication claim is authorized

## 1. Bottom-line verdict

PAIR-VLA remains the strongest direction found for an organic, related
positive-results paper, but the evidence does **not** support calling it a
high-confidence route yet.

The core idea remains scientifically coherent:

> Measure the harm caused by the exact stale K/V values a deployed cache would
> use, preserve their true source provenance, and train a small pre-decision
> router to avoid the high-regret reuse choices.

The third check found no fatal contradiction in that mechanism and no located
paper with the same combination of actual-stale expert-regret supervision,
exact source provenance, and recursive multi-query reuse contracts. It did find
four design gaps that could otherwise create misleading offline results:

1. raw demonstration frames did not yet match the model's eight-action query
   cadence;
2. independent masks did not reproduce cumulative cache provenance;
3. expert trajectories could not represent state drift caused by the cached
   policy; and
4. a fixed 15% raw speed gate did not guarantee 10% net speed after routing and
   fallback.

All four are corrected in plan version 1.2.

## 2. Logic audit

### 2.1 Does the supervision target the observed failure?

Yes, more directly than the earlier methods.

The local negative evidence showed that BRACE could reduce complete-cycle time
by 12.23%, yet all 168 tested cached actions differed from the same-stack dense
actions. Visual similarity, fixed age, and hand-designed layer/camera rules
therefore did not identify tolerable stale substitutions.

PAIR changes the selector's supervision rather than merely adding another
threshold. Its primary label is

\[
R_q^*(Z)=
\ell(A_q(Z),A_q^*)-\ell(A_q^0,A_q^*),
\]

the incremental expert imitation loss caused by an actual stale-cache
intervention relative to the same-stack dense model. This factors out the dense
policy's pre-existing imitation error and is closer to OpenVLA-OFT's supervised
objective than dense-action equality.

This is a defensible mechanism, but not a terminal-success guarantee. Better
expert imitation at an isolated state may still fail to preserve a long
closed-loop trajectory.

### 2.2 Does offline chronology match deployment?

Version 1.1 did not specify this tightly enough.

The official [OpenVLA-OFT LIBERO evaluation
loop](https://github.com/moojink/openvla-oft/blob/main/experiments/robot/libero/run_libero_eval.py)
requeries only when the action queue is empty and recommends executing the full
chunk. The official
[constants](https://github.com/moojink/openvla-oft/blob/main/prismatic/vla/constants.py)
define the LIBERO chunk as eight actions. The
[RLDS pipeline](https://github.com/moojink/openvla-oft/blob/main/prismatic/vla/datasets/rlds/dataset.py)
creates observation windows and longer future-action windows, but that does not
automatically turn every stored frame into a deployment-equivalent model query.

Primary label sequences must therefore use observations at the actual query
cadence: one query, eight expert actions, the next observation query, and so on.
Adjacent-frame substitution may be retained only as a diagnostic. P0 now
requires exact validation of padding, terminal truncation, no-op filtering,
normalization, gripper conversion, and the eight-action target window.

### 2.3 Does one intervention reproduce deployed cache state?

Not by itself.

If query \(q\) reuses a token from query \(q-1\), then the source available at
query \(q+1\) depends on whether that token was refreshed at \(q\). Cache
contents, source age, and layerwise hidden states are recursive. Generating
every label from a fresh independent cache would train on a treatment that the
deployed router never sees.

Version 1.2 therefore defines one-, two-, and four-query contracts. Each starts
from a clean anchor, executes the real cache suffix, and carries the actual
mixed cache and SourceLedger forward. Primary labels include per-query regret
and contract-level upper-tail regret. Single-query interventions remain useful,
but they cannot be the sole justification for multi-query deployment.

### 2.4 Do expert demonstrations solve closed-loop distribution shift?

No.

Demonstration observations were produced by expert actions. They let the study
isolate the computation error caused by stale K/V substitutions at meaningful
task states, but the observations at the next query are still expert-trajectory
states—not states caused by the cached policy's previous actions.

The corrected interpretation is:

- demonstration contracts establish label identifiability and train the first
  router;
- P6 tests whether that router transfers to closed-loop policy states; and
- at most one predeclared on-policy consistency/aggregation round is allowed if
  P6 reveals a clear transfer gap.

That round uses same-state dense-action distortion because expert actions are
unavailable, preserves the original router as an immutable comparator, and
cannot touch the final confirmatory population. Repeated on-policy redesign
would invalidate the clean paper claim.

### 2.5 Is expert action loss sufficient?

It is the correct primary offline target, but not sufficient alone.

The eight delta actions can accumulate into a materially different end-effector
endpoint even when average coordinate error looks modest. Version 1.2 therefore
predeclares integrated translation/rotation displacement and gripper-transition
error as secondary diagnostics. They may explain failures but may not replace
the primary loss after results are seen.

Signed regret is preserved. Negative regret—where caching accidentally moves
closer to the expert—is reported rather than silently clipped into a positive
claim. The risk model may target the positive tail only after the full signed
distribution is documented.

### 2.6 Are router inputs truly available before the decision?

This must be enforced feature by feature. Raw current/source patch change,
provenance, source age, current proprioception, and prior-query summaries are
chronologically available. A current projected visual feature is admissible
only if its producing computation occurs before the cache choice under the real
stack and its synchronized cost is charged. PAIR cannot use a feature produced
by the exact computation it claims to skip.

Demonstration replay also cannot use the expert's previous action as an oracle
router input. Recursive contracts now carry the model-produced previous action
summary, while expert actions remain supervision labels. Instruction features
receive no-instruction/task-identity ablations to detect task memorization.

## 3. Physical efficiency audit

### 3.1 Existing headroom is real but narrow

The best local BRACE point has:

- 18.71% accelerated-query reduction;
- 12.23% complete-cycle reduction; and
- approximately 1.220 seconds median same-stack cached query time.

The complete-cycle result proves that the substrate can save real time. It does
not leave a reliable margin for a 10% net paper result after router features,
provenance, dense fallbacks, and resets.

### 3.2 Why a fixed 15% raw gate is insufficient

If a raw profile saves 15%, is served on 70% of eligible queries, and adds only
1.5% total overhead, its approximate net saving is

\[
0.70(15\%)-1.5\%=9.0\%.
\]

Version 1.2 replaces the nominal raw gate with

\[
S_{\mathrm{net,LB}}=
\rho_{\min}S_{\mathrm{raw,LB}}-
O_{\mathrm{router,UB}}-
O_{\mathrm{reset/fallback,UB}}.
\]

Every term is frozen before outcome labels. P2 passes only if this conservative
lower bound is at least 10%; P5 must then measure at least 10% net complete-cycle
acceleration for the actual routed mixture.

This may require a raw profile above 15%, a high router service rate, extremely
small overhead, or some combination of the three. P2 is therefore the first
decisive feasibility test.

### 3.3 Architecture-specific ceiling

[CrossVLA](https://arxiv.org/html/2605.21854) reports that prefix processing is
only about 21% of a flow-matching \(\pi_{0.5}\) inference call on its H20 system,
so prefix caching has little maximum leverage there. That number does **not**
transfer directly to the autoregressive OpenVLA-OFT checkpoint or TITAN
hardware. It reinforces the correct principle: PAIR must use the local measured
latency anatomy and cannot infer speed from token counts or another backbone.

## 4. Resource feasibility

### 4.1 Verified disk state

A read-only check limited to /home/ved/SAVR found:

| Item | Measured state |
|---|---:|
| Current project | 32,316,980 KiB (30.82 GiB) |
| Filesystem free space | 380,667,804 KiB (363.03 GiB) |
| Checkpoints | about 14.84 GiB |
| Environments | about 8.73 GiB |
| Project cache | about 3.82 GiB |
| Optional compressed LIBERO training data | about 9.53 GiB |

Disk capacity is adequate. The dataset's unpacked and generated-cache size is
not yet known, so P0 still requires an estimate and a new explicit
project-local transfer/storage cap. The historical 25-GiB cap applied only to a
completed setup phase and is not authority for this download.

### 4.2 GPU memory

The project remains restricted to one responsibly selected idle GPU and a
strict 23-GiB peak-reservation cap. Prior BRACE execution peaked near 18.4 GiB,
leaving useful but not unlimited room.

PAIR is feasible only if label masks are evaluated serially, with one bounded
live source ring and one active contract. Holding many alternative K/V banks on
GPU simultaneously is unnecessary and could exceed the cap. Full-backbone
training, model sharding, and a new timing methodology remain outside this
direction.

Full K/V banks also must not be archived per training example: even a few
hundred MiB per source query would make a large intervention set impractical.
The corrected plan streams a bounded trajectory window, records only compact
features/provenance/labels, and discards reproducible K/V tensors after the
contract.

### 4.3 Runtime

At roughly 1.220 seconds per measured cached query, 1,000 pure model calls are
about 20 minutes of raw inference time. Data loading, clean controls, multiple
masks, model startup, contract resets, and analysis make real elapsed time
longer.

The plan is still practical on one GPU because:

- P3 starts with a small identifiability pilot;
- contracts are bounded to one, two, and four queries;
- the router trains separately as a small model;
- scale-up occurs only after label and physical gates pass; and
- every GPU phase receives a frozen query, hour, storage, and memory cap.

The longest likely cost is not router training but the final multi-method,
four-suite closed-loop evaluation.

## 5. Statistical feasibility

OpenVLA-OFT reports 500 trials per suite, but copying that count does not prove
a two-percentage-point paired non-inferiority claim is powered.

The variance of a paired success difference depends mainly on the fraction of
dense/PAIR discordant episode pairs. As an approximate illustration, with 10%
discordance, a two-point margin, one-sided 5% type-I error, and 80% power, the
normal approximation requires roughly 1,550 paired episodes for an overall
test—not 500. The exact frozen analysis may differ, but the dependency is real.

Version 1.2 therefore uses:

- a suite-stratified overall paired non-inferiority estimand;
- P6 discordance to freeze the confirmatory count before P7;
- mandatory per-suite/task intervals without claiming each suite is separately
  non-inferior unless powered; and
- a stop or narrower claim if the required trial count is unaffordable.

The planned 2,000 paired trials across all four suites may be adequate for the
overall estimand under plausible discordance, but that conclusion must come
from the frozen calculation, not this illustration.

Comparator points and the latency/reuse matching rule are frozen from the
outcome-blind cost table before closed-loop outcomes. The confirmatory tests use
a hierarchy—dense non-inferiority, then net latency, then matched-comparator
frontier—so a later favorable test cannot rescue an earlier failed paper gate.
Existing dense episodes may reduce duplicate compute only if checkpoint, code,
preprocessing, task, seed, and environment identity are exact and this reuse is
decided before PAIR outcomes.

## 6. Novelty and comparator audit

The nearest located methods still leave a defensible gap:

- [VLA-Cache](https://arxiv.org/abs/2502.02175) uses visual stability and
  relevance heuristics for temporal reuse.
- [LAC](https://arxiv.org/html/2602.00686) learns allocation with frozen-VLA
  task-loss gradients and optical-flow features.
- [Action-JND](https://arxiv.org/html/2608.21247) learns generic feature
  perturbation tolerance and uses it to rank reuse candidates.
- [Gated VLA-Cache](https://arxiv.org/abs/2608.10824) adds output-confidence
  invalidation at query level.

PAIR's remaining organic novelty is narrower:

> **actual-stale, source-resolved expert-regret supervision over recursive
> multi-query cache contracts.**

That is moderate, credible novelty—not proof of priority and not permission to
claim the first learned or action-aware VLA cache. A primary-source collision
search and faithful code/license audit remain mandatory at P0 and before
submission.

Because the router learns from expert demonstrations, the method is no longer
“training-free” in the strict sense. The VLA backbone is frozen and only a small
router is trained. The paper must also state whether router training includes
demonstrations from the same LIBERO task identities used in evaluation; task-
held-out and no-instruction ablations constrain, but do not erase, that scope.

If the newest comparator code is unavailable, the paper can still compare
against VLA-Cache, SAVR/BRACE, age-only, visual-only, attention, random, and a
faithfully implemented generic-perturbation control. It must then narrow any
state-of-the-art claim.

## 7. Predicted failure modes and when they become visible

| Possible failure | Earliest honest detection | Consequence |
|---|---|---|
| No raw profile supports a 10% routed lower bound | P2 | Stop PAIR; substrate lacks sufficient speed reserve |
| Actual-stale regret is mostly numerical noise | P3 pilot | Stop; supervision is not identifiable |
| Regret is real but cheap features cannot predict it | P3/P4 | Stop or report mechanism study; no deployable router |
| Router falls back on almost every query | P4/P5 | Stop; positive result would be a disguised dense policy |
| Demonstration performance does not transfer on-policy | P6 | Use the one frozen aggregation allowance or stop |
| PAIR matches dense but not a strong cache baseline | P6/P7 | No positive comparative method paper |
| Required confirmatory sample is unaffordable | P7 freeze | Widen margin/claim with advisor approval or stop |
| Recent work collides with the contribution | P0/submission search | Reframe or stop before spending full compute |

These are scientific stop rules, not prompts to rename the same idea and retry
until a positive point appears.

## 8. Plausibility judgment

No defensible exact probability can be computed before P2 and the P3 pilot.
Assigning a percentage now would be subjective precision, not evidence.

| Dimension | Plausibility |
|---|---|
| Method can be implemented in the existing repository | High |
| Dataset fits available disk with a new bounded approval | High |
| One-GPU frozen-label workflow fits the known memory boundary | Moderate-high |
| A physically valid raw profile can support 10% net speed | Moderate-low, unresolved |
| Actual-stale regret is measurable and predictable | Moderate, unresolved |
| Offline router transfers to closed-loop states | Moderate-low, highest scientific risk |
| Contribution remains organically novel | Moderate |
| Direction can produce the targeted positive paper | **Plausible but conditional, not yet high-confidence** |

The direction is worth executing because it is logically connected to the
observed failure, materially distinct from the closest work, implementable with
the current code and one-GPU boundary, and protected by early stop gates.

It should **not** be represented as likely to succeed merely because related
learned selectors report positive results. The route becomes genuinely
promising only after:

1. P2 proves a conservative 10% net-speed path;
2. the P3 pilot proves stable, source-aware regret prediction; and
3. P6 shows the locked router improves the real closed-loop
   success--latency frontier.

Until those three results exist, PAIR-VLA is a well-founded experiment plan,
not a positive-result conclusion.
