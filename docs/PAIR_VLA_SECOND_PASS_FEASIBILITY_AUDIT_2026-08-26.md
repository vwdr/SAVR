# PAIR-VLA Second-Pass Feasibility Audit

> **Supersession note (2026-08-26):** The later
> PAIR_VLA_THIRD_PASS_LOGIC_RESOURCE_AUDIT_2026-08-26.md and execution-plan
> version 1.2 supersede this document's nominal 15% raw-speed gate and add
> deployment-aligned query timing, recursive cache contracts, resource evidence,
> and powered confirmatory sampling. This file is retained as an audit trail.

**Date:** 2026-08-26

**Reviewed plan:** PAIR_VLA_RESEARCH_DIRECTION_AND_EXECUTION_PLAN_2026-08-26.md

**Result:** **PAIR-VLA remains the recommended direction, but only after four material corrections incorporated into plan version 1.1.**
**Authorization:** Research review only; no implementation, dataset download, GPU use, or outcome run is authorized

## 1. Verdict

PAIR-VLA remains the best available positive-paper route under the user's
criteria of organic novelty, positive results, and close relevance to SAVR.
The second review did not find a stronger locally feasible alternative.

Version 1.0 was nevertheless too optimistic in two places: its primary label
imitated the dense model rather than the expert action, and its only demonstrated
speed point left almost no overhead reserve. Version 1.1 fixes those problems
and also tightens treatment fidelity and comparator requirements.

The corrected paper thesis is:

> Exact provenance reveals which stale representation will be substituted;
> controlled actual-stale interventions reveal whether that substitution
> increases expert imitation loss; and a lightweight router can distill this
> signal into a better success--latency frontier than proxy-based cache routing.

This is a plausible positive mechanism. It is not a promise that the final
closed-loop gates will pass.

## 2. Material findings from the second review

| Finding | Severity in v1.0 | Correction in v1.1 |
|---|---|---|
| Dense-action parity is not task quality | High | Use cached-minus-dense **expert imitation loss** on sequential demonstrations as the primary offline label; keep dense distortion secondary |
| 12.23% raw saving leaves only 2.23 points above the paper floor | High | Add an outcome-blind profile-frontier phase requiring at least 15% raw complete-cycle reduction before labels or outcomes |
| Isolated K/V replacement does not reproduce compounded reuse | High | Primary labels must execute the real cache suffix with the complete source ledger; isolated swaps are diagnostic only |
| Token effects may interact non-additively | High | Train on full structured profile masks and use a set-level calibrator; do not justify deployment from one-token effects |
| Demonstration states differ from policy states | Medium-high | Train primary task-aligned labels on demonstrations, add dense-consistency data from policy states, and permit one frozen aggregation round only after the first screen |
| The strongest new comparator may lack usable code | Medium-high | Audit code/license before implementation; narrow claims if a faithful Action-JND/LAC comparison cannot be run |
| High dense success creates a ceiling | Medium | Evaluate at physically meaningful aggressive reuse where cache baselines lose success; require non-inferiority to dense and superiority on the matched cache frontier |
| Router data rows are strongly correlated | Medium | Split by whole trajectories and task prefixes; never randomize token rows across train/test |

## 3. Why expert regret is a better target

Let \(A_t^*\) be the expert action chunk, \(A_t^0\) the same-stack dense output,
and \(A_t(Z)\) the output under an actual stale cache mask. The revised target is

\[
R_t^*(Z)=\ell(A_t(Z),A_t^*)-\ell(A_t^0,A_t^*).
\]

This measures the incremental supervised-task harm caused by caching while
factoring out the dense policy's existing error. It distinguishes three cases
that action parity conflates:

1. cache changes the action and worsens expert imitation;
2. cache changes the action but improves expert imitation; and
3. cache matches a dense action that is already imperfect.

The official [OpenVLA-OFT dataset
pipeline](https://github.com/moojink/openvla-oft/blob/main/prismatic/vla/datasets/rlds/dataset.py)
constructs trajectory windows and future actions, so this target is technically
available once the exact pinned normalization and eight-action semantics are
verified.

The target still cannot prove terminal success. It is a stronger offline
surrogate, followed by a real closed-loop test.

## 4. Positive-result mechanism versus the closest work

- [LAC](https://arxiv.org/html/2602.00686) learns cache selection from task-loss
  gradients and demonstrates that learned caching can improve both speed and
  success. It does not supervise a router with the measured effect of the exact
  stale values and their source provenance.
- [Action-JND](https://arxiv.org/html/2608.21247) learns how much generic feature
  perturbation a VLA action can tolerate, then applies one shared token ranking
  to previous-step stale K/V reuse. PAIR instead measures the actual stale
  direction, source age, camera, and layer, and trains on its excess expert
  regret.
- [Learning What Matters](https://arxiv.org/html/2607.21692) and
  [CausalGate](https://arxiv.org/html/2607.22720) support intervention-derived
  supervision over attention or observational importance, but neither studies
  temporal VLA caching.
- [Error Certificates for KV-Cache Eviction](https://arxiv.org/html/2607.21475)
  warns that cache-side attribution is not general failure prediction. PAIR
  therefore uses interventions offline and keeps terminal success as the final
  outcome.

The remaining novelty is moderate rather than revolutionary, but it is organic
and testable: **actual-stale, provenance-resolved, expert-regret supervision for
VLA cache routing.**

## 5. Feasibility assessment

| Dimension | Assessment | Basis |
|---|---|---|
| Cache integration | High | BRACE already has a working partial-reuse path, exact token/layer source tracking, camera identity, age caps, and measured speedup |
| Label generation | Moderate-high | Frozen inference avoids 7B backpropagation; official sequential demonstrations contain expert actions, but exact preprocessing alignment must pass P0 |
| Router training | High | Only a compact group/set model is trained from stored features and labels |
| Raw physical headroom | Unresolved and decisive | 12.23% is demonstrated but insufficiently buffered; P2 must find at least 15% raw complete-cycle saving |
| Offline predictability | Unresolved and decisive | Intervention supervision is better motivated than proxies, but cheap online features may still fail to predict regret |
| Closed-loop transfer | Moderate risk | Expert demonstrations do not cover all policy-induced states; the staged on-policy consistency check is required |
| Organic novelty | Moderate | Distinct from generic perturbation tolerance and task-gradient selectors, but close enough that careful wording and strong ablations are mandatory |
| Positive-paper plausibility | Credible but conditional | External methods show large gains from learned/action-aware selection; local stack has some real speed headroom; both must combine successfully |

No defensible numeric success probability is assigned. Previous exact-looking
probabilities changed when missing constraints were discovered and were not
statistical estimates.

## 6. Decisive experiment chain

PAIR should be considered increasingly credible only as this chain passes:

1. **Physical reserve:** a profile achieves at least 15% raw complete-cycle
   saving without outcomes.
2. **Label validity:** actual cache-suffix interventions produce repeatable,
   nontrivial excess expert regret.
3. **Predictability:** provenance-aware features beat visual change, attention,
   age, VLA-Cache heuristics, and the strongest faithfully runnable action-aware
   control on held-out trajectories.
4. **Net efficiency:** the complete deployed router retains at least 10%
   complete-cycle acceleration after fallbacks and overhead.
5. **Closed-loop development:** PAIR improves the paired success--latency
   frontier rather than preserving success by falling back almost everywhere.
6. **Independent confirmation:** across four LIBERO suites, PAIR is within the
   frozen 2-point dense-success margin, at least 10% faster, and better than the
   strongest cache comparator at matched compute.

Only step 6 establishes the targeted positive paper.

## 7. Outcomes that would invalidate the route

- No profile reaches the raw speed-reserve gate: the local cache substrate is
  too weak for a 10% net paper result.
- Actual-stale expert regret is dominated by repeat/numerical noise: the label
  is not identifiable enough to train the router.
- Provenance features do not beat cheap proxies: the proposed contribution is
  not useful even if the labels are scientifically interesting.
- The router passes demonstrations but fails policy-state transfer: the method
  does not solve the closed-loop distribution problem.
- Net speed falls below 10% after fallbacks: the router is too conservative or
  expensive.
- A direct paper appears before the implementation freeze: the novelty claim
  must be reassessed.

These are stop conditions, not prompts for immediate threshold changes.

## 8. Remaining reviewer attacks even if results are positive

1. **One checkpoint and simulator.** The main claim must remain OpenVLA-OFT/
   LIBERO-specific unless a second backbone is later feasible.
2. **Training-data dependence.** Expert-regret labels use the task's training
   demonstrations. This is legitimate for a learned router but weakens claims
   of task-free or training-free deployment.
3. **Very recent competitors.** Action-JND, LAC, and Gated VLA-Cache require
   careful, version-pinned comparison and cannot be cited only from abstracts.
4. **Surrogate-to-success gap.** Ablations must show that better expert regret
   translates into terminal success, not merely a better offline metric.
5. **Hardware specificity.** Physical latency gains must be reported on the
   actual TITAN hardware without generalizing the numeric speedup to other
   accelerators.
6. **No formal safety.** Terminal success is reliability evidence, not collision
   or constraint-satisfaction evidence.

These limitations do not prevent a positive paper if the locked frontier is
strong; they control how broadly it can be framed.

## 9. Final decision

The direction remains **PAIR-VLA**, with the version 1.1 plan replacing the
original formulation. The immediate next phase, if authorized later, is P0:
protocol/data/collision freeze. The method should not yet be coded or run.

The strongest reason to continue is that PAIR now tests a direct, task-aligned
question using infrastructure already built by this project. The strongest
reason for caution is that neither raw latency reserve nor offline-to-closed-
loop transfer has yet been demonstrated. The revised gates test those two
uncertainties before the expensive paper evaluation.
