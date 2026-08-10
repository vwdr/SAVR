# IR-SA-ACR Manuscript Translation and Claim Guide

Status: **WRITING GUIDE ONLY — MANUSCRIPT NOT MODIFIED**

Date: 2026-08-10

## 1. Purpose

This guide maps the implemented method and future evidence into a formal paper.
It prevents the manuscript from mixing the original SAVR proposal, negative
V3/V4 development evidence, the V5 correction, and results that have not yet
been measured.

## 2. Recommended naming hierarchy

- **State-Aware Visual Refresh (SAVR):** the umbrella research idea.
- **Asymmetric Camera Refresh (ACR):** the architectural decomposition that
  may cache the scene camera while keeping wrist perception fresh.
- **Isolated-Reuse State-Aware ACR (IR-SA-ACR):** the exact V5 method under
  evaluation.

Use `IR-SA-ACR` when making claims about the final controller. Do not call a
legacy V3/V4 result a V5 result.

## 3. Paper-level contribution statements

These are defensible method contributions now:

1. a training-free asymmetric visual-refresh wrapper that treats scene and
   wrist camera features differently;
2. state-aware reuse eligibility combining scene change, normalized robot
   translation, and an action-history-derived gripper-transition veto;
3. an isolated-reuse state machine that forces a completed refresh after each
   scene reuse and cross-checks internal state against cache age; and
4. an auditable, fail-closed controller with explicit reasons and a hard
   episode-prefix reuse cap.

These are not yet defensible empirical contributions:

- preserved LIBERO success;
- reduced visual computation by a specific percentage;
- reduced sequential or wall-clock latency;
- superiority to Full Refresh, Periodic Refresh, VOR, or recent literature;
- generalization across tasks, models, robots, or hardware; or
- a “positive-results paper.”

## 4. Section-by-section manuscript map

### Title

Keep the broad title only if the final evaluation supports meaningful
efficiency without unacceptable task-success loss. Otherwise explicitly name
the scope, for example “Isolated-Reuse State-Aware Camera Refresh for VLA
Inference.” Final wording waits for results.

### Abstract

Use four evidence-linked sentences:

1. problem: repeated VLA visual encoding is costly, but stale scene features
   can harm reactive control;
2. method: IR-SA-ACR reuses only the scene-camera representation when scene,
   normalized translation, action-derived gripper, temporal-isolation, and
   hard-cap gates permit;
3. protocol: paired evaluation against Full Refresh with success as the
   primary constraint and efficiency as secondary outcomes; and
4. measured result: insert only reconciled final numbers and confidence
   intervals.

Do not write the fourth sentence until confirmation is complete.

### Introduction

Motivate the tension between redundant visual computation and closed-loop
reactivity. State that visual similarity alone cannot establish safe reuse.
Introduce two core ideas: asymmetric camera treatment and controller-owned
temporal isolation. End with contribution bullets that distinguish method,
software invariants, and measured results.

### Related work

Organize by concept rather than chronology:

- VLA caching and token reuse: VLA-Cache, FlashVLA;
- adaptive multimodal compute: AC2-VLA and related work;
- stale/open-loop correction and temporal alignment: VLA-Corrector, VLASH;
- action or intermediate caching: ActionCache, explicitly outside this
  method's action-fresh boundary; and
- event-triggered control as an architectural analogy, not a transferred
  stability theorem.

Use the applicability limitations from `docs/ACR_V5_RESEARCH_AUDIT.md`. Never
imply that a cited paper proves one-step reuse safe for OpenVLA-OFT/LIBERO.

### Problem formulation

Define scene image $S_t$, wrist image $W_t$, state $q_t$, instruction $x$,
visual encoders $\phi_s,\phi_w$, cached scene representation $C_t$, and action
chunk $A_t$. Present the refresh and reuse equations from
`docs/ACR_V5_FORMAL_METHOD_SPECIFICATION.md`.

The optimization objective should be constrained, not a single weighted score:

\[
\min \; \text{visual work and latency}
\quad \text{subject to} \quad
\Delta\text{Success} \ge -\delta,
\]

where $\delta$ is a predeclared non-inferiority tolerance. Do not select
$\delta$ after observing final outcomes.

### Method

Use this order:

1. asymmetric scene/wrist computation boundary;
2. refresh references and cache lifetime;
3. scene change, normalized translation, and the action-derived gripper veto;
4. threshold and hard-cap eligibility;
5. isolated-reuse latch and cache-age agreement;
6. completed-query observation semantics;
7. ordered fail-closed reason codes; and
8. computational paths and theoretical work accounting.

Include the exact finite-prefix trace grammar
$R^*(UR^+)^*(\epsilon\mid U)$, equivalently “no `UU`,” and explicitly say that
a failed or unobserved refresh cannot clear the latch.

### Algorithm

Translate the normative pseudocode verbatim in meaning, but use publication
notation. The algorithm must expose controller state, `DECIDE`, `OBSERVE`, and
episode reset. Avoid presenting `horizon=1` alone as the isolation mechanism;
the controller-owned latch is primary, while horizon and cache-age agreement
are defense in depth.

### Experimental setup

Report:

- exact OpenVLA-OFT checkpoint and source revision;
- LIBERO version, suites, task IDs, seeds, and episode counts;
- image/state/action preprocessing;
- controller/config semantic hashes;
- candidate-selection population versus protected confirmation population;
- Full Refresh oracle and matched baselines;
- hardware, selected GPU ID, driver/CUDA/PyTorch versions;
- warm-up, synchronization, timing boundaries, and repetition policy;
- technical-failure handling and immutable restart rules; and
- statistical estimands, intervals, gates, and multiplicity policy.

### Results

The primary table should use paired episodes:

| Policy | Success n/N | Paired difference vs FR | 95% CI | Scene reuse | Visual-work reduction | Sequential/FR | Wall/FR | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Full Refresh | TBD | reference | TBD | 0 | 0 | 1.000 | 1.000 | oracle |
| IR-SA-ACR | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

The required ablation table should isolate mechanisms:

| Variant | Scene | Translation | Action-derived gripper veto | Latch | Hard cap | Success delta | Reuse | Wall/FR |
|---|---|---|---|---|---|---:|---:|---:|
| Visual-only | on | off | off | on | on | TBD | TBD | TBD |
| + translation | on | on | off | on | on | TBD | TBD | TBD |
| IR-SA-ACR | on | on | on | on | on | TBD | TBD | TBD |
| Direction-reversal diagnostic | on | on | on + reversal veto | on | on | TBD | TBD | TBD |

Report negative and null results alongside positive ones. Do not omit a frozen
candidate because its result is inconvenient.

### Discussion

Separate:

- what the paired data directly show;
- plausible mechanism interpretations;
- executor-specific versus controller-specific effects;
- where asymmetric refresh succeeds or fails; and
- whether the final contribution is positive efficiency, a bounded trade-off,
  or a well-characterized negative result.

### Limitations

At minimum cover whole-scene cache granularity, single-stack evaluation,
threshold dependence, absence of a safety theorem, simulator-to-real transfer,
timing dependence on hardware/executor, limited statistical power, and the
fact that temporal isolation does not detect all relevant within-step changes.

### Conclusion

Restate only claims that survive the final claim audit. If task success is
preserved but wall time is not improved, the conclusion must say so. If a
positive route succeeds, include both its magnitude and evaluation scope.

## 5. Development history disclosure

V3 and V4 are not final-method evidence, but they are scientifically relevant
development history:

- V3 established that the tested route could preserve paired success while
  missing predeclared efficiency gates.
- V4 exposed a temporal-semantics mismatch and selected no eligible redesign.
- V5 corrected that mechanism prospectively with a separate controller.

The main paper may summarize this history as method-development motivation and
place detailed tables in an appendix. It must not describe the V5 correction
as having been designed before V3/V4 observations. Protected confirmation data
must remain independent of this development process.

## 6. Claim-to-evidence rules

| Claim type | Minimum evidence |
|---|---|
| Code implements isolated reuse | Frozen protocol + tests + machine verifier |
| Maximum completed reuse streak is one | Corrected trace and adversarial observation tests |
| Success is preserved | Predeclared paired non-inferiority result and CI |
| Visual work is reduced | Instrumented completed-query accounting under frozen formula |
| End-to-end inference is faster | Synchronized measured wall time with warm-up and paired repetitions |
| Method generalizes | Independent tasks/suites or models not used for selection |
| Mechanism causes improvement | Predeclared ablation, not correlation alone |

Every numerical manuscript claim must map to a versioned result file, config,
code revision, and analysis script. “Approximately” does not remove this
requirement.

## 7. Prohibited writing shortcuts

- Do not call CPU tests experimental validation.
- Do not report theoretical skipped work as measured speed.
- Do not use the best observed threshold from an opened population as if it
  were predeclared.
- Do not merge V3/V4/V5 samples into one final estimate.
- Do not claim a literature method is directly comparable without matching
  model, task, compute boundary, and metric definitions.
- Do not conceal technical failures, reruns, or stopped gates.
- Do not modify the abstract/title to imply positive results before the final
  claim audit.

## 8. Artifact map for formal reporting

| Reporting need | Authoritative artifact |
|---|---|
| Literature rationale | `docs/ACR_V5_RESEARCH_AUDIT.md` |
| Exact method and equations | `docs/ACR_V5_FORMAL_METHOD_SPECIFICATION.md` |
| Design frozen before code | `docs/ACR_V5_ISOLATED_REUSE_PROTOCOL.md`; freeze JSON |
| Implementation provenance | `docs/ACR_V5_IMPLEMENTATION_AND_PROVENANCE_LEDGER.md` |
| CPU result | `reports/PHASE_V5_A_CORRECTION_REPORT.md`; runtime JSON |
| Future evaluation rules | `docs/ACR_V5_GATED_EVALUATION_ROADMAP.md` |
| Project decisions | `docs/DECISIONS.md` |
| Phase chronology | `docs/MILESTONES.md` |

The manuscript should not be changed until a specific writing phase is
authorized and the evidence required for that section exists.
