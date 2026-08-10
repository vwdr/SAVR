# IR-SA-ACR: Formal Method Specification

Status: **IMPLEMENTED AND CPU-VERIFIED; NOT EXPERIMENTALLY VALIDATED**

Specification date: 2026-08-10
Controller identity: `acr-isolated-controller-v1`
Method name: **Isolated-Reuse State-Aware Asymmetric Camera Refresh**
(`IR-SA-ACR`), within the State-Aware Visual Refresh (`SAVR`) project.

## 1. Claim boundary

This document specifies the exact method that exists in code. It does not
claim that the method preserves task success, reduces benchmark visual work,
or lowers end-to-end latency. The verified result is narrower: for accepted,
successfully observed queries, the controller never permits two consecutive
scene-feature reuses and fails closed when its internal temporal state
disagrees with the adapter's cache age.

The method is training-free. It does not change model weights, reuse actions,
prune tokens, alter the action head, or use task-specific rules.

## 2. Problem and computation boundary

At control query $t$, let:

- $S_t$: whole-scene camera image;
- $W_t$: wrist-camera image;
- $q_t$: current robot proprioceptive state;
- $x$: language instruction;
- $\phi_s$, $\phi_w$: scene and wrist visual encoders;
- $h$: the unchanged downstream VLA computation and action head; and
- $A_t$: the newly predicted action chunk.

Full refresh computes

\[
Z_t^{R} = [\phi_s(S_t);\phi_w(W_t)], \qquad
A_t = h(Z_t^{R},q_t,x).
\]

When reuse is permitted, only the scene-camera block is cached:

\[
Z_t^{U} = [C_t;\phi_w(W_t)], \qquad
A_t = h(Z_t^{U},q_t,x),
\]

where $C_t=\phi_s(S_r)$ is the scene representation from the most recent
successfully completed scene refresh $r<t$. Thus the wrist image, current
proprioception, downstream computation, and action prediction remain fresh at
every query.

## 3. Controller inputs and state

### 3.1 Inputs at query $t$

The decision receives:

- query index $t$;
- episode context identifier;
- whether a compatible scene cache exists;
- external scene-cache age $a_t$;
- scene-change signal $d_t^S$;
- normalized end-effector translation $d_t^Q$;
- action-history-derived gripper-transition flag $g_t$; and
- validity flags for every required signal.

The scene signal is the mean of the four largest patch-level absolute
image-change scores relative to the last successfully refreshed scene
reference. The state signal is Euclidean end-effector translation after
normalization. The two latest completed action chunks provide a gripper-
transition veto. Translation-direction reversals are also computed and logged,
but the V5 controller does not use them as a refresh veto. There is no
continuous action-change threshold in this implementation. Exact preprocessing
and normalization are part of the adapter/configuration provenance and must be
frozen before any outcome-bearing run.

### 3.2 Persistent controller state

For each episode the controller maintains:

- $L_t\in\{0,1\}$: post-reuse refresh latch;
- $N_t^R$: completed refresh count;
- $N_t^U$: completed reuse count;
- last refresh references for scene and robot state;
- completed action history needed by $g_t$; and
- active episode context.

Reset clears all of this state. It is forbidden to carry the latch, cache, or
reference state across episodes.

## 4. State-aware reuse eligibility

Let the frozen thresholds be $\tau_S,\tau_Q$, and let $\rho$ be the hard
maximum prefix reuse fraction. Define the prospective fraction after reusing
query $t$ as

\[
\widehat r_t = \frac{N_t^U+1}{N_t^R+N_t^U+1}.
\]

A reuse is eligible only when every condition below is true:

\[
E_t = \mathcal C_t \land \mathcal V_t \land \mathcal W_t
\land \mathcal H_t \land \mathcal P_t \land \mathcal T_t,
\]

where:

- $\mathcal C_t$: cache exists, is compatible with the current episode, and the
  context matches;
- $\mathcal V_t$: required scene/state values and two-chunk action history are
  present, finite, and valid;
- $\mathcal W_t$: warm-up is complete ($t\ge2$);
- $\mathcal H_t$: $d_t^S\le\tau_S$, $d_t^Q\le\tau_Q$, with no
  action-derived gripper transition ($g_t=0$);
- $\mathcal P_t$: $\widehat r_t\le\rho$; and
- $\mathcal T_t$: the temporal isolation checks in Section 5 pass.

Any missing, invalid, non-finite, incompatible, or contradictory input causes
a refresh. The controller never interprets uncertainty as permission to
reuse.

## 5. Isolated-reuse temporal contract

### 5.1 Latch rule

The decision is

\[
D_t =
\begin{cases}
U, & E_t=1,\\
R, & E_t=0.
\end{cases}
\]

The latch dominates ordinary threshold eligibility:

\[
L_t=1 \Longrightarrow D_t=R
\]

with reason `post-reuse-refresh`.

### 5.2 Cache-age agreement

When a compatible cache exists, the expected external age is

\[
a_t =
\begin{cases}
0, & L_t=0,\\
1, & L_t=1.
\end{cases}
\]

Any disagreement forces refresh with reason `isolation-state-mismatch`. The
mandatory configuration horizon is also one. The latch, age cross-check, and
horizon are deliberately redundant defenses.

### 5.3 Completed-query transition

The controller state changes only after the adapter successfully completes and
observes the query:

\[
L_{t+1}=
\begin{cases}
1, & D_t=U \text{ and query }t\text{ completed},\\
0, & D_t=R \text{ and query }t\text{ completed},\\
L_t, & \text{query }t\text{ failed or was not observed}.
\end{cases}
\]

A completed reuse increments $N^U$; a completed refresh increments $N^R$.
An unobserved or technically failed query cannot clear a safety latch. The
observation API rejects a caller that attempts to report a consecutive reuse
in violation of the state machine.

### 5.4 Trace language and invariant

After reset, every finite accepted completed-query trace belongs to

\[
R^*(UR^+)^*(\epsilon\mid U).
\]

This exact language permits multiple refreshes between reuses and a run that
ends immediately after a reuse. Equivalently, `UU` is absent from every
accepted trace. The shorter frozen design shorthand `R* (U R)*` expressed the
intended alternation but omitted these valid finite-prefix cases; this
specification records the precise invariant without changing the controller.
The maximum completed reuse streak is one.

## 6. Ordered fail-closed decision logic

The implementation evaluates safety-relevant causes in a deterministic order
so each refresh has an auditable reason. Conceptually:

1. reject configuration or episode-context mismatch;
2. refresh for missing/incompatible cache;
3. refresh for invalid or non-finite required signals/action history;
4. refresh during warm-up;
5. refresh for scene change, end-effector translation, or the action-derived
   gripper-transition veto;
6. refresh when the post-reuse latch is set;
7. refresh for latch/cache-age disagreement;
8. refresh for the horizon defense or prospective prefix cap; otherwise
9. reuse the scene representation.

Several reasons may apply simultaneously. Returned reasons use the fixed order
in `src/savr/acr/types.py`; the base decision is produced first and the V5
latch/mismatch reasons are then added by
`src/savr/acr/isolated_controller.py`. This executable ordering is part of the
versioned method.

## 7. Reference and history updates

After a successfully completed refresh:

- store the newly encoded scene representation;
- update the scene and robot-state refresh references;
- set external cache age to zero;
- clear $L$; and
- append the fresh action output to completed action history.

After a successfully completed reuse:

- retain the existing scene representation and refresh references;
- increment external cache age to one;
- set $L$; and
- append the newly computed action output to completed action history.

The action history is updated after either path because the action head is
always recomputed. Failed queries do not become completed history.

## 8. Normative pseudocode

```text
reset(episode):
    latch = false
    completed_refreshes = 0
    completed_reuses = 0
    clear references and action history

decide(inputs):
    require controller identity = acr-isolated-controller-v1
    require horizon = 1
    if context/cache/signal/warm-up rule fails: return REFRESH(reason)
    expected_age = 1 if latch else 0
    if cache_age != expected_age: return REFRESH(isolation-state-mismatch)
    if latch: return REFRESH(post-reuse-refresh)
    if any state-aware threshold or gripper veto fires: return REFRESH(reason)
    if prospective_prefix_reuse > hard_cap: return REFRESH(hard-cap)
    return REUSE(reuse)

observe(decision):
    # The adapter calls observe only after successful query completion.
    if decision is REUSE:
        reject if latch is already true
        latch = true
        completed_reuses += 1
    else:
        latch = false
        completed_refreshes += 1
```

## 9. Software and evidence invariants

The implementation must preserve all of the following:

1. controller identity and horizon are fixed;
2. maximum completed reuse streak is one;
3. a failure cannot clear a pending refresh;
4. internal latch and external cache age agree before reuse;
5. prefix reuse never exceeds the frozen hard cap;
6. episode reset removes all previous state;
7. wrist, proprioception, and action output are fresh on both paths;
8. legacy V1/V2/V3 controller behavior and immutable V3/V4 evidence are not
   rewritten; and
9. a machine-readable snapshot exposes the state needed to audit decisions.

CPU verification establishes these software invariants. Only paired,
predeclared benchmark experiments can establish empirical performance.

## 10. Relationship to earlier versions

| Version | Scene reuse | Temporal rule | Status |
|---|---|---|---|
| V3 SA-BDP-ACR | Asymmetric scene reuse; fresh wrist/action path | Legacy `cache_age >= horizon`; horizon 2 permits two reuses | V3-D completed negative on frozen efficiency gates |
| V4 redesign | Same base path plus predeclared candidate vetoes | Intended maximum streak one, but frozen candidates retained legacy horizon-2 semantics | V4-A stopped negative; no candidate selected |
| V5 IR-SA-ACR | Same asymmetric computation boundary | Controller-owned post-reuse latch, horizon 1, age agreement, forged-reuse rejection | Implemented and CPU-verified; benchmark performance unknown |

V5 is a separately versioned correction. It does not retroactively alter the
meaning or result of V3/V4.

## 11. Evidence currently established

Verified facts:

- the separate controller runs through the existing batched adapter;
- deterministic verification completed 128 decisions with 51 reuses;
- every corrected prefix stayed at or below the 0.40 test cap;
- the corrected maximum reuse streak was one;
- a preserved legacy horizon-2 trace reached streak two; and
- adversarial tests cover mismatch, failed observation, forged reuse, reset,
  gripper/context rules, randomized traces, and adapter integration.

Unverified hypotheses:

- one-step isolated scene reuse preserves task success;
- a useful threshold region exists under the isolation rule;
- reduced scene encoding produces sufficient visual-work reduction; and
- the optimized implementation lowers measured wall time rather than merely
  reducing theoretical work.

## 12. Required empirical evaluation

Before any positive claim, the project must predeclare and execute:

1. output-blind candidate generation using development-only inputs;
2. wrapped full-refresh parity and cache-correctness checks;
3. paired task-success evaluation against Full Refresh;
4. scene-reuse, visual-work, sequential-time, and wall-time measurements;
5. bootstrap confidence intervals and predeclared non-inferiority margins;
6. ablations for scene-only, normalized translation, the action-derived
   gripper veto, latch, prefix cap, and executor optimization;
7. sensitivity across eligible thresholds without opening protected outcomes;
8. failure analysis by task, seed, episode length, and reuse position; and
9. independent confirmation on a population not used for selection.

Task success is the primary constraint. Efficiency alone cannot promote a
method that violates the frozen success tolerance.

## 13. Limitations

- The one-reuse limit is a conservative project hypothesis, not a theorem.
- Whole-scene block reuse may still discard task-relevant visual change within
  a single interval.
- Signal thresholds may be stack-, task-, and preprocessing-dependent.
- The hard cap limits aggregate reuse but cannot prove local semantic safety.
- CPU semantic verification does not measure CUDA synchronization, kernel
  launch overhead, memory pressure, or simulator latency.
- The method currently targets the verified OpenVLA-OFT/LIBERO interface; its
  transfer to other VLAs or robots is unknown.

## 14. Primary research basis

The design is motivated, but not validated, by primary work on
[VLA-Cache](https://arxiv.org/abs/2502.02175),
[FlashVLA](https://arxiv.org/abs/2505.21200),
[AC2-VLA](https://arxiv.org/abs/2601.19634),
[VLA-Corrector](https://arxiv.org/abs/2607.01804),
[VLASH](https://arxiv.org/abs/2512.01031), and
[event-triggered control](https://arxiv.org/abs/2002.00058). The source-by-
source applicability limits are recorded in `docs/ACR_V5_RESEARCH_AUDIT.md`.

## 15. Normative implementation references

- controller: `src/savr/acr/isolated_controller.py`
- decision types/reasons: `src/savr/acr/types.py`
- adapter path: `src/savr/integrations/openvla_oft/batched_dual_path.py`
- frozen design: `configs/acr/v5_isolated_reuse_freeze.json`
- semantic verifier: `scripts/verify_acr_v5_isolation.py`
- machine evidence: `reports/runtime/acr_v5_cpu_verification.json`
- acceptance report: `reports/PHASE_V5_A_CORRECTION_REPORT.md`

If prose and executable behavior diverge, stop the project, preserve the
evidence, and reconcile the discrepancy before further evaluation.
