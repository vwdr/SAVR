# ACR Version 3 Execution Protocol

Status: **FROZEN; V3-C COMPLETE POSITIVE; V3-D UNAUTHORIZED**

Freeze date: 2026-08-04

Method: State-Aware Batched Dual-Path Asymmetric Camera Refresh
(`SA-BDP-ACR`)

Machine freeze: `configs/acr/v3_freeze.json`

## 1. Purpose and evidence boundary

V3 is a new positive-paper attempt after SA-DP-ACR V2 stopped negative at its
latency gate. It preserves V2-C exactly and may not rerun, delete, retime, or
reinterpret those 48 queries.

V2 showed a real `50.12%` reuse-path visual CUDA reduction but approximately
`41%` weighted wall slowdown. A CPU diagnostic found expensive evidence
hashing in the timed adapter. More importantly, the deterministic V3-A bound
shows that removing all scene-camera visual time at the frozen reuse weight
could reduce wall time by at most `1.6202%`, below the `2%` gate. V3 must
therefore remove evidence work from timing **and** accelerate refreshes.

This protocol authorizes no implementation, GPU/model query, simulator
episode, population access, or manuscript change. Each phase requires user
authorization.

## 2. Frozen research question

Can an unchanged, training-free state-aware camera controller be paired with
batched two-camera refresh execution and wrist-only reuse to:

1. preserve actions and paired task success;
2. reduce expected visual CUDA work by at least 10%;
3. reduce synchronized query wall time by at least 2% versus the pinned
   upstream sequential implementation; and
4. avoid a wall-time regression versus an implementation-matched Batched Full
   Refresh baseline?

Positive evidence requires all four. Batching and reuse contributions must be
reported separately.

## 3. Frozen controller

Retain `acr-t25-h2-b30` exactly:

| Field | Value |
|---|---:|
| Scene threshold | 0.2476380718954248 |
| Translation threshold | 0.5479944908411765 |
| Horizon | 2 |
| Hard scene-reuse cap | 0.30 |
| Warm-up | queries 0 and 1 refresh |
| Transition gate | existing gripper-phase veto |
| Wrist | always fresh |

No signal, threshold, veto, task rule, learned router, mask, or candidate may
be added. Controller and reference-update semantics remain Version 1 exactly.

## 4. V3 execution method

### 4.1 Batched refresh

Given the ordered upstream tensor `[1,12,H,W]`:

1. split scene and wrist into two `[1,6,H,W]` tensors;
2. split each camera into its three-channel SigLIP and DINOv2 inputs;
3. concatenate scene then wrist on the batch dimension, producing one
   `[2,3,H,W]` input per tower;
4. invoke SigLIP once and DINOv2 once;
5. concatenate tower features per batch sample;
6. reshape without camera reordering from `[2,P,Dv]` to `[1,2P,Dv]`;
7. call the unchanged projector once;
8. require `[1,2P,Dllm]`, split the first scene block for the cache, and return
   the combined scene-then-wrist tensor.

Physical calls are SigLIP `1`, DINOv2 `1`, projector `1`; logical camera work
remains scene `1`, wrist `1`. No camera or token is pruned.

### 4.2 Reuse

Use the established V2 wrist-only execution: compatible cached projected
scene block plus current wrist SigLIP, DINOv2, and projector, followed by the
unchanged current proprioception and downstream policy. Physical calls are
SigLIP `1`, DINOv2 `1`, projector `1`; logical scene work is `0` and wrist work
is `1`.

### 4.3 Fail-closed rules

- install and restore at episode scope;
- reject nested/concurrent queries;
- preserve shape, dtype, device, patch count, camera order, context, finite
  action, cache ownership, and one-downstream-call invariants;
- force refresh on a missing/incompatible cache;
- change no upstream source, checkpoint, weights, prompt, preprocessing,
  proprioception, action head, or controller;
- preserve V1/V2 modules and records unchanged.

## 5. Required comparisons

1. **Sequential FR:** exact pinned upstream behavior.
2. **Batched FR (BFR):** V3 batched encoding on every query, no controller or
   cache. This isolates batching.
3. **SA-BDP-ACR:** batched refresh plus asymmetric reuse.
4. **Historical V2:** immutable negative evidence, never rerun under V3.

Any paper-level evaluation must report all three live paths. V3 may not claim
that camera reuse caused acceleration measured only against sequential FR.

## 6. Fair timing and evidence rules

- Use identical pixel tensors, prompts, state, action inputs, model weights,
  and synchronized start/end boundaries.
- Include controller, cache, split/concat, tower, projector, proprioception,
  and all downstream inference required by each live path.
- Exclude audit hashing, correctness scans, JSON serialization, file I/O, and
  report construction from every timed boundary.
- Validate actions after the timed boundary and before possible execution.
- Record CUDA events and wall time; report every repetition, median, mean, and
  physical/logical work.
- Delete no timed outlier and silently retry no scientific failure.
- Consume query/episode budgets before starting each attempt.

## 7. Phase V3-A — diagnosis and freeze

Authorized by the user on 2026-08-04 and complete when:

- immutable V2 evidence and pinned source are reconciled;
- the scene-skip-only ceiling is deterministically reproduced;
- primary research and official numerical guidance are reviewed;
- this method, BFR ablation, gates, populations, and resources are frozen;
- no V3 implementation, GPU/model query, simulator episode, or new outcome
  occurs.

## 8. Phase V3-B — CPU-only implementation

Administrative status: authorized by the user on 2026-08-04 and completed
without changing this frozen method, gate, population, or resource design.
Evidence is recorded in `reports/PHASE_V3_B_REPORT.md`.

Requires new authorization. Implement separate BFR and SA-BDP-ACR adapters
without changing V1, V2, or upstream files.

Minimum tests:

1. exact ordered batch construction and scene/wrist reconstruction;
2. refresh physical call truth `1/1/1` and logical camera truth `1/1`;
3. BFR contains no controller/cache work;
4. V3 refresh and forced-refresh cache ownership;
5. V3 reuse exact equivalence to V2 reuse on fake tensors;
6. structural failure, cache fallback, action-finite, and exception restoration;
7. nested/concurrent rejection;
8. no audit hash, serialization, file write, or full tensor scan in the timed
   production path;
9. immutable record identities and query-budget fail-closed behavior;
10. all repository tests plus static/build/bootstrap checks pass locally and
    in TITAN's pinned CPU environment.

Resources: zero GPU, zero model query, zero rollout, no download, `512 MiB`
new artifact cap. Stop after verification.

## 9. Phase V3-C — bounded real-model correctness and latency

Administrative status: authorized by the user on 2026-08-04. The exact
64-query identities and timing order are frozen in
`configs/acr/v3_c_gate.json` and `docs/ACR_V3_C_PREFLIGHT.md` before model
execution. The phase completed with every correctness and latency gate passing;
evidence is in `reports/PHASE_V3_C_REPORT.md`.

Requires new authorization after V3-B. Use one responsibly selected idle GPU,
one model process, zero simulator episodes, at most `64` model queries,
`3,600 s`, and `512 MiB` artifacts.

### 9.1 Exact query design

- `8` correctness queries;
- `2` untimed warm-ups for each of four paths (`8` total);
- `12` timed repetitions for each path (`48` total);
- deterministic Latin/cyclic counterbalancing;
- total planned and maximum consumption: `64` queries.

Every attempted model call consumes budget before invocation. No recovery may
exceed 64 cumulative calls.

### 9.2 Correctness gate

- BFR/V3 refresh shape, dtype, device, patch count, and scene-wrist order
  exactly equal upstream;
- bfloat16 projected tokens satisfy the predeclared PyTorch tolerance
  `rtol=0.016`, `atol=1e-5` on two deterministic inputs;
- BFR and V3 refresh action chunks are bitwise upstream-identical;
- V3 reuse tokens/actions are bitwise identical to V2 reuse for the same input
  and owned cache;
- expected physical/logical calls, fresh wrist/proprioception/downstream work,
  fail-closed cache behavior, exception restoration, source/checkpoint hashes,
  and finite outputs all pass.

Any correctness failure stops before accepting a positive latency result.

### 9.3 Latency gate

At the fixed reuse weight `0.26055045871559634`, all must pass:

| Gate | Maximum/Minimum |
|---|---:|
| BFR / sequential-FR median wall ratio | ≤ 0.98 |
| V3 refresh / BFR median wall ratio | ≤ 1.02 |
| V3 reuse / BFR median wall ratio | ≤ 0.98 |
| V3 weighted / sequential-FR wall ratio | ≤ 0.98 |
| V3 weighted / BFR wall ratio | ≤ 1.00 |
| V3 weighted visual CUDA reduction vs sequential FR | ≥ 10% |

Failure stops before rollout and preserves the V3-C result as negative.

## 10. Phase V3-D — fresh Object development

Requires authorization only after V3-C passes. Use Object tasks `0-9`, states
`3-9`, seed `0`, whose ACR outcomes remain unopened. Run BFR and the single V3
method once, contemporaneously and counterbalanced: `70` episodes per policy,
`140` attempts total. Retain sequential FR's immutable A4 evidence only for
the system-latency reference; do not rerun it for success selection.

Remain outcome-blind until all terminal records exist. Pass only with:

- `70/70` terminal episodes per policy and zero technical failures;
- V3 success no more than two episodes below BFR;
- no task loses more than one success;
- scene reuse at least 20%;
- visual CUDA reduction versus BFR at least 10%;
- V3 wall ratio versus sequential FR at most 0.98;
- V3 wall ratio versus BFR at most 1.00;
- every work/cache/restoration invariant passes.

Failure stops before Goal.

## 11. Phase V3-E — independent Goal confirmation

After separate authorization, run sequential FR, BFR, and frozen V3 once on
Goal tasks `0-9`, states `0-9`, seed `0`: `100` episodes per policy. No Goal
outcome may tune V3.

Use the V3-D success/reuse/visual/wall gates plus a one-sided 97.5% lower
confidence bound for the sequential-FR latency reduction above zero. Failure
stops before paper baselines, power planning, or final evaluation.

## 12. Later paper evidence and protected populations

Only after confirmation may a separately frozen phase add matched controller
ablations, PR/VOR-style camera policies, a renewed official VLA-Cache audit,
power planning, and a final evaluation. All four-suite states `10-49`, primary
seed `7`, and reserve seeds `17/27` remain protected. The manuscript remains
unchanged until evidence supports the exact claims to be written.

## 13. Current stop point

V3-C is complete with a positive correctness/latency result. Stop before V3-D
and request separate authorization.
