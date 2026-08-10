# ACR Version 4 Evidence-Gated Redesign Protocol

Status: **FROZEN PLANNING DOCUMENT; V4-A NOT YET AUTHORIZED**

Freeze date: 2026-08-10

Machine freeze: `configs/acr/v4_redesign_freeze.json`

## 1. Purpose and non-negotiable boundary

V3-D is a valid negative result and may not be rerun, retimed, filtered,
relabelled, or rescued by changing its gates. It established that SA-BDP-ACR
preserves task success and beats sequential Full Refresh, but does not beat the
implementation-matched Batched Full Refresh baseline on the complete frozen
efficiency gate.

V4 is allowed only as a materially changed method that directly addresses both
observed failure mechanisms:

1. `25.2434%` scene reuse produced only `8.4618%` visual-CUDA reduction versus
   BFR; and
2. the V3 reuse/controller path left weighted wall time `0.2260%` slower than
   BFR.

At the observed V3-D efficiency slope, approximately `29.8323%` reuse would
only touch the old 10% visual gate. That is not enough margin for a new run.
V4 must establish a conservative efficiency margin before any simulator
episode and must preserve success as the primary constraint.

Creating this protocol authorizes no V4 diagnosis, implementation, GPU/model
query, simulator episode, population access, download, or manuscript change.
Each phase requires explicit authorization.

## 2. Provisional V4 research hypothesis

The provisional route is **Safety-Constrained High-Reuse ACR with an Optimized
Single-View Executor** (`SC-HR-ACR`). This is a design target, not yet a frozen
method.

V4 must combine two independently measurable contributions:

- **Controller contribution:** raise safe realized scene reuse above V3's
  25.24% without using task IDs, memorized state IDs, benchmark-specific rules,
  or protected outcomes.
- **Executor contribution:** make wrist-only reuse materially faster than BFR
  after including controller/cache/concat/downstream work, instead of relying
  on a near-zero wall-time difference.

Possible mechanisms may be evaluated in V4-A, but only one controller and one
executor may be frozen for V4-C onward. Permitted mechanism families are:

1. risk-budgeted or uncertainty-gated reuse using signals already available at
   inference time;
2. phase-aware reuse vetoes derived from generic action/state transitions, not
   task identity;
3. a fixed-shape optimized wrist-only visual path, including a project-owned
   CUDA-graph or compiler path if numerical and restoration requirements pass;
4. removal or deferral of nonessential Python/controller/cache work outside
   the synchronized inference boundary only when the fair-timing definition
   permits it for every compared policy.

V4 may not use training, weight changes, task-specific lookup tables, oracle
success labels online, asynchronous stale actions, camera/token pruning on
refresh, lower image resolution, a different checkpoint, or weaker evidence
standards.

## 3. Required changes before another rollout

### 3.1 Increase efficiency margin

The design target is at least `35%` realized scene reuse on development traces,
not merely the estimated `29.8323%` break-even point. Before rollout, the
frozen weighted microbenchmark must show:

- at least `12%` visual-CUDA reduction versus BFR;
- wall ratio at most `0.98` versus BFR;
- wall ratio at most `0.95` versus sequential FR; and
- no refresh-path regression beyond `1%` versus BFR.

These are promotion margins, not revised interpretations of V3-D.

### 3.2 Strengthen success safety

The V4 controller must fail closed on missing/invalid signals and preserve:

- queries 0 and 1 as forced refreshes;
- current wrist image, proprioception, and downstream policy on every query;
- immediate refresh on cache mismatch, generic gripper transition, or
  predeclared unsafe action/state transition;
- a hard maximum reuse streak and hard aggregate reuse cap; and
- episode-scoped cache ownership and exception-safe restoration.

Offline replay may eliminate unsafe candidates but may not establish closed-loop
success. No candidate can advance because it merely predicts a positive result.

### 3.3 Separate attribution

Every V4 gate must distinguish:

1. upstream sequential FR;
2. Batched Full Refresh;
3. V3 SA-BDP-ACR, preserved and never rerun for selection unless a phase
   explicitly freezes a diagnostic-only comparison;
4. V4 controller with the V3 executor;
5. V3 controller with the V4 executor; and
6. complete V4.

This factorial structure prevents controller gains, batching gains, and
executor gains from being conflated.

## 4. Evidence and population policy

- V3-D Object tasks `0-9`, states `3-9`, seed `0` are opened development
  evidence and may inform V4-A diagnosis.
- Existing A4/A5 Object states `0-9`, seed `0` remain development evidence.
- LIBERO-Goal tasks `0-9`, states `0-9`, seed `0` remain unopened for V4 and
  are reserved for independent confirmation.
- All four-suite states `10-49`, primary seed `7`, and reserve seeds `17/27`
  remain protected final populations.
- No development or Goal result may select among candidates after the relevant
  freeze point.

## 5. Phase V4-A — immutable diagnosis and mechanism selection

Requires separate authorization. CPU/read-only analysis only; no GPU, model
query, simulator reset, or new outcome.

Required work:

1. independently reconcile V3-D queries, timings, decisions, paired outcomes,
   order, work counts, and both failed gates;
2. decompose BFR, V3 refresh, and V3 reuse wall/CUDA distributions by task,
   query index, refresh reason, cache age, and success stratum without deleting
   any sample;
3. calculate visual and wall break-even reuse curves with uncertainty and
   conservative lower bounds;
4. quantify action divergence around V3 reuses and identify generic—not
   task-specific—failure-risk patterns;
5. replay a small predeclared controller family on immutable traces as a
   screening test only;
6. inspect the pinned visual path and evaluate executor mechanism feasibility;
7. review relevant primary research and official framework guidance;
8. reject mechanisms that cannot plausibly reach the V4-C promotion margins;
9. freeze exactly one controller, one executor, ablations, numerical
   tolerances, populations, timing boundaries, resources, and recovery rules;
10. produce a machine-readable diagnosis and a V4 method proposal before any
    implementation.

Candidate screening must be declared before candidate outputs are examined.
If no single design has a conservative predicted `>=12%` visual reduction and
`>=2%` wall reduction versus BFR, V4 stops before implementation.

Resources: zero GPU, zero model queries, zero simulator episodes, zero download,
`512 MiB` new artifacts.

## 6. Phase V4-B — CPU-only implementation

Requires V4-A passage and separate authorization.

Implement the frozen V4 controller/executor in new modules without modifying
V1, V2, V3, upstream sources, or immutable evidence. Required tests include:

1. exact refresh token order, shape, dtype, device, and camera work;
2. exact frozen controller decisions on deterministic traces;
3. fixed hard cap/streak, warm-up, transition, invalid-signal, and cache rules;
4. executor numerical equivalence to its declared oracle;
5. physical/logical call truth for refresh and reuse;
6. current wrist/proprioception/downstream work on every query;
7. nested/concurrent rejection and episode-scoped restoration;
8. no hashing, serialization, file I/O, or full-tensor scan in production
   timing;
9. immutable query/episode budgets and recovery identities;
10. all repository, static, compilation, bootstrap, package-build, and pinned
    TITAN CPU tests.

Resources: zero GPU, zero model queries, zero simulator episodes, zero download,
`512 MiB` new artifacts.

## 7. Phase V4-C — bounded correctness and efficiency margin

Requires V4-B passage and separate authorization. Before the first model call,
freeze exact inputs, path order, warm-ups, repetitions, expected work, and the
reuse weight derived in V4-A.

Use one selected GPU, one model process, zero simulator episodes, at most `96`
model queries, `3,600 s`, and `512 MiB` artifacts. Include sequential FR, BFR,
V3 executor/controller ablations, and complete V4 in deterministic cyclic
order.

Correctness must pass before timing is accepted. Promotion requires every
condition:

- refresh tokens meet the predeclared bfloat16 tolerance and refresh actions
  are bitwise BFR-identical;
- reuse output is numerically/action equivalent to its frozen oracle;
- complete physical/logical work and restoration truth;
- V4 refresh/BFR median wall ratio `<=1.01`;
- V4 reuse/BFR median wall ratio `<=0.95`;
- weighted V4/BFR wall ratio `<=0.98`;
- weighted V4/sequential-FR wall ratio `<=0.95`;
- weighted visual-CUDA reduction versus BFR `>=12%`; and
- both controller-only and executor-only ablations are reported.

Any failure stops before rollout. No timing outlier may be removed.

## 8. Phase V4-D — paired Object development

Requires V4-C passage and separate authorization. Freeze the complete schedule
before execution. Use the already-opened Object development population,
tasks `0-9`, states `0-9`, seed `0`, with one contemporaneous counterbalanced
BFR and V4 episode per identity: `100` episodes per policy, `200` attempts.

Remain outcome-blind until all terminal records exist. Passage requires:

- `100/100` terminal episodes per policy and zero technical failures;
- V4 success no more than two episodes below BFR;
- no task loses more than one success;
- realized scene reuse at least `35%`;
- visual-CUDA reduction versus BFR at least `12%`;
- wall ratio versus sequential FR at most `0.95`;
- wall ratio versus BFR at most `0.98`; and
- every work, cache, timing, record, source, checkpoint, and restoration
  invariant passes.

Failure stops before Goal. Scientific failures are terminal and never retried.

Resources: one GPU, one model process, at most `200` episode attempts,
`43,200 s`, `3 GiB`, no download.

## 9. Phase V4-E — independent Goal confirmation

Requires a positive V4-D result and separate authorization. Freeze the final
V4 method before any Goal outcome. Run sequential FR, BFR, and V4 once on
LIBERO-Goal tasks `0-9`, states `0-9`, seed `0`: `100` episodes per policy,
`300` attempts total.

Use the V4-D gates plus predeclared paired confidence intervals and a one-sided
97.5% lower confidence bound showing V4 wall reduction versus BFR above zero.
No Goal outcome may tune V4. Failure stops before paper baselines, power
planning, or final evaluation.

Resources: one GPU, one model process, at most `300` episode attempts,
`86,400 s`, `4 GiB`, no download.

## 10. Later paper-level evidence

Only after V4-E passes may a separately authorized protocol add matched
PR/VOR baselines, controller/executor ablations at paper scale, power analysis,
four-suite transfer, final protected populations, and manuscript changes.
Every claim must map to immutable evidence. A positive-results goal never
authorizes suppressing a negative result or weakening a frozen criterion.

## 11. Checkpoints and on-task controls

Every phase must:

- start from a clean, synchronized GitHub/Mac/TITAN revision;
- publish its freeze before accessing new outcomes;
- record exact budgets before consumption;
- use monotonically increasing immutable attempt identities;
- stop immediately on a scientific or unhandled technical gate failure;
- document any technical recovery before executing it;
- report planned versus consumed resources and protected-population status;
- update `PROJECT_STATUS.md`, `docs/MILESTONES.md`, and `docs/DECISIONS.md`;
- stop at the phase boundary and request authorization for the next phase.

## 12. Current stop point

This protocol and machine freeze are planning artifacts only. V4-A has not
started. The next possible action is explicit authorization for CPU-only V4-A
diagnosis. No V4 implementation, GPU/model query, simulator episode, Goal
access, protected outcome, or manuscript edit is authorized.
