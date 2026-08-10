# ACR Version 4 Phase V4-A Preflight

Status: **FROZEN BEFORE V4-A CANDIDATE OUTPUTS**

Authorization date: 2026-08-10

Machine freeze: `configs/acr/v4_a_diagnosis_preflight.json`

## Boundary

The user authorized Phase V4-A on 2026-08-10. This phase is CPU-only and may
read only already-opened project evidence. It may not use a GPU, load or query
the model, reset the simulator, create a new task outcome, download an asset,
inspect Goal or a final population, implement V4 production code, or modify the
manuscript.

V3-D remains an immutable negative result. Phase V4-A may diagnose it but may
not rerun, retime, filter, or change its gates.

## Inputs

The analysis is limited to:

- complete V3-D recovery-2 Object records for tasks `0-9`, states `3-9`, seed
  `0`;
- complete A4 Full Refresh Object traces for tasks `0-9`, states `0-9`, seed
  `0`;
- complete A5 Stage-1 records for the three previously frozen candidates;
- pinned project and upstream source code;
- primary papers and official framework documentation.

All input hashes must reconcile before results are accepted. Goal and
states `10-49` remain unopened.

## Predeclared controller screening family

Replay uses the exact A4 scene representations, normalized end-effector
positions, and action chunks. Every candidate forces queries 0 and 1 to
refresh, keeps horizon 2 (no consecutive reuse), uses a hard 40% prefix reuse
budget, refreshes on missing/invalid signals and gripper transitions, and is
independent of task/state identity.

Let the conservative and aggressive ACR thresholds be:

- scene: `0.2476380718954248` and `0.30046895424836606`;
- translation: `0.5479944908411765` and `0.685919037527938`.

For interpolation fraction `a`, each threshold is `low + a * (high - low)`.
The six candidates are the Cartesian product of `a in {0.50, 0.75, 1.00}` and
the following generic transition policies:

1. refresh on gripper transition only; or
2. refresh on gripper transition or any translation direction reversal.

No candidate may be added, removed, or changed after replay outputs are read.
Offline replay is a screening estimate, not a task-success result.

## Frozen analyses

1. Re-run the V3-D reconciliation independently and verify all population,
   order, timing, work, cache, success, and restoration facts.
2. Report complete refresh/reuse timing distributions by task, query index,
   cache age, reason, and episode success without deleting outliers.
3. Compute episode-cluster bootstrap 95% intervals with seed `4102026` and
   exactly `10,000` resamples.
4. Derive visual and wall break-even curves using all immutable records.
5. Replay all six candidates twice and require byte-identical results.
6. Quantify candidate exposure to gripper transitions, direction reversals,
   high action-chunk change, early queries, and failure-associated strata.
7. Audit the pinned reuse execution path for static shapes, control flow,
   synchronization, allocation, input-address, numerical, and restoration
   constraints.
8. Review primary VLA caching/efficiency work and official PyTorch/NVIDIA CUDA
   Graph guidance, separating supported facts from project hypotheses.

## Controller eligibility and selection

A candidate is controller-eligible only if all conditions hold:

- replay reuse-rate point estimate is at least 35%;
- the episode-bootstrap 95% lower bound on replay reuse is at least 30%;
- maximum reuse streak is exactly one;
- no reuse occurs on a gripper transition;
- no invalid/cache/warm-up/horizon/budget invariant fails; and
- the conservative predicted visual-CUDA reduction point estimate is at least
  12%, with its bootstrap lower bound reported.

Among eligible candidates, selection is lexicographic:

1. prefer the direction-reversal veto when it remains eligible;
2. minimize transition-risk exposure;
3. choose the smallest threshold interpolation fraction; and
4. choose the smallest reuse estimate above the 35% target.

This preference protects task success; it does not use episode identity or
success labels to tune a threshold.

## Executor decision tree and complete-method gate

Executor feasibility is evaluated in this fixed order:

1. a project-owned static-buffer CUDA Graph or `torch.compile` region spanning
   the complete fixed-shape V4 reuse query after the CPU controller;
2. if the complete query is not graph-safe, a static-buffer compiled/graphed
   wrist encoder plus projector;
3. if neither path can conservatively support the required wall margin, stop
   V4 before implementation.

Source inspection must prove the selected boundary has fixed shapes and
control flow, no unsafe input-pointer reuse, no capture-time allocation or CPU
synchronization, and an exact eager fallback. V4-B must implement correctness
tests before V4-C measures performance.

For each controller candidate, the analysis must solve for the maximum
per-reuse wall ratio versus BFR that would make the complete method reach a
weighted wall ratio of 0.98, using the conservative refresh estimate and
candidate reuse lower bound. A complete V4 design advances only if:

- controller eligibility passes;
- predicted visual-CUDA reduction is at least 12%; and
- source evidence supports a testable executor whose required per-reuse wall
  ratio is no lower than 0.90.

The `0.90` feasibility floor is not a claimed speedup. It prevents advancing a
design that would require an implausibly large unmeasured gain. V4-C retains
the stricter measured gates from the master protocol.

## Outputs and stop rule

V4-A must produce a machine-readable diagnosis, a human report, primary-source
ledger, exact candidate replay results, executor source audit, and—only if all
gates pass—one frozen V4 method proposal. If no design passes, V4 stops before
implementation. V4-B always requires separate authorization.

Resource cap: zero GPUs, zero model queries, zero simulator episodes, zero
downloads, and at most `512 MiB` of new artifacts.
