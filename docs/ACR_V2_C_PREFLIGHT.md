# ACR Version 2 Phase V2-C Preflight

Date: 2026-08-03

Status: **FROZEN BEFORE REAL-MODEL EXECUTION**

Phase V2-C is authorized under `docs/ACR_V2_EXECUTION_PROTOCOL.md`. This
preflight resolves the exact query schedule without changing the method or
gates. The machine-readable authority is `configs/acr/v2_c_gate.json`.

## Exact 48-query schedule

1. Six correctness queries:
   upstream refresh oracle, dual-path refresh, Version 1 reuse, dual-path
   reuse, cache-mismatch forced refresh, and expected-exception restoration.
2. Six untimed warm-up queries: two each for upstream FR, dual-path refresh,
   and dual-path reuse.
3. Thirty-six timed queries: twelve per path in a three-order deterministic
   counterbalance repeated four times.

No simulator is created or reset. Deterministic synthetic images and the
midpoint of the checkpoint's pinned proprioception bounds are used. Reuse
queries are prepared without a model query by replaying the frozen controller's
first three refresh decisions and storing an owned scene block from an existing
correctness oracle. Every real policy invocation consumes one query identity.

## Identical timing boundary

Each timed path uses one outer synchronized CUDA-event/wall-clock timer. The
timer begins immediately before either upstream FR or the complete dual-path
`run_query` call and ends immediately after it. Episode installation and
offline cache/controller preparation occur outside the boundary, matching the
frozen episode-scoped production design. No outlier is removed.

## Mechanical stop

V2-C passes only if all correctness/integrity proofs pass and:

- dual-path refresh median wall time is at most 1.05 times FR;
- dual-path reuse median wall time is at most 0.98 times FR;
- the median weighted by the fixed A5 reuse rate
  `0.26055045871559634` is at most 0.98 times FR.

Any failure stops before a rollout, V2-D, Goal, or protected population.
