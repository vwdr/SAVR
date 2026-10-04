# PAIR-VLA P3R Overhead-Optimization Protocol V1

Date frozen: 2026-08-29  
Authority: user-approved single optimization attempt after P3 scientific stop  
Protected population: outcome-blind timing only

## Purpose and boundary

P3 established real cache headroom but no point met the frozen 2% decision-path
overhead ceiling. P3R tests one implementation-level correction: replace the
per-tile Python/GPU scoring loop with a vectorized scorer while leaving PAIR's
cache semantics, source provenance, masks, profiles, model, inputs, and gates
unchanged.

P3R is not allowed to change the scientific method, relax a threshold, inspect
actions, access expert labels, use simulator outcomes, add a learned shortcut,
or try multiple optimization families. It stops before P4 in all cases.

## Authenticated diagnosis

The complete V03 population showed:

- router p99: approximately 0.127 ms;
- reset p99: approximately 0.002 ms;
- provenance: approximately 0.27 ms per query;
- tile scoring and mask construction: approximately 64.6 ms median per reused
  query, with 95th percentile approximately 83.1 ms; and
- strongest physical point: `D62_BAL_PT1`, horizon 4, with 20.89% raw saving,
  20.54% one-sided 95% raw lower bound, and 14.37% net lower bound.

Thus the only approved redesign is to batch current-versus-actual-source raw
and projected tile comparisons across all four onset layers and both cameras,
precompute tile indices, and eliminate per-tile kernel launches. Router,
provenance, sidecar, cache update, and fallback semantics remain unchanged.

## Equivalence contract

The vectorized path must match the authenticated legacy path exactly for:

- ordered reusable patch positions;
- per-layer reuse proportions;
- camera/tile onset groups;
- protected tiles;
- reused tile counts; and
- recursive source-ledger updates and digests.

CPU synthetic tests cover zero tiles, mixed sources, all source ages, protected
allocations, ties, both cameras, every onset layer, and invalid states. Before
timing, GPU equivalence controls run both `D59_BAL_PT1` and `D62_BAL_PT1`
recursively through horizon 4. Any difference is a technical stop.

No action value or action-hash comparison is permitted.

## Frozen timing population

- Model, checkpoint, stack, input manifest, preprocessing, and source revisions:
  identical to P3 V03.
- Profiles: `D59_BAL_PT1` and `D62_BAL_PT1` only.
- Horizons: 2 and 4 queries.
- Repetitions: 6 per profile/horizon.
- Randomized paired dense/vectorized-cache arm order.
- Model warm-ups: 4 queries.
- Sidecar/all-fresh controls: 4 queries.
- Recursive GPU-equivalence controls: 10 queries total.
- Timed model queries: 192.
- Exact total: 210 model queries; hard cap 220.

Profile selection uses only the already-open outcome-blind P3 timing frontier.
It does not use action quality, regret, expert data, or task success.

## Measurement and gates

Complete-cycle wall time is primary. Every measurement is CUDA-synchronized.
The analyzer uses 20,000 paired block-bootstrap replicates with seed 20260901.

A point passes only if all conditions hold:

1. positive raw complete-cycle saving with one-sided 95% lower bound;
2. conservative net-saving lower bound at least 10%;
3. total feature/router/provenance/reset overhead upper bound at most 2%;
4. service rate at least 70%;
5. exact vectorized/legacy equivalence controls;
6. source, shape, sequence, and all-fresh invariants;
7. peak selected-GPU memory strictly below 23,552 MiB; and
8. all query, time, artifact, and protected-population caps.

The 2% overhead and 10% net thresholds are unchanged from P3.

## Resource and stop rules

- one GPU and one model process;
- at most 220 model queries;
- at most 2 wall-clock hours;
- at most 1 GiB compact artifacts;
- zero downloads;
- zero expert-action or terminal-outcome access;
- no simulator;
- no automatic retry; and
- write only inside `/home/ved/SAVR`.

Any equivalence, correctness, memory, resource, or terminalization failure stops
P3R. If no point passes, PAIR is stopped and no second optimization family is
attempted. If a point passes, P3R reports the result and still stops before P4
for separate user review and approval.
