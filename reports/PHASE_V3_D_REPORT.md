# ACR Version 3 Phase V3-D Report

**Date:** 2026-08-04

**Disposition:** **STOPPED NEGATIVE — V3-E INELIGIBLE**

## Outcome

The complete outcome-blind recovery matrix produced 70 terminal BFR episodes
and 70 terminal SA-BDP-ACR episodes with zero technical failures and every
work, cache, record, source, checkpoint, and restoration invariant passing.
Both policies achieved 67/70 success. V3 therefore passed aggregate and
per-task success preservation, achieved 25.24% scene reuse, and reduced wall
time 3.96% versus immutable sequential FR.

V3 failed two predeclared efficiency gates. Its visual CUDA reduction versus
BFR was 8.46%, below 10%, and its wall ratio versus BFR was 1.00226, above
1.00. The mechanical disposition is therefore negative and stops before
V3-E. No threshold, result, timing sample, or gate was changed; no outlier was
removed.

## Frozen gate reconciliation

| Gate | Required | Result | Status |
|---|---:|---:|---|
| Terminal episodes per policy | 70 | 70 / 70 | PASS |
| Technical failures | 0 | 0 | PASS |
| V3 success loss vs BFR | at most 2 | 0 (67 vs 67) | PASS |
| Per-task success loss | at most 1 | maximum 1 | PASS |
| Scene reuse | at least 20% | 25.2434% | PASS |
| Visual CUDA reduction vs BFR | at least 10% | 8.4618% | **FAIL** |
| Wall ratio vs sequential FR | at most 0.98 | 0.960431 | PASS |
| Wall ratio vs BFR | at most 1.00 | 1.002260 | **FAIL** |
| All invariants | pass | pass | PASS |

Per-task BFR successes were `5,7,7,7,7,7,7,7,7,6`; V3 successes were
`6,7,7,7,7,7,7,7,7,5`. V3 executed 337 scene reuses across 1,335 queries.

## Timing and work points

| Policy | Steady queries | Wall ms/query | Visual CUDA ms/query |
|---|---:|---:|---:|
| Immutable sequential FR | 1,256 | 1,240.0362 | — |
| Batched FR | 1,251 | 1,188.2837 | 114.9292 |
| SA-BDP-ACR | 1,332 | 1,190.9691 | 105.2042 |

## Execution and recovery provenance

The first start stopped before a completed query because a valid action list
reached a tensor-only finite checker. Recovery 1 fixed that boundary, completed
one BFR episode, then stopped before any V3 query because the V3 context used
the method label instead of controller configuration identity. Both technical
stops were frozen before correction, restored all protected state, and remained
outcome-blind. Recovery 2 reran the entire 140-episode scientific matrix in one
model process; neither earlier partial run contributes to the official result.

Recovery 2 used GPU 1, TITAN RTX UUID
`GPU-a9e134d4-93b7-df8d-5de3-26a8c705943a`, for 7,211.47 seconds and produced
19,541,445 result bytes. It ended at 6 MiB and 0% utilization. Cumulative starts
were 143/143; wall and artifact use remained below 43,200 seconds and 2 GiB.
No download, Goal outcome, final holdout, manuscript edit, or write outside
`/home/ved/SAVR` occurred.

The compact analysis is `reports/runtime/acr_v3_d.json`, byte SHA-256
`c8e3b9e6534950225e76b71d0cd08cd6e882baa1b430dab50646e4cbae96c586`
and semantic SHA-256
`1653feb24f6396de2c8766532995a2acb0f4bc06f1abd8cf407d184fcfff0370`.

## Boundary decision

V3-D is complete and negative. V3-E is not authorized and is ineligible under
the frozen protocol. Preserve this result without rerun or reinterpretation.
