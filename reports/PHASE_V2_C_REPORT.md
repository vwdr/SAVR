# ACR Version 2 Phase V2-C Report

Date: 2026-08-03

Status: **STOPPED NEGATIVE — LATENCY GATES FAILED**

## Scope and integrity

V2-C used one selected TITAN RTX GPU, one model process, exactly 48 cumulative
real-model queries, zero simulator resets, zero rollout episodes, no download,
and no benchmark or protected-population outcome. GPU 0 was selected from
aggregate-only evidence at 0% utilization and 6 MiB used. It returned to the
same aggregate state after execution.

Checkpoint metadata restored to its accepted hashes. The SAVR,
OpenVLA-OFT, and LIBERO worktrees are clean. Nothing outside
`/home/ved/SAVR` was modified.

## Preserved technical stop

The original attempt is preserved as technically failed after 7/48 queries.
All six correctness assertions completed before the first upstream-FR warm-up
reached a mistaken component-count assertion. The runner expected one
low-level tower invocation for the two-view path, while the pinned backbone
correctly invoked SigLIP and DINOv2 once per camera. No timed sample was
accepted from that attempt.

The frozen recovery corrected only this accounting truth, counted the completed
warm-up, and executed the remaining five warm-ups plus all 36 timed queries.
Recovery used 41 queries, so cumulative use is exactly 48/48. No further
recovery is available.

Correctness is adjudicated as passing by immutable program control flow: the
runner could not enter timing until exact upstream/V2 refresh token and action
parity, exact refresh return identity, Version 1/V2 reuse token and action
parity, camera-work truth, cache fail-closed behavior, and exception
restoration had all passed. A limitation is that the original technical stop
occurred before the per-proof token/action hashes were written to a final
record; this is disclosed rather than reconstructed.

## Paired timing result

Twelve synchronized timed queries were retained for each path; no outlier was
deleted.

| Path | Median wall ms | Mean wall ms | Median visual CUDA ms | Mean visual CUDA ms |
|---|---:|---:|---:|---:|
| Upstream FR | 1213.519 | 1213.537 | 150.566 | 150.634 |
| Dual-path refresh | 1703.028 | 1704.567 | 149.655 | 149.577 |
| Dual-path reuse | 1735.271 | 1700.782 | 75.104 | 75.053 |

The reuse path reduced its visual tower/projector CUDA time by **50.12%**
relative to FR. At the fixed A5 reuse weight `0.26055045871559634`, expected
visual CUDA time fell by **13.51%**. This compute reduction did not translate
to end-to-end acceleration.

## Frozen latency gates

| Gate | Required | Observed | Result |
|---|---:|---:|---|
| Dual refresh / FR median wall ratio | ≤ 1.05 | 1.40338 | FAIL |
| Dual reuse / FR median wall ratio | ≤ 0.98 | 1.42995 | FAIL |
| A5-weighted median wall ratio | ≤ 0.98 | 1.41030 | FAIL |

All three latency gates fail. SA-DP-ACR is approximately 41% slower at the
frozen weighted operating point despite avoiding half the visual work on reuse.

## Diagnostic interpretation

The external synchronized timer included the complete adapter call. A bounded
CPU-only diagnostic on the same deterministic inputs measured a median
`395.36 ms` for the three audit hashes performed per dual-path query, versus
`1.98 ms` for scene-representation preparation. The audit implementation
serializes image values through Python/JSON and is a major measured source of
wrapper overhead. This diagnosis does not change or excuse the gate result;
the frozen V2 implementation included that work in its production query path.

All 12 timed action hashes per path were identical, and every physical/logical
camera-work invariant passed.

## Evidence hashes

- Original failure record:
  `745a8cff68921190acc6d738c8febf1667de44b3891d683a377e60172e5354ad`
- Recovery final byte SHA-256:
  `9e4d8e0034c4410dbc35f4e4a2b987eda4a0645ab603c3479da05a97bd6f1ae6`
- Recovery semantic SHA-256:
  `e834b5dc04385ec6b5d2385cff4098016427f5ce0c6e2d77744f0c9f1b76afc6`
- Tracked machine record: `reports/runtime/acr_v2_c_recovery.json`

## Decision

V2-C stops negative. Phase V2-D is ineligible and Object states `3-9` remain
unopened for SA-DP-ACR. Goal and all protected final populations remain
untouched. No positive latency or end-to-end efficiency claim is supported.

Any further positive-paper attempt requires a newly predeclared implementation
that removes record-generation work from the critical path and proves its
timing before receiving a new query budget. It may not reinterpret or rerun
this V2 result.
