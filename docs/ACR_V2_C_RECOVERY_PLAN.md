# ACR Version 2 Phase V2-C Technical Recovery

Date: 2026-08-03

Status: **FROZEN BEFORE RECOVERY EXECUTION**

## Preserved stop

The original V2-C attempt is immutable and technically failed after 7/48
queries. The runner expected one low-level SigLIP and DINOv2 invocation for
upstream FR, but the pinned fused backbone legitimately invokes each tower
twice—once per camera—through one upstream two-view boundary. No method result
or timed sample was accepted.

The failure record has SHA-256
`745a8cff68921190acc6d738c8febf1667de44b3891d683a377e60172e5354ad`.
It records all six correctness labels followed by the first upstream-FR
warm-up label. Program control can reach that warm-up only after refresh token
and action parity, refresh object identity, Version 1/V2 reuse parity, cache
fail-closed behavior, work accounting, and exception restoration all pass.
The first warm-up completed its model query and synchronized timing before the
component-count assertion stopped the attempt.

## Recovery

The recovery changes only the expected low-level call truth:

- upstream FR and dual-path refresh: two SigLIP, two DINOv2, one projector;
- dual-path reuse: one SigLIP, one DINOv2, one projector.

It performs the remaining five warm-ups and all 36 frozen timed queries. The
recovery uses 41 queries, so cumulative usage is exactly 48/48. Correctness is
not rerun, no warm-up is discarded, the counterbalance and latency gates are
unchanged, and no additional query is available. Any further technical or
scientific failure ends V2-C.
