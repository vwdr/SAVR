# ACR Phase A3 Frozen Preflight

**Date:** 2026-08-02

**Status:** frozen before any A3 model load or GPU query

**Governing protocol:** `docs/ACR_EXECUTION_PROTOCOL_V1.md`

## Scope and caps

- Run ID: `acr-a3-correctness-none-v01`
- Inputs: deterministic synthetic 256×256 scene and wrist images, one fixed
  instruction, and two states derived from checkpoint q01/q99 statistics
- Benchmark populations: none
- Simulator resets and rollout episodes: zero
- Real-model query plan: 12; hard cap: 16
- GPU/model processes: one
- Wall-time cap: 3,600 seconds
- Artifact cap: 512 MiB
- Downloads: prohibited; all model/cache access is offline
- Numerical tolerance: prohibited

The four unplanned query slots are a safety margin, not a tuning or rerun
budget. A parity or scientific failure is not rerun. An interrupted attempt is
preserved and cannot resume from guessed cache state.

## Frozen query matrix

| Global query | Path | Purpose |
|---:|---|---|
| 0 | upstream FR, cameras A, state A | exact token/action oracle |
| 1 | camera-factorized FR, cameras A, state A | bitwise token/action parity |
| 2 | camera-factorized FR, scene variant only | scene-block isolation |
| 3 | camera-factorized FR, wrist variant only | wrist-block isolation |
| 4 | Scene-Visual ACR, cameras A/state A | warm-up refresh 0 |
| 5 | Scene-Visual ACR, cameras A/state A | warm-up refresh 1/cache reference |
| 6 | Scene-Visual ACR, cameras A/state B | required scene reuse/current proprio |
| 7 | upstream FR, cameras A/state B | reuse action/current-state oracle |
| 8 | Scene-Visual ACR with injected cached shape mismatch | fail-closed refresh |
| 9 | Scene-Visual ACR with injected cached dtype mismatch | fail-closed refresh |
| 10 | Scene-Visual ACR with injected cached device mismatch | fail-closed refresh |
| 11 | Scene-Visual ACR after context identity change | invalidation/refresh |

Scene-Visual ACR is used for the isolated reuse proof because it preserves the
same projected-scene cache semantics while avoiding an action-transition veto
that could prevent the deliberately requested reuse. SA-ACR state and
transition semantics were already exhaustively CPU-tested in A2; no candidate
or threshold is selected in A3.

## Hard proofs

1. Query 1 projected tokens equal Query 0 bitwise.
2. Query 1 actions equal Query 0 bitwise.
3. Query 2 changes only the scene token block; Query 3 changes only the wrist
   token block.
4. Query 6 executes zero scene SigLIP, DINOv2, and projector calls.
5. Every ACR query executes exactly one fresh wrist tower/projector path.
6. Query 6 uses current state B and its actions equal upstream Query 7 bitwise.
7. Queries 8-11 fail closed to a full scene refresh.
8. Pinned source trees and checkpoint metadata/files are identical before and
   after the run.
9. Every immutable record, component count, query count, hash, and resource cap
   reconciles.

Any failure stops before a rollout or protected-population access. A tolerance,
additional query, altered input, or recovery run requires a root-cause report
and new user authorization.
