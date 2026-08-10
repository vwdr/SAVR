# ACR Version 5 Isolated-Reuse Correction Protocol

Status: **FROZEN BEFORE V5 CPU IMPLEMENTATION**

Freeze date: 2026-08-10

Machine freeze: `configs/acr/v5_isolated_reuse_freeze.json`

## Purpose

Correct the temporal semantics discovered by V4-A without changing or
reinterpreting any V3/V4 result. The method remains State-Aware Visual Refresh
with asymmetric scene/wrist camera computation. This checkpoint corrects only
the maximum-reuse-streak mechanism.

## Frozen controller contract

Create a separate `IsolatedACRController` and leave `ACRController` behavior
unchanged. The corrected controller must:

- accept only `controller_version="acr-isolated-controller-v1"`;
- accept only `horizon=1`;
- retain warm-up queries 0 and 1, scene/state/action signals, gripper veto,
  prefix reuse cap, episode context, and fail-closed invalid-signal behavior;
- latch a required refresh after every successfully completed reuse;
- clear that latch only after a successfully completed refresh;
- force `post-reuse-refresh` whenever the latch is set;
- require cache age 0 after refresh and 1 after reuse;
- force `isolation-state-mismatch` on any cache/latch disagreement;
- expose the latch and completed refresh/reuse state in an auditable snapshot;
- remain compatible with the existing dual-path and batched dual-path adapter
  interfaces; and
- never alter V1/V2/V3 decisions or immutable evidence.

## CPU acceptance matrix

Before the correction is accepted, tests must prove:

1. horizon 2 and the old controller identity are rejected;
2. stable synthetic traces follow `R* (U R)*` and never contain `U U`;
3. a post-reuse refresh is forced even when the caller falsely reports age 0;
4. any cache/latch disagreement forces refresh;
5. a technical failure or missing observation cannot clear the latch;
6. reset removes all prior-episode latch state;
7. warm-up, gripper, invalid-signal, hard-cap, and context rules still apply;
8. deterministic randomized traces satisfy maximum streak one and the prefix
   cap at every completed query;
9. the legacy `ACRController` with horizon 2 retains its historical semantics,
   demonstrating version separation; and
10. the complete repository and bootstrap validation pass locally and in the
    pinned TITAN CPU environment.

## Exclusions

This authorization does not permit:

- replaying A4/A5/V3/V4 records to select a threshold;
- adding a candidate or reporting a predicted reuse/efficiency result;
- GPU use, model loading/querying, simulator reset/episode, or download;
- Goal or final-population access;
- executor/CUDA-Graph implementation;
- action reuse, token pruning, training, weight changes, or task-specific
  rules; or
- manuscript modification.

## Stop and next gate

Any failed CPU invariant stops the correction. Passing CPU tests establishes
only that the temporal contract is correctly implemented. A future V5
screening/efficiency phase requires a separate output-blind protocol and
authorization before reading any replay output or using a GPU.

Resource cap: zero GPUs, zero model queries, zero simulator episodes, zero
downloads, zero new outcomes, and at most 256 MiB of new artifacts.
