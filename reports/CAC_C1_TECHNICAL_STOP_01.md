# CAC C1 Technical Stop 01

**Date:** 2026-08-31  
**Run:** `cac-c1-tensor-feasibility-v01`  
**Classification:** technical integration failure before any VLA call  
**Scientific result:** none

## What happened

The worker loaded the authenticated OpenVLA-OFT checkpoint on TITAN GPU 0, then
prepared seven observation queries before beginning the 95-call schedule. Query
preparation was accidentally executed with gradient tracking enabled. Retaining
those prepared queries therefore retained their vision-backbone autograd graphs,
and the seventh preparation exhausted GPU memory.

The worker stopped fail-closed with:

- model calls: 0;
- training: none;
- simulator outcomes: none;
- expert actions: unopened;
- terminal outcomes: unopened; and
- automatic retry: none.

The immutable machine record is
`results/cac-c1-tensor-feasibility-v01/technical_stop.json` with SHA-256
`79c83fe2791301578d301418fb657a116d4486d3bca88e4d0477c5f715991df7`.

## Checkpoint restoration

The standard model loader temporarily staged `config.json` and
`modeling_prismatic.py` inside the project-local checkpoint. Their exact
pre-load copies were restored. The loader-created copies were preserved under
the failed run root. Restored hashes are:

- `config.json`: `edd5c5cf6d7927e07465cf086ebe41f7b3ec8f3b128a51f71d6db14dad7ad8b1`;
- `modeling_prismatic.py`: `f40ee7883e16aab1a2d89b6e8f31cc81f6b8055120b1fefe169e05c7031098fa`.

## Recovery correction

Recovery 01 makes only two technical changes:

1. query preparation is forced to run under inference mode, and the shared
   helper now fails closed if gradients are enabled; and
2. the worker captures the checkpoint metadata baseline before model loading
   and restores it exactly in a `finally` block on success or failure.

The scientific method, D62 profile, observations, timing schedule, 95 planned
calls, 160-call hard cap, 4-GiB artifact cap, and every C1 gate remain unchanged.
Recovery requires explicit approval and must use a new immutable result root.
