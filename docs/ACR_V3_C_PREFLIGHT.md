# ACR Version 3 Phase V3-C Preflight

Date: 2026-08-04

Status: **FROZEN BEFORE REAL-MODEL EXECUTION**

Phase V3-C is authorized under `docs/ACR_V3_EXECUTION_PROTOCOL.md`. This
preflight resolves the exact 64-query schedule without changing the method,
controller, gates, populations, or resource caps. The machine-readable
authority is `configs/acr/v3_c_gate.json`.

## Exact correctness schedule: 8 queries

Two deterministic synthetic two-camera inputs, A and B, each consume one
sequential-FR oracle, one Batched-FR query, and one V3-refresh query. These six
queries establish two-input bfloat16 projected-token closeness at the frozen
`rtol=0.016`, `atol=1e-5` tolerance and bitwise action equality. Input A then
consumes one historical V2 wrist-only reuse and one V3 reuse with the same
owned scene block and current input. Their returned tokens and actions must be
bitwise identical.

The V3-B CPU suite remains the independent authority for cache mismatch,
exception restoration, nesting/concurrency, action-finite failure, immutable
identity, and query-budget behavior. Those proofs require no additional model
query and may not be replaced by a ninth correctness call.

## Exact latency schedule: 56 queries

- Two untimed warm-ups for each of sequential FR, BFR, V3 refresh, and V3
  reuse: 8 queries.
- Twelve timed repetitions for each path: 48 queries.
- Timed repetitions use the four cyclic orders frozen in the machine config,
  each exactly three times.

Every attempted model call consumes its unique identity before invocation.
No scientific failure is retried. A technical recovery, if required, must
preserve the parent attempt and may consume only unconsumed frozen labels;
cumulative use may never exceed 64.

## Fair timing boundary

All paths use identical model inputs and one synchronized CUDA-event boundary.
Sequential FR reports the outer synchronized wall time. BFR and V3 report the
adapter wall time that includes required batching, controller, cache,
proprioception, and downstream inference while excluding only the frozen
post-boundary action-finite validation. CUDA visual work is measured from
actual SigLIP, DINOv2, and projector hooks. Audit hashing, tensor comparison,
serialization, file I/O, and report construction occur after every timed
boundary. Every repetition is retained.

## Mechanical gate

Correctness must pass before latency can be accepted. At reuse weight
`0.26055045871559634`, every frozen latency condition must pass:

- BFR/sequential-FR median wall ratio at most `0.98`;
- V3-refresh/BFR median wall ratio at most `1.02`;
- V3-reuse/BFR median wall ratio at most `0.98`;
- V3-weighted/sequential-FR wall ratio at most `0.98`;
- V3-weighted/BFR wall ratio at most `1.00`;
- V3-weighted visual CUDA reduction versus sequential FR at least `10%`.

Any failure preserves the complete result and stops before V3-D, simulator
execution, Object states `3-9`, Goal, or a protected population.

## Resource and safety boundary

Use one responsibly selected GPU and one model process for at most 64 model
queries, 3,600 seconds, and 512 MiB of artifacts. Use no simulator, rollout,
download, training, manuscript edit, or write outside `/home/ved/SAVR`.
