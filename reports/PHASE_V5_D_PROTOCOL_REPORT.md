# Phase V5-D Protocol-Freeze Report

Status: **COMPLETE — PROTOCOL FROZEN; GPU PHASE NOT STARTED**

Date: 2026-08-10

## 1. Outcome

The bounded real-tensor V5-D feasibility study is now fully predeclared before
backend implementation, GPU selection, model loading, or measurement. The
freeze preserves `v5-a100-b40` and the V5-C split-core executor contract.

Normative artifacts:

- `docs/ACR_V5_D_RESEARCH_AND_MEASUREMENT_DESIGN.md`;
- `docs/ACR_V5_D_GPU_FEASIBILITY_PROTOCOL.md`;
- `configs/acr/v5_d_gpu_feasibility_freeze.json`; and
- `tests/acr/test_v5_d_gpu_protocol.py`.

Freeze semantic SHA-256:
`b7b3bc058aa800ba9409e82288f637555c8ffe242d2320631c58685d548eb39a`.

## 2. Frozen experiment

V5-D will compare four paths—Batched FR, V5 refresh, eager reuse, and optimized
reuse—using exact deterministic A/B inputs and real pinned-model tensors. It
contains:

- seven correctness queries;
- two untimed warm-ups for each path;
- all 24 lexicographic path permutations, producing 96 timed queries;
- a hard cap of 111 full queries;
- 10,000 paired bootstrap resamples with seed `20260810`; and
- conjunctive numerical, wall-time, CUDA-time, visual-work, ordering, memory,
  lifecycle, restoration, and resource gates.

The compiler backend is attempted first. Raw CUDA graphs are a technical
fallback only if compilation is unavailable before any correctness or timing
record. They cannot replace a parity or performance failure, and mixed
backends are prohibited.

## 3. Verification and correction

The protocol tests validate the semantic digest, selected method, backend
waterfall, static tensors, seven-query correctness schedule, all 24 balanced
permutations, 111-query arithmetic, derived reuse wall target, deferred GPU
selection, resource caps, and claim boundary.

Validation passed 11 dedicated V5-D safeguards and the complete configured
repository suite passed 304 tests. Formatting, lint, compilation, and bootstrap
validation also passed. An initial full-suite invocation through an
unconfigured system Python could not import the local `src` package; the
repository-declared source path was then used and all 304 tests passed. No code
or criterion was changed in response.

During preparation, a hash reconciliation check detected that several
initially copied hash suffixes did not match the authoritative repository or
pinned TITAN files. Before publication, each value was re-read from its exact
source and corrected. This produced no experimental output and demonstrates
the intended fail-closed drift control.

## 4. Resources and protection

This checkpoint used:

- zero GPUs selected or queried;
- zero model loads or model queries;
- zero simulator resets or episodes;
- zero task-success, reward, or protected outcomes;
- zero downloads or environment changes; and
- no manuscript modification.

Only read-only pinned-file hashes beneath `/home/ved/SAVR` were checked on
TITAN. No unrelated university file, job, allocation, process identity,
permission, service, or configuration was inspected or changed.

## 5. Claim and next-step boundary

This is a protocol result, not a speed or task-performance result. It does not
support a positive-results paper claim.

The next eligible action is a separately authorized V5-D backend
implementation checkpoint. That implementation must pass local/fake-backend
tests and merge before the mandatory user-coordinated one-GPU selection. No
simulator work is eligible unless the eventual real-tensor phase passes every
frozen gate.
