# Phase V5-D V10 Downstream-Only Graph Implementation Report

Status: **PRE-GPU CHECKPOINT PASSED — GPU SELECTION NOT AUTHORIZED**

Date: 2026-08-13

## Outcome

V10 is implemented as an isolated successor to immutable V09. It replaces the
failed two-graph shared-pool backend with one eager wrist core and exactly one
downstream-only CUDA graph. The ACR method, model, tensors, default allocator,
inference-mode lifecycle, 111-query schedule, correctness tolerances,
statistical gates, 23 GiB cap, and scientific exclusions remain unchanged.

The implementation is fail-closed on capture, pointer, stream, order, and
reentrancy violations. It records the exact three wrist warm-ups, three
downstream warm-ups, single downstream capture, graph count, memory stages,
and published preparation labels. The independent verifier rejects missing or
mutated hybrid provenance.

## Technical rationale

V08 and V09 both completed wrist capture and failed at the second downstream
capture despite different allocators and more than 7.6 GiB of reservation
headroom. V10 removes that repeated-capture boundary while leaving the
downstream graph body unchanged. Prior V3-C timing attributes 93.43% of reuse
CUDA time to the downstream portion, so this narrow architecture remains a
meaningful efficiency test if capture succeeds.

## Verification

- Resolved configuration semantic SHA-256:
  `b078b57dff6f8a9548a75e5b611375ce71e9066eccd950c4f1ea22e6abbac083`.
- Deterministic preflight semantic SHA-256:
  `3cf4ecc4cbe40a8d771fcad2ecadc90b92c0ffdf25f7a6b806b94f6725f22132`.
- Curated pre-GPU verification semantic SHA-256:
  `f4da5caf3a552f188a39ffdcbbaa5677e26c28b1b3c8515fdd89dcda16db1b0a`.
- Thirteen focused tests and all 391 repository tests passed locally.
- Ruff, formatting, repository-diff integrity, and two byte-identical
  preflight evaluations passed.
- Both GitHub validation jobs for PR #100 passed; implementation revision is
  `f587d13b4f439fd075dc53e890641115b6abce1e`.
- TITAN reproduced 13 focused tests, all 391 tests plus 9 subtests, and all 15
  deterministic preflight checks on the exact merged revision.
- TITAN's pinned PyTorch `2.2.0+cu118` reported CUDA uninitialized with
  `CUDA_VISIBLE_DEVICES` empty.

All work was confined to the SAVR repositories. It used zero GPU inspection or
selection, CUDA initialization, model query, simulator episode, download,
task outcome, protected population, or manuscript change.

## Checkpoint

V10 is ready for a separate explicit one-GPU execution decision. This report
does not authorize selection or launch. A future attempt must retain the
frozen compiler-to-raw waterfall, exact 111-query cap, one downstream capture,
no retry, and every correctness, efficiency, memory, restoration, and resource
gate. Passing V10 would establish real-tensor feasibility only; V5-E would
still require a separately frozen and authorized protocol.
