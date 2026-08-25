# BRACE-B3 v05 Audited Tensor-Path Recovery Freeze

Date: 2026-08-25

Status: Completed; stopped negative

## Sole correction

The patch-change range now calls binary `torch.maximum` and `torch.minimum`
with the two live tensor operands. Sidecar index tensors and ordered profile
positions inherit the device of their input tensors instead of assuming a
literal device name. These are integration corrections only: they do not
change scores, ordering, budgets, profiles, actions, or acceptance criteria on
the physical CUDA path.

The resolved configuration semantic SHA-256 is
`f3c21dd00ca314c17d0c6044536ff242ce5099336edc4550d136a167c60882ff`.

## Exhaustive preflight basis

The accepted pre-v05 audit exercised all CPU-testable B3 paths using real
PyTorch tensors, all profiles and horizons, every scheduled cycle, source
provenance, query accounting, the synthetic terminal analyzer, pinned private
interfaces/backends/repositories, and immutable evidence. The TITAN audit
passed 43 tests with CUDA hidden and uninitialized, zero model loads and
queries, and semantic SHA-256
`35cb1448b3c2ec80f02bcbfa3fc9c09b927dd0d435fe021d43c1b070f55da273`.

## Unchanged boundary

Every scientific and resource setting remains identical to the v01 base:
method, four clean profiles, thresholds, horizons, timing, parity, comparator
dispositions, the 388-query plan and 420-query hard cap, one aggregate-idle
GPU, the strict 23 GiB memory gate, zero simulator outcomes, no protected-
outcome access, and conjunctive acceptance gates.

This authorizes one v05 attempt only. There is no automatic retry. B4 remains
unauthorized, including if v05 is accepted.

## Completion

The single attempt completed 356 queries without a technical stop. P2-D50
passed both speed thresholds but failed action parity; no profile passed the
joint gate. The analyzer disposition is `stopped_negative`. Evidence is in
`reports/BRACE_B3_V05_REPORT.md` and `results/brace-b3-physical-v05/`. No retry
or B4 work is authorized.
