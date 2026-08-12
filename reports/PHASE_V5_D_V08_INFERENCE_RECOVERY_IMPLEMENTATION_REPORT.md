# Phase V5-D V08 Inference-Recovery Implementation Report

Status: **PRE-GPU CHECKPOINT PASSED — GPU SELECTION NOT AUTHORIZED**

Date: 2026-08-11

## Outcome

V08 is implemented as an isolated successor to immutable V07. It preserves
V07's allocator, model, tensors, raw backend, lifecycle, query schedule, gates,
and 23 GiB cap. Its only execution change is a fail-closed whole-raw-attempt
`torch.inference_mode()` lifecycle entered after transition revalidation and
before model initialization.

The runner attests that inference mode is active and grad mode disabled when
raw evidence is written. A separate create-once lifecycle record requires the
prior thread-local state to be restored in `finally`. The compiler process is
unchanged.

## Research basis

The pinned official OpenVLA action path uses inference mode, while the custom
V5-D path previously relied only on `model.eval()`. PyTorch 2.2 documents that
evaluation mode does not disable gradient tracking and recommends inference
mode for model evaluation that has no autograd interaction.

Pinned PyTorch CPU verification reproduced the relevant static-copy behavior:

- default mode: `CopyBackwards` remained attached and reachable backward nodes
  grew `10 -> 16 -> 22` across three repeated copies;
- inference mode: `requires_grad=False`, no `grad_fn`, and zero reachable
  backward nodes for all three repetitions; and
- CUDA remained uninitialized.

This provides a direct mechanism for V07's active-memory growth and a strong,
but not guaranteed, reason V08 should fit.

## Verification

- Resolved configuration semantic SHA-256:
  `20d51bb3429270665f77bd3a05d5986defdbffbce8d3f9044861870832e3eaa3`.
- Deterministic preflight semantic SHA-256:
  `6c4c6dbfaf01549c2fb58ba330feb952dfc402064ea4d7e2fca81d4ea1782503`.
- CPU mechanism semantic SHA-256:
  `154cc8cf8005f90ee3df7669c823f53048d9b3c2f95662538a2810e6faf7eff5`.
- Six focused V08 tests passed locally and on TITAN.
- All 372 repository tests passed locally.
- All 372 repository tests plus nine subtests passed on TITAN.
- Both GitHub validation jobs for PR #91 passed.
- Ruff and diff-integrity checks passed.

All TITAN verification was CUDA-hidden and project-confined. It used zero GPU
inspection, CUDA initialization, model query, simulator episode, download,
task outcome, protected population, or manuscript change.

## Checkpoint

V08 is ready for an explicit one-GPU execution decision. This report does not
authorize selection or launch. If authorized, the selector must create a new
immutable V08 manifest and the attempt must stop fail-closed on any inference
state, memory, capture, correctness, restoration, or resource violation.
