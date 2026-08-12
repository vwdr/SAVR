# Phase V5-D V09 Default-Allocator Implementation Report

Status: **PRE-GPU VERIFICATION COMPLETE; GPU SELECTION NOT AUTHORIZED**

Date: 2026-08-11

## 1. Research conclusion

V08 removed the prior memory blocker and left 8,292,139,008 bytes below the
unchanged reservation cap. Its remaining failure occurred only during the
second shared-pool graph capture. Because V07's experimental expandable
allocator is no longer needed, reverting to PyTorch's default native allocator
is the narrowest falsifiable correction that does not modify the model or
measurement method.

PyTorch 2.2 documents the native allocator as the default,
`expandable_segments` as experimental, and ordered nonconcurrent graph-pool
sharing as supported. PyTorch's issue tracker contains related CUDA-graph
failures with expandable segments. This motivates V09 but does not prove that
the allocator caused V08's exact error.

## 2. Frozen change

V09 removes only the raw-process
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` setting. Both compiler and
raw processes reject any inherited allocator override. Raw/final evidence must
record an absent environment variable and PyTorch's observed `native` backend.

V08's inference mode, model/checkpoint, tensors, graph bodies, two-graph shared
pool, stream/order, warm-up lifecycle, 111-query schedule, gates, 23 GiB cap,
restoration rules, and scientific exclusions remain unchanged.

## 3. Implementation

- Immutable V09 recovery configuration and semantic validation.
- Isolated V09 adapter layered after V08, retaining its inference lifecycle.
- V09 selector, LIBERO configuration, runner, launcher, finalizer, and
  deterministic preflight wrappers.
- Fail-closed allocator absence and actual backend attestation.
- Six adversarial tests covering configuration inheritance, allocator scope,
  launch ordering, inference preservation, provenance, and pre-GPU stopping.

Resolved configuration semantic SHA-256:
`893012bebd2ed157ecaf0be71b48fd582b65ed85607ef0a73e781ddd9f918e98`.

## 4. Verification completed

- Six focused V09 tests passed.
- All 378 local tests passed.
- Ruff passed for every V09 Python file.
- Repository diff validation passed.
- Deterministic V09 preflight passed with semantic SHA-256
  `fa9596664fb831b46c75a6738e312277a0bf4a13bc38279e072abee3479750b2`.
- Zero GPU inspection/selection, CUDA initialization, model query, simulator,
  download, outcome access, or manuscript change occurred.

## 5. Current checkpoint

PR #95 passed both validation jobs and merged at revision
`65b2d22997350c620223accabad20811f6177890`. TITAN reproduced six focused
tests, all 378 tests plus nine subtests, and deterministic preflight with CUDA
hidden. Pinned PyTorch `2.2.0+cu118` reported no allocator environment,
allocator backend `native`, zero visible GPUs, and uninitialized CUDA.

Curated pre-GPU verification is stored at
`reports/runtime/acr_v5_d_v09_pre_gpu_verification.json` with semantic SHA-256
`92123fdc19a0ac5de703e82340edd15ac0c859e6b607549fa28d8cee6e48d4ec`.

Stop before GPU inspection or selection. A GPU attempt needs separate explicit
coordination and a newly sealed execution authorization.
