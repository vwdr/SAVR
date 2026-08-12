# Phase V5-D v07 Allocator-Recovery Implementation Report

Status: **FROZEN FOR ONE-GPU EXECUTION; PRE-GPU VERIFICATION IN PROGRESS**

Date: 2026-08-11

## 1. Purpose

V06 remains an immutable technical stop before correctness. Its raw process
reported approximately 648.56 MiB reserved but unallocated, while exceeding
the 23 GiB cap by 245,366,784 bytes. V07 tests one PyTorch-documented allocator
change intended to reduce unusable free slices:

`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

PyTorch 2.2 documents this experimental native-allocator option, and TITAN's
pinned `2.2.0+cu118` library contains the implementation. This is a bounded
fragmentation hypothesis, not a guaranteed fix.

## 2. Isolation

V07 uses a new run identity and applies the exact setting only to the fresh raw
fallback process before PyTorch import. The compiler process remains unchanged.
V06's model, checkpoint, tensors, lifecycle, CUDA graphs, one stream, shared
pool, capture/replay order, transition gate, 111-query schedule, scientific
gates, and 23 GiB cap remain identical.

No `empty_cache`, allocator combination, retry, GPU switch, multi-GPU mode,
offload, quantization, model change, simulator access, outcome access, or
manuscript change is permitted.

## 3. Frozen identities and verification

- V06 configuration semantic SHA-256:
  `7d0976512b15c6d14486f9e83e5b14513ab7fc919bbf9b55b75c9536b90b92e6`.
- V06 technical-stop semantic SHA-256:
  `0588f628a118a2f467215c2337bc23452f3b8e98d0b5865c37be0d2892a18edb`.
- V07 overlay semantic SHA-256:
  `5088f08c98a53bd15d891081ebabfccfc94c5f3684fdbcc9700529e72597698d`.
- V07 resolved configuration semantic SHA-256:
  `fa9a5785eb2cb885bd98f05355db61251b840b4c6e5ce19760cff088880b88d5`.
- Deterministic preflight semantic SHA-256:
  `3ba2dc3b814799325dd553cc25a59894e7121e77488058a32596bf2a4a531b08`.
- All 366 local repository tests pass; all 5 focused V07 tests pass.
- Ruff, focused mypy, shell syntax, and deterministic preflight pass.

## 4. Execution boundary

After CI and CUDA-hidden TITAN verification pass, the user-authorized protocol
allows exactly one aggregate-selected GPU attempt. Any unsupported allocator,
OOM, cap breach, integrity failure, correctness failure, or completed negative
timing gate stops the run without tuning or retry. V5-E remains unauthorized.
