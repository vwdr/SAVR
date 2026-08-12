# ACR V5-D v07 Expandable-Segments Protocol

Status: **FROZEN FOR ONE-GPU EXECUTION**

Date: 2026-08-11

## 1. Decision and scope

V06 remains an immutable pre-correctness memory technical stop. V07 creates a
new run identity and changes only the raw fallback process's native PyTorch
allocator configuration, set before importing PyTorch:

`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

The compiler process remains unmodified. V07 retains V06's model, checkpoint,
tensors, pre-capture warm-up lifecycle, graph bodies, one stream, shared pool,
capture/replay order, transition rule, 111-query schedule, correctness and
timing gates, 23 GiB cap, and claim boundary.

## 2. Evidence and mechanism

V06 failed during downstream pre-capture warm-up with 24,259,268,096 bytes
peak-allocated and 24,941,428,736 bytes peak-reserved. The OOM message reported
approximately 648.56 MiB reserved but unallocated. The unchanged 23 GiB cap was
exceeded by 245,366,784 bytes (234 MiB).

PyTorch 2.2 documents `expandable_segments` as an experimental native-allocator
option that grows fewer per-stream segments and reduces unusable free slices
between independently allocated segments. This directly targets the observed
reserved-but-unallocated margin. It is plausible but not guaranteed because
PyTorch presents the option primarily for workloads whose allocation sizes
change over time.

Primary reference:

- [PyTorch 2.2 CUDA memory management](https://docs.pytorch.org/docs/2.2/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf)

TITAN's pinned environment is PyTorch `2.2.0+cu118`, CUDA `11.8`, and its
installed CUDA library contains the `expandable_segments` implementation.

## 3. Frozen lifecycle

1. Select one eligible GPU with the unchanged three-sample aggregate rule.
2. Run the unchanged compiler process with no allocator override.
3. Require an exact zero-output compiler failure, checkpoint restoration, and
   raw-transition permit.
4. Start one fresh raw process with the exact allocator setting present before
   Python imports PyTorch.
5. Require the unchanged V05 sustained-idle transition gate.
6. Require the raw attempt record to attest the exact allocator setting.
7. Execute V06's exact three-wrist then three-downstream pre-capture warm-ups,
   followed by wrist/downstream shared-pool capture and ordered replay.
8. Enforce peak reservation at or below 23 GiB.
9. Only if preparation passes, execute the unchanged 7 correctness, 8 schedule
   warm-up, and 96 paired timed queries.

## 4. Exclusions and stop rules

V07 does not permit `empty_cache`, a second allocator option, allocator tuning,
automatic retry, GPU switching, multiple GPUs, offload, quantization, model or
precision changes, fewer warm-ups, cap increases, simulator access, outcome
access, or manuscript changes.

- Unsupported allocator, warning/error, OOM, or cap breach: preserve and stop.
- Capture/order/stream/pool/pointer failure: invalidate and stop.
- Correctness failure: stop before timing.
- Complete negative timing gate: retain as valid engineering evidence.
- Any restoration or integrity uncertainty: stop without analysis.

Only a complete 111-query record may be analyzed. V5-E remains unauthorized.
