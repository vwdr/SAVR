# ACR Version 3 Diagnosis Report

Date: 2026-08-04

Status: **V3-A COMPLETE — NEW METHOD FROZEN BEFORE IMPLEMENTATION**

## Decision

Pursue **State-Aware Batched Dual-Path Asymmetric Camera Refresh
(`SA-BDP-ACR`)** as a new, separately gated positive-paper attempt. Preserve
the complete V2-C negative result. Do not rerun or reinterpret V2.

This is a technically plausible hypothesis, not a positive result.

## Why V2 could not pass by camera skipping alone

The immutable V2-C record measured upstream Full Refresh at `1213.519 ms`
median wall and `150.566 ms` median visual CUDA. Wrist-only reuse used
`75.104 ms` visual CUDA, implying `75.462 ms` of measured scene-camera visual
work.

At the fixed A5 scene-reuse weight `0.26055045871559634`, even an impossible
zero-overhead reuse path that removes all `75.462 ms` can reach only a
`0.983798` weighted wall ratio: a maximum `1.6202%` reduction. The frozen
paper gate is `2%`. Therefore, scene skipping alone cannot reach the gate; a
new route must also accelerate refresh queries.

The deterministic derivation is
`reports/runtime/acr_v3_feasibility.json`. It is an optimistic ceiling, not a
measured V3 result.

## Pinned-source finding

At OpenVLA-OFT revision `e4287e94541f459edc4feabc4e181f537cd569a8`, the
two-image `PrismaticVisionBackbone.forward` splits scene and wrist and loops
over them. It invokes SigLIP and DINOv2 once per camera, then concatenates the
camera token blocks and projects once. V2-C independently observed exactly
two SigLIP, two DINOv2, and one projector call for Full Refresh.

The two camera samples have the same tensor shape and independent tower
semantics. V3 will stack scene and wrist on the batch dimension, invoke each
tower once at batch size two, reconstruct the original scene-then-wrist token
sequence, and invoke the unchanged projector once. Refresh therefore removes
sequential per-camera tower launches without removing either camera.

## Scientific controls

- **Upstream sequential FR** remains the deployment baseline.
- **Batched Full Refresh (BFR)** uses the V3 batched path on every query. It
  isolates generic batching acceleration from asymmetric reuse.
- **SA-BDP-ACR** uses the exact conservative controller and batched refresh,
  with cached scene plus fresh wrist on reuse.
- Historical V2 remains a disclosed negative result only.

A combined V3 speedup is not sufficient by itself. BFR must independently
accelerate upstream FR, V3 reuse must accelerate BFR, V3 must reduce weighted
visual CUDA by at least 10%, and V3 may not be slower than BFR at the frozen
reuse weight. This prevents a caching claim from taking credit for batching.

## Numerical-equivalence boundary

PyTorch does not guarantee bitwise equality between batched and per-sample
floating-point computations. V3 therefore freezes the documented bfloat16
`torch.testing.assert_close` tolerances (`rtol=0.016`, `atol=1e-5`) before
implementation. Shape, dtype, device, scene/wrist order, and actions remain
strict: action chunks must be bitwise identical to upstream for the bounded
correctness inputs. Reuse remains bitwise identical to the established V2
wrist-only path for identical inputs/cache.

## Timing integrity

Controller and required cache/tensor operations remain inside the synchronized
policy-query boundary. Evidence hashing, JSON serialization, and file I/O are
performed after timing for every path. These operations do not affect policy
semantics and are not deployment inference work. No timed outlier may be
deleted.

## Research basis

- The official OpenVLA-OFT source establishes the sequential multi-image loop
  and exact interception boundary:
  <https://github.com/moojink/openvla-oft/blob/e4287e94541f459edc4feabc4e181f537cd569a8/prismatic/extern/hf/modeling_prismatic.py>.
- VLA-Cache establishes temporal visual redundancy as a VLA acceleration
  target, though its method and evaluator are not treated as evidence for V3:
  <https://arxiv.org/abs/2502.02175>.
- Selective Perception treats multi-view computation as task-dependent rather
  than uniformly necessary, but uses a trained router unlike V3:
  <https://arxiv.org/abs/2602.15543>.
- Adaptive Visual Token Caching and DySta show the broader move toward dynamic
  visual computation, but both differ materially from this training-free,
  camera-level execution redesign:
  <https://arxiv.org/abs/2602.00686> and
  <https://arxiv.org/abs/2602.03983>.
- PyTorch explicitly warns that batched and slice computations need not be
  bitwise identical and documents the bfloat16 closeness defaults:
  <https://docs.pytorch.org/docs/2.2/notes/numerical_accuracy.html> and
  <https://docs.pytorch.org/docs/2.2/testing.html>.

## Boundary and next step

No V3 implementation, GPU use, model query, simulator episode, new scientific
outcome, download, protected-population access, or manuscript edit occurred in
V3-A. The next phase is CPU-only V3-B implementation and requires a new user
authorization.
