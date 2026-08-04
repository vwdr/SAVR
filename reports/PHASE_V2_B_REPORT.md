# ACR Version 2 Phase V2-B Report

Date: 2026-08-03

Status: **COMPLETE — CPU IMPLEMENTATION AND VERIFICATION PASSED**

## Scope

The user authorized Phase V2-B on 2026-08-03. This phase implemented and
CPU-tested the frozen State-Aware Dual-Path Asymmetric Camera Refresh
(`SA-DP-ACR`) execution architecture. It did not authorize or perform a GPU
query, model load, simulator reset, rollout episode, download, threshold
change, protected-population access, or manuscript edit.

The exact Version 1 controller and its frozen `acr-t25-h2-b30` parameters are
unchanged. Version 1 implementation and evidence files are unchanged.

## Implementation

The new project-owned adapter is `src/savr/acr/dual_path.py`.

- Installation is episode-scoped and exception-safe.
- Refresh calls the saved original upstream `_process_vision_features` method
  exactly once on the ordered 12-channel scene+wrist tensor.
- The exact original combined projected tensor is returned to downstream code
  without replacement, copying, or reordering.
- The projected tensor is split only to store an owned, detached scene block.
- Reuse performs no scene tower/projector work and executes one current wrist
  SigLIP path, one current wrist DINOv2 path, and one unmodified projector.
- Current proprioception and all downstream policy work remain outside the
  intercepted boundary and therefore remain fresh.
- Physical module invocations and logical camera work are separate immutable
  fields in `DualPathWork`.
- Structural context, shape, dtype, device, patch-count, token-order, and cache
  compatibility checks remain online.
- Production mode performs no projected-token finite scan. It retains the
  required terminal action-finite check. Correctness mode performs complete
  projected-block finite validation.
- Nested/concurrent episode or query use is rejected. Any query failure
  invalidates the scene cache, preserves a classified failure object, and does
  not advance controller state.

No upstream OpenVLA-OFT or LIBERO source file was modified.

## Frozen minimum-test reconciliation

| Requirement | Evidence | Result |
|---|---|---|
| Original refresh invoked exactly once | Fake upstream call counters | PASS |
| Original refresh output returned unmodified | Object-identity assertion | PASS |
| Scene split, cache ownership, token order | Detached-clone and exact-block assertions | PASS |
| Wrist-only reuse truth table | Refresh/reuse path matrix | PASS |
| Physical versus logical accounting | Exact `DualPathWork` validation | PASS |
| Episode installation and exception restoration | Class/instance method restoration assertions | PASS |
| Nested/concurrent rejection | Nested episode, nested query, and other-thread tests | PASS |
| Context/shape/dtype/device/patch failures | Fail-closed parameterized tests | PASS |
| No production intermediate finite synchronization | Finite-operation call audit | PASS |
| Correctness-mode full finite validation | Combined/scene/wrist/action call audit | PASS |
| Action-finite failure preservation | Non-finite action failure record and state test | PASS |
| Immutable recovery identities | Write-once record and monotonic attempt test | PASS |
| Existing and new tests | 172 tests plus 9 subtests on TITAN | PASS |
| Ruff, format, mypy, bootstrap, compilation, package build | Local checks and clean wheel build | PASS |

## Verification

- Local full suite: `172 passed`.
- TITAN pinned environment full suite: `172 passed, 9 subtests passed`.
- New V2-B matrix: `14 passed` locally and on TITAN.
- Changed-file Ruff: pass.
- Changed-file formatting: pass.
- Changed-source mypy: pass.
- Python byte compilation: pass.
- Bootstrap validation: pass, 68 required files.
- Wheel build: pass; wheel SHA-256
  `c2976900dd98dc4814bce7ca13bda4ed32dc7f2c0dccf81457804b7229019309`.
- New/changed implementation and test bytes before reporting: 40,244 bytes,
  below the 512 MiB phase cap.

Pre-publication file hashes:

- `src/savr/acr/dual_path.py`:
  `00ee207812bf0320fbcdb62bed0c76165c1d9f4c3c80d8a834d6f87a28a1896c`
- `src/savr/acr/__init__.py`:
  `503da967036b50571a5988ed2890f612beba1929348d7ae0d6c6326487e747f3`
- `tests/acr/test_dual_path_adapter.py`:
  `f04ff39db366719b76f55720cde13a0fac16a2a8b1c1cd49e24a6e65861ba9c0`

## Exit decision

Every V2-B CPU, static, integrity, and resource gate passes. Phase V2-B is
complete. This is implementation evidence, not evidence that the method is
bitwise equivalent or faster on the real model and not a positive method
result.

Phase V2-C remains unauthorized. It requires separate user approval and must
apply the frozen 48-query, zero-episode correctness and paired-latency gates
before any rollout can begin.
