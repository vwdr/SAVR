# Phase V5-D v04 TITAN Memory-Remediation Implementation Report

Status: **COMPLETE PRE-GPU CHECKPOINT; GPU EXECUTION NOT STARTED**

Date: 2026-08-11

## 1. Outcome

The V04 same-TITAN remediation is researched, frozen, implemented, reviewed,
merged, and verified without GPU visibility. It is ready for a separately
coordinated one-GPU feasibility attempt.

The change is narrow: the two sequential raw CUDA graphs share one PyTorch
private memory pool. The compiler-first/raw-second waterfall, selected method,
checkpoint, tensors, 111-query schedule, tolerances, timing design, statistics,
gates, and 23 GiB reservation cap remain unchanged.

This checkpoint contains no method result and does not predict that the memory
gate will pass. It establishes only that the proposed remediation follows the
pinned API and that its lifetime assumptions are mechanically enforced.

## 2. Research decision

V03 assigned separate private pools to wrist and downstream graphs and stopped
during the second capture at 24,937,234,432 reserved bytes. Returning to the
unchanged 23 GiB cap requires a reduction of 241,172,480 bytes.

PyTorch 2.2 explicitly permits graphs to share a private pool when they are
always replayed in capture order and never concurrently. The SAVR optimized
path is wrist then downstream on one thread for every query. V04 therefore
uses the documented `g1.pool()` to capture downstream in wrist's pool and
enforces the missing safety assumptions instead of relying on convention.

The higher-memory environment plan is retained but marked superseded after the
user clarified that the available cluster is the existing TITAN host.

## 3. Implementation

V03's hashed implementation files remain byte-identical. V04 is isolated in:

- `src/savr/acr/v5_d_v04_runtime.py`: provenance-checked overlay and resolved
  configuration;
- `src/savr/acr/v5_d_v04_torch_backend.py`: shared pool, shared capture stream,
  capture-stage memory trace, pointer checks, replay-order checks, one-stream
  checks, lock, and terminal invalidation;
- `src/savr/acr/v5_d_v04_adapter.py`: V04-only adaptation of the immutable V03
  runner and evidence writer;
- V04-only selection, config, run, launch, analysis, and preflight scripts; and
- focused recovery and fake-CUDA test modules.

The first implementation extended shared V03 files. The full regression suite
correctly rejected that design because it changed hashes in the immutable V03
preflight. The implementation was restructured into isolated V04 modules; all
shared V03 files were restored exactly before publication.

## 4. Verification

- Local full suite: `347 passed`.
- Focused Ruff: pass.
- Focused mypy: pass; the repository's pre-existing unrelated override warning
  remains outside the V04 modules.
- GitHub PR #79: both validation jobs passed; merge commit
  `34de55fbf6012705d0254231a3c15120e71f9412`.
- TITAN focused suite: `7 passed`, including V04 recovery/backend tests and the
  immutable V03 preflight verifier.
- TITAN full-suite repeat: exited successfully before the chained deterministic
  and import preflights ran. A prior attempt omitted `PYTHONPATH`, failed during
  collection with zero executed tests, and is excluded.
- Deterministic V04 preflight semantic SHA-256:
  `30899e753a50f0d8e293f81f435de56a4f51dccf41511b50316a4051c2719dda`.
- Pinned import/API preflight semantic SHA-256:
  `b467a4783d8dc67b5a6e445a6099cc12109f222ca3184f0240bec61ef22df019`.
- Curated CPU-verification semantic SHA-256:
  `60a5c44647ad1d699ee32ddbe2bd64da95bcb020a33145fda6c35c04299105cd`.

The pinned environment attested PyTorch `2.2.0+cu118`, a `pool` parameter on
both graph capture APIs, `CUDAGraph.pool()`, OpenVLA import, zero visible CUDA
devices, and CUDA uninitialized before and after import.

## 5. Resources and integrity

- GPU inspection/selection: `0`.
- CUDA initialization: `0`.
- Model loads/queries: `0`.
- Simulator instances/resets/episodes: `0`.
- Downloads and new task outcomes: `0`.
- Manuscript changes: `0`.
- Writes outside `/home/ved/SAVR` on TITAN: `0`.

The initial TITAN full-test command and all corrected temporary roots were
inside `/home/ved/SAVR/results`. No unrelated process, allocation, university
file, environment, service, permission, or server configuration was inspected
or changed.

## 6. Predetermined next step

Stop for explicit user coordination before running
`scripts/select_acr_v5_d_v04_gpu.py`. After coordination, select at most one
eligible TITAN GPU using three aggregate-only samples and execute V04 once.

If shared-pool capture exceeds the unchanged memory cap, fails correctness, or
violates restoration/integrity, stop without tuning. Only a complete verified
111-query result may advance to V5-E planning.
