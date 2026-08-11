# Phase V5-D v05 Transition-Recovery Implementation Report

Status: **PRE-GPU ACCEPTED; GPU NOT INSPECTED OR SELECTED**

Date: 2026-08-11

## 1. Purpose

V04 remains an immutable zero-query technical stop. Its raw process rejected
one immediate aggregate sample at 33% utilization four seconds after the
compiler process ended, although memory was already 6 MiB and later telemetry
was 6 MiB and 0%. V04 did not test the shared-pool backend.

V05 creates a new run identity and corrects only that transition gate. NVIDIA
documents `utilization.gpu` as recent-window activity with a product-dependent
sample period of one sixth to one second. V05 therefore discards two seconds,
then requires three new aggregate-only samples five seconds apart, with every
sample still constrained to at most 512 MiB and 5% utilization.

## 2. Frozen implementation

- New immutable run ID: `acr-v5d-real-tensor-feasibility-v05`.
- V04 configuration semantic SHA-256:
  `f5bd1e2c1622acbade05f9b67d8e7ee6e53dee935d9d315ecd7614f5a62f22e3`.
- V04 technical-stop semantic SHA-256:
  `a3515180022df7938b50956851a2ca05b698819da38b387ddc23b54e59769811`.
- V05 recovery-overlay semantic SHA-256:
  `41f516ff17af73dc20e68f8094ec94661c7905a4651aa2bb03ccd04417249065`.
- Resolved V05 configuration semantic SHA-256:
  `b34c1d70bbc7163419597148906c22daa82cea3b497405aeeb82afcb4802b2cf`.

The transition sampler runs only in the fresh raw process and only after
semantic validation of the compiler attempt and raw-transition permit. It
runs before PyTorch import, CUDA initialization, and model load. It records the
complete fixed window once, caches a passing final sample, and fails closed
without another window on identity, memory, utilization, sampling, or evidence
failure.

V04's shared-pool backend is reused unchanged. The compiler-first order,
checkpoint, method, tensors, 7 correctness queries, 8 warm-ups, 96 timed
queries, statistics, gates, and 23 GiB cap are unchanged.

## 3. Verification completed

- 353 complete local repository tests passed.
- All 6 V05 transition tests passed.
- Focused Ruff passed.
- Focused mypy with skipped imports passed; the ordinary focused invocation
  reached the repository's pre-existing `batched_dual_path.py` override error,
  unrelated to V05.
- Deterministic CUDA-free preflight passed with semantic SHA-256
  `67c641228f406b8048cacf813b52cc66ef9cc6e7249ab99c0512a7d1fc4cf101`.
- Query identities remain exactly 7 correctness, 8 warm-up, and 96 timed.
- V05 run, analysis, and verification paths are absent.
- PR #82 and the import-path correction PR #83 each passed both GitHub
  validation jobs.
- TITAN's compliant focused repeat passed 6 tests; the complete repeat passed
  353 tests and 9 subtests with CUDA hidden.
- The corrected closed-stdin TITAN import/API preflight passed with semantic
  SHA-256
  `0b71455a193e906fd68b05e89d48b72277b91a2554440ea31e0be85bd050fdb2`.
- Curated CPU-verification semantic SHA-256:
  `7fad244f58140616ef7abebfb8a907b78f156bd58586e5ab096f5a46260c3dab`.

The first TITAN full-suite command used two basetemps nested under one parent;
pytest removed that parent between invocations, causing 22 setup-only errors.
The independent top-level repeat passed fully. The first import preflight then
exposed a relative `PYTHONPATH` after changing directories. That v01 directory
is preserved; a separately identified v02 repeat used the absolute project
source path and passed. Neither excluded invocation exposed a GPU or produced
scientific output.

## 4. Safety and scientific boundary

This checkpoint performed zero GPU inspection/selection, CUDA initialization,
model load/query, simulator operation, download, reward/success access, or task
outcome access. It did not modify the manuscript. No positive or negative
method result was created.

## 5. Advancement boundary

All pre-GPU gates pass. The repository must now stop for explicit V05 GPU
selection coordination. No selector, model, compiler, raw capture, query, or
simulator work is authorized by this checkpoint.
