# Phase V5-D v03 Recovery Implementation Report

Status: **COMPLETE AND VERIFIED — GPU EXECUTION NOT AUTHORIZED**

Date: 2026-08-10

## 1. Outcome

The separately authorized v03 pre-GPU correction is implemented. It retains
the complete v02 scientific contract and changes only checkpoint-loader
restoration plus its preflight coverage. The new immutable run ID is
`acr-v5d-real-tensor-feasibility-v03`.

Resolved configuration semantic SHA-256:
`a9447cd385b4229e54cf85ba8fc7e06e4b4d283b9ac5c655e0c5201fb5d3f297`.

V03 overlay semantic SHA-256:
`a1b30fdd7337558281882438c28c917de52b90805a670a8eb8ec791dc30f146a`.

## 2. Exact correction

The runner now captures the complete top-level checkpoint inventory, protected
bytes/hashes, and non-protected entry signatures before model initialization.
Restoration then:

1. rejects missing baseline entries, symlinks, directories, and baseline
   drift;
2. accepts only new regular files using the exact protected-name backup forms
   frozen in the v03 overlay;
3. requires every backup's content hash to equal its protected baseline;
4. validates all new artifacts before deleting any;
5. restores protected bytes and removes only verified new backups;
6. requires exact final inventory and protected hashes; and
7. supports a second no-change validation.

The attempt evidence separately records protected-byte restoration, removed
loader backups, cleanup completion, and inventory equality. A restoration
failure clears raw-transition permission.

The runner also records the selected device's compute capability during a
future authorized execution. It does not silently bypass the compile-first
waterfall.

## 3. Verification

The regression suite covers the two exact v02 artifacts, all permitted suffix
forms, content mismatch, unrelated files, directories, symlinks, changed
protected and non-protected entries, an injected partial-cleanup failure, and
idempotent repeat validation.

- Local complete suite: 341 tests passed.
- TITAN scope-compliant complete suite: 341 tests and 9 subtests passed in
  38.24 seconds.
- Deterministic preflight semantic SHA-256:
  `f25d7f2dd743bf0c4fbe8a56420ba94d136c8cfc6b5baaeedf473e5a5a6ab163`.
- Closed-stdin import-preflight semantic SHA-256:
  `f8b9002d0998345c7f1a423003a180cfc5a406d52f85e1f3ac2d1f60fe22cdc5`.
- Curated CPU-verification semantic SHA-256:
  `37587ccd329cfc68672ef048a7422fb8825944a2a61260d1631b8532c2efdf95`.

The TITAN import preflight used `CUDA_VISIBLE_DEVICES=""`. CUDA remained
uninitialized before and after importing the pinned upstream evaluation
module. It used zero GPU inspection, model loads/queries, simulator instances,
downloads, or outcomes.

## 4. Process-boundary correction

The first TITAN pytest invocation inadvertently used pytest's default
temporary directory outside `/home/ved/SAVR`. No external path was inspected
or manually modified after detecting this. The entire suite was repeated with
both `TMPDIR` and `--basetemp` fixed beneath
`/home/ved/SAVR/results/acr-v5d-v03-cpu-test-preflight-v02/`; only this
scope-compliant repeat is acceptance evidence. This process error had no
scientific effect, GPU use, or outcome access.

## 5. Scope and disposition

No GPU was inspected or selected. No model was loaded, no model query ran, no
simulator was created, no task outcome was accessed, and the manuscript was
unchanged. V01 and v02 evidence remain immutable.

`STOP_FOR_EXPLICIT_USER_COORDINATION_BEFORE_V03_GPU_SELECTION`

This checkpoint does not authorize v03 GPU selection or execution.
