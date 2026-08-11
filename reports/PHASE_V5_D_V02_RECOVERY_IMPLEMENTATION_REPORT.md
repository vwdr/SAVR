# Phase V5-D v02 Recovery Implementation Report

Status: **COMPLETE — PRE-GPU RECOVERY VERIFIED; GPU NOT INSPECTED**

Date: 2026-08-10

## 1. Outcome

The v01 launch/preflight defect is corrected without changing the V5-D
scientific experiment. v02 uses the new immutable run ID
`acr-v5d-real-tensor-feasibility-v02` and resolves the original frozen contract
through recovery overlay semantic SHA-256
`5c92f49601c1c04a87670ccd4f0cfd5a3057dcd7b5825d5defeb2e32a9437aaf`.
The resolved experiment semantic SHA-256 is
`4ae65dda537a5b6dcdf9abd34d79e0a9d7defee834a2a8cc2f7107a659f36076`.

Every selected-method, stack, input, tensor, backend, correctness, timing,
bootstrap, threshold, memory, resource, recovery, and claim field is exactly
equal to the v01 base freeze. Only these two recovery changes are permitted:

1. create and attest a canonical run-local LIBERO config before upstream
   import; and
2. record any uncaught pre-model failure as an immutable zero-query stop that
   cannot trigger raw fallback.

## 2. Canonical LIBERO configuration

The project-owned helper writes canonical JSON, which is valid YAML, to
`<v02-run>/cache/libero/config.yaml`. It contains exactly the pinned benchmark,
BDDL, init-state, dataset, and asset paths. It rejects escaping paths,
symlinks, changed keys or bytes, and any second creation attempt. The launch
wrapper calls this helper before either backend process. The runner verifies
both the file and its separate immutable attestation before importing LIBERO.

## 3. Pre-model failure envelope

Manifest, configuration, pinned-source, aggregate launch-snapshot, LIBERO
attestation, environment, and import failures are now enclosed by a
dependency-light outer stop handler. An uncaught failure records zero model,
backend, correctness, warm-up, timing, simulator, download, and outcome counts
and exits with status 4. The launcher permits raw transition only for exact
status 20, so pre-model failures cannot become backend retries.

## 4. Verification

- Complete local suite: **329 tests passed**.
- Deterministic v02 preflight semantic SHA-256:
  `d7c3ed40cc9d5760a846cb15c688fa5c776cbac8f243d948376d16e64427a695`.
- TITAN closed-stdin import preflight semantic SHA-256:
  `a3ffc574631e8e250ab8021c0f8b99e0bf329a1e82d085499fb8e19747dd3490`.
- The pinned upstream module imported successfully with interactive input
  forbidden and stdin connected to `/dev/null`.
- `torch.cuda.is_initialized()` was false before and after import.
- GPU inspections, model loads/queries, simulator instances/resets/episodes,
  downloads, and task outcomes were all zero.
- Formatting/lint, Python compilation, shell syntax, bootstrap validation,
  semantic-record reconciliation, and repository diff checks passed.

## 5. Boundary

This checkpoint authorizes no GPU selection or execution. It advances only to
explicit user coordination before the v02 aggregate-only selector. Even a v02
feasibility pass would permit only preparation of the later online protocol;
it would not establish task success or a positive-paper result.
