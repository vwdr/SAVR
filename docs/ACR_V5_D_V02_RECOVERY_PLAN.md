# ACR V5-D v02 Recovery Plan

Status: **PROPOSED AFTER v01 TECHNICAL STOP; NOT AUTHORIZED FOR EXECUTION**

Date: 2026-08-10

## 1. Objective and invariant

V5-D v02 will answer the same frozen feasibility question as v01. The selected
method `v5-a100-b40`, tensor cores, backend waterfall, correctness tolerances,
111-query schedule, bootstrap analysis, efficiency gates, memory limits, and
claim boundary must remain unchanged. v02 is not an opportunity to tune the
method or browse backends.

The only permitted experiment-level change is deterministic, non-interactive
initialization of LIBERO's configuration before its upstream module is
imported, plus stronger pre-import failure recording.

## 2. Required implementation correction

Before Python imports LIBERO, a project-owned helper must create a canonical
`config.yaml` beneath the v02 run-local `LIBERO_CONFIG_PATH`. It must contain
only the pinned repository paths for LIBERO's benchmark root, BDDL files, init
states, datasets, and assets. The helper must:

1. refuse any root outside `/home/ved/SAVR/results/<v02-run-id>`;
2. refuse symlinks or paths escaping `/home/ved/SAVR`;
3. create the file once and never overwrite it;
4. read it back and verify exact keys and canonical paths;
5. record its bytes and SHA-256 in the launch evidence; and
6. perform no dataset, simulator, network, or GPU access.

The runner must validate the same file before the upstream import. It must
also place manifest validation, pinned hashes, aggregate launch telemetry, and
all imports inside the technical-stop envelope so that any pre-model exception
produces an immutable terminal summary rather than only a shell traceback.

## 3. New pre-GPU gates

Before a v02 device may be inspected or selected:

- a unit test must begin with a genuinely absent LIBERO config and prove the
  canonical helper creates the expected file exactly once;
- a second creation attempt must fail without changing bytes;
- wrong keys, escaping paths, symlinks, and pre-existing mismatched bytes must
  fail closed;
- an import-only subprocess with stdin closed must import the pinned upstream
  evaluation module without prompting, loading a model, initializing CUDA, or
  touching a simulator;
- the launch wrapper must be tested to prove config preparation precedes the
  runner;
- an injected import failure must produce an immutable zero-query technical
  summary and must not trigger raw fallback; and
- the complete repository regression suite and deterministic preflight must
  pass on the merged execution revision.

## 4. Evidence and run identity

v01 remains immutable and excluded from v02's query budget. v02 requires:

- a new run ID, `acr-v5d-real-tensor-feasibility-v02`;
- a new machine freeze referencing v01's technical-stop digest;
- refreshed source and preflight hashes;
- a clean reviewed/merged/synchronized execution commit; and
- a new aggregate-only launch manifest and GPU selection after separate user
  authorization.

The statistical seed, sample size, thresholds, and all method identities stay
exactly frozen. No v01 timing or correctness output exists and none may be
imputed.

## 5. Authorization sequence

1. Implement and CPU/import-preflight the correction without GPU inspection.
2. Publish, review, merge, and synchronize that checkpoint.
3. Stop and obtain explicit user authorization for v02 GPU selection.
4. Run v02 exactly once under its new immutable ID.

This plan does not itself authorize implementation or execution.
