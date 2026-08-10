# Phase V5-D Pre-GPU Implementation Report

Status: **COMPLETE — IMPLEMENTATION VERIFIED; STOPPED BEFORE GPU SELECTION**

Date: 2026-08-10

## 1. Outcome

The frozen V5-D real-tensor feasibility experiment is implemented end to end:
backend waterfall, mixed-dtype static executor, exact model cores, aggregate
GPU selector, project-local launch, immutable runner, paired analyzer,
independent verifier, and deterministic preflight evidence.

Preflight record: `reports/runtime/acr_v5_d_preflight.json`

- status: `pass`;
- semantic SHA-256:
  `db097ca8cab44d474a65e22888a72da8c4c6e2489a31188abea67c7ed55bff98`;
- query-label SHA-256:
  `eccd0e91acddf131411dbd916a1eb13bfa178b3e36c216bd4146bce1dfdf3291`;
- 111 exact labels: 7 correctness, 8 warm-up, and 96 timed; and
- all pre-GPU checks passed.

## 2. Pre-output corrections

Two issues were detected and corrected before backend publication or GPU
output:

1. The dependency-free V5-C executor assumed one fake dtype. V5-D adds a
   separate mixed-dtype subclass for real bfloat16 plus int64/bool tensors,
   leaving V5-C source/evidence unchanged.
2. Direct deterministic regeneration found copied-suffix errors in the six
   A/B image, instruction, and state hashes. They were corrected against the
   immutable V3-C machine record. The freeze semantic digest changed to
   `f445cf5d1a5ec6877ebea46ccc3883a11a676b38cb33a711ee4b74baf22f53f8`;
   no method, threshold, schedule, tolerance, gate, or resource changed.

The pinned PyTorch API was also checked read-only on TITAN. Version 2.2 has no
public `CUDAGraph.instantiate()` method, so the raw backend accepts only a
replayable graph instantiated implicitly at capture completion and records that
mode.

## 3. Verification

The implementation has dedicated adversarial tests for:

- exact query identities and budget exhaustion;
- compile-first/fresh-process raw transition;
- prohibition of raw fallback after correctness;
- cumulative preparation, wall, artifact, and memory caps;
- exact restoration after exceptions;
- real prompt/mask mixed dtypes and A/B/A current-value behavior;
- exact compiler arguments and both raw captures;
- pointer-drift rejection;
- deterministic positive and negative statistical records;
- independent gate recomputation;
- three-sample lowest-index GPU selection;
- simulator/outcome exclusion and project-local cache roots; and
- deterministic preflight publication.

The complete local regression suite passed **321 tests**. Change-scoped
formatting and lint, Python compilation, shell syntax, bootstrap validation,
the deterministic V5-D preflight verifier, and repository diff checks also
passed.

## 4. Resources and protection

This checkpoint used:

- zero GPU selections or workloads;
- zero model loads or queries;
- zero simulator resets or episodes;
- zero task success, reward, or protected outcomes;
- zero model, dataset, or TITAN downloads or environment changes; and
- no manuscript modification.

Local package validation used an isolated temporary build environment. It did
not modify the repository or either persistent Python environment.

TITAN access was limited to read-only pinned source/API checks inside
`/home/ved/SAVR`. No unrelated file, process, job, allocation identity,
permission, service, or configuration was inspected or changed.

## 5. Next boundary

The implementation advances only to explicit user coordination before GPU
selection. The one-GPU phase must begin with the aggregate-only selector and
cannot be inferred from this implementation approval.
