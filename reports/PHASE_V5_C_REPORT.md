# Phase V5-C Static Executor CPU-Correctness Report

Status: **COMPLETE — CPU CONTRACT PASSED; V5-D PROTOCOL PREPARATION ELIGIBLE**

Date: 2026-08-10

Implementation commit: `7159f4d475e90ac2f3454be298d4bc92939e69c0`

## 1. Outcome

V5-C implemented and mechanically verified the frozen split-core static-buffer
executor for the selected `v5-a100-b40` IR-SA-ACR controller. The reference
and static executors were exactly equivalent on deterministic wrist tokens,
scene-first combined tokens, and normalized actions. The static executor kept
owned buffer identities stable while current caller objects and values changed.

This is a **software-correctness result only**. It does not show that the pinned
OpenVLA model compiles or captures, fits GPU memory, runs faster, or preserves
task success. Those questions remain gated behind a separately frozen V5-D
one-GPU protocol.

## 2. Implemented contract

- reference executor: `acr-reuse-executor-eager-v1`;
- static executor: `acr-reuse-executor-static-v1`;
- integration adapter: `ir-sa-acr-static-executor-v1`;
- lifecycle: `UNPREPARED → PREPARED → ACTIVE`, with fail-closed
  `INVALIDATED` after postlaunch failure;
- complete 24-field compatibility key;
- owned wrist, cached-scene, prompt, mask, proprioception, wrist-output,
  combined-output, and action-output buffers;
- one fresh wrist core and one fresh downstream action core per completed
  reuse, with zero scene-core calls;
- prelaunch rejection to explicit eager refresh without consuming reuse;
- postlaunch failure invalidation with no controller observation or retry; and
- exception-safe episode method restoration.

The selected controller thresholds, signals, cache semantics, and prior V5-B
evidence were not changed. Shared `types.py` was deliberately left unchanged
after the full regression suite showed that modifying it would invalidate a
source digest embedded in immutable V5-A evidence. Executor reason insertion
is therefore isolated inside the new V5 integration adapter.

## 3. Acceptance matrix

| # | Frozen requirement | Result |
|---:|---|---|
| 1 | Exact identities and selected controller | PASS |
| 2 | Eager/static wrist parity | PASS |
| 3 | Scene-first combined-token parity | PASS |
| 4 | Normalized-action parity | PASS |
| 5 | One wrist/downstream, zero scene calls | PASS |
| 6 | Current prompt/proprioception affect result | PASS |
| 7 | Stable owned-buffer identities | PASS |
| 8 | Changed values cannot return stale output | PASS |
| 9 | Every key field rejects mismatch prelaunch | PASS |
| 10 | Unsupported modes/shapes/metadata fail closed | PASS |
| 11 | Nested/concurrent/reset-active use rejected | PASS |
| 12 | Prelaunch unavailable forces refresh | PASS |
| 13 | Postlaunch failure invalidates without observe | PASS |
| 14 | Method restoration survives exceptions | PASS |
| 15 | Episode reset clears bindings/buffers/counters | PASS |
| 16 | Selected controller trace matches established path | PASS |
| 17 | Reuse streak one; prefix reuse at most 0.40 | PASS |
| 18 | No hot-path hash/I/O/full-tensor audit/device sync | PASS |
| 19 | Legacy adapter/controller regressions | PASS |
| 20 | Local repository and pinned TITAN CPU verification | PASS |

## 4. Deterministic machine evidence

Machine record:
`reports/runtime/acr_v5_c_cpu_executor_verification.json`

- semantic SHA-256:
  `f7a8d11d4574add57caa630c03463375421d9482984478be769f497b1c9d0b66`;
- file SHA-256:
  `15c2249d59e7d196c0652b42fc713469071e95355420e401c6b2b86500d6030c`;
- compatibility-key SHA-256:
  `500432c7e1fc6636ede47f5bdccc4e5bd8637804594a3b841409c19bf88dc54b`;
- reference/static completed queries: `3/3`;
- each executor: zero scene, three wrist, three downstream calls;
- 128-query selected-controller trace: 51 reuses, maximum streak `1`, maximum
  prefix reuse `0.40`; and
- legacy controller separation: maximum streak `2` versus isolated `1`.

## 5. Verification

Local:

- complete repository: `293 passed`;
- focused V5-C plus legacy-adapter matrix: `47 passed`;
- `ruff check src tests`: pass;
- all changed V5-C files: Ruff format check pass;
- new executor/adapter mypy check: pass;
- bootstrap validation: pass (`85` required files);
- package wheel build: pass; and
- dependency-free verifier deterministic across repeated calls.

The repository-wide Ruff inventory still reports two lint and thirty format
findings in pre-existing legacy files. None is in a V5-C file, and changing
those historical files would violate this phase's scope and evidence
preservation boundary. The change-scoped and source/test lint gates pass.

TITAN (`/home/ved/SAVR` only):

- exact commit `7159f4d475e90ac2f3454be298d4bc92939e69c0`;
- system Python `3.10.12`;
- dependency-free verifier produced the identical semantic SHA-256;
- compilation passed;
- bootstrap validation passed; and
- clean detached checkout after verification.

As previously documented, TITAN's system Python has no `pytest`. No package or
environment change was permitted or performed; full pytest verification was
therefore local/CI, while the deliberately dependency-free semantic verifier
was the pinned TITAN CPU gate.

## 6. Resources and protected boundaries

- GPUs: `0`;
- model queries: `0`;
- simulator episodes/resets: `0`;
- downloads or environment changes: `0`;
- new task outcomes or success fields: `0`;
- protected Goal/reserve/final access: `0`; and
- manuscript modifications: `0`.

All remote work stayed inside `/home/ved/SAVR`. No unrelated university file,
process, service, permission, environment, GPU allocation, or server
configuration was inspected or changed. The temporary Git transfer bundle was
created only inside each SAVR repository and removed from both locations after
verification.

## 7. Disposition

`ADVANCE_ONLY_TO_V5_D_PROTOCOL_PREPARATION`

V5-D must be researched and frozen before any GPU selection. It must test
real-tensor parity, compilation/capture feasibility, memory, and timing under
the predeclared compiler-then-raw-CUDA-graph waterfall. No online rollout,
task-success claim, or manuscript result is authorized by V5-C.
