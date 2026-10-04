# CAC C1H Recovery 01 Comprehensive Readiness Audit

**Date:** 2026-08-31  
**Recovery:** `cac-c1h-headroom-v02-recovery01`  
**Verdict:** TECHNICALLY READY; EXPLICIT RECOVERY AUTHORIZATION STILL REQUIRED

## Scope and invariants

This audit changes no population, cache method, outcome, threshold, stopping
rule, or claim. C1H remains a paired dense-versus-`D62_BAL_PT1` repair-
opportunity screen. Stage 1 is exactly 120 conditions/240 episodes on state IDs
0--2. IDs 3--5 open only under the frozen ambiguity rule, for a cumulative
maximum of 240 conditions/480 episodes. C2 and state IDs 10--49 remain blocked.

## Failures anticipated and controls added

| Failure class | Preventive or fail-closed control |
|---|---|
| Wrong upstream import | Import the environment helpers from the pinned `libero_utils` module; regression-test the exact AST import and execute a real pinned-runtime import |
| Incorrect suite task order | Resolve every C0 task by exact LIBERO task name, never by global index or modulo arithmetic |
| Missing/corrupt initial state or renderer | Create all 40 tasks and validate every state ID 0--5 before authorization; 240/240 setups passed |
| Observation-layout mismatch | Require two 224x224x3 images and one 8-D robot state in the exhaustive setup audit |
| Dense helper integration drift | Before any terminal episode, compare the helper action with the official OpenVLA-OFT evaluator on the same simulator observation; require max absolute error <=1e-6 |
| Wrong normalization key | Resolve and validate every suite normalization key immediately after model load, before terminal execution |
| Checkpoint loader mutation | Authenticate three protected checkpoint files before loading, reject stale backups, capture the full inventory, and restore bytes/inventory exactly on success or failure |
| Cache reset memory spike | Drop the old cache, tracker, and anchor tensors and clear released CUDA allocations before each mandatory dense reset |
| Recursive source-age drift | Preserve the validated physical source tracker; it independently rejects any source age above four queries |
| Query/episode overrun | Prove the exact two-stage worst case is 19,922 model calls including two controls; enforce 20,000 calls and 480 episodes before invocation/scheduling |
| Slow per-step monitoring | Use a 0.2-second aggregate selected-GPU memory sampler plus PyTorch peak reservation instead of launching `nvidia-smi` every simulator step |
| GPU interference | Launcher and worker both require one explicit GPU at <=1,024 MiB and <=5% utilization; strict runtime stop remains 23,552 MiB |
| Disk/cache spill | Require at least 2 GiB free; force Hugging Face, Torch, Matplotlib, XDG, temporary, Python, and W&B caches under the new result root; force offline mode |
| Hanging process | Use a fail-closed external 36,120-second timeout in addition to the internal 36,000-second wall boundary |
| Partial-result steering | Keep episode outcomes in worker memory and write them only after an exact 240-episode stage; progress records contain counts/timing only |
| Extension failure after Stage 1 opens | Record that protected outcomes have opened and prohibit recovery; never incorrectly label it outcome-blind |
| Analyzer mismatch | Recompute schedules from C0, match every terminal row to its scheduled arm/condition, verify counts/hashes/resources/restoration/parity, and mechanically reapply Gate H |
| Detached-launch failure | Use one versioned launcher with the exact config, runtime, result root, GPU variables, and timeout instead of a compound remote shell command |

## Completed verification

- The original v01 technical stop is immutable and authentic: zero episode
  attempts, zero model queries, zero progress records, and no partial outcomes.
- All 240 permitted simulator setups across 40 tasks passed without a visible
  GPU or model query. The evidence is
  `results/cac-c1h-sim-preflight-v02/result.json`.
- Both analyzer branches passed synthetic end-to-end reconciliation:
  direct Stage-1 proceed and ambiguous Stage 1 followed by extension proceed.
- 37 focused C1H/C1/P3 tests pass on TITAN.
- The complete repository suite produced 511 passes and 9 subtest passes. Its
  only failure is an unrelated historical ACR-V10 pre-attempt assertion that
  expects the now-completed V10 result directories not to exist; every other
  V10 check passes. It does not touch shared C1H code or runtime behavior.
- The recovery config and every authenticated code hash reconcile locally and
  on TITAN. In-memory authorization proves that all non-authorization checks
  pass, while the persisted config remains blocked.
- The three protected checkpoint hashes match the completed C1 restoration;
  no stale loader backup exists.
- The recovery result root does not exist. Available project storage is
  348,200,075,264 bytes. At the final audit snapshot GPUs 0, 1, and 3 used 6 MiB
  at 0% utilization and GPU 2 used 64 MiB at 0%.

## Outcome and analysis lifecycle

1. Launch only after explicit Recovery 01 authorization is written into the
   config and its semantic hash is recomputed.
2. Run two pre-episode model controls. Any import, model, normalization,
   official-helper parity, action-processing, memory, checkpoint, or simulator
   failure stops before terminal evidence.
3. Run exactly 240 Stage-1 episodes while exposing only process health, progress
   count, bytes, elapsed time, and aggregate selected-GPU telemetry.
4. Write/open Stage-1 outcomes only after exactly 240 records. Apply Gate H once.
5. Stop, proceed, or open the frozen extension mechanically. No discretionary
   rerun, threshold change, or architecture choice is allowed.
6. Restore the checkpoint, run the independent analyzer, sync evidence locally,
   and stop before C2.

## Residual risks that cannot be honestly eliminated beforehand

The 7B model could still encounter a hardware/driver failure, the official
dense parity control could reveal an unforeseen pinned-runtime mismatch, a
long rollout could hit a simulator defect not present during state setup, or
the server could be interrupted externally. These are not being ignored: each
has a pre-outcome check or fail-closed record, and none permits an automatic
retry. Scientific failure at Gate H is also possible and must be reported as a
result rather than treated as a technical error.

No GPU/model recovery attempt has been launched by this audit. The persisted
recovery config remains unauthorized, and C2 remains unauthorized.
