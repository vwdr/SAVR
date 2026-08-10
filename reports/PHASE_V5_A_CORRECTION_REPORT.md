# ACR Version 5 Phase V5-A Correction Report

Status: **COMPLETE; ISOLATED-REUSE SEMANTICS VERIFIED; PERFORMANCE UNEVALUATED**

Date: 2026-08-10

## Outcome

The State-Aware Visual Refresh route remains intact. V5 corrects the V4
temporal specification error through a separate Isolated-Reuse State-Aware ACR
controller (`IR-SA-ACR`). The research audit and controller contract were
published before implementation in PR #64 at merge commit
`6007d6fd834b8b1e464275d2ab36b304c04cbbe8`.

The legacy ACR controller and every V3/V4 result remain unchanged.

## Implemented correction

- `IsolatedACRController` accepts only the state-aware policy, horizon 1, and
  controller identity `acr-isolated-controller-v1`.
- A completed reuse sets a controller-owned `refresh_required_after_reuse`
  latch.
- While latched, the next query is forced to refresh with
  `post-reuse-refresh`.
- Only a successfully observed refresh clears the latch. A decision that is
  computed but fails before observation cannot clear it.
- Cache age must be 0 after refresh and 1 after reuse. Disagreement forces
  `isolation-state-mismatch`.
- `observe` rejects a forged consecutive reuse even if a caller bypasses the
  decision returned by the controller.
- The latch and completed refresh count are included in the controller
  snapshot and reset at episode boundaries.
- Existing batched dual-path code accepts the new controller without changing
  its refresh/reuse computation paths.

## Verification

The frozen CPU matrix covers invalid configurations, warm-up, stable traces,
post-reuse forcing, cache/latch mismatch, unobserved technical failure,
forged decisions, gripper transitions, context mismatch, prefix budget, reset,
randomized invalid inputs, legacy-version separation, and batched-adapter
integration.

The dependency-free mechanical verifier produced:

- corrected trace: 128 queries, 51 reuses, maximum prefix reuse fraction 0.40,
  maximum reuse streak 1;
- preserved legacy trace: maximum reuse streak 2;
- mismatch reasons: `post-reuse-refresh` and
  `isolation-state-mismatch`; and
- semantic SHA-256:
  `7dcde7e8b96ba7fe79f1eed0cd6a73661e0d0977678f3581062902b445f7de2b`.

All 253 repository tests pass locally. TITAN's project-scoped Python 3.10
verification reproduced the tracked machine record exactly and compiled the
new controller/verifier without writing outside `/home/ved/SAVR`; no GPU or
shared process inspection was performed.

Machine evidence is in `reports/runtime/acr_v5_cpu_verification.json` and is
reproducible with `scripts/verify_acr_v5_isolation.py`.

## Scientific boundary

This is a positive software-correctness result only. It establishes that V5
mechanically enforces isolated whole-scene reuse and preserves the existing
fresh wrist/proprioception/downstream architecture. It does not establish a
safe threshold, realized reuse rate, task success, visual-CUDA reduction,
latency improvement, executor speedup, or positive paper result.

No GPU, model query, simulator episode, download, new benchmark outcome,
protected population, or manuscript change was used.

## Next gate

Stop after the correction checkpoint. Any attempt to select thresholds,
estimate efficiency from replay, implement the optimized executor, use a GPU,
or run a simulator requires a new output-blind V5 protocol. Goal and all final
populations remain unopened.
