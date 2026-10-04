# Live checkpoint — 2026-09-26

## COMPLETED — supersedes dispatch instructions below

S1 PID 3074656 exited after successful completion at 19:34:20 UTC. All 168
episodes and 240 timing records completed, 3728 calls. Frozen CPU analyzer and
independent verifier passed. Read `reports/ADAPTIVE_SCREEN_V03_RESULT.md`.
Heartbeat `monitor-adaptive-comparison-v03` has been deleted. Do NOT relaunch.
No confirmation is running or frozen. Fixed384 and adaptive256 each obtained
39/40, versus dense39/40, with roughly15.5% lower controlled mean query time.
Adaptive384 obtained38/40 and6.76% reduction. Positive development tradeoff is
not a novel-method or paper-ready confirmation. All subsequent text records
the historical launch; it is not a current execution instruction.

User requested continued work on the authorized diagnosis → four-arm screen →
one justified confirmation plan. No new training or manuscript activity.

## Current stage: S1 dispatched — do not launch another worker

- Owned S1 PID **3074656**, launched 2026-09-26 after completed S0 verification
  and successful S1 CPU preflight (`preflight_passed=true`, zero model queries).
- Script `scripts/run_adaptive_screen.py`, configuration
  `configs/openvla/adaptive_screen_v3.json`, SHA256
  `ae1cbd7d0dbaccb3ff14a335cae6b451baa2b6ea62f25968e52c5ea5e3c8c768`.
- Output `results/adaptive-screen-v03`, terminal log
  `reports/adaptive-screen-v03-terminal.log`.
- Fresh pre-dispatch GPU0 identity matched; 6 MiB memory, 0% utilization. The
  worker repeats its own idle gate immediately before loading the model.
- The worker can spend minutes verifying inputs before creating the result
  directory. Monitor the owned PID in that interval, not unrelated processes.
- Heartbeat `monitor-adaptive-comparison-v03` is active every ten minutes, quiet
  on ordinary progress. It must use this S1 PID, not relaunch S0 or S1.

While active inspect only count-only streams, owned process health, artifact
bytes, elapsed time and aggregate selected-GPU0 telemetry. Do not read partial
successes, timing values, parity contents or aggregates. Require exactly 168
episode records, 240 timing rows and immutable completed summary before opening
outcomes. If stopped early, preserve and inspect technical_stop/technical_traceback
only; terminal log is allowed for a pre-result-directory technical failure.
No automatic retry. Verify all frozen hashes, counts, caps, controls and four
arms before CPU analysis; sync evidence/status locally. Delete heartbeat when
completed or technically stopped. No GitHub push or manuscript activity.

## Completed qualification dispatch

- Host: ssh titan; all remote operations inside /home/ved/SAVR.
- Selected GPU: 0, UUID GPU-bb2451d6-2989-a112-5c18-8892943710e4.
- Fresh read-only pre-dispatch observation: 6 MiB, 0% utilization. Worker repeats
  its own idle/identity gate immediately before model loading.
- CPU preflight completed successfully with zero model calls.
- Source hashes match frozen local and server copies; CPU tests 27 passed,
  eight skipped (35 discovered).
- Worker PID **3072450**, `scripts/run_adaptive_qualification.py`.
- Config `configs/openvla/adaptive_qualification_v3.json`, SHA256
  `6f3a36decbb18706aa6cf39f2a727eb64601fd53de8efa19b334bf87ea10fc63`.
- Output `results/adaptive-qualification-v03`; terminal log
  `reports/adaptive-qualification-v03-terminal.log`.
- No automatic retry. While active inspect only health, counts and bytes.

## S0 verification completed

S0 completed 176 calls and 16 records, zero episodes, all 16 exact command
matches and zero hidden/normalized/raw parity error. Frozen CPU analyzer and
independent local reconciliation both passed. Summary SHA256
`55af20761ff4bbba9a30229b53bee50db6a9fedffbc7f21eb81145a098832948`.
See `reports/ADAPTIVE_QUALIFICATION_V03_VERIFICATION.md`. Local evidence synced.

## Gated sequence (S0 and S1 dispatch below already done)

Require 176 model calls, 16 check records, immutable completed summary, unchanged
sources and artifact hashes. Run analyze_adaptive_qualification.py CPU-only,
CUDA hidden, bytecode disabled, one BLAS thread. Independently reconcile exact
zero-pruning commands, numerical parity and both actual pruning budgets.

Then run S1 CPU preflight and dispatch one GPU-0 screen after its fresh idle gate:
`scripts/run_adaptive_screen.py --config configs/openvla/adaptive_screen_v3.json`.
S1 config SHA256
`ae1cbd7d0dbaccb3ff14a335cae6b451baa2b6ea62f25968e52c5ea5e3c8c768`.
Output `results/adaptive-screen-v03`. No duplicate dispatch or automatic retry.
Do not inspect outcomes until 168 episode records, 240 timing rows and a complete
immutable worker summary exist. Reconcile all four arms and controls, CPU-analyze,
sync complete evidence locally, then decide whether one confirmation is warranted.

The adaptive arm is a native-SDPA local selection adaptation, not exact upstream
VLA-Pruner reproduction. See protocol V2. No premature positive-paper claim.
