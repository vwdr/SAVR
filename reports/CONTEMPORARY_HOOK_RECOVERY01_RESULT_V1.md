# Stored-image recovery 01: technical qualification passed

Date: 2026-09-12 local. Completion: 2026-09-13 02:40:22 UTC.

## Result

The single authorized recovery completed 80/80 model calls and all 24 ordered
trajectory/arm checks. Native inference on frame 0, and dense/SpecPrune inference
on two-frame traces, produced byte-identical processed float32 commands with and
without diagnostic hooks. Selection metadata and prior-token indices matched.
Instrumented calls observed the required 32 layers. The independent CPU reconciler
verified complete counts, all comparisons, resource conditions and source/artifact
hashes. There were no simulator episodes or training updates.

Elapsed after launcher initialization: 139.73 seconds. Peak aggregate selected-
GPU memory: 15,679 MiB. Owned PID 1251217 exited successfully. GPU 0 returned to
6 MiB and 0% utilization. No automatic retry or heartbeat was used.

## What changed

The new offline helper validates and copies raw 128x128 demonstration images
without resizing or cropping them. The original policy preprocessing handles
current and historical images exactly once. The separate live-camera helper
still requires 224x224 evaluator output. No model weights, attention semantics,
selection rules, controller rules, task population or success criteria changed.

Before model loading, CPU checks validate actual image/state schemas, finite
state values and frames 0/1 from all eight consumed training trajectories. The
actual released preprocessing and existing current/history wrapper matched
pixel-for-pixel for all 16 frames, without mutating inputs. The worker compared
their hashes against the frozen CPU report before GPU/model setup.

96 targeted CPU tests passed with the GPU hidden, including 10 new recovery
tests for raw/live separation, preparation count, state/frame validation,
immutable copying, preflight ordering and complete hook-result reconciliation.
These establish the specified tested contracts, not absence of all future bugs.

## Provenance

- Protocol: `docs/CONTEMPORARY_HOOK_RECOVERY01.md`.
- Config: `configs/openvla/contemporary_hook_qualification_v2_recovery01.json`.
  SHA-256: `e585c3341921f86dd88c2f3af62622a909a7aacc8e230b299a6cf67e3d190b9e`.
- Worker: `scripts/run_contemporary_reference_recovery01.py`.
  SHA-256: `99e784f05d71820c285c4bb0481baa88b83aca61b30c57841dc871ca70f95d94`.
- CPU input report: `reports/CONTEMPORARY_RECOVERY01_REAL_INPUT_PREFLIGHT_V1.json`.
- Immutable run: `results/contemporary-hook-qualification-v02-recovery01`.
- Detailed checks and artifact hashes accompany this report in JSON.

Execution used only `ssh titan` and project-local files in `/home/ved/SAVR`, on
one selected GPU. No unrelated processes or allocations were inspected or changed.
All code and evidence are copied to `/Users/veddwivedi/Documents/VLA/SAVR` and
checked against the recorded hashes. No GitHub push, downloads or installations.
The failed v01 source/config/evidence remain byte-for-byte unchanged and failed.

## Meaning and next checkpoint

This resolves the observed stored-image integration error and supports removing
diagnostic hooks from measured primary queries without changing the tested
outputs. It does not establish task success, deployment speedup or learned-
corrector efficacy. The 139.73-second runtime is a qualification duration, not a
policy-latency result.

Stop here as authorized. The next experiment is the already designed 88-episode
contemporaneous dense/SpecPrune comparison, with native controls and separate
fixed-trace timing. Its executable configuration must authenticate this completed
qualification and the v2 analyzer before dispatch. That comparison was not launched
in this recovery. Corrector training and positive-method evaluation remain ahead.
