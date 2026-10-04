# Learning qualification: stopped at the GPU availability guard

2026-09-14. Owned launcher PID 1446582 exited before result-directory creation
or model initialization. This is a resource-availability stop, not a failed
action-correction experiment or a demonstrated integration fault.

## Completed preparation

- New actual-feature extractor and corrected deployment wrapper, preserving the
  existing qualified fixed-compression implementation unchanged.
- 33 targeted CPU tests passed in the original TITAN runtime with CUDA hidden.
- Full source/checkpoint/16-frame input preflight passed, without GPU use.
- Three additional deterministic sampling tests passed (36 CPU tests total).
  CPU-only 4,000-frame manifest preparation completed and independently verified
  locally: 3,200 fitting observations from 1,054 trajectories, 800 validation
  observations from 271 disjoint trajectories, 80/20 observations per task.
  All 40 source hashes and actual camera/state schemas were checked. No model
  features or labels have yet been generated from these observations.
  Manifest `configs/openvla/current_frame_training_inputs_v1.json`, SHA-256
  `2fd13ced3e5a9a2b810eb299a5c1667f8c1de3ee2c5487a4a0683e61847d9ae8`.
  The role split is within the original training trajectories: lowest 28 split
  hashes per task for fitting, remaining seven for validation. Samples ranked
  by a fixed salted hash, without looking at actions or outcomes. Calibration
  and locked observations remain closed. This validation is not a final test.

## What stopped

The worker independently checks the selected GPU immediately before allocation.
Its permitted idle condition is <=1,024 MiB aggregate memory and <=5% utilization,
with the exact coordinated UUID. The condition was not met. The launch log ends
with `ValueError: selected GPU not idle`. The failed guard did not print the exact
launch-time telemetry, so no launch-time values are inferred.

A subsequent permitted aggregate-only reading was 1,227 MiB and 96% utilization
on GPU 0, UUID `GPU-bb2451d6-2989-a112-5c18-8892943710e4`. This confirms that the
GPU was busy at that later check. No other user's process or allocation was
inspected, identified, terminated, or changed. No other GPU was selected.

No backbone call, optimizer step, simulator episode, or new scientific outcome
was produced. `results/current-frame-learning-qualification-v01` does not exist.
There is no completed summary to analyze. No automatic retry was launched.

## Preserved evidence and continuation boundary

- Config: `configs/openvla/current_frame_learning_qualification_v1.json`.
  SHA-256 `b82b4ec88792d9253a7a90bbf7f4bdad90e83b2dcf384d5cdb91df1a84a3c8b5`.
- Launch log: `reports/current-frame-learning-qualification-v01-terminal.log`.
  SHA-256 `9b4fa77680b0293a69a553c5add5b870669183c3e24d87e01f11feab4fe8e1c1`.
- New plan: `docs/CURRENT_FRAME_LEARNING_PILOT_V1.md`.

Retain the config and log. Any authorized resource-recovery dispatch must use a
new launch-log identity, record that the first attempt performed zero calls,
and freshly check coordinated availability. Do not lower the idle threshold,
overwrite evidence, or bypass the GPU guard. No scientific gates need changing.
The 384-token screen remains promising; learned-adapter efficacy remains untested.

All TITAN activity used ssh titan and project-local paths. No unrelated university
files, processes, environments, allocations or configuration were modified.
