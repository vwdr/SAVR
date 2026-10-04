# Original-runtime dense baseline: fixed forty-task development check

Prepared: 2026-09-09; pre-launch verification: 2026-09-10.
User authorization: “Okay go for it,” following the proposed
broader original-baseline evaluation. Single GPU selection remains delegated.

## Purpose and scope

Establish task coverage for the unmodified released OpenVLA-OFT policy before
choosing another acceleration method. The preceding attention-only diagnostic
gave 7/8 original and 5/8 causal-control successes. That small diagnostic neither
qualifies the full baseline nor validates caching. This phase has no cache,
adapter, training, speedup claim, or newly protected test-set consumption.

## Frozen design

- Original project-local OpenVLA-OFT runtime, checkpoint and action head used
  by the completed diagnostic; authenticate their bytes before launch.
- Four existing LIBERO suites: Spatial, Object, Goal and Long, ten tasks each.
- Exactly one episode per task, initial-state ID 0 and seed 7, selected from
  previously consumed headroom_stage1 conditions. Sort names within each suite,
  use fixed suite order, and resolve task names against the actual benchmark.
  There must be exactly ten unique tasks per suite, matching its full inventory.
- Exactly 40 episodes; original attention only during closed-loop evaluation.
  Retain ten settling steps, eight-action chunks, released preprocessing and
  gripper conversion, and horizons 220/280/300/520 steps.
- Repeat the already-tested 32-call offline reference/restoration check on
  eight consumed training observations before simulation. Its causal calls are
  reference diagnostics only, not another closed-loop causal-control arm.
- Preserve the tested attention observer and episode function. No installed
  dependency or checkpoint edits. Disable loader metadata writes and TensorFlow
  GPU allocation as in the successful pilot.

## Verification and technical protections

CPU tests must cover the existing attention/episode contracts, full-population
selection, rejection of duplicate/missing/wrong-seed conditions, and analyzer
rejection of incomplete, tampered or inconsistent records. Authenticate the
worker, analyzer, tests, protocol, original inputs and model files in the frozen
configuration. Test the original runtime with CUDA hidden; use preflight-only
mode to hash inputs without loading a model. Recheck hashes before launch.

GPU 0: GPU-bb2451d6-2989-a112-5c18-8892943710e4. Aggregate memory must be at most
1,024 MiB and utilization at most 5% immediately before loading. If unavailable,
stop without displacing work. One model on one GPU; PyTorch allocation capped
at 23,000 MiB and aggregate memory strictly below 23,552 MiB.

Hard ceilings: 40 episodes, 32 offline calls, 1,692 total model calls
(32 + 10 × (28+35+38+65)), two hours after preflight, 512 MiB run artifacts.
No retry, installation, download, process interference or automatic extra run.
All server work and temporary/cache files remain inside /home/ved/SAVR.
The elapsed-time ceiling is not an expected duration or latency measurement.

## Blinded monitoring and completion

While active, inspect only owned process health, progress-record count, artifact
bytes, elapsed time and aggregate selected-GPU telemetry. Do not read progress
contents, terminal outcomes, reference summaries or partial success aggregates.
Keep episode outcomes in memory until all 40 complete and integrity/resource
checks pass. Write a complete worker summary with hashes; never count technical
exceptions as task failures.

On early stop inspect only technical_stop.json and technical_traceback.log,
preserve evidence, report and stop without retry. On completion reconcile exact
counts, schedule, hashes, finite parity values, native attention contract,
action-queue accounting, task inventory and resource ceilings before reporting.
Retain each initial-state array hash for subsequent reproducibility checks.

## Analysis and decision

Report success counts overall and per suite, all failures and their executed
steps. Compare outcomes on the eight repeated condition IDs with the completed
pilot only after completion; report any disagreement without tuning or rerunning.
Review the previously common Goal failure in that context. Terminal records
alone cannot identify its physical cause, so do not invent a failure mechanism.

One fixed state per task and one seed give coverage, not a precise benchmark
estimate or independent generalization evidence. Do not compare these 40
episodes with historical 120-episode percentages as matched populations.
Do not pool repeated pilot conditions as independent evidence.

No post-hoc success threshold certifies the baseline. Technical completion and
scientific performance are separate. Poor task performance requires diagnosis,
not selection of easier tasks. Strong coverage supports planning a published
comparator, but cannot establish a positive acceleration result.

Stop after this phase and its report. No comparator, compression, adapter or
cache experiment is authorized here. Sync non-cache evidence and status to
/Users/veddwivedi/Documents/VLA/SAVR. Preserve historical artifacts and unrelated
local changes. No manuscript/poster work or GitHub push in this phase.
