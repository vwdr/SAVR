# Current-frame adapter fitting v1

2026-09-18 UTC. Implements step 3 of CURRENT_FRAME_LEARNING_PILOT_V1.md following
the verified 4,000-record collection. No further backbone calls or simulator
episodes. No change to the 384-token budget or the qualified adapter architecture.

## Scientific question and interpretation

Can a downstream action residual learn a useful mapping from current visual
patches and compressed-policy features to the dense policy's action? The training
target is the dense policy, not expert action or optimal behavior. The teacher
can itself fail. Reduced offline error does not demonstrate closed-loop recovery,
novelty, deployment speed or a positive paper result. The existing 39/40 screen
has a ceiling and will not establish the adapter's incremental benefit.

## Frozen design

Use exactly the completed collection and its immutable 3,200 fitting / 800
validation manifest. Trajectories are disjoint, with 80/20 observations per task
over 40 tasks; both pools originate from historical training data. No calibration
or locked observations. The validation pool is an offline development pool, not
a final independent test. Opaque file hashing is permitted in preflight; no
validation tensor decoding until all three final checkpoints are saved and sealed.

Three arms, in this fixed order: action-only adapter; full visual adapter; visual
adapter with a fixed within-task derangement of the 3,200 dense teacher chunks.
The third is an observation/label-alignment diagnostic, not a deployable candidate
or a guarantee against leakage. It preserves each task's action distribution and
may still learn common corrections. It has the same features, budget and initial
visual weights. No validation target can enter its permutation.

The qualified default architecture is unchanged: width 256, two blocks, eight
heads, eight action queries, 4,096-dimensional features, all 512 current projected
patches, and a zero-initialized seven-dimensional residual output. Copy identically
named and shaped shared parameters from the seeded visual initialization into
the action-only arm, including the common feed-forward blocks. The action-only
arm omits visual attention, so parameter counts differ; this is a modality
ablation, not a parameter-matched capacity experiment.

All arms: seed 7; AdamW, learning rate 1e-4, betas (0.9,0.999), epsilon 1e-8,
weight decay 0; FP32 weights/optimizer; gradient norm clipping at 1.0; batch 16;
exactly 10 epochs and 2,000 updates per arm (6,000 total). No scheduler, AMP,
early stopping or hyperparameter search. Each epoch includes every fitting
observation exactly once, ordered by a fixed sample-ID SHA-256 schedule that is
identical across arms. Optimize mean L1 over all 8x7 normalized action components.
No gripper reweighting, residual clipping, cache reconstruction or action routing.

Use final checkpoints only. Save initialization identities, final state dictionaries
and metadata; require finite gradients/parameters and exact save/reload predictions
on a fitting batch. Seal all three checkpoint hashes in checkpoints_frozen.json
before decoding any validation features/labels. Then compute all predictions on
the same 800 validation observations once, without any later optimizer update.
Record per-observation L1, gripper-sign disagreement and full normalized predictions
for later audit. Gripper-sign disagreement here is a teacher-agreement diagnostic,
not robot task success. Report task-balanced mean L1 (equivalent to observation
mean because task counts match), each suite's mean and all arms, including control.

## Predeclared next-step rule

This is modest engineering triage, not a statistical efficacy test: permit a
separately frozen closed-loop development evaluation only if visual mean L1 is
strictly lower than the uncorrected 384-token output and the shuffled-label control.
If the dense/384 discrepancy is exactly zero, the offline test is uninformative.
Always report comparison to the action-only arm; beating it offline is not required
to measure closed-loop behavior, but lack of benefit weakens the visual mechanism.
No numerical margin is invented after observing results. No significance or
generalization claim follows from strict ordering. If triage fails, stop and report;
do not silently extend epochs, tune, switch to 256 or retry to obtain a pass.

Before any robot run freeze its populations, success/latency criteria and compute
budget. Include all current feature extraction/copies/correction overhead. No
learner-state refinement is authorized by this fitting run; any such extension
needs a separate prospective protocol. Single-seed training is only a pilot.

## Engineering safeguards

Authenticate collection summary, analysis, all records and frozen source identities
before training. Each decoded record must match its digest and approved sample.
Enforce fitting/validation roles in the reader, not only in an outer loop. Cache
at most 32 fitting records in host memory; zero data-loader subprocesses; read only
generated features, never raw demonstration actions. Fit one small model at a time;
the 7B backbone is not loaded. Use the original Torch 2.2.0+cu118 environment.

Single coordinated GPU 0, exact UUID; fresh idle check before allocation. Disable
TF32, Flash and memory-efficient attention for adapter training; use deterministic
math attention and CUBLAS_WORKSPACE_CONFIG=:4096:8. These training-only backend
settings do not modify the released backbone. Enforce finite values, model/device
and dtype checks; retain technical traceback on failure. Math attention avoids
unsupported deterministic backward kernels. Exact numerical reproducibility across
different hardware/software is not claimed.

Caps: eight hours; aggregate GPU memory below 23,552 MiB; own peak allocated GPU
memory below 4,096 MiB; output below 2 GiB; preserve at least 10 GiB free disk.
No downloads, installs, other GPU, process interference or automatic retry.
Checkpoint selection, source code and sample/schedule manifests are frozen before
launch. Run CPU synthetic training/serialization/role tests before freezing.
No research labels are used to tune the optimizer before this freeze.
After freezing, CPU preflight additionally decodes one real 16-record fitting
batch to check batching and shapes; no values/losses are printed, no optimization
is performed and no validation record is decoded. All settings remain unchanged.

While active inspect only owned health, update count, artifact bytes, elapsed
time and aggregate GPU-0 telemetry. Do not inspect losses, predictions or validation
outcomes. After completed immutable summary, verify 6,000 updates, 3 checkpoints,
800 validation rows, zero backbone calls/episodes, checkpoint-before-validation
ordering and every hash. Run CPU-only reconciliation, sync evidence/status, report
the offline triage honestly, and stop before closed-loop evaluation.
