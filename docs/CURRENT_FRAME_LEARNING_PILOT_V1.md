# Current-frame learned correction: bounded execution plan

2026-09-14. User requests continued, efficient execution toward a positive paper.
This document supersedes the screen's stop-before-preparation boundary only.
All previous frozen experiments and their decisions remain unchanged.

## Question and ceiling

The selected 384-token baseline remains primary. It matched dense on all 40
consumed development conditions, so those conditions cannot by themselves
establish recovery of compression-induced terminal failures. Do not silently
switch to 256, select only failed tasks, or call offline loss a positive method.

The hypothesis is that a small downstream adapter uses all current visual patches
to correct the action produced by a physically shortened transformer sequence.
The backbone, preprocessing, original positions, released action head and eight-
action execution queue remain unchanged. This is current-frame compression,
not temporal caching. The existing research audit supplies motivation, not a
guarantee of novelty, closed-loop benefit, or publication.

## Integrated work sequence

1. Qualify real features, record round-trip, actual optimizer updates, saved-model
   reload and zero-initialized deployment in one bounded GPU job. Use the already
   exposed 16 demonstration frames, not a new benchmark campaign. No adapter from
   this engineering check becomes the research model.
2. Build one deterministic training sample manifest, at most 4,000 queries,
   balanced across the 40 tasks, using training trajectories only. Verify source
   hashes and exposure ledger before opening new observations. Split offline
   validation by trajectory, never adjacent frames. Keep locked data closed.
   Generate dense teacher actions and actual 384-token features from identical
   observations. Store numeric detached features with hashes. One model loaded,
   sequential inference; train the small module separately.
3. Fit the visual corrector and action-only ablation with identical labels,
   sample order, seed and optimizer budget. Freeze optimizer, steps, checkpoint
   rule and validation roles before fitting. Include a shuffled-label diagnostic.
   No tuning on closed-loop results. One fixed learner-state refinement may follow
   only if prospectively defined; it is not an unlimited recovery mechanism.
4. Compare dense, 384 alone, action-only correction and visual correction on a
   prospectively frozen, exposure-audited development population covering all
   four suites and more than state 0. Report the old 40 conditions separately.
   A new state is not an unseen task and not automatically a held-out test.
   Freeze exact state/seed identities and evaluation thresholds before launch.
   Include controlled complete-query timing with the corrector and all copies,
   all episode costs/failures, paired differences, and native controls.
5. Decide once. Incremental closed-loop benefit over 384 alone must survive its
   added latency and comparison with the action-only ablation. If no benefit is
   observed, say so; do not rename compression's existing gain as adapter credit.
   Paper-level validation then requires broader independent evaluation, adapter
   seeds, a dense-plus-corrector control and competitive comparisons. The previous
   SpecPrune results are development context, not a matched new-population control.

The executable manifest/config for each workload is frozen before that workload.
Routine code/tests/preparation need no additional micro-approvals. Stop on a
technical correctness failure, conflicting resources or a material scope change.
No automatic GPU retry, installs, downloads, or GitHub push. Single coordinated
GPU 0 only; ssh titan and /home/ved/SAVR exclusively on the server.

## Exact first integrated qualification

16 frames = frames 0 and 1 from the eight authenticated training trajectories in
the previous qualification. Per frame: qualified plain 384, qualified dense 512,
feature-extracting 384, zero-init action-only 384, zero-init visual 384. Exactly
80 backbone calls, no simulator episodes. Require byte-identical processed
commands for all three new 384 paths versus plain 384, unchanged observations,
finite exact-shape features, and lossless numeric serialization/reload.

Action features are the input to the released head's final 4096-to-7 linear layer,
shape 8x4096. They are not an arbitrary tail slice. Visual context is the first
512 current projected tokens, excluding proprio. Instruction context is the FP32
mean of instruction-only input embeddings, rounded once to BF16 both offline and
online. State is the actual normalized BF16 model input promoted to FP32.
Base/teacher normalized actions are promoted to FP32, with identical unnormalization
and gripper processing at deployment. Store BF16 bit patterns losslessly.

Train each existing default-size adapter for exactly 64 AdamW steps (learning rate
0.0001, weight decay 0, batch two, seed 7) on the first two qualification frames.
This is a same-sample implementation diagnostic, not generalization evidence.
Require finite updates, no backbone gradients, nonzero output gradients, and
finite strictly reduced same-batch L1 error; save final weights, reload strictly,
require identical predictions. No best-checkpoint selection or retry to pass.
If there is zero initial teacher discrepancy, report an uninformative fit check
and stop before claiming the optimization check passed.

Caps: 80 backbone calls, 128 optimizer updates total, 0 episodes, 30 minutes,
aggregate selected-GPU memory <23,552 MiB and artifacts <512 MiB. Backbone weights
are frozen and remain resident during tiny fitting, providing a conservative
memory check. Larger fitting will not need the backbone resident. Retain all 16
numeric records and both diagnostic adapters, labeled not-for-evaluation.

Unblind checks/losses only after completed immutable summary, exact counts and
hashes. Active monitoring: owned health, progress count, bytes, elapsed time and
aggregate GPU telemetry only. On failure preserve technical evidence and stop.
CPU analysis must independently reconcile all checks before bulk data work.

## Practical expectations

The previous 1–3 day estimate was a planning estimate, not measured throughput.
Use this run and subsequent extraction throughput to update the schedule. Keep
the number of scientific decisions small; a hard stop protects evidence, not an
excuse for repeated proxy-only phases. No assurance of a positive paper is made.
