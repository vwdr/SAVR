# Fixed current-frame compression screen, version 1

2026-09-13. Authorized by the user after completed contemporaneous-reference v2.
This protocol implements the previously proposed 384/256-token development screen.
No training, new downloads, hidden evaluation data or automatic retry is authorized
by this phase. Historical source, configuration and evidence remain unchanged.

## Scientific question and exact intervention

Does modest current-frame compression provide at least 10% complete-query timing
headroom while losing no more than 15 percentage points of observed task success?
This is a substrate triage rule, not a claim that action correction will work.

Encode both current camera views with the unchanged released vision encoder and
projector. Of their 512 projected visual tokens, keep 384 or 256 using the existing
fixed per-camera 2x2-stratified selector. Drop unselected visual positions before
decoder layer zero, not after an unspecified layer. Preserve all nonvisual tokens,
camera order, original rotary positions, original bidirectional SDPA and the exact
released 56-position action readout. Never mask in place of actual sequence
shortening. No stale features, cache, adaptive selector, routing or replanning.
The 512-token arm uses exactly the same query implementation without deletion.

No novelty is claimed for uniform spatial deletion. The later research hypothesis
is that a small current-visual correction module can recover errors left by this
compression. This screen cannot establish that hypothesis.

## 1. Technical integration, before task outcomes

CPU tests use the original project Torch/Transformers runtime with CUDA hidden.
Check all-token exact equality to the qualified direct decoder, all 32 actual
layer sequence lengths, original position IDs, all protected/readout positions,
variable prompts, no cache or persistence, bidirectional attention and rejected
invalid inputs. Retain and rerun the existing preparation/action-loop tests.

Then one frozen real-checkpoint qualification on all 16 already-consumed stored
frames (eight train trajectories, frames 0 and 1). Per frame: one audited qualified
dense reference, then plain and audited calls of new 512/384/256 arms. Total 112
calls, zero simulator episodes. Compare reference versus new 512 head/normalized/
raw action values within 1e-6 and processed float32 commands byte-for-byte. Require
exact commands with/without hooks for each new arm, finite outputs, unchanged
input pixels/state and correct 32-layer compressed shapes/positions. Do not
impose equality between compressed and dense actions. That difference is intended.
Caps: 112 calls, 1,800 seconds, <23,552 MiB aggregate GPU memory, <256 MiB artifacts.
An incomplete/failed qualification stops the version without retry.

Before GPU allocation, verify source/checkpoint identities and the actual raw
128x128 stored input schema against the completed preprocessing audit. Keep
the separate live-image path unchanged. Check the selected GPU is freshly idle.
No partial action-error/outcome inspection while the qualification is active.

## 2. Frozen robot and timing screen after qualification

The full screen requires its own executable configuration and analyzer after
qualification passes. Its exact design is fixed here, not chosen after success.

Use all 40 consumed task/state-0/seed-7 conditions, including known failures, with
fresh 512/384/256 episodes per condition. Rotate the three arms by global condition
index (512,384,256), (384,256,512), (256,512,384). Retain the frozen suite/task order.
Eight native controls bracket the ten triples in each suite, on the same sentinel
conditions as contemporaneous-reference v2. Each native query has a same-input
512 shadow. Thus 120 primary episodes plus eight controls = 128 episodes.
No historical result substitutes for the current dense denominator.

Use the original simulator, initial-state hashes, settling actions, horizons,
eight-action chunk and float32/gripper processing. All three primary arms use
the same measured loop with controller disabled. Hook diagnostics are restricted
to the qualification and native controls, not hidden from timed production calls.

Matched timing: same eight consumed two-frame traces, four measured rounds,
three arms, balanced rotating order by trace/round. Two warm-up traces per arm
on the first trajectory (12 calls), then 192 measured calls = 204 total.
Every query is current-only; resetting trace state does not create a cache.
Include preprocessing, transfers, selection/mapping, decoder/head, validation,
float32 conversion and CUDA synchronization. Report first/second query separately
as well as combined mean/median/p95. Include all primary episode costs/failures.

Conservative screen caps: 7,000 model calls including shadows and timing; 128
episodes; six hours; <23,552 MiB aggregate memory; <512 MiB artifacts. The maximum
episode call count under the frozen horizons is 5,644 including native shadows
(3 x 10 x (28+35+38+65) + 4 x (28+35+38+65)); with 204 timing calls this is 5,848,
below the cap. No capacity assumed beyond one
coordinated GPU. Do not retry because a scientific result is poor.

## 3. Analysis and selection, fixed before reading outcomes

Unblind only after completed immutable summary and exact counts/hash verification.
Reconcile every condition, call, action queue, timing slot and resource limit.
Use the same native success-discordance flag as reference v2, report independent
trace variation, retain every suite. A flag or a current dense result below 39/40
requires review before selecting a training substrate; it is not comparator credit.

For each budget report paired success difference against current 512; all per-suite
counts, discordant tasks, episode costs and controlled-query time reduction.
Descriptive intervals: 10,000 task bootstrap draws stratified by suite, seed 7;
timing bootstrap clusters all paired rounds/frames by the eight trajectories.
No significance/noninferiority claim; no success-only timing or omitted failures.

A budget is eligible if observed success loss is at most six of 40 conditions
(15 percentage points), controlled mean complete-query saving is at least 10%,
and no correctness/resource/baseline concern blocks interpretation. If both are
eligible, choose 384 (less deletion). If neither is eligible, stop before training.
Do not lower these criteria or automatically add budgets based on the result.
Eligibility does not establish closed-loop correctability or a positive method.

## 4. Next boundary and monitoring

Record commands, frozen source/input/configuration hashes, checkpoint/runtime,
selection layout and complete evidence under new exclusive result directories.
Use only ssh titan and /home/ved/SAVR; no unrelated university access or changes.
Copy reviewed evidence/status to /Users/veddwivedi/Documents/VLA/SAVR.
While active inspect only owned health, counts, bytes, elapsed time and aggregate
selected-GPU telemetry. On technical stop inspect only technical evidence, preserve
and stop without retry. Do not push to GitHub or train at the end of the screen.

After eligible selection, separately freeze the actual-data feature extraction,
tiny-batch training/serialization/reload qualification, then the matched learned
pilot. Offline action loss alone never counts as the desired positive result.
