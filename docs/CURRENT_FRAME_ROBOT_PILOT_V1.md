# Current-frame correction: matched closed-loop development pilot v1

2026-09-19. User authorized continuation after the completed adapter-fit pilot.
This document freezes the next development comparison before observing its
rollouts. The executable configuration additionally authenticates code, trained
weights, initial states and schedules before launch. No training or retuning.

## Question and fixed policies

Does learned correction improve terminal task success beyond fixed current-frame
compression while retaining useful total-query speed? Compare four policies:
dense (512 visual tokens); compression alone (384); compression plus action-only
correction; compression plus current-visual correction. Use exactly the three-arm
fit's final action-only and visual weights. The shuffled-label control is not a
deployment candidate. Do not change weights, token counts or action execution.

The full two-camera encoder still runs in every policy. The visual adapter reads
all 512 current projected patches, whereas only 384 enter the transformer.
Corrected query timing includes feature capture, checks, copies, adapter execution,
unnormalization and the existing redundant base-action transfer in the qualified
wrapper. Do not remove costs after seeing timing. Eight-action chunks, original
preprocessing/normalization, original positional indices and released action head
are preserved. No stale visual cache, action routing or adaptive execution queue.

## Population and exposure

Primary population: all 40 tasks over the four LIBERO suites, initial states 1,
2 and 3, seed 7: 120 paired conditions, four policies each, 480 primary episodes.
Use the existing frozen simulator-population ledger: states 1/2 from
headroom_stage1, state 3 from headroom_extension. Select by identity, never by
previous failure. Some conditions were exposed in earlier investigations; this
is explicitly a development comparison, not a held-out or novel-task test.
States 6 onward and final-confirmation populations remain unused.

The old state-0 40-condition compression screen remains historical context and
is reported separately; do not pool its successes with this new comparison.
Within each suite sort tasks by name then states 1/2/3. Rotate the four-arm order
by global condition index to balance order. Reset each episode with the same
seed/state. Pairing controls starting conditions, not future observations.

Eight additional native dense controls: before/after each suite, using the first
task's old state 0. At every native-control query run a shadow 512 query on the
identical observation and require released-head parity and identical processed
commands. Only native actions are executed; exclude shadow overhead from primary
timing. Report independent native trajectory variation, not just success agreement.
Total 488 episodes. Hash selected initial arrays before GPU launch and before use.

## Integrated technical qualification before episodes

Use the already exposed 16 qualification observations (eight trajectories, frames
0/1), not new validation labels. Per observation run six paths: native, 512,
plain 384, zero-initialized visual correction, trained action-only, trained visual.
Exactly 96 backbone calls. Require native/512 readout errors <=1e-6 and identical
processed commands; zero correction must match plain 384 commands exactly.
The trained wrappers must match direct adapter execution on the authenticated
stored qualification features and the same normalization key, with identical
processed command bytes. Compare all extracted feature tensors byte-for-byte
with the qualified stored records. Require finite 8x7 outputs, no input mutation,
exact query accounting, eval mode and no model gradients.

This is the deployed batch-one path, not a claim that offline batch-16 floating
point predictions must be bitwise identical. Any qualification error stops this
run before robot episodes. Preserve evidence; no automatic repair/retry.

## Timing

Before episodes, run a balanced controlled timing block over those eight
two-frame trajectories: four measured rounds for every arm, plus two warmup
traces per arm. Exactly 272 calls (16 warmup, 256 measured). Rotate arm order by
trajectory and round. Synchronize CUDA before/after each complete query and
include history capture and output conversion. These fixed-input measurements
exclude simulator observation resizing because inputs are already prepared;
label them controlled policy-query times, not complete episode latency.

Also record in every primary episode the full query interval including evaluator
observation preparation, all policy/corrector work and output queue preparation;
record simulator, controller, non-query preparation and whole-episode time.
Report all episodes, including failures, without success-only timing selection.
Episode timing follows different trajectories across policies and is descriptive,
not a causal same-input comparison. Do not equate faster query time with faster
successful task completion. Report mean, median and p95, counts and all costs.

## Prospective interpretation

All requirements below define an initial development signal, not a publication
claim or statistically established noninferiority:

- Technical qualification, exact evidence accounting and all resource checks pass.
- Dense succeeds on at least 108/120 (90%) primary conditions; native controls
  succeed on all eight episodes and before/after success agrees. Otherwise report
  baseline concerns and do not declare a positive development signal.
- Visual correction has at least three more successes than compression alone
  (a 2.5-percentage-point net improvement) and at least one more than action-only.
- Visual correction loses no more than two successes relative to dense overall,
  and no suite loses more than two relative to compression alone.
- Visual corrected controlled mean query time is at least 10% below dense.

These practical pilot margins are fixed now; they are not a power calculation.
With 120 conditions and clustering within tasks, uncertainty may remain large.
If compression is at ceiling so the improvement margin cannot be met, report
insufficient room to establish this benefit, not proof that every adapter fails.
If visual ties or loses to action-only, report that visual input has not established
an incremental terminal-success benefit. Do not credit the old compression gain
to the adapter, lower the thresholds, select favorable subsets or try another
checkpoint. One completed run, one decision.

Report per-suite successes, paired win/loss counts, and descriptive paired 95%
bootstrap intervals using 10,000 draws, seed 7, task clusters stratified by suite
(all three states kept together). Bootstrap controlled timing by the eight paired
trajectory clusters. Degenerate intervals near ceiling are not zero uncertainty.
Report all pairwise candidate comparisons; no confirmatory significance claim.
Paper-level confirmation would still require independent populations, training
seeds, dense-plus-adapter and contemporary method controls. No such extension or
learner-state collection is authorized by this run.

## Resource, monitoring and stop boundaries

One coordinated GPU 0, original Torch 2.2.0+cu118 / Transformers 4.40.1 runtime,
one frozen backbone plus small FP32 adapters. Fresh idle check and exact GPU UUID.
At most 22,000 backbone calls, 488 episodes, 16 hours, <23,552 MiB aggregate GPU
memory, <512 MiB artifacts, >=10 GiB remaining disk. The worst-horizon schedule
requires at most 20,952 calls including shadows, timing and qualification.
No videos, downloads, installs, other GPUs, training or unrelated server access.

Run CPU schedule/reconciliation tests, authenticated-source/data/initial-state
preflight, then the integrated qualification before any robot episode. All remote
operations through ssh titan and inside /home/ved/SAVR. No automatic retries.
While active inspect only owned health, count-only progress/terminal record
counts, artifact sizes, elapsed time and aggregate GPU-0 telemetry. Do not inspect
successes, timing values or partial aggregate outcomes. Unblind only after complete
immutable summary, exact 488 episodes, 272 timing and 16 qualification records.
On an early stop inspect technical evidence only, preserve and report, without retry.
After completion run frozen CPU analysis, independently reconcile hashes and
counts, sync evidence/status locally, report the outcome and retire monitoring.
