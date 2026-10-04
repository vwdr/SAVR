# Contemporaneous-reference development evaluation

Date: 2026-09-12. User approved this prospective methodological revision.
Scope: same-checkpoint SpecPrune comparator evaluation plus native repeatability
controls. Prepare the corrector/data pipeline alongside this work. This document
and its schedule are not sufficient to launch: the new worker, analysis code,
source hashes and preflight must be completed before dispatch.

## 1. What changes, and what does not

The historical forty-task dense result remains 39/40. The two stopped integration
runs remain stopped under their original gates. Do not edit their configurations,
summaries, source or raw evidence, or retrospectively mark them passed.

The targeted trace observed zero native/bridge action differences on twenty
matched inputs and byte-identical processed commands. Separate rollouts still
diverged after eight identical query/command prefixes; both succeeded at 79
steps rather than the earlier 78. This supports separating conditional inference
equivalence from independent-rollout repeatability. It does not prove the precise
source of rollout variation, or conditional equivalence on every possible input.

Prospectively, independent episodes are NOT required to reproduce historical
step counts or observation hashes. Their task success, length and query count
are measured outcomes. Native-versus-dense-bridge comparisons on the SAME
observation still require the frozen 1e-6 head/action tolerance and byte-identical
float32, gripper-processed command chunks. No tolerance is selected after results.

This is not accepting a failed check under new wording. It is a new evaluation
with a new configuration, contemporaneous reference data and explicit limits.

## 2. Arms and fixed development population

- D: qualified same-checkpoint dense bridge, all current visual tokens, original
  bidirectional SDPA, original head/readout, float32 execution boundary. No
  controller-induced replanning. The direct decoder avoids unused logits/history.
- S: same-checkpoint SpecPrune-OFT adaptation, pinned selection/controller rules,
  current camera history, original head/readout, same float32 execution boundary.
  Include selection and all replanning costs. Do not alter its preset after data.
- N: original released native inference with the authenticated baseline's
  float32 execution conversion. This is a repeatability/conditional-parity control,
  not the deliberately slower primary timing denominator.

Use exactly the forty previously consumed task/state-0/seed-7 conditions from
`configs/openvla/original_baseline_40task_v1.json`, including its known failed
Goal task. Same original simulator initialization, settling actions, horizons,
image processing, action chunks, gripper transform, source/runtime/checkpoint.
No held-out data or new task/state selection. These are development results.

Primary comparison: one D and one S episode per condition, 80 episodes total.
Within each suite, the sorted frozen condition order is preserved; alternate
D/S versus S/D execution, with five of each order among its ten pairs. Vary the
initial order by suite index. The attached schedule specifies every episode.
Reset model/controller/history state independently for every episode and arm.

Native controls: the first frozen condition in each suite, run once immediately
before and once immediately after that suite's ten primary pairs. Eight N
episodes, making 88 scheduled episodes overall. On every N policy query, run D
as a shadow on the same prepared observation, compare head/action/command values,
but execute only N. The shadow cannot change the simulator or N's action queue.
These diagnostic episodes include instrumentation and are excluded from primary
latency estimates BY DESIGN, not selected for exclusion after outcomes.

N-control condition IDs:

- Spatial: `d78f949af14d367e4a3057d13b211cc12f9ea7cbe00b6401c9bc5e1f74787bb9`.
- Object: `48ea27cca0ebf01549167a8190b268521d8512e50550c3819072cae2761b2536`.
- Goal: `f99df5534771ab0c6a8401f7cbdbbf9c32f326d5962c63ed3bbddac870c2d3cf`.
- Long: `6f1ef16636c45c3452c9912959342edd802ed51ade9ff45bf077d4e3692f539c`.

## 3. Distinct technical and scientific decisions

Technical stop conditions remain strict: nonfinite outputs; incorrect tensor or
action shapes; changed source/checkpoint/runtime identity; dropped nonvisual
tokens; wrong camera order; incomplete/reset-leaking state; changed horizons;
incorrect episode/call accounting; same-observation N/D command mismatch; or
resource-cap violation. Preserve evidence and stop without automatic retry.
Do not turn exceptions into task failures or missing rows into exclusions.

Scientific outcomes do not trigger early data-dependent stopping. Finish the
fixed population without inspecting partial success/latency aggregates. A
completed episode failing its task is a valid failure record, not an excuse
to rerun it. After completion:

- Report all native-control success, step, query and observation/command-trace
  variation. Two repeats per suite provide a diagnostic, NOT a precise estimate
  of success probability or an adequate calibration of simulator randomness.
- If before/after N success differs, or D and the N controls disagree on the
  sentinel task, flag that suite as repeatability-limited. Retain its results;
  do not exclude the suite or average away the disagreement. Do not advance a
  claimed reliability-preserving improvement on that suite without a separately
  prespecified replicated evaluation.
- A markedly weaker contemporaneous D result is a baseline concern, not evidence
  that S improved. Report the new D result alongside 39/40 historically. Do not
  substitute the historical denominator to make an arm look better. The final
  interpretation must explain any observed degradation before further claims.
- S may be better, similar or worse. A poor comparator result does not license
  deliberate weakening, tuning on these outcomes, or presenting its port as an
  exact paper reproduction. Its technical fidelity remains independently audited.

No exact-step tolerance or artificial minimum-success gate is introduced to
manufacture an acceptance. This phase characterizes the comparator; it cannot
by itself establish the proposed corrector's efficacy.

## 4. Cost measurements and their limits

Primary episodes record complete policy-query wall time, action count, replans,
discarded actions, simulator/controller/history overhead and total episode wall
time. Include unsuccessful episodes. Use D's equally optimized direct wrapper
as the primary denominator. Never compare a stripped S wrapper against unused
logit/history costs in N and attribute that difference to pruning.

Keep heavy boundary hooks and shadow queries in the separate N controls. Before
launch, CPU and fixed-input checks must establish that disabling audit hooks
does not change actions or the selected path. All production-required validation,
camera preparation, transfer/synchronization and selection costs remain timed.

Add a bounded matched-input timing module using frames 0 and 1 from the eight
already authenticated training trajectories. Each two-query trace starts with
reset state; second-query selection uses the prior frame/index state. Freeze the
coarse controller profile for this timing module. This measures a specified short
trace, NOT the full deployment mix of precise-mode changes and replanning.

Warm-up: two complete two-query traces per arm on the first frozen trajectory,
eight calls total, excluded by their predefined warm-up label. Then four rounds
of all eight two-query traces in each arm: 128 measured calls. Alternate arm
order by trace/round index, balanced within each round. The identical image,
state and language sequences are supplied to D and S. Reset state between traces
and rounds. Include actual history-image preprocessing and selection/attention
work. Synchronize CUDA immediately around the timed complete query and store
all individual times. Teacher/shadow inference is not part of this module.

Report first-query and second-query results separately, plus the combined mean,
median and p95. Compare query costs within matched inputs/rounds, not unmatched
success-only episode distributions. Do not use this coarse-profile timing alone
to claim deployment acceleration. The episode accounting above is required,
particularly if S changes query frequency. A broader timing corpus is required
for a paper-level result and the trained-method evaluation.

## 5. Analysis and honest uncertainty

Primary unit: the paired task/state condition, not a policy query. Publish the
40 paired D/S success outcomes, discordant counts, suite-specific counts and
task lengths/costs. Native repeats are controls, not eight more independent
benchmark tasks. Do not pool their successes into the 40-condition denominator.

Report paired success differences with a task-level resampling interval stratified
by suite, alongside the raw paired table. Its interpretation is descriptive for
this development task population and seed, not a guarantee about unseen tasks,
training seeds or simulator realizations. Do not report hundreds of timed calls
as hundreds of independent task-success observations. Use trace-level resampling
for timing summaries and disclose the tiny eight-trajectory corpus.

Separate technical qualification, development suitability and confirmatory
method evidence in the report. No significance or noninferiority claim from
40 conditions and two sentinel repeats. Final paper evaluation needs a frozen
candidate, exposure-audited conditions, repeated training seeds and sample-size
planning based on development variability.

## 6. Stay on the route to the actual learned method

Corrector feature/trainer implementation continues while this evaluation runs.
The new module has passed CPU structural and synthetic optimizer checks, but
not actual-data training or GPU timing. Do not wait for another literature audit
or a perfect comparator result before preparing its data pipeline.

The next compression screen uses the already specified 384/256 current-token
settings, actual hard-compressed features and a separately frozen schedule.
Do not fold those unqualified arms into this comparator run. Triage targets from
the research audit remain at least 10% complete-query headroom and roughly no
more than 15 percentage points of success loss before correction, evaluated
as resource-allocation criteria rather than proofs of learnability.

The first trained pilot compares compression alone, action-only correction and
fresh-visual correction against contemporaneous dense and the existing-method
reference. Same teacher inputs and fitting budget; at most 4,000 initial training
queries. Require closed-loop improvement and remaining speed benefit, not only
offline action-loss reduction. At most one prospectively specified learner-state
refinement. A negative pilot ends or revises the method version transparently.

## 7. Launch safeguards and work boundary

No dispatch from this document. The attached schedule is a design artifact and
has no executable worker identity yet. Before launch, freeze the new worker and
analyzer hashes, exact input/state hashes, 88 episode slots, 136 timing calls,
seed/order, source identities, telemetry selection and all stop conditions.
Validate malformed schedules/counts and audit-hook neutrality on CPU first.

Planned upper limits: one selected GPU; 6,000 total model calls (including native
shadows and timing warm-up), 88 episodes, six hours, aggregate GPU memory strictly
below 23,552 MiB, 512 MiB output artifacts. These are caps, not promises that a
pathological replanning policy fits. An exceeded cap stops fail-closed with
partial evidence, not an invented success aggregate. Data/feature-generation
storage is separate and not authorized by this comparator output budget.

While active, monitor only owned process health, counts, bytes, elapsed time and
aggregate selected-GPU telemetry. Reconcile outcomes only after immutable worker
completion. Preserve all slots, warm-ups and technical failures. No resumption,
automatic retry, unplanned model/data download, shared-process inspection,
system installation or access outside `/home/ved/SAVR` on TITAN. Sync reviewed
evidence/status to `/Users/veddwivedi/Documents/VLA/SAVR`. No blanket Git push.

Routine implementation/preflight work remains covered by the user's phase-level
authorization. Stop for resource conflicts, failed correctness checks or a
material methodological change. Do not ask approval after every routine edit.

### Bounded hook-neutrality preflight (implementation detail)

Before the 88-episode dispatch, run a separate fixed-input qualification on the
same eight consumed training trajectories. For each trajectory, compare native
inference with/without audit hooks on frame 0, and compare D and S with/without
hooks on complete frame-0/frame-1 traces. Reset selection state between traces.
Require byte-identical processed commands, identical selection metadata/prior
indices and 32 observed layers. This is 80 calls, 24 arm/trajectory checks, zero
simulator episodes and no training. Caps: 1,800 seconds, aggregate GPU memory
strictly below 23,552 MiB, output below 256 MiB. Stop without retry on failure.
It has a separate versioned configuration and result directory. Its calls do
not enter the 136-call evaluation timing denominator or the 88-episode sample.
The primary evaluation must authenticate its completed summary and same worker
source hash. No historical failed run is reclassified by this new preflight.
