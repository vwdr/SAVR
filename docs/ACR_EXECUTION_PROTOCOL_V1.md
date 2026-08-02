# Asymmetric Camera Refresh Execution and Results Protocol

Status: FROZEN MASTER PLAN BEFORE ACR IMPLEMENTATION OR ACR OUTCOME COLLECTION

Protocol version: 1.0

Freeze date: 2026-08-02

Project: Asymmetric Camera Refresh for Efficient Multi-View VLA Inference

Parent proposal: `docs/ASYMMETRIC_CAMERA_REFRESH_PROPOSAL.md`

Negative-results archive: `docs/NEGATIVE_RESULTS_PAPER_ARCHIVE.md`

## 1. Purpose

This document converts the Asymmetric Camera Refresh (ACR) proposal into a
phase-gated program for obtaining publishable evidence. It is the authoritative
plan for ACR implementation, correctness testing, development, confirmation,
final evaluation, analysis, and manuscript decision-making.

The purpose is not to guarantee a positive paper. The purpose is to give the
proposed method a rigorous opportunity to succeed while preventing outcome
chasing, split leakage, silent scope changes, unsupported claims, and unsafe
use of the university server.

This document freezes:

- the claim boundary;
- the primary method family;
- development, confirmation, transfer, and final populations;
- candidate-generation and selection rules;
- correctness and scientific gates;
- primary and secondary metrics;
- statistical and resource rules;
- evidence, checkpoint, and stop requirements.

Numerical controller thresholds are intentionally not present at freeze time.
They will be generated once by the deterministic procedure in Section 10 from
the frozen development Full Refresh (FR) traces, written to a hashed
configuration, and frozen before the first ACR rollout.

Creating or approving this document does not itself authorize implementation,
GPU use, simulator rollouts, downloads, final-holdout access, or manuscript
changes. Each execution phase begins only after the user authorizes that phase.

## 2. Authority and status verification

Use this precedence order:

1. system, developer, and current explicit user instructions;
2. repository `AGENTS.md`;
3. this ACR protocol;
4. frozen phase-specific configuration and manifest files;
5. `PROJECT_STATUS.md`, `docs/DECISIONS.md`, reports, and immutable artifacts;
6. remembered conversation context.

At the start of every ACR phase:

1. read this protocol completely;
2. read `AGENTS.md`, `PROJECT_STATUS.md`, the ACR proposal, and the latest
   relevant report;
3. verify the actual local, GitHub, and TITAN revisions;
4. verify that the prior phase exit gate passed;
5. verify the exact authorized scope and resource cap;
6. record the phase as the only `IN_PROGRESS` phase;
7. stop if repository state, evidence, or authorization is ambiguous.

Plans never establish completion. Only direct records and reconciled evidence
establish completion.

## 3. Non-negotiable safety boundary

- The only permitted writable university-server path is `/home/ved/SAVR`.
- Do not inspect, modify, move, rename, delete, copy, or change permissions on
  unrelated university files or directories.
- Do not inspect or interfere with unrelated processes, jobs, users,
  containers, services, environments, mounts, or GPU allocations.
- Never use `sudo` or perform a system-wide installation.
- Never terminate, reprioritize, or attach to another process.
- Use at most one explicitly selected GPU and one model process.
- GPU selection may use only user-authorized aggregate device utilization and
  memory information; do not inspect process identities.
- Keep environments, caches, checkpoints, temporary files, logs, and results
  inside `/home/ved/SAVR`.
- Do not launch a GPU phase unless its time, storage, episode, and query limits
  are approved.
- Stop if an action could affect shared university work.

Every phase report must state whether anything outside `/home/ved/SAVR` was
modified. The expected answer is no.

## 4. Scientific claim boundary

### 4.1 Primary research question

Can a training-free multi-view VLA preserve task success while reducing visual
computation by always refreshing the dynamic wrist camera and selectively
reusing only the fixed scene-camera representation?

### 4.2 Primary method

The primary method is State-Aware Asymmetric Camera Refresh (SA-ACR):

- scene-camera projected patch tokens may be reused;
- wrist-camera projected patch tokens are recomputed on every query;
- the two blocks retain scene-first, wrist-second token order;
- current proprioception is appended on every query;
- the language model, action head, prompt, preprocessing, and weights remain
  unchanged;
- the scene gate uses scene-image change, accumulated normalized
  end-effector translation since the scene refresh, one fixed gripper-phase
  transition veto, and a finite cache horizon;
- invalid inputs or context mismatches fail toward scene refresh.

### 4.3 Co-primary hypotheses

Both hypotheses must pass for a positive method result.

**H1 — Success non-inferiority.** SA-ACR task success is non-inferior to
upstream two-view FR within an absolute margin of two percentage points on the
frozen primary final population, with no individual suite showing an observed
loss larger than five percentage points.

**H2 — Meaningful realized efficiency.** SA-ACR achieves all of:

- at least 40% scene-camera reuse;
- at least 15% measured visual-backbone-plus-projector CUDA-time reduction;
- at least 2% point reduction in synchronized end-to-end policy-query latency;
- a confidence bound excluding zero end-to-end latency improvement.

The efficiency thresholds are deliberately below the theoretical maximum but
high enough to prevent a negligible reuse rate from being described as a
positive efficiency result.

### 4.4 Secondary mechanism hypotheses

1. The scene camera supports substantially more reuse than the wrist camera.
2. Keeping the wrist view fresh avoids the major failure mode of whole-prefix
   SAVR.
3. The frozen state/transition signals improve safety or the success-compute
   frontier relative to scene-image-only ACR.
4. SA-ACR outperforms periodic scene refresh at matched scene-camera compute.
5. A frozen SA-ACR configuration transfers to goal-oriented and long-horizon
   task suites without retuning.

Failure of Hypothesis 3 removes a state-awareness contribution but does not
retroactively change the primary controller. The paper title and claims must
then emphasize ACR rather than the value of state signals.

### 4.5 Claims that remain prohibited

Do not claim:

- positive ACR performance before the final gates pass;
- universal safety or generalization beyond the evaluated model and tasks;
- learned routing, token-level caching, or VLA-wide acceleration;
- end-to-end speedup from call counts or theoretical FLOPs alone;
- superiority to an external method that was not compatibly executed;
- independence for a split that informed thresholds or method selection;
- novelty before the Phase A0 full-text and code audit passes.

## 5. Novelty gate and research positioning

Before ACR code is implemented, Phase A0 must compare the complete methods and
available code of at least:

- OpenVLA and OpenVLA-OFT;
- VLA-Cache;
- Learning to Accelerate VLAs through Adaptive Visual Token Caching;
- VLA-ADP;
- VLA-IAP;
- Selective Perception for Robot;
- multi-camera distillation and multi-camera view scaling;
- any newer work found by repeating the frozen search strategy at execution
  time.

The search log must record databases, exact queries, dates, URLs, versions,
and inclusion/exclusion decisions. Use primary papers and official code.

The intended distinction is:

- temporal reuse of a complete scene-camera token block rather than view
  removal or attenuation;
- a fresh wrist block on every query;
- both camera blocks remain present in their original positions;
- training-free, deterministic, fail-closed control rather than a learned
  router;
- reuse of camera encoder/projector computation rather than only language
  model KV tokens;
- direct closed-loop evaluation in a chunked OpenVLA-OFT policy.

**Novelty exit gate:** a comparison matrix reviewed before implementation must
show that no identified method already contains this complete contribution.
If the distinction is too narrow or already established, stop and reframe the
project before collecting ACR outcomes.

## 6. Frozen technical system

- Base policy: OpenVLA-OFT.
- Checkpoint: `openvla-7b-oft-libero-four-suite`.
- Checkpoint revision:
  `638918f3d1c2e43a39a8a20772bdb8b91835e4b7`.
- OpenVLA-OFT revision:
  `e4287e94541f459edc4feabc4e181f537cd569a8`.
- LIBERO revision: `8f1084e3132a39270c3a13ebe37270a43ece2a01`.
- Environment: existing project-local `envs/openvla-oft`.
- Images: scene `full_image` followed by `wrist_image`.
- Proprioception: current eight-dimensional LIBERO state.
- Action head: continuous L1 regression.
- Action chunk: eight actions.
- FiLM: disabled.
- Diffusion: disabled.
- Center crop: enabled.
- Model/checkpoint/prompt/preprocessing: unchanged.
- Primary cache boundary: projected scene-camera patch embeddings before
  current proprioception.

The installed combined checkpoint and pinned LIBERO source support Spatial,
Object, Goal, and LIBERO-10. No new model, checkpoint, or standard LIBERO
dataset download is part of the core protocol.

## 7. Evidence-reuse and population ledger

### 7.1 Historical evidence only

LIBERO-Spatial tasks `0-9`, initial-state IDs `0-9`, seed `0` have informed
SAVR design, calibration, diagnosis, and redesign. They may be used for
historical motivation and negative-result comparison but never as fresh ACR
confirmation or final evidence.

The SAVR3 population, Spatial states `3-9`, is also consumed. SAVR3 must not be
rerun or retuned.

### 7.2 Frozen ACR populations

| Role | Suite | Tasks | Initial-state IDs | Seed | Maximum episodes per policy | Use |
|---|---|---:|---:|---:|---:|---|
| Development | LIBERO-Object | 0-9 | 0-9 | 0 | 100 | FR traces, candidates, selection |
| Confirmation | LIBERO-Goal | 0-9 | 0-9 | 0 | 100 | One frozen primary confirmation |
| Transfer | LIBERO-10 | 0-9 | 0-9 | 0 | 100 | Long-horizon transfer, no tuning |
| Primary final | LIBERO-Spatial | 0-9 | 10-49 | 7 | 400 | Co-primary paper claims |
| Primary final | LIBERO-Object | 0-9 | 10-49 | 7 | 400 | Co-primary paper claims |
| Primary final | LIBERO-Goal | 0-9 | 10-49 | 7 | 400 | Co-primary paper claims |
| Primary final | LIBERO-10 | 0-9 | 10-49 | 7 | 400 | Co-primary paper claims |
| Reserve | All four suites | 0-9 | 10-49 | 17, 27 | 3,200 | Untouched unless separately justified |

The development population is subdivided before outcomes:

- Stage 1: Object states `0-2`, 30 episodes per candidate;
- Stage 2: Object states `3-9`, 70 episodes per advancing candidate.

### 7.3 Independence rules

- Candidate thresholds and selection may use only Object development data.
- Goal confirmation cannot modify thresholds, rules, margins, metrics, or
  candidate identity.
- LIBERO-10 transfer cannot modify the method or final analysis.
- States `10-49`, seed `7`, in all four suites cannot be inspected or executed
  before the final configuration, statistical plan, and semantic configuration
  hash are frozen.
- Seeds `17` and `27` remain reserve evidence. Do not use them merely because
  the primary outcome is disappointing.
- Repeated seeds on an identical saved initial state must not be treated as
  independent without a documented source of stochastic variation and a
  cluster-aware analysis.

## 8. Exact SA-ACR controller family

At policy query `t`, let the last scene refresh be `tau`. SA-ACR refreshes the
scene block if any condition is true:

1. the scene cache is empty, invalid, or context-incompatible;
2. scene local-image change from query `tau` exceeds
   `gamma_scene`;
3. normalized end-effector translation from query `tau` exceeds
   `gamma_pos`;
4. the fixed gripper-phase transition veto is active;
5. scene-cache age reaches `H_scene`;
6. the episode-local hard scene-reuse cap would be violated;
7. any required signal or metadata is invalid or non-finite.

The wrist block is always recomputed. Current proprioception and all downstream
model computation are always recomputed.

### 8.1 Scene-image score

1. Convert the current and reference scene images to `[0,1]`.
2. Produce the existing deterministic `32 x 32` representation.
3. Divide it into an `8 x 8` grid of `4 x 4` patches.
4. Compute mean absolute pixel difference per patch.
5. Define the scene score as the mean of the four largest patch scores.
6. Preserve all 64 patch scores and the top-four aggregate in development
   records.

No wrist score is used to approve scene reuse because the wrist view is always
fresh. No robot-region mask is part of Version 1.

### 8.2 Scene-relative translation score

Normalize end-effector position with the checkpoint proprioception `q01/q99`
statistics. Compute L2 distance between the current normalized position and
the position stored at the last actual scene refresh.

Orientation and gripper joint positions are logged but do not independently
veto scene reuse in Version 1.

### 8.3 Fixed transition veto

Use the existing validated gripper-command binarization rule. Refresh the
scene camera if the newest predicted action chunk contains both open and closed
commands or if its final binary gripper command differs from the preceding
chunk's final command. Insufficient action history during the first two policy
queries forces scene refresh.

Translation-direction reversal is logged for analysis but is not a Version 1
gate because it vetoed 412 SAVR3 queries and would risk recreating the
overconservative whole-prefix controller.

### 8.4 Cache and temporal semantics

- Episode/context start always refreshes both cameras.
- Minimum query index for scene reuse is `2` (zero-based).
- Consecutive scene reuse is permitted until `H_scene`.
- Scene age is measured in policy queries since the last scene refresh.
- Scene reference image, reference position, and transition context update
  only on an actual scene refresh.
- Task, instruction, episode, checkpoint, model configuration, dtype, device,
  patch count, or token-shape mismatch invalidates the scene cache.
- A recovered/interrupted run must restore exact controller state or restart
  the incomplete episode with a recorded new attempt; it must not continue
  from guessed cache state.

## 9. Required policies and comparisons

### 9.1 Primary oracle

**Upstream two-view FR:** the original two-camera path, refreshed on every
query. This is the primary success and end-to-end latency oracle.

### 9.2 Correctness-only oracle

**Camera-factorized FR:** separately execute the scene and wrist camera paths,
concatenate their projected tokens, and refresh both every query. This must
match upstream FR outputs and quantifies factorization overhead. It is not a
replacement for upstream FR in the headline latency comparison.

### 9.3 Primary method

**SA-ACR:** the frozen selected candidate from Section 10.

### 9.4 Required matched baselines after confirmation eligibility

- **Scene-Periodic ACR:** wrist always fresh; scene refreshes at a fixed
  interval selected to match SA-ACR's scene refresh rate within two absolute
  percentage points when feasible.
- **Scene-Visual ACR:** wrist always fresh; scene uses the same image score,
  horizon, hard cap, and cache behavior as SA-ACR but omits accumulated state
  and transition vetoes.
- **Historical SAVR3:** report the immutable prior result only; do not rerun it
  on consumed states or present it as a matched final baseline.

Wrist-only, scene-only, masks, pre-projector caching, learned routing, and
token-level caching are optional post-primary ablations. They cannot select or
redefine the primary method.

## 10. Deterministic candidate derivation and selection

### 10.1 Development FR trace

Run upstream FR once on all 100 Object development episodes. Each immutable
query record must contain enough compact information to replay scene caching:

- deterministic `32 x 32` scene representation;
- normalized end-effector position;
- gripper-phase data;
- query/environment-step indices;
- action hash and required transition summary;
- original component calls and synchronized timings.

The development FR run must achieve at least 90/100 success, at least 8/10
success for every task, 100 terminal records, and zero technical/accounting
failures. Otherwise Object is not a valid development population and the
protocol stops for a discrepancy review before any ACR rollout.

### 10.2 Frozen candidate templates

Create exactly three candidates:

| Candidate | Offline target scene reuse | Horizon | Hard online scene-reuse cap |
|---|---:|---:|---:|
| `acr-t25-h2-b30` | 25% | 2 | 30% |
| `acr-t50-h4-b55` | 50% | 4 | 55% |
| `acr-t70-h8-b75` | 70% | 8 | 75% |

For each template:

1. construct empirical adjacent-query distributions for the scene-image score
   and normalized translation score from the complete development FR trace;
2. use one shared quantile grid `q = 0.500, 0.505, ..., 0.995`;
3. set `gamma_scene = Q_scene(q)` and `gamma_pos = Q_pos(q)`;
4. replay the exact scene-cache, transition, horizon, warm-up, and prefix-cap
   semantics for every `q`;
5. choose the `q` whose aggregate replay reuse is closest to the template
   target without exceeding its hard cap;
6. break ties by lower replay reuse, then lower `q`;
7. write all three configurations, replay estimates, trace hash, code revision,
   and semantic SHA-256 before the first ACR outcome.

The derivation must be deterministic and byte-identical on two independent
executions. If no valid quantile exists for a template, retain it as
`DERIVATION_INELIGIBLE`; do not invent a replacement target.

### 10.3 Stage 1 safety screen

Run every derivation-eligible candidate once on Object states `0-2` (30 fixed
episodes each). A candidate advances only if:

- 30/30 terminal episodes reconcile;
- 30/30 episodes succeed;
- every task succeeds 3/3;
- aggregate scene reuse is at least 15%;
- wrist refresh count equals policy-query count;
- every scene reuse skips exactly one scene encoder/projector path;
- zero technical, cache, counter, timing, or schema errors occur.

If no candidate advances, stop. Do not tune thresholds, add candidates, relax
the gate, or consume Stage 2.

### 10.4 Stage 2 development selection

Run advancing candidates once on Object states `3-9` (70 episodes each).
Combine each candidate's fixed Stage 1 and Stage 2 records into its complete
100-episode development result.

A candidate is development-eligible only if:

- paired success is no more than two episodes below Object FR;
- no task loses more than one success relative to FR;
- aggregate scene reuse is at least 40%;
- measured visual CUDA time falls by at least 10%;
- every component/cache invariant passes;
- zero technical failures occur.

Select one candidate mechanically:

1. highest paired success difference versus FR;
2. among candidates within one percentage point, highest synchronized
   policy-query latency reduction;
3. then highest visual CUDA-time reduction;
4. then lowest horizon;
5. then lexicographic configuration ID.

Freeze the selected configuration and semantic hash. All other candidates
remain reported; none may be revisited after confirmation begins.

## 11. Metrics and records

### 11.1 Task outcomes

- aggregate success;
- paired success difference from FR;
- per-task and per-state success;
- horizon failures and technical failures separately;
- episode length and policy-query count;
- first action divergence from FR where trajectories remain comparable;
- failure timing relative to scene-cache age and scene reuse.

### 11.2 Camera-specific computation

- scene and wrist refresh counts;
- scene reuse rate;
- per-camera SigLIP and DINOv2 calls;
- per-camera projector calls;
- reproducible visual FLOPs proxy;
- per-camera and total visual CUDA time;
- gate, cache-copy, and concatenation overhead;
- synchronized total policy-query CUDA and wall time;
- complete episode wall time;
- peak allocated and reserved GPU memory;
- scene-cache memory.

### 11.3 Correctness and provenance

- token shapes, order, dtype, and device;
- scene reference and cache age;
- fresh wrist-image hash at every query;
- current proprioception hash at every query;
- downstream language/action-head calls;
- context identity and invalidation reason;
- checkpoint, source-tree, configuration, and record hashes;
- selected GPU ID and bounded aggregate before/after state;
- commands, timestamps, revisions, seeds, and host information.

Warm-up queries and timing exclusions must be predeclared and identical across
policies. Report both inclusive and steady-state timing.

## 12. Statistical analysis

### 12.1 Analysis unit and pairing

The primary unit is one episode. Methods are paired by suite, task,
initial-state ID, and seed. Query-level observations are nested within episodes
and cannot be treated as independent success samples.

### 12.2 Primary success analysis

For the final task-balanced four-suite states `10-49` pool, compute:

- `success_SA-ACR - success_FR`;
- a Newcombe score confidence interval for the paired risk difference;
- a suite/task-stratified paired bootstrap sensitivity interval with at least
  10,000 deterministic resamples;
- an exact paired-discordance sensitivity analysis;
- suite-specific paired differences and intervals;
- per-task paired differences.

Use one-sided alpha `0.025`. H1 passes only if the Newcombe lower confidence
bound is strictly greater than `-0.02` and no suite's observed paired
difference is below `-0.05`. The Newcombe implementation must be unit-tested
against an independent statistical implementation before final outcomes.

### 12.3 Primary efficiency analysis

For each paired episode, aggregate steady-state synchronized policy-query
latency and visual CUDA time. Use a suite/task-stratified paired bootstrap with
at least 10,000 resamples.

H2 passes only if:

- aggregate scene reuse is at least 40%;
- visual CUDA-time point reduction is at least 15%;
- end-to-end policy-query latency point reduction is at least 2%;
- the one-sided 97.5% lower confidence bound for latency reduction is above
  zero;
- all call-count and timing invariants reconcile.

Both H1 and H2 are co-primary and both must pass. Passing only one is a mixed
or negative result, not a positive method result.

### 12.4 Power and sample-size gate

After Goal confirmation and LIBERO-10 transfer, but before opening any final
population, use only the 200 Goal/10 paired episodes to calculate the sample
size needed for 90% power at one-sided alpha `0.025` and margin `0.02`.
Development Object outcomes are excluded from this power estimate because
they selected the configuration.

Let `pD` be the paired discordance rate. Use the larger of the observed pooled
Goal/10 discordance, its two-sided 95% Wilson upper bound, and `0.01`. The
frozen normal-approximation planning calculation is:

`n = ceil(pD * (z_0.975 + z_0.90)^2 / 0.02^2)`.

- Maximum primary final sample: 1,600 distinct suite/task/state episodes.
- If required sample size is at most 1,600, round upward to the smallest
  multiple of 40 and freeze that exact `N` by assigning the same number of
  ascending state IDs to each of the 40 suite/task strata.
- If required sample exceeds 1,600, stop and request a design decision. Do not
  silently treat reserve seeds as independent or enlarge the margin.
- The sample-size script, inputs, result, and hash must be committed before
  any final outcome.

### 12.5 Secondary analyses

- Compare SA-ACR with Scene-Visual ACR and Scene-Periodic ACR at matched
  scene-camera compute.
- Report Goal and LIBERO-10 transfer descriptively with confidence intervals.
- Test state/transition signal value only as a secondary mechanism claim.
- Correct families of secondary hypothesis tests using Holm's procedure.
- Label failure-mode and task-phase analyses exploratory unless separately
  predeclared.

## 13. Phase plan and gates

Only one phase may be `IN_PROGRESS`.

### Phase A0 — Novelty, source, and split audit

Actions:

- complete the literature/code comparison in Section 5;
- repeat the literature search to the execution date;
- verify pinned per-camera factorization in source;
- verify all four suite mappings and initial-state counts;
- audit every historical result to confirm split consumption;
- confirm no ACR outcome exists;
- prepare the exact implementation design and novelty matrix.

Exit gate:

- novelty distinction survives;
- source boundary is feasible;
- population ledger is verified;
- no outcome, GPU, simulator, or final-holdout access occurred.

### Phase A1 — Protocol acceptance and resource freeze

Actions:

- review this protocol for scientific, statistical, and resource adequacy;
- calculate bounded phase estimates from prior measured runtimes;
- freeze schemas, run IDs, recovery rules, and artifact limits;
- obtain user authorization for Phase A2 only.

Exit gate:

- protocol and resources are accepted;
- local, GitHub, and TITAN `main` agree;
- manuscript and historical evidence remain unchanged.

### Phase A2 — Implementation and CPU verification

Actions:

- add separate camera-factorized adapter and ACR controller modules;
- preserve SAVR1-3 code and tests;
- implement compact replayable FR records;
- implement component-level camera accounting and timing;
- add deterministic candidate derivation and statistical-analysis tests;
- pass all existing and new unit/static/build tests.

Minimum CPU tests:

- token order and block shape;
- cache identity and invalidation;
- always-fresh wrist truth table;
- every scene refresh condition and exact boundary;
- gripper transition and warm-up;
- horizon and hard prefix-cap semantics;
- interrupted-run recovery;
- candidate derivation determinism and tie-breaking;
- immutable record validation;
- unchanged SAVR1-3 behavior.

Exit gate:

- all tests pass;
- no upstream source is modified;
- no model/GPU/simulator outcome was used;
- code is reviewed and synchronized.

### Phase A3 — Bounded real-model correctness

Use at most 16 real-model policy queries and zero rollout episodes.

Required proofs:

1. camera-factorized FR projected tokens are bitwise identical to upstream FR;
2. camera-factorized FR actions are bitwise identical to upstream FR;
3. changing only one camera affects only its pre-language-model token block;
4. scene reuse performs zero scene tower/projector work;
5. every query performs exactly one fresh wrist path;
6. scene reuse uses current proprioception and unchanged downstream execution;
7. context/shape/device/dtype errors fail closed;
8. checkpoint metadata and upstream trees restore exactly.

Any parity failure stops before rollout. A numerical tolerance cannot be added
without a root-cause report, protocol amendment, and new user approval.

Exit gate:

- all correctness proofs pass within 16 queries;
- immutable records reconcile;
- no simulator outcome or protected split is touched.

### Phase A4 — Development FR and candidate freeze

Actions:

- run upstream FR once on the 100 Object development episodes;
- apply the FR feasibility gate in Section 10.1;
- derive exactly three candidates twice;
- freeze configurations and semantic hashes;
- stop before ACR rollout if derivation or feasibility fails.

Exit gate:

- 100 FR terminal records and all query traces reconcile;
- all feasibility thresholds pass;
- candidate derivations are byte-identical;
- no ACR outcome has yet been observed.

### Phase A5 — Staged ACR development

Actions:

- execute Stage 1 for all eligible templates;
- apply the 30/30 safety gate immediately;
- execute Stage 2 only for advancing candidates;
- apply the fixed development eligibility and selection rule;
- freeze exactly one primary SA-ACR configuration.

Exit gate:

- one configuration meets every development gate and is frozen; or
- apply the negative stop with no tuning or replacement candidate.

### Phase A6 — Independent Goal confirmation

Run upstream FR and the single frozen SA-ACR configuration once on the 100
Goal confirmation episodes. No other method runs until the primary
confirmation gate is applied.

Confirmation eligibility requires:

- 100/100 paired terminal records;
- SA-ACR success no more than two episodes below FR;
- no task loses more than one success relative to FR;
- scene reuse at least 40%;
- visual CUDA-time reduction at least 15%;
- end-to-end query-latency point reduction at least 2%;
- zero technical or invariant failures.

If any condition fails, stop before baselines, transfer, power planning, and
final evaluation. Do not retune on Goal.

### Phase A7 — Matched baselines and frozen transfer

This phase runs only after Phase A6 passes.

1. Derive Scene-Periodic ACR's interval mechanically from the Goal SA-ACR
   scene refresh rate without accessing final outcomes.
2. Run Scene-Periodic ACR and Scene-Visual ACR on the same 100 Goal episodes.
3. Run upstream FR and frozen SA-ACR on the 100 LIBERO-10 transfer episodes.
4. Run the frozen power calculation and determine final `N`.
5. Freeze the final configuration, analysis script, table shells, and semantic
   hashes.

Baseline or transfer failure does not license retuning. It changes the allowed
secondary claims and must remain visible.

Exit gate:

- all planned records reconcile;
- final `N <= 1,600` or execution stops for a design decision;
- the final analysis package is frozen before Spatial outcomes.

### Phase A8 — Primary final four-suite evaluation

This phase requires separate explicit user authorization after reviewing all
prior reports and the power result.

Run only:

- upstream FR; and
- the single frozen SA-ACR configuration

on the frozen task-balanced selection from states `10-49`, seed `7`, across
LIBERO-Spatial, Object, Goal, and LIBERO-10, up to the frozen `N`. Do not
inspect seed `17` or `27` outcomes.

Run order must be deterministically interleaved or counterbalanced by paired
episode to reduce temporal hardware bias. Policy identity, order, and recovery
rules must be frozen in the manifest.

After every batch of 50 paired episodes, perform only technical reconciliation
and resource checks. Do not compute success or efficiency hypothesis results
until the complete frozen sample finishes, except for a predeclared technical
or safety stop.

Exit gate:

- all paired records reconcile;
- both co-primary analyses execute once from the frozen script;
- the result is classified positive, mixed, or negative mechanically.

### Phase A9 — Robustness, evidence package, and manuscript decision

Core actions after final classification:

- publish complete tables, plots, confidence intervals, and failure analysis;
- reconcile compute estimates with measured speed;
- update the claims-evidence matrix;
- preserve both SAVR negative and ACR evidence;
- decide the manuscript route from Section 17.

Conditional robustness extension:

- audit LIBERO-PRO or the most current primary robustness benchmark;
- estimate any download/environment cost separately;
- run a bounded FR feasibility pilot only after approval;
- compare ACR only if FR has adequate success for a meaningful test;
- label robustness evidence secondary and never use it to replace a failed
  primary result.

No manuscript result, discussion, limitation, title, or abstract change occurs
until the complete evidence package is reviewed and the user authorizes
manuscript editing.

## 14. Resource limits

These are maximum planning caps, not automatic authorization.

| Phase | GPU/simulator cap | Time cap | New artifact cap | Downloads |
|---|---|---:|---:|---|
| A0-A2 | none | CPU only | 512 MiB | none |
| A3 | 1 GPU, 16 queries, 0 episodes | 1 hour | 512 MiB | none |
| A4 | 1 GPU, 100 FR episodes | 8 GPU-hours | 1 GiB | none |
| A5 | 1 GPU, at most 300 ACR episodes | 24 GPU-hours | 2 GiB | none |
| A6 | 1 GPU, 200 paired-policy episodes | 16 GPU-hours | 2 GiB | none |
| A7 | 1 GPU, at most 400 episodes | 32 GPU-hours | 3 GiB | none |
| A8 | 1 GPU, at most 3,200 policy episodes | 200 GPU-hours | 12 GiB | none |
| A9 robustness | separately proposed | separately proposed | separately proposed | separate approval |

The episode cap counts every attempt, including technical failures. A failed
episode is never silently rerun. A technical recovery must preserve the first
attempt and follow a predeclared recovery rule.

## 15. Checkpoint and on-task controls

At every phase boundary, answer:

1. What was authorized?
2. What actually ran?
3. Which files/configurations/revisions were used?
4. Which populations have now been consumed?
5. Did all records and counters reconcile?
6. Did any scientific or technical gate fail?
7. Were any rules changed after seeing outcomes?
8. Is the next population still untouched?
9. Did actual GPU time/storage remain within cap?
10. Was anything outside `/home/ved/SAVR` modified?

Techniques that keep the project on task:

- one active phase at a time;
- semantic configuration hashes before outcomes;
- immutable attempt, query, episode, progress, and manifest records;
- deterministic derivation and analysis scripts with unit tests;
- table shells and statistical code frozen before final outcomes;
- consumed-population ledger updated after every run;
- mechanical advancement and stop rules;
- no failed-episode reruns or selective exclusions;
- no threshold, margin, metric, or claim changes after confirmation begins;
- clean upstream-tree and checkpoint-hash checks;
- local/GitHub/TITAN revision reconciliation at every accepted checkpoint;
- explicit separation of facts, hypotheses, interpretations, and decisions.

## 16. Stop, recovery, and amendment rules

Stop immediately for:

- novelty gate failure;
- factorized-FR parity failure;
- unexpected upstream or checkpoint modification;
- cache/token/component invariant failure;
- an unsafe or unavailable shared-resource situation;
- phase resource-cap exhaustion;
- split leakage or premature final-outcome access;
- no candidate passing Stage 1 or Stage 2;
- confirmation failure;
- required final sample size above 1,600;
- any attempt to relax a gate after an unfavorable outcome.

A protocol amendment is allowed only before the affected population is
opened. It must:

1. preserve all prior evidence;
2. explain the factual cause;
3. identify whether the change is technical or scientific;
4. define newly consumed populations;
5. receive user approval;
6. create a new version rather than silently modifying Version 1;
7. prohibit relabeling prior exploratory outcomes as confirmatory.

## 17. Result classification and manuscript route

### Positive ACR paper

Allowed only if both final H1 and H2 pass with zero unresolved integrity
failure. Secondary results determine whether state-aware and transfer claims
are included.

The manuscript may then be rewritten around:

- camera-factorized temporal caching;
- always-fresh wrist perception;
- success non-inferiority;
- measured camera-specific and end-to-end efficiency;
- comparison with periodic and visual-only camera refresh;
- transparent motivation from failed whole-prefix SAVR.

### Mixed result

Examples:

- success passes but realized latency does not;
- efficiency passes but success non-inferiority does not;
- the pooled primary test passes but one suite has a materially worse result;
- ACR works but state signals add no measurable value.

The paper must narrow its title and claims. A mixed outcome cannot be called a
fully positive method result.

### Negative ACR result

If either co-primary gate fails, preserve the ACR result with the existing
SAVR archive. The project may pivot to a negative-results paper about the
safety-efficiency limits of whole-prefix and sensor-granular visual reuse.

No additional local redesign may use the final population. A new method would
require a new question, protocol, and genuinely untouched evidence source.

## 18. Planned artifacts

Before each relevant phase, create and hash:

- `docs/ACR_NOVELTY_AUDIT.md`;
- `docs/ACR_IMPLEMENTATION_DESIGN.md`;
- `docs/ACR_SPLIT_AND_RESOURCE_AUDIT.md`;
- `configs/acr/development_fr.json`;
- `configs/acr/candidates.json`;
- `configs/acr/confirmation.json`;
- `configs/acr/transfer.json`;
- `configs/acr/final.json`;
- `schemas/acr_query.schema.json`;
- `schemas/acr_episode.schema.json`;
- phase-specific analysis scripts and tests;
- immutable result directories and run registry entries;
- one report per phase;
- final claims-evidence table and manuscript migration checklist.

Large raw results, caches, images, checkpoints, and environments remain
uncommitted. Compact configurations, schemas, reports, aggregate tables,
plots, and hashes are committed.

## 19. Primary sources at protocol freeze

- OpenVLA: https://arxiv.org/abs/2406.09246
- OpenVLA-OFT: https://arxiv.org/abs/2502.19645
- LIBERO: https://arxiv.org/abs/2306.03310
- VLA-Cache: https://arxiv.org/abs/2502.02175
- Adaptive Visual Token Caching:
  https://arxiv.org/abs/2602.00686
- VLA-ADP: https://arxiv.org/abs/2509.22093
- VLA-IAP: https://arxiv.org/abs/2603.22991
- Selective Perception for Robot:
  https://arxiv.org/abs/2602.15543
- Multi-Camera View Distillation:
  https://arxiv.org/abs/2303.07026
- Multi-Camera View Scaling:
  https://arxiv.org/abs/2604.00557
- LIBERO-PRO: https://arxiv.org/abs/2510.03827
- Newcombe, *Improved Confidence Intervals for the Difference Between Binomial
  Proportions Based on Paired Data*:
  https://doi.org/10.1002/(SICI)1097-0258(19981130)17:22%3C2635::AID-SIM954%3E3.0.CO;2-C
- Official OpenVLA-OFT code: https://github.com/moojink/openvla-oft
- Official LIBERO code:
  https://github.com/Lifelong-Robot-Learning/LIBERO

## 20. Current acceptance checkpoint

This protocol is ready for review. No ACR code, numerical threshold, ACR
rollout, simulator outcome, GPU workload, final-holdout access, or manuscript
change is authorized or claimed by its creation.

The next executable step after user acceptance is **Phase A0 only**: complete
the novelty/source/split audit and produce the exact implementation design.
