# Cache Action Correction Execution Protocol V1

**Protocol version:** 1.0  
**Date created:** 2026-08-30  
**Status:** SUPERSEDED by `docs/CACHE_ACTION_CORRECTION_EXECUTION_PROTOCOL_V2.md`; do not execute  
**Method working name:** Cache Action Correction (CAC)  
**Parent research assessment:**
`docs/LOW_DIMENSIONAL_CACHE_ACTION_CORRECTION_FEASIBILITY_2026-08-30.md`

## 1. Purpose and authorization boundary

This protocol defines a phase-gated study of whether action-relevant error from
recursively mixed-age visual K/V reuse can be repaired by a small action-space
corrector without reconstructing the complete fresh cache.

This document freezes the scientific question, permitted method family,
development sequence, evidence rules, early-stop conditions, and positive-
result criteria. It is designed to give the method a fair opportunity to
succeed while preventing outcome chasing, repeated redesign, split leakage,
silent metric changes, and unnecessary GPU work.

Creating or approving this document does **not** authorize implementation,
server access, GPU work, feature collection, training, simulator rollouts,
locked-test access, repository publication, or manuscript modification.

Each phase requires separate explicit user approval. Approval of one phase
does not authorize the next phase.

## 2. Authority and phase-start procedure

Use this precedence order:

1. system, developer, and current explicit user instructions;
2. repository `AGENTS.md`;
3. this protocol;
4. the frozen phase configuration and manifest;
5. `PROJECT_STATUS.md`, `docs/DECISIONS.md`, reports, and immutable evidence;
6. remembered conversation context.

At the start of every phase:

1. read this protocol completely;
2. read `AGENTS.md`, `PROJECT_STATUS.md`, the parent feasibility assessment,
   and the latest phase report;
3. verify the actual local and TITAN repository revisions;
4. verify that the preceding phase exit gate passed;
5. verify the exact authorized phase, resource limit, and evidence population;
6. record only that phase as `IN_PROGRESS`;
7. verify that no later-phase outcome or locked population is accessible; and
8. stop if authorization, repository state, or evidence provenance is
   ambiguous.

Plans and conversation summaries do not establish completion. Only reconciled,
hashed evidence establishes completion.

## 3. Critical TITAN boundary

- Connect only through the existing `ssh titan` host.
- The only permitted writable server path is `/home/ved/SAVR`.
- Do not inspect, modify, move, rename, delete, copy, or change permissions on
  unrelated university files or directories.
- Do not inspect or interfere with unrelated processes, jobs, users,
  containers, services, environments, mounts, or GPU allocations.
- Never use `sudo` or perform system-wide installation or configuration.
- Never terminate, reprioritize, attach to, or alter another process.
- Use at most one explicitly selected GPU and one VLA model process.
- Select a GPU only using user-authorized aggregate utilization and memory
  information. Record the GPU ID and telemetry without inspecting process
  identities.
- Store environments, logs, compact features, checkpoints, and results only
  under `/home/ved/SAVR`.
- Do not download a new model, checkpoint, dataset, or large dependency without
  separate approval and a written storage estimate.
- Stop if an action could affect shared university work.

Every phase report must state whether anything outside `/home/ved/SAVR` was
modified. The expected answer is no.

## 4. Scientific question and claim boundary

### 4.1 Primary research question

Can the action-relevant effect of high-dimensional, recursively mixed-age
visual K/V reuse in the pinned OpenVLA-OFT LIBERO stack be repaired by a small
frozen-backbone action corrector using already-available fresh and cached
representations, while retaining meaningful measured acceleration?

### 4.2 Central hypothesis

Selective K/V reuse may create a high-dimensional internal representation
error, but the part of that error that changes the served robot action is
substantially lower-dimensional and predictable from:

- current projected visual tiles;
- tile-aligned fresh-minus-source projected-feature deltas;
- cached-path action hidden states and base actions;
- current proprioception; and
- exact camera/tile/layer source provenance.

Therefore, reconstructing every fresh K/V tensor may be unnecessary for
preserving task behavior.

### 4.3 Positive paper claim

If every primary gate passes, the supported claim is:

> On the pinned two-camera OpenVLA-OFT checkpoint and the evaluated LIBERO
> population, a small action-space corrector recovers task reliability from a
> fixed recursively mixed-age visual K/V reuse profile without reconstructing
> the full fresh cache, while preserving a statistically supported
> complete-cycle latency reduction.

### 4.4 Prohibited claims

Do not claim:

- that low offline action error proves closed-loop reliability;
- formal robot safety, collision avoidance, or constraint satisfaction;
- universal VLA, robot, task, or checkpoint generalization;
- higher success than dense OpenVLA-OFT unless independently demonstrated;
- end-to-end speedup from FLOPs, call counts, or CUDA kernels alone;
- full K/V reconstruction or cache-state correction—the method corrects the
  action output;
- novelty based merely on using a residual head, frozen backbone, fresh
  vision, DAgger, or caching individually;
- superiority to a paper whose official implementation was not comparably
  executed; or
- a positive CAC result before independent closed-loop and physical-efficiency
  gates pass.

### 4.5 Feasibility judgment and connection to prior failures

This is a feasible **study**, not a guaranteed successful method. It is worth a
bounded test because the existing stack already exposes every proposed
deployment input, a compact adapter fits the one-GPU constraint, and supervised
dense-minus-cached labels can be generated without expert teleoperation.

The design directly responds to the project evidence:

- SAVR/ACR established that uncontrolled stale visual state can destroy
  closed-loop success;
- BRACE and PAIR established that a physically useful reuse substrate exists;
- PAIR showed that handcrafted low-dimensional harm proxies do not reliably
  identify safe reuse; and
- CAC therefore learns the action-relevant correction from current visual
  evidence, corrupted-path context, and exact provenance instead of trying to
  rank risk from handcrafted summaries.

The unresolved risk is whether the action error is predictable and whether an
offline correction remains stable under its own induced state distribution.
C2 is the inexpensive predictability test; C5 and, only if justified, C6 test
the closed-loop distribution risk before independent confirmation.

## 5. Research positioning and novelty audit

Before implementation, Phase C0 must repeat a primary-source search and compare
the exact mechanism, inputs, training distribution, and evaluation of at least:

- OpenVLA and OpenVLA-OFT;
- VLA-Cache;
- LAC and AC2-VLA;
- Action-JND and Gated VLA-Cache;
- Latent Bridge;
- CloudEdgeVLA;
- A2C2 / Leave No Observation Behind;
- Action ControlNet;
- ViTaR; and
- any newer cache-repair, VLA residual-adaptation, action-distillation, or
  frozen-feature-control paper found at execution time.

The literature record must distinguish peer-reviewed papers from preprints and
reported results from locally reproduced results.

The intended original contribution is limited to:

1. same-query correction of selective token/layer mixed-age K/V corruption;
2. prediction of only the action-relevant effect rather than full K/V state;
3. reuse of the current projected OpenVLA-OFT tokens already computed by the
   accelerated stack;
4. explicit conditioning on exact recursive cache provenance; and
5. controlled comparison of demonstration-only versus one-round on-policy
   dense-teacher training.

If the C0 audit finds a prior method with all five elements on a substantially
identical architecture and problem, stop for user/advisor review before code.

## 6. Inherited evidence and fixed system identity

CAC inherits the following completed findings without retesting them during
method selection:

- Full Refresh is the correctness and reliability oracle.
- The pinned stack uses two current images and proprioception.
- The projected vision/proprioception tensor has shape `1 x 513 x 4096`.
- The cached downstream path exposes 56 action hidden states of width 4096.
- The frozen L1 action head produces an `8 x 7` normalized action chunk.
- Eight actions are executed before the next observation query.
- Reuse provenance is resolved by camera, 4x4 tile, onset layer, and source
  query.
- BRACE measured a 12.23% complete-cycle time reduction for a valid cached
  configuration.
- PAIR P3R selected physical profile `D62_BAL_PT1` and estimated a 17.05%
  conservative net saving.
- PAIR P4B found that compact proxy features did not independently validate
  reliable harm ranking.
- P4B contained 776 valid intervention records, but it did not retain raw
  56-dimensional residual vectors and contained no designated gripper-
  transition contracts.
- The authenticated P1 data index contains 2,000 trajectories across 40 tasks:
  1,400 train, 320 calibration, and 280 locked-test trajectories, with 41,447
  eligible query points.

Phase C0 must record exact hashes for:

- repository revision;
- OpenVLA-OFT source revision;
- LIBERO revision and dataset identity;
- checkpoint and normalization statistics;
- P1 split manifest;
- P3R profile and physical-timing evidence;
- Python/environment lock; and
- this protocol.

Any identity mismatch stops execution until resolved in a new report.

## 7. Frozen method family

### 7.1 Deployment cache policy

The primary reuse substrate is the physically validated
`D62_BAL_PT1` profile. Phase C0 must authenticate its exact token/layer schedule
and hash; it may not retune the profile using CAC outcomes.

The initial CAC study uses:

- one fixed reuse profile;
- a maximum source age of four model-query intervals;
- a mandatory dense reset at or before the frozen age boundary;
- current proprioception on every query;
- the original two-camera token ordering;
- the original prompt, preprocessing, normalization, action head, and action
  queue; and
- no learned cache selector, risk router, or fallback policy.

If C1 physical reconstruction shows that `D62_BAL_PT1` plus the mandatory reset
does not preserve at least 8% gross complete-cycle latency headroom before the
adapter, stop. A different profile requires a new protocol version.

### 7.2 CAC inputs

At model query $q$:

- $V_q$: current projected visual tokens;
- $V_q^{src}$: projected source tokens aligned to the actual reused tile
  sources;
- $H_q^C$: cached-path action hidden states;
- $A_q^C$: cached normalized `8 x 7` action chunk;
- $s_q$: current normalized proprioception; and
- $p_q$: camera, tile, onset layer, source query, and source age.

Future observations, future proprioception, expert outcomes, terminal success,
rewards, dense hidden states unavailable at deployment, and later action chunks
are prohibited inputs.

The dense action $A_q^D$ is a training label only. It is never a CAC deployment
input.

### 7.3 Compact representation

The full candidate uses:

1. sixteen 4x4 spatial tiles per camera, giving 32 current visual tokens;
2. aligned current-minus-source tile deltas;
3. four onset-layer source ages and camera identity for each tile;
4. 56 cached action hidden states reshaped into eight groups of seven and
   pooled into eight action-step tokens;
5. the cached base `8 x 7` action; and
6. current proprioception.

The learned width is at most 256. The learned corrector contains at most two
cross-attention/transformer blocks and at most 8 million trainable parameters.

### 7.4 Output

CAC predicts a bounded normalized residual:

$$
\widehat A_q = A_q^C + b\odot\tanh(R_\phi(X_q)),
$$

where $X_q$ contains only the permitted inputs and $b$ is a frozen
per-dimension bound derived from training-split residual quantiles before model
training.

The output projection is zero-initialized. Dense/all-fresh queries bypass CAC
completely, guaranteeing exact dense identity by construction.

The six motion coordinates use a continuous residual head. The gripper
coordinate uses a separately reported bounded head and threshold-disagreement
metric. The original downstream action processing remains unchanged.

### 7.5 Training target

The primary label is

$$
\Delta A_q^* = A_q^D-A_q^C.
$$

Dense OpenVLA-OFT is the primary teacher because CAC aims to preserve the dense
policy under caching. Expert demonstration actions are a secondary diagnostic
and may not be combined as an equally weighted conflicting target.

The frozen loss contains:

- dimension-normalized Smooth L1 for six motion coordinates;
- a separately weighted gripper-coordinate loss;
- fixed action-step weights declared in C0, with the first action no lower than
  any later action because all eight are executed open-loop; and
- no terminal outcome, reward, success predictor, or post-hoc CVaR tuning.

Loss weights and correction bounds are derived from training data only and
frozen before calibration access.

## 8. Allowed model-selection screen

Only Phase C2 may compare model classes. The permitted sequence is:

1. ridge regression;
2. a two-layer MLP with at most 2 million parameters; and
3. one compact cross-attention model satisfying Section 7.3.

Models are evaluated in this order. The smallest model passing every C2 gate is
selected. A larger model may be selected only if all smaller models fail and it
passes. No fourth model family, alternate backbone tap, recurrent module,
diffusion head, reinforcement-learning objective, learned selector, or external
encoder is permitted under V1.

After C2, architecture, initialization, optimizer, schedule, features, loss,
bounds, and seeds freeze. C3 may train the frozen architecture at the
predeclared larger data scale but may not redesign it.

## 9. Data governance and split rules

### 9.1 Demonstration data

The existing P1 split remains authoritative:

- `train`: 1,400 trajectories, available for C2/C3 fitting and internal
  validation;
- `calibration`: 320 trajectories, opened once in C3 after the architecture is
  frozen; and
- `locked_test`: 280 trajectories, unavailable until C7.

Phase C0 deterministically divides the 1,400 training trajectories within each
task into:

- 28 trajectories per task in `adapter_fit` (80%); and
- 7 trajectories per task in `adapter_validation` (20%).

The split unit is a complete trajectory. Queries, tiles, horizons, and action
dimensions from one trajectory may never cross subsets.

### 9.2 C2 screen population

C2 uses exactly 3,600 non-control cache contracts:

- 30 contracts per task at horizon 1;
- 30 contracts per task at horizon 2; and
- 30 contracts per task at horizon 4.

For each task and horizon, exactly 24 contracts come from `adapter_fit` and six
come from `adapter_validation`, yielding 2,880 fit and 720 validation primary
contracts. Models may fit only on the 2,880 fit contracts. C2 gates use only
the 720 validation contracts.

Within each task, at least eight contracts must be centered on or immediately
precede a demonstrated gripper transition where sufficient transitions exist.
If a task contains fewer eligible transitions, use every eligible transition
and report the shortfall before training. Where at least two validation
transitions exist, at least two of the eight transition contracts must be in
`adapter_validation`.

The schedule is selected by a frozen SHA-256 ordering over dataset revision,
task, trajectory, query, horizon, and C0 seed. Gripper-transition status may be
used only for the prespecified stratum; residual magnitude and dense/cached
error may not influence selection.

C2 also includes exactly:

- 240 all-fresh controls: two per task and horizon, selected by the same frozen
  ordering; and
- 360 exact repeat executions drawn from the 3,600 primary contracts and
  balanced by suite, horizon, and transition stratum.

Thus C2 contains 3,600 primary cache contracts and 600 prespecified auxiliary
executions, for exactly 4,200 scheduled contract executions. C0 must freeze the
corresponding number of dense, cached, source-establishment, and control model
calls before C1. It may not reinterpret a contract after outcomes are visible.

### 9.3 C3 training population

After C2 selects the architecture, C3 may use all eligible `adapter_fit`
queries up to a hard cap of 26,000 cache contracts. Sampling is balanced as far
as data permit across suite, task, horizons 1/2/4, and transition/non-transition
strata using frozen identity-based ordering.

`adapter_validation` is used for checkpoint selection and early stopping.
Calibration trajectories remain closed until the final R0 checkpoint is
immutable.

### 9.4 Rollout populations

Phase C0 generates disjoint, outcome-blind simulator-seed manifests:

- `development_r0`: 6 seeds per task, exactly 240 paired conditions for C5;
- `development_r1_holdout`: 4 seeds per task, exactly 160 paired conditions
  reserved for C6 and never used for R0 selection or R1 fitting; and
- `confirmatory_rollout`: a power-determined number of seeds per task, at most
  50, disjoint from both development partitions and all prior CAC tuning seeds.

The complete development budget is therefore at most 10 seeds per task. C6
may label the actual R0 states recorded during C5 with the dense teacher, but
it may not fit on `development_r1_holdout` states or outcomes.

Power planning uses only the frozen non-inferiority margin, target power, and
prior dense success evidence. It may not use CAC outcomes. The final seed count
and manifest freeze before C5 begins. Target power is at least 80% for both G3
and G4 under the predeclared paired-outcome assumptions. If the required sample
exceeds 50 seeds per task or available compute, C0 must stop rather than run an
underpowered confirmatory study.

## 10. Primary and diagnostic metrics

### 10.1 Offline primary metrics

- normalized dense-action error over valid coordinates;
- first-action dense error;
- 95th-percentile per-query dense error;
- gripper-coordinate error;
- gripper threshold/sign disagreement; and
- proportion of residual variance explained by PCA dimensions.

All error improvements are computed relative to the identical uncorrected
cached action.

### 10.2 Offline diagnostic metrics

- translation and rotation error separately;
- action-step index 1--8;
- suite, task, horizon, transition status, source age, camera, and phase;
- residual singular-value spectrum;
- residual norm versus fresh--stale feature-delta norm;
- exact-repeat noise; and
- current-only, cached-only, no-provenance, and no-source-delta ablations.

### 10.3 Closed-loop primary metrics

- terminal task success;
- paired success difference versus dense;
- paired success difference versus uncorrected cache;
- complete-cycle wall-clock latency per model query;
- synchronized CUDA query latency;
- realized cache reuse and dense-reset frequency; and
- adapter-only latency and peak allocated/reserved memory.

Task success is primary. Offline fidelity and efficiency cannot override a
failed success gate.

### 10.4 Statistical rules

- Use identical task, initialization, and seed for paired policy comparisons.
- Report suite-stratified and task-level estimates and intervals.
- Use complete episodes as the resampling unit for terminal outcomes.
- Use complete trajectories as the resampling unit for offline metrics.
- Primary confidence level is 95%; one-sided intervals are permitted only for
  predeclared directional gates.
- No optional stopping based on ordinary fixed-sample p-values.
- If sequential evaluation is used, it must implement a separately verified
  anytime-valid or alpha-spending design frozen before the first outcome.
- Every method, task, episode, exclusion, technical stop, and failure remains
  in the record.

## 11. Principal risks and frozen safeguards

| Risk | Why it could invalidate the direction | Frozen safeguard or decision point |
|---|---|---|
| Action residual is not compressible | High-dimensional K/V corruption may remain high-dimensional at the action output | C2 retains raw 56D residuals and reports the full spectrum; PCA is diagnostic, while held-out predictability decides continuation |
| Current projected tokens omit needed information | The fresh signal available before the reused layers may not reveal the downstream action error | Current-only and full cache-aware controls must be separated by the G1 margin |
| Corrector becomes a replacement policy | A small network may learn task actions from current vision without repairing cache-specific error | Cached hidden/action context, source deltas, and provenance ablations are mandatory; claims narrow if G6 fails |
| Gripper errors are sparse but decisive | Mean continuous error can hide open/close mistakes | Transition-stratified sampling, a separate gripper head, disagreement metric, and gate are mandatory |
| Demonstration states do not match corrected-policy states | Low offline error may fail in closed loop | C5 tests R0 first; only one R1 dense-teacher aggregation round is allowed, followed by held-out development evaluation |
| Recursive correction destabilizes later states | A locally good correction may change future observations and compound error | Residuals are bounded, zero-initialized, evaluated across all eight actions, and finally judged by terminal success |
| Dense and cached branches contaminate each other | Shared mutable cache could invalidate labels | C1 requires isolated clean clones, dense-shadow non-mutation tests, all-fresh parity, and provenance checks |
| Apparent speedup disappears after correction | Fresh feature handling and adapter execution may consume the cache saving | C1 requires gross headroom; C4 and C7 require complete-cycle physical benefit and bound adapter overhead |
| One-GPU memory is insufficient | Simultaneous dense/cached paths and feature capture can exceed 24 GB | C1 measures peak allocation; record generation is sequential where necessary; adapter training occurs with the 7B backbone unloaded |
| Split leakage inflates offline results | Queries from the same trajectory are highly correlated | Every split and resample uses complete trajectories; calibration and locked test remain separately sealed |
| Architecture or threshold chasing creates a false positive | Repeated redesign after outcomes invalidates inference | Only three ordered model classes are allowed in C2; the smallest passing model freezes before calibration |
| Development results are too noisy | Six R0 and four R1-holdout seeds per task are not final proof | C5/C6 are decisions only; C7 uses a disjoint power-planned confirmatory population |
| Prior work already covers the mechanism | The claimed contribution may not be original | C0 repeats a current primary-source collision audit and stops on a five-element mechanism match |
| Result is overgeneralized | Evidence uses one checkpoint, benchmark family, and cache profile | Claim language is explicitly limited to the pinned stack and evaluated populations |

No safeguard guarantees a positive result. Their purpose is to distinguish a
failed hypothesis from a technical failure and to prevent a favorable but
non-causal or non-reproducible result from being presented as CAC evidence.

## 12. Positive-result gates

All gates below must pass. Failure cannot be offset by another metric.

### G1 — Cache-specific offline repair

On the once-opened C3 calibration population, frozen R0 must achieve:

1. at least 25% relative reduction in mean dense-action error;
2. at least 20% relative reduction in first-action error;
3. at least 15% relative reduction in 95th-percentile error;
4. at least 20% relative reduction in gripper threshold disagreement on the
   transition stratum; and
5. non-worsening point estimates in every LIBERO suite for mean and first-
   action error.

The full model must improve mean error by at least 10% relative to the
current-only controller and by at least 10% relative to the cached-action-plus-
provenance controller. Otherwise the cache-specific mechanism is unsupported.

### G2 — Physical efficiency

Before scaled rollouts, the integrated CAC path must demonstrate:

1. at least 5% observed complete-cycle query-latency reduction versus dense;
2. a one-sided 95% confidence bound above zero improvement;
3. adapter overhead no greater than 35% of the gross cache saving;
4. peak memory within the selected 24 GB device boundary with a documented
   reserve; and
5. no numerical, cache-provenance, action-processing, or dense-bypass mismatch.

### G3 — Closed-loop repair versus cache

On the independent C7 confirmatory population, final CAC must improve paired
terminal success over the identical uncorrected cache profile by:

- at least 5 percentage points observed; and
- a one-sided 95% lower confidence bound greater than zero.

### G4 — Dense reliability non-inferiority

On the same C7 population, final CAC must be non-inferior to dense
OpenVLA-OFT within an absolute 2-percentage-point margin using the predeclared
paired one-sided 95% interval.

No individual suite may show an observed CAC success loss greater than five
percentage points versus dense, and no task may have zero CAC successes when
dense succeeds on at least half of its paired trials.

### G5 — Confirmed physical benefit

On C7 evaluation queries, final CAC must retain:

- at least 5% observed complete-cycle latency reduction versus dense;
- a one-sided 95% lower bound greater than zero; and
- the frozen reuse/reset behavior without hidden dense fallback.

### G6 — Mechanism support

At least three of the following four full-model comparisons must show the
predeclared favorable direction on independent data, and none may reverse by a
large prespecified margin:

1. remove current fresh features;
2. remove cached hidden/action context;
3. remove exact provenance;
4. remove tile-aligned fresh--source deltas.

The full model must outperform the current-only controller by the frozen G1
offline margin and show the prespecified favorable paired-success direction on
the C7 mechanism subset. Otherwise the result is consistent with small-policy
distillation rather than cache-specific correction and must be reframed.

## 13. Phase C0 — freeze, literature, and manifests

### Work

- repeat the literature collision audit in Section 5;
- authenticate all inherited revisions, hashes, and evidence;
- freeze the exact `D62_BAL_PT1` schedule and age-four reset semantics;
- generate `adapter_fit` and `adapter_validation` trajectory manifests;
- generate the C2 balanced schedule without reading residual outcomes;
- generate disjoint `development_r0`, `development_r1_holdout`, and
  `confirmatory_rollout` seed manifests;
- conduct confirmatory sample-size/power planning;
- freeze correction bounds procedure, loss weights, optimizer candidates,
  seeds, model limits, timing method, storage cap, call cap, and stop rules;
- specify the exact gripper threshold and postprocessing identity; and
- produce a C0 research and design report.

### Exit gate

- no unresolved novelty collision;
- every identity/hash authenticated;
- all manifests deterministic and leak-free;
- every outcome population still closed;
- projected C1/C2 storage at most 10 GiB and one-GPU calls explicitly capped;
- tests and implementation plan reviewable; and
- no server/GPU execution performed.

Stop before C1 and request approval.

## 14. Phase C1 — tensor and systems feasibility

### Work

- implement read-only hooks for current/source projected tile features,
  cached action hidden states, actions, state, and provenance;
- implement isolated dense and cached branches from clean cache clones;
- verify shapes, dtypes, devices, normalization, source identity, and eight-step
  chronology;
- verify the dense shadow branch cannot update or contaminate the deployed
  cache;
- implement compact streaming records and schemas;
- add unit, synthetic-recursion, exact-repeat, all-fresh, and exception-cleanup
  tests;
- run a bounded real-tensor smoke test on one selected GPU; and
- measure peak memory, per-hook overhead, data volume, gross substrate latency,
  and numerical repeatability.

### Hard caps

- at most one GPU and one VLA process;
- at most 120 model calls;
- at most 4 GiB temporary evidence;
- no adapter training;
- no simulator terminal outcomes; and
- no calibration or locked-test trajectories.

### Exit gate

- exact dense parity with hooks disabled/enabled;
- exact all-fresh parity;
- exact cache provenance and source-feature alignment;
- zero nonfinite values;
- repeat noise at numerical tolerance;
- at least 8% measured gross complete-cycle headroom for the fixed profile and
  reset rule;
- memory remains inside the 24 GB boundary with reserve; and
- deterministic, bounded storage estimate for C2/C3.

Failure stops the project or requires a new protocol version. Stop before C2
and request approval.

## 15. Phase C2 — residual structure and predictability screen

### Work

- execute the frozen train-only schedule of 3,600 primary contracts, 240
  all-fresh controls, and 360 exact repeats;
- retain raw 56-dimensional dense-minus-cached residuals and compact permitted
  inputs;
- verify all controls and repeats before analysis;
- run PCA/SVD and report 80/90/95/99% variance dimensions;
- fit ridge, then MLP, then compact cross-attention only as allowed by Section
  8;
- evaluate nested input ablations and every offline metric; and
- select the smallest passing architecture once.

### Protection

During the GPU worker, monitor only process health, record counts, bytes,
elapsed time, and aggregate selected-GPU telemetry. Do not inspect partial
residual summaries, model rankings, or subgroup outcomes until the immutable
worker summary verifies all 4,200 scheduled executions and the separately
frozen model-call counts.

### C2 continuation gate

On `adapter_validation`, the selected model must achieve:

- at least 25% mean-error reduction;
- at least 20% first-action reduction;
- at least 15% 95th-percentile reduction;
- at least 20% gripper-disagreement reduction on transitions;
- favorable mean-error direction in all four suites;
- at least 10% mean-error advantage over current-only and
  cached-action-plus-provenance controls; and
- predicted inference cost consistent with the C4 overhead budget.

PCA rank alone neither passes nor fails the method. Predictable action repair
is the primary question.

If no permitted model passes, stop CAC. Do not invent a new architecture.
If one passes, freeze it and stop before C3 for approval.

## 16. Phase C3 — R0 training and offline confirmation

### Work

- generate the capped, balanced C3 training records from `adapter_fit`;
- train only the frozen adapter with the 7B model unloaded;
- use `adapter_validation` for early stopping and checkpoint selection;
- freeze exactly one R0 checkpoint and its optimizer/data/config hashes;
- open the 320 calibration trajectories once;
- evaluate G1 and all prespecified diagnostics;
- publish the complete calibration report without changing thresholds; and
- preserve calibration records as consumed evidence.

### Hard rules

- no new architecture, features, optimizer search, loss, bound, or seed after
  calibration opens;
- no terminal simulator outcomes;
- no locked-test access;
- no backbone or action-head updates; and
- no retry of calibration with a revised R0.

### Exit gate

- all G1 requirements pass;
- no task/suite subgroup reveals a catastrophic correction pattern;
- checkpoint is immutable and reproducible; and
- exact dense bypass remains guaranteed.

Failure stops CAC before integration. Passing stops before C4 for approval.

## 17. Phase C4 — physical integration and timing

### Work

- integrate frozen R0 after cached action-hidden production;
- validate bounded residuals, gripper processing, action queue, cache reset,
  source tracking, exception cleanup, and dense bypass;
- run warm-up-separated, synchronized physical timing for dense, uncorrected
  cache, and R0 CAC;
- measure complete-cycle wall time, CUDA time, adapter time, memory, and reuse;
- verify no second encoder, hidden dense call, or fallback inflates correctness;
  and
- evaluate G2 once.

### Exit gate

Every G2 requirement passes and all correctness controls pass. Failure stops
before closed-loop work. Passing stops before C5 for approval.

## 18. Phase C5 — R0 paired closed-loop development pilot

### Population

Use only the frozen `development_r0` manifest: six seeds per task across all 40
tasks, giving 240 paired conditions. Every condition runs:

- dense OpenVLA-OFT;
- identical uncorrected `D62_BAL_PT1` plus age-four reset; and
- R0 CAC.

For R0 queries, retain the compact deploy-time inputs and replayable current
observation identities needed to obtain dense-teacher labels later if C6 is
authorized. These records remain sealed during C5 analysis and may not contain
terminal success as a training feature.

### Protection

Do not inspect partial task success or aggregate outcomes. Monitor only process
health, terminal-record counts, artifact bytes, elapsed time, and aggregate
selected-GPU telemetry until all scheduled episodes and immutable summaries
exist.

### Decision

R0 becomes the final candidate without R1 only if the full development
population shows:

- observed success improvement of at least 5 percentage points over
  uncorrected cache;
- observed dense success loss no greater than 2 percentage points;
- no suite loss greater than 5 percentage points versus dense; and
- retained G2 efficiency.

R1 is eligible only if:

- R0 passed C2/C3 strongly;
- R0 is not more than 10 percentage points worse than uncorrected cache;
- R0 produces no catastrophic task subgroup or invalid action behavior; and
- results are consistent with an offline-to-policy distribution gap rather
  than missing information or excessive overhead.

If R0 is catastrophic or worse than uncorrected cache by more than 10 points,
stop. If R0 passes directly, freeze it for C7 and skip C6. Otherwise, stop and
request approval for the single C6 aggregation round.

## 19. Phase C6 — one R1 on-policy aggregation round

### Purpose

C6 addresses only the predeclared demonstration-to-policy distribution shift.
It is not permission to redesign the model.

### Work

- open the sealed deploy-time records from actual R0 states induced during C5;
- replay each recorded current observation through an isolated dense teacher
  to label the dense-minus-R0 residual without changing the recorded rollout;
- compare task/horizon-matched R0 dense-action error on those states with the
  frozen C3 calibration reference;
- only if the distribution-gap diagnostic below passes, train the same
  architecture on a C0-frozen mixture of demonstration and on-policy records;
- freeze one R1 checkpoint; and
- evaluate dense, uncorrected cache, and R1 once on the frozen
  `development_r1_holdout` population of four seeds per task.

### Prohibited changes

- no second aggregation round;
- no architecture, profile, reset, loss, feature, bound, or threshold change;
- no reinforcement-learning reward or terminal-success training;
- no use of confirmatory seeds or locked-test trajectories; and
- no choosing between multiple R1 seeds after outcomes.

C0 must freeze the C6 mixture ratio, teacher-call cap, epoch cap, and R1 seed.
C6 may not use C5 terminal outcomes, C6 holdout outcomes, or future states as
training inputs or labels.

Before R1 fitting, the on-policy records must show at least 15% relative
degradation from the matched C3 calibration reference in mean error,
first-action error, or gripper-transition disagreement. If none degrades by
15%, the hypothesized distribution gap is not supported and C6 stops without
training. This diagnostic threshold is not a positive-result gate; it only
prevents an unjustified aggregation round.

### Exit gate

R1 must satisfy the same C5 development criteria required for direct R0
selection and retain G2 efficiency. Otherwise CAC stops. Passing R1 freezes it
as the final candidate and stops before C7 for approval.

## 20. Phase C7 — independent confirmation

### Preflight

Before any confirmatory outcome:

- authenticate the final R0/R1 checkpoint and every code/config hash;
- verify zero confirmatory-seed use during C0--C6;
- verify the power-planned population and hard episode cap;
- verify dense, uncorrected-cache, and final-CAC policy identities;
- verify complete output sealing and no partial outcome monitoring; and
- verify sufficient disk, time, and one-GPU allocation.

### Execution

Run paired dense, uncorrected cache, and final CAC on the frozen confirmatory
manifest. Required mechanism ablations use a separately frozen bounded subset
or offline locked-test records as declared in C0; they may not alter the primary
population.

Open outcomes only after exact episode counts, terminal summaries, timing
records, exclusions, and hashes reconcile.

### Decision

- **Positive CAC result:** G3, G4, G5, and G6 all pass.
- **Reliability-only result:** success passes but G5 fails; do not claim
  efficient inference.
- **Efficiency-only negative result:** G5 passes but G3 or G4 fails.
- **Mechanism-ambiguous result:** primary performance passes but G6 fails;
  narrow or reframe the contribution.
- **Negative result:** required reliability gates fail.

No confirmatory rerun, threshold change, task deletion, model reselection, or
new R1 round is allowed under V1.

## 21. Phase C8 — evidence and paper decision

### Work

- reconcile every protocol gate against immutable evidence;
- produce full method, data, model, timing, and statistical documentation;
- report all tasks, suites, exclusions, failures, and confidence intervals;
- publish code/config/result manifests appropriate for the repository;
- update `PROJECT_STATUS.md` and `docs/DECISIONS.md`;
- preserve checkpoints and compact reproducibility artifacts under size rules;
- sync approved repository content to GitHub and the local review clone; and
- decide the manuscript route from the evidence only.

If positive, the manuscript should present CAC as the solution motivated by the
negative SAVR evidence. If negative, preserve the complete study without
rebranding another method inside the same protocol.

Manuscript editing and external publication remain separately authorized work.

## 22. Evidence schema and reproducibility

Every phase must record:

- protocol and phase-config hashes;
- repository and dependency revisions;
- model/checkpoint and normalization identities;
- dataset/split/schedule hashes;
- selected GPU ID and non-identifying aggregate telemetry;
- command, start/end timestamps, exit state, and random seeds;
- input/output counts and semantic hashes;
- peak memory, storage bytes, and elapsed time;
- every technical stop, exclusion, and retry decision;
- whether any protected population was opened; and
- confirmation that nothing outside `/home/ved/SAVR` was modified.

Raw images, full K/V tensors, large checkpoints, datasets, credentials, and
unreviewed generated artifacts must not be committed to GitHub. Compact feature
records and adapter checkpoints require size and privacy review before commit.

At every completed phase:

1. write a human-readable report;
2. write a machine-readable summary;
3. hash all evidence;
4. run repository tests and formatting checks appropriate to the phase;
5. commit only scoped, reviewable files;
6. push approved changes to the private GitHub repository; and
7. update the local review clone only after the remote revision is verified.

## 23. Technical-stop and retry rules

A technical stop is limited to failures such as:

- dependency/import failure;
- GPU out-of-memory before scientific records open;
- corrupted or missing input;
- schema/count/hash mismatch;
- nonfinite tensor;
- simulator crash;
- process termination; or
- disk/resource exhaustion.

A technical stop is not an unfavorable scientific result.

Before protected labels or outcomes open, one repaired retry may occur only if
the phase-specific protocol explicitly permits it and the failure leaves no
partial scientific result. After a calibration, development, or confirmatory
population opens, preserve all evidence and stop fail-closed. Any retry then
requires a new versioned protocol identifying the population as consumed.

Never delete or overwrite failed-run evidence.

## 24. Techniques that keep the project on task

1. One fixed cache profile and reset horizon.
2. One bounded model-selection phase with three ordered model classes.
3. Smallest-passing-model rule.
4. Whole-trajectory split inheritance.
5. Explicit gripper-transition coverage.
6. Current-only and cached-only controls against false mechanism claims.
7. Physical timing before closed-loop scale-up.
8. One conditional on-policy aggregation round only.
9. Disjoint development and confirmatory rollout seeds.
10. Outcome sealing during GPU execution.
11. Phase-specific hard call, storage, and episode caps.
12. No threshold relaxation after evidence opens.
13. No phase begins without explicit approval.
14. Stop rules are treated as scientific results, not obstacles to bypass.
15. The central question remains action-space repair versus full cache
    reconstruction; it may not silently become routing, pruning, or backbone
    training.

## 25. Phase summary

| Phase | Purpose | GPU/rollout scope | Main decision |
|---|---|---|---|
| C0 | Literature, hashes, manifests, freeze | No GPU/outcomes | Is the study novel and fully specified? |
| C1 | Tensor/system feasibility | <=120 model calls | Can inputs be captured safely with speed headroom? |
| C2 | Residual structure/predictability | 3,600 primary + 600 auxiliary executions | Is action repair learnable and cache-specific? |
| C3 | R0 train/offline confirmation | Adapter train + calibration | Does frozen R0 generalize offline? |
| C4 | Integration/physical timing | Bounded timing only | Does meaningful net speed remain? |
| C5 | R0 closed-loop development | 6 seeds/task | Does offline repair transfer? |
| C6 | Optional single R1 round | C5-state labels + 4 held-out seeds/task | Does distribution matching close the gap? |
| C7 | Independent confirmation | Power-planned, <=50 seeds/task | Do reliability, speed, and mechanism pass? |
| C8 | Evidence and paper decision | No new experiment | What claim is supported? |

## 26. Protocol acceptance checklist

Before C0 execution is requested, verify that the user understands and accepts:

- CAC is a positive-results hypothesis, not a guaranteed positive outcome;
- the first decisive test is the inexpensive C2 predictability screen;
- terminal success, not offline action error, decides the method;
- R1 is the only permitted distribution-shift correction round;
- failed gates stop the direction rather than trigger an unbounded redesign;
- each phase requires separate approval;
- TITAN work remains confined to `/home/ved/SAVR`; and
- this document itself performed no implementation, training, GPU work, or
  protected-data access.
