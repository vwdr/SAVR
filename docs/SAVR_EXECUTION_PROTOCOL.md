# SAVR Execution and Verification Protocol

Version: 1.0

Protocol date: 2026-07-29

Project: State-Aware Visual Refresh for Efficient VLA Inference

Repository: https://github.com/vwdr/SAVR

TITAN workspace: `/home/ved/SAVR`

Local review copy: `/Users/veddwivedi/Documents/savr`

## 1. Purpose

This document is the authoritative execution guide for the SAVR project. It converts the manuscript's proposed method into a staged, verifiable research program while preventing scope drift, unsupported claims, unsafe server activity, post-hoc experimental changes, and hallucinated status.

When the user supplies this file and explicitly asks to begin, continue, or execute the SAVR project:

1. Read this file completely.
2. Read `AGENTS.md`, `PROJECT_STATUS.md`, the manuscript, the current protocol/status files, and the latest Git history.
3. Verify the actual repository, server, GitHub, and local-copy state before relying on any remembered status.
4. Begin at the first incomplete checkpoint that is safe and authorized.
5. Execute only one phase at a time.
6. Stop at every approval or acceptance gate.

Providing this file authorizes safe, in-scope planning, coding, validation, and documentation steps inside the SAVR project. It does **not** waive the explicit approval gates for downloads, GPU use, large experiments, destructive actions, manuscript changes, or expanded scope.

## 2. Authority and conflict resolution

Use this precedence order:

1. system, developer, and current explicit user instructions
2. `/home/ved/SAVR/AGENTS.md`
3. this protocol
4. the frozen experimental protocol, once created
5. `PROJECT_STATUS.md`, milestone records, decision records, and run manifests
6. remembered conversation context

If two sources conflict:

- do not select the convenient interpretation
- collect direct evidence
- record the conflict
- stop and ask the user if the conflict affects scientific meaning, safety, resource use, or scope

This file defines the stable plan. `PROJECT_STATUS.md` defines the changing project state. Never infer status from this plan alone.

## 3. Non-negotiable server boundary

- The only writable university-server path is `/home/ved/SAVR`.
- Do not inspect, modify, move, rename, delete, copy, or change permissions on unrelated university files or directories.
- Do not inspect or interfere with unrelated processes, jobs, containers, services, environments, users, network settings, or GPU allocations.
- Never use `sudo`.
- Never make system-wide installations or configuration changes.
- Never terminate, reprioritize, or attach to unrelated processes.
- Never run broad cleanup, recursive search, or destructive commands outside `/home/ved/SAVR`.
- Do not infer that a GPU is available by inspecting other users' processes. The user must coordinate and identify the permitted GPU.
- Use at most one explicitly user-approved GPU unless separately authorized.
- Keep all project environments, caches, checkpoints, logs, results, and temporary files inside `/home/ved/SAVR`.
- If any required action may affect shared university work, stop.

Every phase report must explicitly confirm whether anything outside `/home/ved/SAVR` was modified.

## 4. Research objective and claims

SAVR is a training-free controller that decides whether to recompute or reuse visual features at each VLA policy query. Its decision uses:

- image change
- robot-state change
- recent action change
- a maximum reuse horizon

The project must test three claims:

### H1: Safety/performance

SAVR preserves task success within a predeclared non-inferiority margin relative to Full Refresh (FR).

### H2: Efficiency

SAVR reduces visual computation and produces measurable end-to-end inference benefits.

### H3: Value of embodied signals

State and action signals improve the success-efficiency trade-off relative to Periodic Refresh (PR) and Visual-Only Refresh (VOR), especially at matched refresh budgets.

A positive paper requires evidence for both success preservation and meaningful efficiency. A negative or mixed result must still be reported honestly.

## 5. Scope

### In scope

- training-free inference control
- OpenVLA-OFT as the primary base policy
- LIBERO-Spatial, LIBERO-Object, LIBERO-Goal, and LIBERO-10/Long
- FR, PR, VOR, SAVR, signal ablations, and the external VLA-Cache comparison
- task success, latency, FLOPs estimates, refresh rate, control frequency, memory, and failure analysis
- reproducible code, configurations, manifests, tests, plots, tables, and manuscript updates supported by results

### Out of scope unless separately approved

- model training or fine-tuning
- changing policy weights or action heads
- additional VLA architectures before the core study is complete
- unrelated robotics benchmarks
- real-robot experiments
- multi-GPU execution
- broad server administration
- paper claims not tested by the frozen protocol

AC2-VLA, DeeR-VLA, token pruning, early exit, and other efficiency methods are relevant literature but are not implementation targets unless the user expands scope. VLA-Cache is the one required external comparison because it is the closest published training-free visual-caching method with OpenVLA-OFT/LIBERO code.

## 6. Primary sources and relevance

Use official or primary sources:

- Reviewed paper artifact: `output/pdf/SAVR_Negative_Results_Paper.pdf` (editable source retained in the authors' external archive)
- OpenVLA-OFT code: https://github.com/moojink/openvla-oft
- OpenVLA-OFT paper: https://arxiv.org/abs/2502.19645
- LIBERO: https://github.com/Lifelong-Robot-Learning/LIBERO
- VLA-Cache paper: https://arxiv.org/abs/2502.02175
- VLA-Cache code: https://github.com/siyuhsu/vla-cache
- AC2-VLA paper: https://arxiv.org/abs/2601.19634
- DeeR-VLA paper: https://arxiv.org/abs/2411.02359

Record exact upstream revisions. Never silently update dependencies or model revisions after calibration begins.

## 7. OpenVLA-OFT integration design

The pinned OpenVLA-OFT implementation computes projected visual patch embeddings before it appends the current proprioception token. SAVR must cache only the projected visual embeddings.

On every policy query:

1. Process the current two-camera observation for the low-cost refresh signals.
2. Let the selected controller decide refresh or reuse.
3. If refreshing, run the original visual backbone and projector, then cache the projected patch embeddings and reference images.
4. If reusing, return the cached projected patch embeddings without running the visual backbone/projector.
5. Append the **current** proprioception token.
6. Recompute the language-model and action-head path.
7. Save the predicted action chunk for future action-change decisions.

Initial configuration:

- two input images: third-person and wrist camera
- current proprioception enabled
- L1 continuous action head
- FiLM disabled
- diffusion disabled
- center crop enabled
- eight-action chunks
- deterministic action generation

Do not duplicate or rewrite the upstream action-decoding logic unless the adapter approach is proven impossible. FR through the SAVR integration must match unmodified upstream inference.

### Query semantics

For OpenVLA-OFT, the controller operates when the policy constructs a new action chunk. Therefore:

- `query_index`: index of the VLA policy query
- `environment_step`: index of the simulator control step
- one query normally generates eight actions
- cache age and `H_max` are measured in policy queries

Log both indices. Do not call query-level refresh decisions control-step refreshes without explaining the action-chunk relationship.

## 8. Exact signal definitions

These definitions must be implemented, tested, and frozen before final evaluation.

### Image change

- Use both third-person and wrist images.
- Convert pixels to a common `[0, 1]` scale.
- Downsample each camera to a fixed `32 x 32` representation.
- Compare with the corresponding image from the most recent actual refresh, not only the previous query.
- Compute per-camera mean absolute difference and average the two camera scores.
- Preserve a per-camera score in logs for analysis.

### State change

- Use the eight-dimensional LIBERO state:
  - end-effector position
  - end-effector orientation in axis-angle form
  - gripper joint positions
- Normalize dimensions using the checkpoint's proprioception `q01` and `q99` statistics.
- Compute a normalized L2/RMS change between current and previous query states.

### Action change

- Use the two most recent predicted action chunks available before the current prediction.
- Normalize action dimensions using checkpoint action `q01` and `q99` statistics.
- Compare flattened chunks with a normalized L2/RMS score.
- Force conservative refresh during action-history warm-up.

### Forced refresh conditions

- empty cache
- episode, task, checkpoint, or incompatible-configuration change
- insufficient action history during warm-up
- maximum reuse horizon reached
- invalid, missing, non-finite, or shape-incompatible signal/cache data

## 9. Required policies

All policies must use the same checkpoint, preprocessing, prompt, action head, simulator settings, task order, initial states, seeds, and episode limits.

- **FR:** refresh every policy query; correctness oracle.
- **PR(k):** refresh every fixed `k` queries.
- **VOR:** image threshold plus maximum horizon.
- **SAVR:** image, state, action, and maximum-horizon triggers.
- **VLA-Cache:** official external implementation, pinned and run on the same hardware/checkpoint when compatible.

PR and VOR must be evaluated at refresh budgets comparable to SAVR. Do not compare a conservative SAVR configuration with an aggressively skipping baseline and call the difference causal.

## 10. Repository architecture

Planned project-owned code:

```text
src/savr/
  cache.py
  controllers.py
  signals.py
  logging.py
  timing.py
  evaluation.py
  integration/openvla_oft.py
  analysis/aggregate.py
  analysis/statistics.py

configs/
  smoke/
  calibration/
  final/
  ablations/

scripts/
  setup_environment.*
  verify_environment.*
  run_smoke.*
  run_calibration.*
  run_evaluation.*
  analyze_results.*

tests/
  unit/
  integration/

docs/
  SAVR_EXECUTION_PROTOCOL.md
  MILESTONES.md
  DECISIONS.md
  PROTOCOL_V1.md
  CLAIMS_EVIDENCE.md

results/
  RUN_REGISTRY.csv
  <immutable run directories>
```

Third-party source, environments, checkpoints, caches, and datasets remain project-local but are not committed to Git.

## 11. Phase plan and gates

Only one phase may be `IN_PROGRESS`.

### Phase 0: Establish authoritative state

Actions:

- verify GitHub visibility and latest commits
- verify the TITAN branch/worktree
- verify the local review copy
- merge preparation PRs only after user approval
- synchronize `main`
- create/update milestone and decision ledgers

Exit gate:

- GitHub, TITAN, and local `main` agree
- manuscript hash matches provenance
- no uncommitted or unexplained changes

### Phase 1: Environment and storage feasibility

Actions:

- estimate downloads and storage
- obtain user approval
- create a project-local Micromamba installation/environment
- target Python 3.10.14 and the official OpenVLA-OFT dependency family
- pin PyTorch, the custom Transformers revision, OpenVLA-OFT, LIBERO, and checkpoint revisions
- keep all caches inside the project
- skip FlashAttention initially unless inference proves it is required
- verify imports and headless LIBERO rendering without loading the VLA

Exit gate:

- reproducible environment lock/report
- headless simulator smoke test passes
- no system changes
- actual storage usage documented

### Phase 2: Unmodified FR reproduction

Actions:

- obtain a user-coordinated GPU ID
- load the LIBERO-Spatial OpenVLA-OFT checkpoint
- run a fixed FR smoke episode
- collect component timing and peak memory
- verify action-chunk and observation semantics
- run a calibration-split FR pilot across all Spatial tasks

Exit gate:

- checkpoint fits one TITAN RTX
- complete run manifests
- no gross baseline-success discrepancy
- visual-encoder share of latency quantified

If maximum attainable end-to-end benefit appears negligible, report the estimate and request a go/no-go decision before large experiments.

### Phase 3: Controller and cache implementation

Actions:

- implement common controller interface
- implement signal calculations
- implement projected-visual-feature cache
- integrate without altering policy weights/action head
- implement immutable per-query and per-episode logging

Exit gate:

- code review complete
- controller/cache unit tests pass
- no upstream source change unless separately justified and recorded

### Phase 4: Correctness and instrumentation

Required tests:

- FR refreshes every query
- PR cadence is exact
- VOR ignores state/action
- SAVR trigger truth table is correct
- first-query/history warm-up behavior is conservative
- horizon enforcement is exact
- cache resets between episodes/tasks/checkpoints/configurations
- cached tensor shape, dtype, and device remain correct
- current proprioception remains fresh during reuse
- invalid data forces refresh
- wrapped FR actions match unmodified OpenVLA-OFT
- timing includes required synchronization
- manifests and result schemas validate
- interrupted runs remain recoverable and visible

Exit gate:

- all tests pass
- FR numerical/behavioral parity passes
- logging audit passes

### Phase 5: Smoke policies and external-baseline feasibility

Actions:

- run FR, PR, VOR, and SAVR on one task and three fixed initial states
- verify refresh counts and trajectories
- pin and test official VLA-Cache on the same base stack

Exit gate:

- all core policies finish
- VLA-Cache compatibility is established or a documented technical exclusion is reviewed
- no unexplained policy-specific instrumentation difference

### Phase 6: Calibration and power

Calibration data:

- LIBERO-Spatial
- all 10 tasks
- initial-state IDs `0-9`
- calibration seed `0`
- 100 episodes per candidate setting

Actions:

- collect FR signal distributions
- evaluate predeclared skip-rate targets of approximately 25%, 50%, and 75%
- evaluate `H_max` candidates in query units
- select the most efficient SAVR setting that satisfies the calibration success constraint
- match PR and VOR refresh budgets to the chosen SAVR operating point within a declared tolerance
- calculate statistical power from paired pilot discordance

Exit gate:

- one frozen primary configuration per method
- final sample size confirmed
- non-inferiority margin approved
- no final holdout outcome inspected

### Phase 7: Freeze final protocol

Create and commit `docs/PROTOCOL_V1.md` containing:

- hypotheses and ordered primary outcomes
- checkpoints and upstream revisions
- exact policy configurations
- thresholds/horizons
- task suites, task IDs, initial-state IDs, and seeds
- episode limits and success definition
- timing protocol
- statistical analysis
- failure taxonomy
- exclusions and rerun rules
- expected run count and resource estimate

Exit gate:

- user approves protocol
- repository tagged or commit hash recorded
- any later methodological change requires a new protocol version

### Phase 8: Final evaluation

Final holdout for each suite:

- all 10 tasks
- initial-state IDs `10-49`
- seeds `7`, `17`, and `27`
- 1,200 episodes per policy per suite

Core evaluation:

- 4 suites
- FR, PR, VOR, SAVR
- 19,200 core episodes

External comparison:

- VLA-Cache on the same four suites if Phase 5 compatibility passes
- up to 4,800 additional episodes

Run policies in counterbalanced/randomized blocks on the same approved GPU to reduce ordering, temperature, and transient-system bias. Do not omit failed or interrupted episodes.

Exit gate:

- every planned run has a terminal status
- manifests and counts reconcile with `RUN_REGISTRY.csv`
- no protocol deviation is unresolved

### Phase 9: Ablations and sensitivity

Required ablations:

- image only
- image + state
- image + action
- state + action
- full SAVR
- maximum-horizon sensitivity
- per-camera trigger contribution
- refresh-budget sensitivity

Run core signal ablations across all four suites with one frozen seed. Confirm the full SAVR result using the complete three-seed protocol. Do not add post-hoc ablations solely to rescue an unfavorable result; mark exploratory analyses explicitly.

Exit gate:

- each manuscript component claim has a corresponding ablation
- exploratory and confirmatory results are separated

### Phase 10: Analysis and claim audit

Primary outcomes:

1. task-success non-inferiority versus FR
2. end-to-end policy-query speedup versus FR
3. visual-refresh reduction

Secondary outcomes:

- CUDA visual-encoder time
- total CUDA and wall-clock latency
- median and p95 latency
- amortized latency per executed action
- FLOPs estimate
- control frequency
- gate overhead
- peak GPU memory
- trigger frequency/overlap
- cache-age distribution
- task/phase failure categories

Use:

- paired episode comparisons
- paired risk differences
- one-sided non-inferiority confidence intervals
- task-aware/stratified bootstrap
- paired latency distributions
- success-efficiency Pareto curves

Measure latency in two ways:

- online closed-loop timing
- offline replay of fixed observation traces for isolated model-compute comparison

Exit gate:

- analysis scripts regenerate every table/figure
- raw records trace to aggregates
- `CLAIMS_EVIDENCE.md` maps every claim to evidence

### Phase 11: Manuscript completion

Only after Phase 10:

- write Results
- write Discussion
- write Limitations
- write Conclusion
- revise the abstract and introduction to match measured findings
- update related work and citations as needed

Do not conceal null, negative, inconsistent, or failure results. Do not describe calibration findings as final evaluation.

Exit gate:

- every quantitative manuscript value is reproducible
- unsupported claims are removed
- limitations include benchmark, simulation, architecture, hardware, action-chunk, and threshold-generalization constraints

## 12. Statistical commitments

Recommended non-inferiority margin:

- absolute task-success decrease no larger than 2 percentage points relative to FR

This margin must be approved and frozen before final evaluation.

If pilot-based power is inadequate:

- increase sample size or narrow the claim before final evaluation
- do not enlarge the margin after seeing final results

Treat episodes as paired by suite, task, initial-state ID, and seed. Report suite-level and task-level results, not only a pooled average.

## 13. Efficiency measurement rules

Refresh percentage is not sufficient evidence of acceleration.

Record:

- preprocessing time
- gate time
- visual-backbone/projector time
- downstream VLA/action-head time
- total policy-query time
- environment-step time
- episode wall time
- CUDA memory allocation

Use CUDA events or equivalent synchronized device timing. Warm up the model before timing. Do not mix model-loading time into steady-state inference. Use the same GPU and dependency environment for comparisons.

Estimate FLOPs separately for:

- refresh query
- reuse query
- gate

Report actual latency and estimated compute; do not substitute one for the other.

## 14. Failure analysis

Predeclare categories:

- stale visual representation near contact
- missed object or scene motion
- gripper-transition failure
- directional correction/reversal failure
- excessive/unnecessary refresh
- action instability after reuse
- simulator failure
- model/environment compatibility failure
- timeout/interruption/resource failure
- unclassified, with preserved evidence

Save a bounded sample of success and failure videos. Do not save every rollout by default.

## 15. Anti-hallucination verification protocol

When uncertain, do not improvise. Apply this sequence:

1. **State the uncertain claim.**
2. **Identify the authoritative evidence source.**
3. **Collect direct evidence.**
4. **Compare evidence with the claim.**
5. **Record the result and source.**
6. **Proceed only if verified.**

Evidence hierarchy:

1. current file contents, hashes, Git state, test output, and run artifacts
2. official upstream source code and documentation at a recorded revision
3. primary papers
4. project decision records
5. conversation summaries
6. memory

Mandatory rules:

- Never invent results, status, file existence, successful tests, citations, repository visibility, GPU availability, or completed uploads.
- Never say a command passed without its output or recorded exit status.
- Never claim a file is synchronized without comparing commit/hash state.
- Never claim a result belongs to the final protocol without a valid manifest and protocol version.
- Never infer missing values from nearby runs.
- Never silently repair or delete failed results.
- Never cite a search-result summary when the primary source is available.
- Mark statements as `FACT`, `DECISION`, `HYPOTHESIS`, `RESULT`, or `BLOCKER` in decision/status records when ambiguity is possible.
- If verification is impossible, report `UNVERIFIED` and stop the dependent action.

## 16. Techniques that keep the project on task

Maintain:

- `PROJECT_STATUS.md`: current phase, completed work, next gate, blockers
- `docs/MILESTONES.md`: phase checklist and exactly one active milestone
- `docs/DECISIONS.md`: decisions, alternatives, evidence, and approver
- `docs/PROTOCOL_V1.md`: frozen confirmatory protocol
- `results/RUN_REGISTRY.csv`: every run and terminal status
- `docs/CLAIMS_EVIDENCE.md`: manuscript claims mapped to artifacts

At the start of every work session:

1. read the governing files
2. verify branch and worktree
3. identify the single active milestone
4. state the intended checkpoint

At the end of every work session:

1. update status and decision records
2. run appropriate validation
3. inspect the scoped diff
4. commit/push only intentional changes
5. report checkpoint status and blockers
6. synchronize the local review copy

Change-control rules:

- one active phase at a time
- no final evaluation before protocol freeze
- no result-dependent benchmark expansion
- no threshold changes after final outcomes are viewed
- no unrelated model/benchmark work before the core study
- a correctness bug invalidates all affected comparisons, not only unfavorable runs
- any protocol deviation is logged before analysis
- exploratory work is labeled and cannot replace confirmatory results

## 17. Run integrity

Each run must have:

- immutable run ID
- protocol version
- SAVR Git commit
- clean/dirty status
- upstream commits
- checkpoint identifier and immutable revision
- environment lock/hash
- suite, task, initial-state ID, and seed
- complete controller configuration
- exact command
- start/end UTC timestamps
- terminal status
- raw logs and result paths

Run statuses:

- `PLANNED`
- `RUNNING`
- `COMPLETED`
- `FAILED`
- `INTERRUPTED`
- `INVALIDATED`

Never reuse a run ID. Never overwrite a completed run directory.

## 18. GitHub and local-review workflow

TITAN `/home/ved/SAVR` is the authoritative execution workspace.

For each scoped phase:

1. create or use a dedicated branch
2. inspect status/diff before staging
3. stage only intended files
4. run relevant checks
5. commit tersely
6. push through the authenticated Mac workflow without copying GitHub credentials to TITAN
7. open/update a draft PR
8. wait for CI
9. synchronize the pushed branch to `/Users/veddwivedi/Documents/savr`
10. stop if the local review copy has uncommitted user changes
11. merge only after review/approval
12. synchronize `main` everywhere after merge

Do not claim the local review copy is current until its commit matches the pushed branch.

## 19. Approval gates

Explicit user approval is required before:

- merging a draft PR
- downloading/installing the project environment, checkpoints, or substantial assets
- using a GPU or selecting the permitted GPU ID
- launching multi-episode calibration
- freezing the final non-inferiority margin and protocol
- launching final or ablation experiments
- changing the manuscript
- expanding beyond this scope
- deleting material project data

The agent may perform read-only verification and lightweight CPU-only project validation while preparing an approval request.

## 20. Stop conditions

Stop and report if:

- any action may affect resources outside `/home/ved/SAVR`
- the GPU allocation is unclear
- system-wide changes appear necessary
- storage/network requirements exceed approval
- FR cannot reproduce a credible baseline
- wrapped FR differs from upstream
- the intended visual cache boundary is absent or semantically unsafe
- results cannot be traced to manifests
- final data were exposed before threshold/protocol freeze
- a material protocol conflict appears
- the study is statistically underpowered
- the local review copy contains uncommitted user changes
- a claim would require evidence not produced by the protocol

## 21. Required phase report

Every checkpoint report must contain:

- outcome
- evidence and validation
- files changed
- Git branch/commit/PR
- current phase and gate status
- experiments/runs performed
- resources used
- failures or uncertainty
- server-safety confirmation
- local-sync confirmation
- next proposed action and required approval

## 22. Definition of project completion

The project is complete only when:

- all planned core runs have valid terminal statuses
- all required baselines and ablations are accounted for
- statistical analyses are reproducible
- every paper claim maps to evidence
- results include negative findings and limitations
- manuscript sections reflect actual results
- code, configurations, manifests, tables, and plots are committed
- private GitHub, TITAN, and local review copies are synchronized
- no unresolved safety, provenance, or protocol issue remains

Until then, never describe the project or paper as complete.
