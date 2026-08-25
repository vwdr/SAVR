# BRACE Execution Protocol V1

**Method:** Branch-Rollout Adaptive Cache Execution  
**Date frozen:** 2026-08-24  
**Status:** **SUPERSEDED before implementation. Do not execute V1.** The active
plan is `docs/BRACE_EXECUTION_PROTOCOL_V2_1.md`. No V1 implementation, model
run, GPU operation, or BRACE outcome occurred.
**Authority:** No GPU/model/simulator campaign is authorized merely by this document

## 1. Objective

Determine, with bounded cost and protected holdouts, whether paired terminal
rollout interventions can train a pre-inference cache-profile controller that
improves the task-reliability--latency frontier of OpenVLA-OFT beyond existing
visual-KV caching.

## 2. Invariants

1. The base VLA, action head, proprio projector, checkpoint, prompt, image
   preprocessing, action normalization, and simulator semantics remain frozen.
2. Optimized full refresh is the correctness and timing oracle.
3. BRACE may choose computation profiles but may not alter predicted actions
   after a profile is selected.
4. Every cache path uses current proprioception and the current allowed visual
   tokens exactly.
5. Runtime routing uses only the deployment feature allowlist.
6. Any uncertainty, invariant failure, reset, task change, cache-age violation,
   or OOD rejection causes full refresh.
7. Direct mid-episode `set_state()` restoration alone is not an accepted branch.
8. All paired outcome categories, technical stops, and exclusions are retained.
9. Final thresholds, populations, and analyses are frozen before final outcomes.
10. Nothing outside `/home/ved/SAVR` may be modified on TITAN.

## 3. Protected populations

Historical development outcomes remain development evidence. The existing
untouched initial states 10--49 and seeds 7/17/27 across the supported suites
remain protected until the final protocol explicitly selects and freezes a
subset.

Recommended structure:

- method development: already-opened states 0--9;
- branch training/validation split: group by complete task/initial-state
  episode, never by branch record;
- confirmation: untouched initial states from 10--49;
- optional external-validity test: suite/task group never used to train the
  router.

The exact map is frozen before B4 outcomes.

## 4. Phase B0 — Research and feasibility audit

**Status:** Complete.

Evidence:

- `docs/BRACE_RESEARCH_AND_FEASIBILITY_AUDIT.md`;
- official VLA-Cache/OpenVLA-OFT source inspection;
- pinned LIBERO and robosuite state audit;
- current nearest-work collision search.

No BRACE implementation or outcome was produced.

## 5. Phase B1 — Replay-verified branch harness

**Resource class:** CPU/simulator only; no VLA model or CUDA allocation.

### 5.1 Implementation

Build an episode transcript schema containing:

- task, BDDL, initial-state, seed, environment and dependency revisions;
- every low-level action in execution order;
- policy-query and action-chunk boundaries;
- reward, success, termination, timestep, and action-queue indices;
- selected MuJoCo state components and observation digests at branch points;
- both camera images or lossless hashes needed for comparison.

Build a replay function that starts from the original published initial state and
replays the complete low-level action prefix through normal `env.step()` calls.
Do not patch LIBERO state semantics.

### 5.2 Tests

For at least three tasks with contact, gripper transition, and free-space motion:

1. generate a deterministic scripted transcript;
2. replay it twice in fresh environments;
3. compare every branch checkpoint;
4. apply the same next probe action and compare the resulting transition;
5. repeat with branch points early, middle, and late in the episode;
6. deliberately alter one prefix action and prove the validator rejects it.

### 5.3 Acceptance

- 100% transcript completeness;
- exact equality for discrete state/counters/reward/success;
- exact numeric equality where deterministic, otherwise a pre-outcome tolerance
  supported by repeated negative controls;
- identical next-action transition at every accepted branch;
- corrupted-prefix and direct-state-only negative controls rejected;
- no untracked simulator or system modifications.

Failure to reproduce branches ends BRACE or requires a newly researched full-
state serialization method. Do not continue with approximate labels.

## 6. Phase B2 — Cache substrate integration and CPU correctness

**Resource class:** local/CPU source and synthetic-tensor work only.

### 6.1 Isolation

- pin VLA-Cache source and its Transformers branch by commit;
- compare the fork against the project's pinned OpenVLA-OFT Transformers fork;
- create a project-local isolated dependency plan;
- never overwrite the proven SAVR environment;
- record download and disk estimates before installation.

### 6.2 Components

Implement/test:

- exact `DynamicCache` clone/restore, including K/V tensors, positions, seen
  tokens, attentions, and metadata;
- transactional model-config restoration;
- P0--P3 profile definitions and switching;
- scene/wrist token-index proofs;
- cache-age, provenance, reset, and exception invalidation;
- deterministic profile-disabled equivalence to FR;
- immutable branch identities and treatment assignment.

### 6.3 Acceptance

- all synthetic parity and adversarial tests pass;
- no cache object or global model configuration leaks between arms;
- profile order produces identical results under randomized test ordering;
- exact resource estimate and B3 command are frozen;
- repository tests and static checks remain green.

## 7. Phase B3 — Bounded physical cache microbenchmark

**Resource class:** one explicitly authorized TITAN GPU; no simulator outcomes.

Maximum: 200 real-model queries across balanced FR/P1/P2/P3 conditions.

### 7.1 Required measurements

- synchronized end-to-end query wall time;
- CUDA time by visual encoder, projector, decoder, action head, and cache work;
- peak allocated/reserved memory;
- reusable-token counts by camera/layer;
- action parity with P0 and repeatability by profile;
- cache clone cost for training only;
- profile-switch and reset correctness.

### 7.2 Acceptance

- P0 matches optimized FR actions and tensors within frozen tolerances;
- no profile violates shape, current-state, camera, or cache provenance
  invariants;
- at least one deployable profile achieves at least 10% synchronized end-to-end
  query-wall reduction versus optimized FR;
- projected router budget leaves a credible path to at least 8% net reduction;
- peak reservation remains below the unchanged conservative GPU cap;
- source/checkpoint restoration and GPU release reconcile exactly.

If only P3 is fast enough and it causes extreme action divergence in the
outcome-blind trace, record the evidence but do not call it deployable. B4 may
use it only as a stress treatment.

## 8. Phase B4 — Small branch-label pilot

Requires B1--B3 acceptance and a separate frozen population/resource record.

### 8.1 Provisional bounded design

- 30 development episodes;
- no more than three preassigned branch points per episode;
- P1 and the fastest B3-eligible profile as treatments;
- paired FR/cache arms;
- one treatment chunk followed by common FR continuation;
- maximum 180 paired branch comparisons;
- provisional 3--6 GPU-hour cap, replaced by the B3-based estimate;
- randomized arm order and exact prefix reconstruction.

Reserve a predeclared subset for identical-treatment negative controls: `FR`
versus `FR` and each cache profile versus itself. These controls are additional
to the primary paired comparisons but remain inside the frozen hour/query cap.

No router is trained in B4.

### 8.2 Questions

1. Do harmful cache interventions exist at a learnable prevalence?
2. Does cache profile/age change that prevalence?
3. Does action disagreement miss harmful pairs, as predicted by the literature?
4. Are labels stable under replay and randomized arm order?
5. Is the measured cost compatible with a scaled dataset?

### 8.3 Acceptance

Before outcomes, freeze minimum class-count and cost rules. Recommended defaults:

- at least 20 harmful and 40 neutral-success pairs across the eligible
  deployment profiles, with no single task contributing more than half of the
  harmful class;
- zero unexplained branch-equivalence or arm-order violations;
- at least 95% of planned branches terminal and analyzable;
- scaled development collection projected below 40 GPU-hours and the frozen
  storage cap.

If cache harm is too rare, BRACE has no demonstrated problem to solve on this
stack. If nearly every cache treatment is harmful, no useful coverage exists.
Both conditions stop BRACE.

### 8.4 Conditional development expansion

B4 is a prevalence and noise pilot, not a sufficient router-training dataset.
If it passes, calculate and freeze the additional branch count before collecting
more outcomes. The expansion must provide, after episode-grouped splitting, at
least 100 harmful training pairs, 30 harmful validation pairs, and 30 harmful
held-out test pairs, unless a formal learning-curve/power analysis justifies a
larger requirement. It must
also preserve a deployment-representative evaluation stratum separate from any
rare-regime enrichment.

Stop instead of scaling if the measured prevalence implies more than the
40-GPU-hour development cap, if identical-treatment discordance is not
negligible relative to harmful prevalence, or if any task would dominate the
harmful class. Record selection probabilities and use them when estimating
natural prevalence and risk--coverage.

## 9. Phase B5 — Feature and label-sufficiency test

Freeze the router feature set and architecture before training. Split complete
episodes/initial states, not individual branch records.

### 9.1 Models

- prevalence/profile prior;
- task/cache-metadata baseline;
- visual-motion threshold baseline;
- action-disagreement-supervised router;
- BRACE terminal-effect router.

### 9.2 Metrics

- harmful-class prevalence;
- AUPRC and AUROC with group bootstrap intervals;
- false-negative harm rate at matched cache-service coverage;
- risk--coverage and calibration curves;
- results by task, profile, cache age, episode phase, and camera motion;
- router synchronized wall latency and memory.

### 9.3 Gate freeze

Because meaningful AUPRC depends on pilot prevalence, B4 may determine the exact
B5 numeric gate, but it must be committed before any comparative model outcome
is opened. It must require all of:

- clear improvement over the prevalence and nonvisual baselines;
- a lower confidence bound above random ranking for the primary discriminative
  metric;
- material harmful-intervention reduction at a service rate that still passes
  the B3 net-latency projection;
- router overhead within the frozen budget;
- no dependence on prohibited deployment features.

Failing B5 ends BRACE. Do not add current full-VLA hidden features to rescue it.

## 10. Phase B6 — One on-policy aggregation round

Only after B5 passes:

1. freeze BRACE-v0 threshold/profile policy;
2. collect new rollouts under BRACE-v0 on separate development initial states;
3. sample branches outcome-blind from those cache-induced trajectories;
4. use the current BRACE continuation after the randomized first intervention;
5. aggregate once and train BRACE-v1;
6. compare v1 against v0 on a held-out branch set.

Proceed only if aggregation improves or preserves risk--coverage without losing
the speed-eligible coverage. More than one additional aggregation round requires
a new cost/overfitting justification.

## 11. Phase B7 — Paired closed-loop development

Minimum methods:

1. optimized FR;
2. official VLA-Cache;
3. periodic clean refresh at matched compute;
4. strongest reproducible confidence gate;
5. action-disagreement router;
6. BRACE-v1;
7. BRACE without OOD rejection;
8. BRACE without on-policy aggregation.

LAC and Action-JND receive direct experimental comparisons only if usable code
and compatible weights are available before the baseline freeze. As of the
audit, the LAC repository contains no implementation beyond a README.
AC2-VLA does expose research code, but its released path is CogACT +
Prismatic-7B on SIMPLER with multi-GPU Bridge/OXE training, not a drop-in
OpenVLA-OFT/LIBERO baseline. Audit its design and published results directly;
attempt a head-to-head only if a separately validated compatible checkpoint and
evaluation path exist. Otherwise these results remain context, not fabricated
comparisons.

Provisional advancement targets, finalized by a power calculation before
outcomes:

- success non-inferior to FR within a 2-percentage-point paired margin;
- at least 10% net synchronized query-wall reduction including the router;
- statistically supported Pareto improvement over official VLA-Cache or the
  strongest reproducible gate;
- no task-level catastrophic regression;
- complete profile, rejection, cache-age, and refresh-reason accounting.

## 12. Phase B8 — Sealed confirmation

Only after B7 passes:

- freeze code, model, router, thresholds, profiles, seeds, populations, and
  analyses;
- evaluate across all standard LIBERO suites using untouched initial states;
- determine episode count with an a priori paired non-inferiority power
  calculation; provisional target is 500 episodes per method per suite;
- report binomial success intervals, paired differences, paired bootstrap
  latency intervals, and the complete risk--latency frontier;
- retain all excluded/failed/technical episodes and reasons;
- repeat the novelty/code-availability audit before claims.

No threshold or operating point may be chosen from B8 outcomes.

## 13. Resource and safety policy

- Default to one GPU and select only an aggregate-idle device.
- Record GPU ID/UUID and aggregate pre/post telemetry.
- Do not inspect or interfere with other users' processes.
- Use multiple GPUs only under a separately frozen coordination plan.
- All writes remain inside `/home/ved/SAVR` on TITAN.
- No `sudo`, system-wide installation, permission change, service change, or
  unrelated file inspection.
- Every new source/checkpoint/dataset download requires a provenance, license,
  size, and storage record before execution.
- Every long run gets a fixed identity, query/episode/hour/storage cap, terminal
  record, technical recovery rule, and outcome-blind monitor.

## 14. Statistical and claim policy

- Use task success and reliability language, not safety, unless collisions or
  constraint violations are separately measured.
- Treat branch records from one episode as dependent; use group-aware splits and
  resampling.
- Report effect prevalence and all four paired outcome categories.
- Report full curves, not only the selected threshold.
- Distinguish simulator interventional validity from real-world validity.
- Do not claim novelty for VLA-Cache, DAgger, counterfactual simulation, failure
  detection, or camera asymmetry individually.
- A negative feasibility gate is a valid outcome and must not be bypassed.

## 15. Next authorized-sized task

Prepare B1 only: implement the CPU/simulator transcript and replay-equivalence
design with tests. Do not install the VLA-Cache environment, load a model, use a
GPU, collect policy outcomes, or open protected initial states as part of B1.
