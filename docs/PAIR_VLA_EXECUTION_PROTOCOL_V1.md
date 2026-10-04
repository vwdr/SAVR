# PAIR-VLA Authoritative Research and Execution Protocol

**Project:** SAVR / PAIR-VLA

**Protocol version:** 1.0

**Date:** 2026-08-26

**Status:** Planning authority only. Creating this document does not authorize
dataset transfer, implementation, GPU use, simulator outcomes, a commit, a
push, or the next phase.

**Method identity:** Provenance-Aware Interventional Reuse for
Vision-Language-Action Models, abbreviated PAIR-VLA.

## 0. Purpose, authority, and document hierarchy

This is the single operational plan for evaluating PAIR-VLA as a possible
positive-results paper. It is written to be usable by a future Codex agent
without relying on remembered conversation history.

The execution order is:

1. repository AGENTS.md safety and scientific-integrity rules;
2. explicit instructions in the current user turn;
3. this protocol;
4. the PAIR research plan and third-pass audit;
5. earlier SAVR, ACR, and BRACE documents as historical evidence only.

This protocol consolidates and supersedes operational details in:

- docs/PAIR_VLA_RESEARCH_DIRECTION_AND_EXECUTION_PLAN_2026-08-26.md;
- docs/PAIR_VLA_SECOND_PASS_FEASIBILITY_AUDIT_2026-08-26.md; and
- docs/PAIR_VLA_THIRD_PASS_LOGIC_RESOURCE_AUDIT_2026-08-26.md.

Earlier documents remain immutable audit history. If this protocol conflicts
with an earlier PAIR plan, this protocol controls unless the user explicitly
changes it.

Every phase ends with exactly one disposition:

- PASS: all frozen gates passed;
- FAIL-SCIENTIFIC: the method hypothesis failed;
- STOP-TECHNICAL: the experiment did not validly test the hypothesis; or
- BLOCKED-AUTHORITY: the next required action needs user approval.

A failure is not permission to rename the method, change a threshold, inspect a
protected population, or begin another redesign.

## 1. Goal and positive-paper definition

### 1.1 Research goal

Determine whether a small deployment-time router, trained using the measured
expert-regret effect of the exact stale K/V states available to the cache, can
produce a better closed-loop reliability-efficiency frontier than dense
OpenVLA-OFT and strong cache baselines.

### 1.2 Targeted contribution

PAIR-VLA is not merely a learned cache selector. Its proposed contribution is
the linked combination of:

1. actual-stale intervention supervision;
2. exact per-token, per-layer, per-camera temporal provenance;
3. recursive multi-query cache contracts; and
4. a lightweight risk-calibrated router whose complete overhead is measured.

The narrow novelty claim is:

> PAIR-VLA learns temporal VLA cache decisions from the expert-regret caused by
> the exact provenance-resolved stale states that deployment would reuse,
> including their cumulative behavior across bounded query contracts.

The protocol must never claim:

- the first learned VLA cache;
- the first action-aware token selector;
- formal robot safety;
- guaranteed task success;
- training-free operation;
- universal acceleration across VLA architectures or hardware; or
- a positive result before the locked closed-loop gates pass.

The VLA backbone is frozen, but the router is trained. The accurate description
is frozen-backbone, lightweight offline router training.

### 1.3 Positive-paper gates

PAIR-VLA supports the targeted positive paper only if the locked confirmatory
evaluation establishes all of the following:

1. suite-stratified paired task success is non-inferior to same-stack dense
   inference with a predeclared margin of 2 percentage points;
2. synchronized complete-cycle latency is at least 10% lower than same-stack
   dense inference, with a positive lower confidence bound;
3. PAIR improves the paired success-latency frontier over the strongest
   faithfully runnable cache comparator under a matching rule frozen without
   success outcomes;
4. the actual-stale label, provenance features, and recursive contract
   treatment each contribute under frozen ablations; and
5. no resource, correctness, leakage, or evidence-integrity gate fails.

If PAIR improves a cache baseline but misses dense non-inferiority or the 10%
net-speed floor, the result may be scientifically useful but is not the
targeted positive paper.

## 2. Research basis

### 2.1 Local evidence

The relevant local evidence is fixed:

- BRACE-B3 v05 completed 356 model queries without a technical stop.
- P2-D50 reduced accelerated-query time by 18.71%.
- P2-D50 reduced complete-cycle time by 12.23%.
- Every one of the 168 timed cached actions differed numerically from its
  same-stack dense reference.
- Dense same-stack completion was approximately 1.220 seconds p50.
- Peak aggregate model memory was 18,419 MiB and worker reservation was
  19,094,568,960 bytes, under the strict 23-GiB cap.
- SourceLedger already records query source, age, layer, position, camera,
  patch identity, K/V digest, source record, prior action summary, and cache
  contract state.
- BRACE already supports bounded horizons 1, 2, and 4 and nested reuse sets at
  decoder pruning layers 2, 6, 9, and 11.

The measured problem is therefore selection under stale-state corruption. The
project does not need another visual threshold without task-aligned evidence.

### 2.2 Primary-source research implications

- [VLA-Cache](https://arxiv.org/abs/2502.02175) establishes adaptive
  temporal visual-token K/V reuse and is the required heuristic comparator.
- [OpenVLA-OFT](https://arxiv.org/html/2502.19645) establishes the frozen
  continuous eight-action OpenVLA policy and its L1-regression training
  objective.
- The official [LIBERO evaluation
  loop](https://github.com/moojink/openvla-oft/blob/main/experiments/robot/libero/run_libero_eval.py)
  requeries only when the action queue is empty and defaults to eight open-loop
  actions.
- The official [platform
  constants](https://github.com/moojink/openvla-oft/blob/main/prismatic/vla/constants.py)
  define an eight-action, seven-dimensional LIBERO chunk with eight-dimensional
  proprioception and q99-bound normalization.
- The official [RLDS
  pipeline](https://github.com/moojink/openvla-oft/blob/main/prismatic/vla/datasets/rlds/dataset.py)
  constructs observation and future-action windows, but temporal continuity and
  filtering must be verified rather than inferred.
- [LAC](https://arxiv.org/html/2602.00686) demonstrates that a frozen-backbone
  learned cache policy can improve both task results and latency. It trains
  through task-loss gradients and optical flow rather than measured
  actual-stale interventions.
- [Action-JND](https://arxiv.org/html/2608.21247) is the closest neighbor. It
  learns generic feature-perturbation tolerance and uses the score to rank
  VLA-Cache candidates, including on OpenVLA-OFT. It does not train on the
  regret caused by the exact mixed-source stale K/V state or recursive
  provenance contract proposed here.
- [Gated VLA-Cache](https://arxiv.org/html/2608.10824) uses previous-step
  discrete action-token confidence to invalidate a cache. Its published gate
  is not directly defined for the pinned continuous L1 action head.
- [Learning What Matters](https://arxiv.org/html/2607.21692) and
  [CausalGate](https://arxiv.org/html/2607.22720) support the use of controlled
  interventions rather than attention alone as routing supervision.
- [CrossVLA](https://arxiv.org/html/2605.21854) shows that cache leverage is
  architecture- and hardware-specific. Its approximately 21% prefix ceiling on
  a flow-matching policy is not transferred to OpenVLA-OFT; PAIR uses measured
  TITAN complete-cycle latency only.

All 2026 papers are recent preprints. P0 and the final paper phase must repeat
the collision search and version-pin every citation and comparator.

## 3. Fixed experimental system and boundaries

### 3.1 Base policy

- Backbone: the pinned four-suite OpenVLA-OFT 7B checkpoint already present
  under the project.
- Policy head: continuous L1 regression.
- Inputs: primary camera, wrist camera, proprioception, and instruction.
- Output: eight actions, each with seven dimensions.
- Normalization, unnormalization, gripper inversion, image preprocessing,
  prompt, token map, action mask, dtype, and checkpoint revision must match the
  accepted BRACE/OpenVLA-OFT stack.
- The backbone, vision encoder, action head, and proprio projector remain
  frozen.

### 3.2 Deployment chronology

Let q index model queries and t index environment steps. Under the pinned
evaluator, the normal relationship is:

\[
t_{q+1}=t_q+8,
\]

except for episode termination and explicitly documented evaluator behavior.

Source age is measured in model-query intervals, not raw video frames.

The primary offline demonstration sequence must preserve the same chronology:
query observation, eight executed expert actions, next query observation.

The dataset name containing no_noops is not sufficient evidence of chronology.
If no-op filtering removes transitions or original step indices, taking every
eighth retained record is forbidden. P0/P1 must use one of these valid paths:

1. verify retained original step indices and reconstruct exact eight-step
   spacing;
2. use unfiltered original LIBERO demonstration trajectories;
3. replay authenticated demonstrations in the pinned simulator to recover
   query-aligned observations; or
4. stop if none is reproducible within the project boundary.

No synthetic interpolation may replace missing control steps.

### 3.3 Resource and server boundary

- The only writable server path is /home/ved/SAVR.
- No sudo, system installation, external cleanup, unrelated file inspection,
  process inspection, allocation inspection, or interference is permitted.
- At most one user-coordinated GPU may be used.
- Static GPU identity/capacity checks are allowed; GPU selection requires user
  coordination rather than inspecting other users.
- Peak project worker reservation must stay strictly below 23 GiB.
- One model process may run at a time.
- Dataset/model downloads require a separate explicit transfer and storage
  approval.
- Full-backbone gradients, model sharding, and multi-GPU training are outside
  this protocol.

The last measured server state was approximately:

| Resource | State |
|---|---:|
| Project footprint | 30.82 GiB |
| Filesystem free space | 363.03 GiB |
| Checkpoints | 14.84 GiB |
| Environments | 8.73 GiB |
| Existing project cache | 3.82 GiB |
| Optional compressed LIBERO data | about 9.53 GiB |
| Prior model peak | about 18.4 GiB |

Disk capacity is plausible, but unpacked data, generated indices, and temporary
cache growth must be estimated before approval.

## 4. Formal PAIR-VLA method

### 4.1 Cache and provenance state

At model query q, decoder layer l, and visual token i, freshly computed key and
value states are:

\[
K^q_{l,i},V^q_{l,i}.
\]

The ledger resolves the exact live source query:

\[
\tau_{q,l,i}\le q,
\qquad
d_{q,l,i}=q-\tau_{q,l,i}.
\]

It also records camera, patch identity, source observation hashes, source
proprioception, source prior-action summary, K/V identity, profile, anchor,
horizon, and abort history.

For a valid structured reuse mask Zq:

\[
C_q(Z_q)_{l,i}=
\begin{cases}
(K^{\tau_{q,l,i}}_{l,i},V^{\tau_{q,l,i}}_{l,i}), & Z_{q,l,i}=1,\\
(K^q_{l,i},V^q_{l,i}), & Z_{q,l,i}=0.
\end{cases}
\]

The primary treatment must execute the real partial cache suffix. Post-hoc
replacement of an isolated dense tensor is diagnostic only.

### 4.2 Dense and intervened actions

For current observation Oq, proprioception Sq, instruction L, and frozen model
parameters theta:

\[
A_q^0=f_\theta(O_q,S_q,L;C_q^{fresh}),
\]

\[
A_q(Z_q)=f_\theta(O_q,S_q,L;C_q(Z_q)).
\]

Same-stack dense is the causal reference for intervention labels. Optimized
core-FR is a separate latency comparator because BRACE showed it is not
numerically identical to same-stack dense.

### 4.3 Primary expert-regret label

Let Aq* be the checkpoint-normalized expert action chunk aligned to query q.
The primary loss exactly mirrors the pinned continuous-head training loss:

\[
\ell_{L1}(A,A_q^*)=
\frac{1}{|\mathcal{M}_q|}
\sum_{(h,j)\in\mathcal{M}_q}
|A_{h,j}-A_{q,h,j}^*|,
\]

where the valid action mask Mq, action normalization, padding, and reduction
must reproduce the official training pipeline.

The signed cache-induced expert regret is:

\[
R_q^*(Z_q)=
\ell_{L1}(A_q(Z_q),A_q^*)-
\ell_{L1}(A_q^0,A_q^*).
\]

Positive regret means caching worsens expert imitation relative to same-stack
dense. Negative regret is retained and reported; it is not silently clipped or
counted as evidence of a beneficial policy.

This exact mean L1 loss replaces the weighted custom action loss in the earlier
PAIR research draft. Coordinate-weighted translation, rotation, and gripper
measures remain diagnostics, not the primary label.

### 4.4 Secondary diagnostics

Predeclared diagnostics are:

1. dense-action distortion:

   \[
   D_q(Z_q)=
   \frac{1}{M_q}
   \sum_{(h,j)\in\mathcal{M}_q}
   |A_q(Z_q)_{h,j}-A_q^0{}_{h,j}|;
   \]

2. continuous translation error by chunk step;
3. continuous rotation error using the simulator-faithful composition, not
   naive axis-angle addition;
4. gripper sign/transition mismatch after the official normalization,
   binarization, and inversion path;
5. integrated chunk-end translation and rotation displacement;
6. maximum coordinate error and upper-tail per-step error; and
7. exact action and gripper hashes for repeatability.

No diagnostic may replace the primary label after results are observed.

### 4.5 Recursive contracts

A contract of horizon B is:

\[
\mathcal{Z}_{q:B}=(Z_q,Z_{q+1},\ldots,Z_{q+B-1}),
\qquad B\in\{1,2,4\}.
\]

It starts from a clean dense anchor. Each query carries the actual mixed cache,
SourceLedger, prior model-produced action summary, and remaining contract state
into the next query.

Contract regret is:

\[
R_{q:B}^{*}=
\sum_{b=0}^{B-1}\beta_b
R_{q+b}^{*}(Z_{q+b}),
\]

with beta weights frozen in P0. The primary default is equal weighting. A
discounted alternative may be examined only as a predeclared diagnostic.

The offline observations remain expert-trajectory observations. Therefore,
contract labels measure cumulative computational corruption under an exogenous
state sequence; they do not reproduce state drift caused by executing cached
policy actions.

### 4.6 Atomic reuse groups

The working atomic group is:

camera x spatial tile x reuse-onset layer.

Subject to P0 verification of the 16 by 16 patch map:

- each camera has 256 patches;
- a spatial tile contains 4 by 4 patches;
- each camera therefore has 16 tiles;
- onset layers are 2, 6, 9, and 11; and
- there are at most 2 x 16 x 4 = 128 atomic groups.

Reusing a tile from an onset layer means its entries remain eligible through
the deeper pruning layers in the contract, preserving suffix-safe nested sets.
Profile budgets must remain reachable in whole-tile increments or explicitly
fall back to the existing per-token ranking without changing the treatment.

If runtime mapping, nesting, or physical-kernel behavior contradicts this
grouping, P0 must freeze a corrected grouping before any labels. Grouping may
not change after the first intervention population is opened.

### 4.7 Intervention population

The offline label population contains:

1. repeated all-fresh controls;
2. atomic group interventions;
3. camera-balanced multi-group masks;
4. profile-realistic masks at physically measured budgets;
5. horizons 1, 2, and 4;
6. source-age and mixed-source cases;
7. gripper-transition and contact-adjacent states; and
8. generic-perturbation and previous-frame diagnostic controls.

Every mask has a known sampling probability and immutable identity. Sampling is
stratified by suite, task, trajectory, camera, onset layer, age, horizon,
profile, and transition status.

Pilot allocation begins with:

- 20% all-fresh/repeat controls;
- 30% atomic interventions; and
- 50% multi-group/profile contracts.

Ten percent of non-control masks are exact repeats for technical-noise
estimation. P4 may change later allocation only through a predeclared
variance/coverage rule and must preserve the original pilot.

### 4.8 Deployment-available router features

Primary features may include only values available before the reuse decision:

- exact source query and source age statistics;
- camera, spatial tile, onset layer, and profile identity;
- current-versus-actual-source raw patch L1, cosine, mean, maximum, and
  upper-quantile change;
- current and source normalized proprioception and their L1/Linf changes;
- model-produced previous action summaries and their source/current changes;
- previous executed gripper transition;
- contract horizon, remaining steps, source-mixture count, and abort history;
- previous-query attention/salience if retained without a current dense pass;
  and
- a compact instruction embedding only if it is already computed before the
  cache decision and its cost is charged.

A current projected visual feature is excluded unless P0 proves it is available
before the decision without executing computation the selected cache profile is
supposed to skip.

Oracle expert actions, current dense outputs, terminal success, future frames,
future proprioception, task outcomes, and hidden current dense passes are
forbidden router inputs.

The primary router must carry the model-produced previous action summary during
demonstration contracts. Expert previous actions are labels, not deployment
features.

### 4.9 Router architecture

The primary architecture is a compact group encoder plus a DeepSets-style
contract calibrator:

1. standardize continuous features using training-split statistics only;
2. encode categorical camera/layer/profile/horizon fields with small learned
   embeddings;
3. pass each group through a shared two-hidden-layer MLP;
4. output a signed group-regret estimate and a reusable group embedding;
5. aggregate selected-group embeddings by sum, mean, and maximum;
6. concatenate aggregate, global state, profile budget, camera composition,
   layer composition, horizon, and source-age features;
7. pass the result through a two-hidden-layer set MLP; and
8. output signed mean regret, median positive regret, and 0.9-quantile positive
   contract regret.

The initial frozen implementation target is:

- group hidden widths 64 and 32;
- set hidden widths 64 and 32;
- GELU activation;
- no dropout in the primary architecture;
- fewer than 250,000 trainable parameters;
- no gradient through the VLA; and
- deterministic inference.

Training objectives are:

- Huber loss for signed group and contract regret;
- pinball loss for the 0.5 and 0.9 positive-regret quantiles; and
- an auxiliary Huber loss for dense-action distortion.

Loss weights, optimizer, learning rate, batch construction, early stopping,
maximum epochs, and seeds are frozen in P0. P2 verifies their implementation on
synthetic data. If a P2 correctness contradiction requires a change, execution
stops for a documented protocol amendment before any GPU population is opened.

P0 may freeze a smaller architecture if its outcome-free sample/parameter audit
shows the primary design is unsupported. Once P4 labels are opened, the
architecture may be neither simplified nor enlarged from observed performance;
an underpowered or overfit router is a failed/blocked gate requiring an explicit
protocol amendment.

### 4.10 Selection and fallback

For each frozen physical profile:

1. identify eligible atomic groups from provenance and hard validity rules;
2. rank groups by predicted upper-tail reuse risk;
3. construct a suffix-safe nested mask satisfying the profile budgets;
4. evaluate the set-level predicted contract-risk upper bound; and
5. choose the greatest measured saving whose calibrated risk remains below the
   frozen threshold.

Formally:

\[
Z_q^*=
\arg\max_{Z\in\mathcal{Z}_{frozen}} S(Z)
\quad\text{subject to}\quad
U_\alpha(\widehat R_q^*(Z))\le\delta.
\]

S(Z) is synchronized measured complete-cycle saving, not FLOPs.
Ualpha is an empirical held-out calibration bound, not a formal safety
certificate.

Dense fail-closed inference is mandatory when:

- provenance is missing or inconsistent;
- any input is nonfinite;
- source age exceeds the frozen cap;
- the source record or K/V identity cannot be resolved;
- the router is outside its training support;
- the calibrated bound exceeds delta;
- a frozen gripper-transition veto fires;
- mask nesting or profile accounting fails;
- a contract aborts; or
- memory/timing instrumentation reports invalid state.

Fallback frequency is charged against the net speed result.

## 5. Data design and leakage controls

### 5.1 Required fields

Each demonstration query requires:

- raw primary and wrist images;
- original trajectory and environment-step identity;
- proprioception;
- instruction;
- eight future expert actions;
- episode boundary;
- action padding/validity mask;
- no-op/filter identity;
- dataset and preprocessing revision; and
- normalization statistics matching the checkpoint.

If original timing or either camera cannot be authenticated, that trajectory is
ineligible for primary labels.

### 5.2 Split unit

Individual tokens, tiles, frames, masks, or rows from the same trajectory may
not cross splits.

The primary split is by complete trajectory within each task:

- 70% router training;
- 15% calibration/model selection; and
- 15% locked offline test.

The split is generated once from trajectory hashes using a pinned seed. Every
derived mask inherits its parent trajectory split.

A rotating task-held-out analysis is a predeclared generalization diagnostic,
not the primary deployment claim. The main paper claim is for new trajectories
and paired initial states on known LIBERO task identities whose demonstrations
may be used to train the small router. This scope must be explicit.

### 5.3 Protected populations

Populations are opened in order:

1. CPU synthetic data;
2. outcome-blind physical timing observations;
3. demonstration intervention labels;
4. offline held-out labels;
5. development simulator outcomes; and
6. confirmatory simulator outcomes.

Later populations remain inaccessible until the preceding phase passes. Scripts
must separate outcome-free technical records from success records. An analyzer
must refuse success fields before the run reaches its terminal expected record
count and completed summary.

No final confirmatory episode may be used for:

- router training;
- threshold selection;
- profile selection;
- comparator matching;
- sample-size re-estimation;
- debugging; or
- failure-driven redesign.

## 6. Metrics, estimands, and statistical design

### 6.1 Offline metrics

- signed excess expert L1 regret;
- positive-regret upper tail;
- dense-action distortion;
- gripper-transition error;
- simulator-faithful chunk-end displacement;
- repeat/control noise;
- rank correlation;
- quantile calibration;
- task/trajectory generalization;
- risk-coverage curve; and
- matched-budget regret reduction versus each proxy.

### 6.2 Physical metrics

- synchronized accelerated-query latency;
- synchronized complete-cycle latency;
- router feature time;
- router inference time;
- provenance/mask construction time;
- reset/fallback time;
- profile service rate;
- effective reused-token percentage;
- peak allocated and reserved GPU memory;
- CPU memory and artifact bytes; and
- abort/fallback counts.

Complete-cycle latency is primary. FLOPs and CUDA-kernel time are descriptive.

### 6.3 Closed-loop metrics

- terminal task success;
- paired success difference;
- complete-cycle and episode latency;
- service/reuse/fallback rate;
- task and suite breakdown;
- contract horizon/age distribution;
- action/gripper diagnostic transfer; and
- failure taxonomy after aggregate unblinding.

Task success is reliability evidence, not collision safety.

### 6.4 Net-speed algebra

Before labels or simulator outcomes, define:

\[
S_{net,LB}=
\rho_{min}S_{raw,LB}-
O_{router,UB}-
O_{reset/fallback,UB}.
\]

The screening defaults are:

- minimum routed service rate rho_min = 0.70;
- maximum total router/provenance/reset overhead = 0.02 of dense
  complete-cycle time; and
- required net lower bound = 0.10.

Under those defaults, raw saving must exceed approximately 17.2% before
uncertainty. The actual P3 gate uses measured confidence bounds, not this point
calculation.

These values are engineering screening thresholds, not scientific results. P0
may correct them using outcome-blind cost evidence only. Once P3 begins, they
are frozen.

### 6.5 Paired success inference

The primary confirmatory estimand is the suite-stratified overall paired
success difference between PAIR and same-stack dense inference.

- Non-inferiority margin: -2 percentage points.
- Type-I error: one-sided 0.05.
- Target power: at least 0.80.
- Pairing: identical task, initial state, and supported simulator seed.
- Individual rates: Wilson intervals.
- Paired differences: paired confidence intervals and McNemar-type analysis.
- Latency: episode/query-block bootstrap.

The P7 development discordance rate determines the P8 sample size before P8
outcomes. Five hundred trials per suite is a starting reference, not an
automatic power guarantee.

The hierarchical confirmatory order is:

1. dense non-inferiority;
2. at least 10% net latency reduction;
3. matched-comparator frontier improvement; and
4. mechanistic ablations.

A later favorable result cannot rescue an earlier failed gate.

## 7. Comparators and ablations

### 7.1 Required primary methods

1. same-stack dense refresh;
2. optimized core-FR as a separate implementation/latency reference;
3. official or corrected VLA-Cache;
4. strongest faithfully runnable action-aware or learned cache comparator; and
5. PAIR-VLA.

Comparator code, licenses, checkpoint compatibility, preprocessing, and timing
semantics are audited in P0.

If LAC or Action-JND cannot be reproduced faithfully on the pinned stack, the
protocol must not imply a direct state-of-the-art comparison. A matched
reproduction may be used if clearly labeled and technically validated.

Gated VLA-Cache is not a direct primary comparator for the continuous L1 head
unless a principled continuous confidence analogue is frozen before outcomes.

### 7.2 Zero-cost proxy controls

- random group ranking;
- source age only;
- raw visual change only;
- prior attention only;
- proprio/action-state change only;
- BRACE heuristic ranking; and
- previous-frame rather than exact-source features.

### 7.3 Mechanistic PAIR ablations

- no provenance;
- assume every source is the previous query;
- generic perturbation labels instead of actual-stale labels;
- dense-action distortion instead of expert regret;
- single-query labels only;
- no set-level interaction calibrator;
- shared camera model;
- no source age;
- no proprio/action history;
- no instruction feature;
- no upper-tail calibration; and
- fixed profile without router.

Ablations are frozen before the confirmatory phase. Smaller predeclared
populations may be used, but ablations cannot replace the three primary methods.

## 8. Phase map

| Phase | Purpose | Outcome access | Main gate |
|---|---|---|---|
| P0 | Research, protocol, data, comparator, and resource freeze | None | All semantics and budgets resolvable |
| P1 | Data acquisition and query-alignment validation | Demonstration metadata only | Exact eight-step chronology and expert chunks |
| P2 | CPU/synthetic implementation correctness | Synthetic only | Provenance, masks, router, and fail-closed tests |
| P3 | Outcome-blind GPU headroom frontier | Timing only | Conservative path to 10% net speed |
| P4 | Bounded intervention-label pilot | Demonstration regret | Stable, nontrivial, predictable labels |
| P5 | Scaled labels and locked router training | Offline labels | Held-out source-aware improvement |
| P6 | Fully routed physical timing | Timing only | At least 10% measured net speed |
| P7 | Paired closed-loop development | Development success | Better success-latency frontier |
| P7R | Optional one-round on-policy correction | Development states only | Corrected router passes immutable comparator |
| P8 | Locked four-suite confirmation | Confirmatory success | All positive-paper gates |
| P9 | Paper and artifact release | All locked evidence | Reproducible, scoped claims |

### 8.1 Approval matrix

Every phase requires explicit approval unless the user later grants a clearly
bounded multi-phase authorization. Approval for a phase includes only the work
and caps frozen for that phase; it never implies approval for the next phase.

| Phase | Material authority required |
|---|---|
| P0 | Read-only research and protocol/config/report creation |
| P1 | Exact network-transfer and project-storage allowance |
| P2 | Repository code/tests only; no GPU or outcomes |
| P3 | One coordinated GPU and frozen timing cap |
| P4 | One coordinated GPU and frozen pilot-label cap |
| P5 | One coordinated GPU and scaled-label/training cap |
| P6 | One coordinated GPU and outcome-blind timing cap |
| P7 | Development simulator outcomes and episode cap |
| P7R | Separate approval; allowed only if eligibility is documented |
| P8 | Confirmatory outcomes and powered episode cap |
| P9 | Manuscript/release edits, commit, push, and local synchronization |

No phase approval permits sudo, unrelated server access, another user's GPU,
destructive cleanup, hidden outcome access, or a scientific threshold change.

## 9. Detailed phases

### Phase P0 — research and freeze

#### Work

1. Repeat primary-source novelty search using actual-stale, source provenance,
   recursive cache, expert regret, action-aware token compression, and learned
   VLA caching terms.
2. Record title, version, date, code link, license, backbone, label, router
   inputs, timing metric, and evaluation scale for every nearest paper.
3. Audit official code/license availability for VLA-Cache, LAC, Action-JND,
   Gated VLA-Cache, and existing project comparators.
4. Authenticate checkpoint, OpenVLA-OFT, LIBERO, Transformers, PyTorch, CUDA,
   and project revisions.
5. Verify the exact continuous L1 training loss, action mask, q99
   normalization, padding, and gripper path against pinned source.
6. Determine whether available demonstrations preserve original step indices
   and unfiltered chronology.
7. Choose the valid data path and calculate compressed, unpacked, index,
   temporary, and artifact storage.
8. Freeze atomic groups, masks, horizons, beta weights, feature chronology,
   router architecture, optimizer, splits, seeds, thresholds, query/hour caps,
   statistical tests, and comparator matching.
9. Define schemas and protected-outcome enforcement.
10. Map planned code to existing BRACE components and list every required new
    module.
11. Produce:
    - docs/PAIR_P0_FREEZE.md;
    - configs/pair/p0_freeze_v1.json;
    - reports/PAIR_P0_REPORT.md; and
    - a semantic manifest.

#### Gate P0

PASS only if:

- novelty remains defensible against the dated source set;
- data chronology has a valid reproducible path;
- primary loss exactly matches the checkpoint objective;
- every feature is available before its decision and costed;
- recursive actual-cache contracts are implementable without VLA gradients;
- planned storage, runtime, and 23-GiB memory boundary are credible;
- at least one strong comparator is faithfully runnable or the claim is
  explicitly narrowed;
- every population, split, cap, and stop rule is frozen; and
- the user has not yet been asked to authorize a download or GPU run.

Otherwise STOP before implementation.

#### Authority

P0 requires explicit phase approval. It authorizes read-only research and
documentation only.

### Phase P1 — data acquisition and semantic alignment

#### Preconditions

- P0 passed.
- The user explicitly approved exact transfer and project-local storage caps.
- No GPU is used.

#### Work

1. Download only the approved data into /home/ved/SAVR.
2. Authenticate archive identity, license, byte count, and source.
3. Preserve original trajectory and step identities.
4. Verify both cameras, proprioception, language, expert actions, episode
   boundaries, and timestamps/indices.
5. Reconstruct query observations at exact eight-step deployment spacing.
6. Reproduce the official future eight-action chunk and valid mask.
7. Reproduce normalization, unnormalization, gripper conversion, and terminal
   truncation with pinned code.
8. Build an index containing hashes and metadata, not duplicate images.
9. Generate trajectory-level train/calibration/test splits from hashes.
10. Run a CPU-only audit on examples from every suite/task and every terminal
    boundary class.
11. Record actual project growth and remove nothing outside the approved
    temporary files.

#### Gate P1

PASS only if:

- query q and q+1 are separated by exactly eight original environment actions
  for eligible records;
- no filtered-frame shortcut is used;
- expert chunks match official preprocessing byte-for-byte or within a frozen
  exact numerical tolerance;
- every primary record has both cameras and correct proprioception;
- split inheritance prevents trajectory leakage;
- hashes and counts reconcile; and
- storage remains within the approved cap.

If original chronology cannot be recovered, FAIL-SCIENTIFIC/TECHNICAL and stop;
do not substitute adjacent filtered frames.

### Phase P2 — CPU/synthetic implementation correctness

#### Planned code

- src/savr/pair/types.py
- src/savr/pair/query_alignment.py
- src/savr/pair/interventions.py
- src/savr/pair/features.py
- src/savr/pair/router.py
- src/savr/pair/calibration.py
- src/savr/pair/records.py
- src/savr/pair/runtime.py
- tests/pair/
- schemas/pair/

The implementation must extend BRACE interfaces without mutating accepted BRACE
evidence.

#### Work

1. Implement query-aligned dataset indexing.
2. Implement atomic group and profile-contract masks.
3. Implement recursive cache/ledger clone, advance, reset, and abort.
4. Implement exact normalized L1 regret and all diagnostics.
5. Implement chronological router features.
6. Implement the compact group/set router and deterministic serialization.
7. Implement empirical quantile calibration and out-of-support detection.
8. Implement outcome-free intervention records and schemas.
9. Add adversarial tests for:
   - scene/wrist swaps;
   - layer/token off-by-one;
   - missing source record;
   - mixed-source age;
   - ring eviction of live sources;
   - non-nested masks;
   - expert-action leakage;
   - future-frame leakage;
   - current-dense feature leakage;
   - padding and terminal chunks;
   - nonfinite features;
   - unsupported categories;
   - contract abort/reset;
   - deterministic seeds; and
   - malformed result records.
10. Prove that all-fresh masks reproduce same-stack dense on synthetic tensors.
11. Prove actual-stale tensors resolve to requested ledger sources.
12. Run existing BRACE regression tests.

#### Gate P2

- All new and existing relevant tests pass.
- CUDA remains hidden/uninitialized.
- No model/checkpoint/simulator outcome is accessed.
- Every malformed or unsupported state fails closed.
- Router inference is deterministic from a serialized checkpoint.
- Schema validation and semantic hashes reproduce.

Any correctness failure stops before GPU work.

### Phase P3 — outcome-blind physical headroom

#### Preconditions

- P2 passed.
- User coordinated one GPU.
- One model process, 23-GiB cap, no simulator outcomes, no downloads.
- Frozen query/hour/artifact caps.

#### Work

1. Load the pinned stack once.
2. Reverify same-stack dense, sidecar, sequence-map, and all-fresh controls.
3. Benchmark a bounded profile frontier extending beyond P2-D50 through:
   - reuse budget;
   - scene/wrist allocation;
   - onset layer;
   - horizon;
   - source age; and
   - protected-token allocation.
4. Do not use expert regret, action parity, or terminal outcomes to select the
   timing frontier.
   Action arrays are retained only as authenticated hashes/finite-status fields
   until P4; their values and comparisons are not opened in P3.
5. Measure raw accelerated-query and complete-cycle saving.
6. Measure feature extraction, router mock inference, provenance, mask, reset,
   and fallback upper-bound costs.
7. Estimate physically eligible service rate from outcome-free inputs.
8. Compute Snet,LB under the frozen service/fallback rule.
9. Reconcile peak reservation, query identities, randomized interleaving,
   synchronization, and artifact bytes.

#### Initial cap

- at most 800 model queries;
- at most 6 wall-clock hours;
- one GPU and one process;
- peak reservation strictly below 23 GiB;
- at most 2 GiB compact artifacts;
- zero success outcomes; and
- no automatic retry.

P0 may lower the cap. Raising it requires a new user-approved freeze.

#### Gate P3

At least one technically valid profile must:

- have positive raw complete-cycle saving with a bootstrap lower bound;
- satisfy Snet,LB at least 10%;
- fit below 23 GiB;
- preserve provenance, contract, and reset invariants; and
- leave enough source diversity for PAIR routing.

If none passes, PAIR stops before demonstration labels. The cache substrate
does not have enough physical reserve for the targeted paper.

### Phase P4 — bounded expert-regret pilot

#### Preconditions

- P3 passed.
- User approved one bounded GPU pilot.
- No simulator outcomes.

#### Work

1. Select query-aligned anchors from training/calibration trajectories under
   the frozen stratification.
2. Generate same-stack dense actions and exact expert L1.
3. Run all-fresh repeats, atomic interventions, and profile contracts.
4. Carry actual cache and ledger state for horizons 1, 2, and 4.
5. Store compact features, labels, masks, hashes, and timings; do not store a
   full K/V bank per example.
6. Quantify:
   - repeat/control noise;
   - signed-regret distribution;
   - positive-tail prevalence;
   - age/camera/layer/horizon coverage;
   - interaction residual;
   - dense-action and endpoint diagnostics;
   - correlation with provenance features; and
   - correlation with cheap proxy baselines.
7. Train only a small pilot router on the training split.
8. Inspect calibration labels only after the complete pilot summary exists.

#### Initial cap

- at most 400 anchor contracts;
- at most 3 non-control masks per anchor on average;
- at most 4,000 total frozen-model calls;
- at most 8 GPU hours;
- one process and 23-GiB cap;
- zero simulator outcomes;
- no automatic retry.

#### Gate P4

PASS only if:

- exact repeated masks are stable relative to between-mask variation;
- all-fresh controls match their frozen tolerance;
- signed regret varies beyond technical noise;
- positive regret is neither almost absent nor dominated by a few corrupt
  records;
- actual-source/provenance features reach held-out trajectory Spearman
  correlation at least 0.30 with structured-contract regret;
- a source-aware pilot reduces held-out upper-tail positive regret by at least
  15% versus the strongest zero-cost proxy at matched physical budget;
- cumulative contracts show measurable structure beyond single-query labels
  and the locked router can represent that structure; and
- no leakage or resource gate fails.

Thresholds are screening rules, not final claims. Failure stops PAIR or returns
to advisors; it does not authorize another label definition.

If the only failed item is that cumulative contracts are empirically
indistinguishable from single-query labels, P4 is BLOCKED-AUTHORITY rather than
PASS. A single-query PAIR method would require a new novelty check, an explicit
protocol amendment, and user approval before P5; it cannot silently inherit the
recursive-contract contribution.

### Phase P5 — scaled label corpus and locked router

#### Preconditions

- P4 passed.
- P4 population and report are immutable.
- P0 design includes the scaling rule.
- User approved the scaled query/hour budget.

#### Work

1. Use P4 noise and effect estimates to calculate the required intervention
   count for held-out regret and calibration precision.
2. Scale only within the frozen population, masks, horizons, splits, and
   maximum cap.
3. Train the primary router under three pinned seeds.
4. Select epochs and calibration thresholds using calibration trajectories
   only.
5. Evaluate once on the locked offline test split.
6. Compare against every zero-cost proxy and faithfully runnable action-aware
   control.
7. Run the predeclared offline ablations.
8. Freeze:
   - one router architecture;
   - one serialized checkpoint or deterministic seed ensemble;
   - one risk threshold;
   - one fallback rule;
   - at most three physical profiles; and
   - one profile-selection policy.

#### Maximum cap

- at most 20,000 frozen-model calls;
- at most 30 GPU hours;
- compact stored features/labels below the P0-approved artifact cap;
- one GPU/process and 23-GiB cap;
- no simulator outcomes.

The cap may be smaller after the P4 power calculation. It may not be raised
because test performance is disappointing.

#### Gate P5

- Held-out positive-tail regret improves by at least 15% over the strongest
  zero-cost proxy at matched measured compute.
- The upper-risk calibration curve is monotone and within the frozen empirical
  coverage tolerance.
- Performance is not produced only by dense fallback or one task identity.
- No-instruction, no-provenance, previous-source-only, and single-query
  ablations support the claimed mechanism.
- Router size, deterministic inference, and feature availability comply.
- Offline test is opened once.

After PASS, router/profile identity is locked.

### Phase P6 — fully routed physical timing

#### Work

1. Run same-stack dense, optimized core-FR, frozen cache comparators, and the
   locked PAIR router over the same outcome-free balanced query set.
2. Replay the actual recursive cache/reset lifecycle.
3. Include real risk fallbacks at their observed rate.
4. Synchronize every measured interval.
5. Randomize/interleave methods under a pinned schedule.
6. Measure full feature, router, provenance, sorting, cache, reset, and fallback
   cost.
7. Bootstrap by query/contract blocks.

#### Gate P6

- Net complete-cycle reduction versus same-stack dense is at least 10%.
- The saving lower confidence bound is above zero.
- Peak reservation remains below 23 GiB.
- Service rate is at least the frozen minimum.
- Dense fallback is not masking near-zero useful reuse.
- Comparator matching is performed using the pre-outcome rule.

Failure ends PAIR before simulator success.

### Phase P7 — paired closed-loop development

#### Population

Use a development population disjoint from P8. Begin with one predeclared suite
chosen in P0 for its combination of high baseline reliability, interaction
states, and adequate episode length. Do not choose the suite from PAIR outcomes.

#### Methods

- same-stack dense;
- strongest matched cache comparator; and
- locked PAIR.

#### Work

1. Pair identical task/initial-state/seed conditions.
2. Run the complete episode lifecycle and measure success plus physical timing.
3. Do not inspect success until the exact terminal episode count and completed
   run summary exist.
4. Analyze paired success, latency, service, fallback, and outcome-free
   diagnostics.
5. Inspect failure videos only after the frozen aggregate report.
6. Estimate discordant-pair rate for P8 power.

#### Initial development size

The starting design is 30 paired trials per task for the selected ten-task
suite per primary method, subject to a P0 feasibility estimate. The exact count
is frozen before launch.

#### Gate P7

- PAIR retains at least 10% net complete-cycle acceleration.
- PAIR improves the paired success-latency frontier over the strongest cache
  comparator.
- PAIR does not preserve success only by dense fallback on most queries.
- No task has a PAIR paired success drop greater than 20 percentage points, and
  no task has zero PAIR successes when dense succeeds on at least half of its
  paired episodes.
- Gripper-transition failures are not concentrated beyond the P0-frozen
  diagnostic threshold.
- Demonstration-regret risk meaningfully transfers to policy states.

A merely good offline metric does not pass.

### Phase P7R — optional single on-policy correction

This phase exists only for a clear predeclared demonstration-to-policy transfer
failure.

It is allowed once and only once.

1. Preserve the original router and P7 results.
2. Gather bounded query states produced by the locked PAIR policy from a
   development-only population.
3. At those fixed states, compute same-state dense versus cached action
   distortion. Expert actions are unavailable.
4. Add this auxiliary consistency data under a frozen mixing weight.
5. Retrain/refreeze one corrected router.
6. Compare corrected router against the original router on a second disjoint
   development population.

P7R cannot:

- access P8;
- change the actual-stale expert-regret label;
- add a new architecture;
- repeat;
- tune from terminal success; or
- erase the original P7 result.

If the corrected router does not pass the original P7 gates, PAIR stops.

### Phase P8 — locked confirmatory evaluation

#### Design

- all four standard LIBERO suites;
- official tasks and pinned evaluator;
- three primary methods;
- paired initial states;
- final count from the P7 discordance-based power calculation;
- starting reference of 50 trials per task, 500 per suite, and 2,000 across
  suites per method;
- suite-stratified overall primary estimand; and
- mandatory task/suite intervals.

Per-suite two-point non-inferiority is claimed only if separately powered.

#### Outcome protection

During execution, monitoring is limited to:

- owned runner health;
- terminal record count;
- artifact byte growth;
- elapsed time; and
- aggregate selected-GPU telemetry permitted by the frozen run.

Episode success fields and aggregate outcomes remain unread until every expected
terminal episode record and completed summary exist. Early stops may inspect
technical logs only.

#### Positive-paper gate

All Section 1.3 gates must pass in the predeclared hierarchy. Every task,
failure, interval, and deviation is reported.

No rerun is allowed for an unfavorable valid result. A technical stop follows a
frozen fail-closed recovery protocol that changes no scientific choice.

### Phase P9 — paper and artifact release

#### Work

1. Write the method from this frozen specification.
2. Write results only from locked analyzer outputs.
3. Report exact data dependence and known-task scope.
4. Report reliability-efficiency, not safety-efficiency.
5. Report synchronized complete-cycle latency and hardware identity.
6. Include success and paired intervals, negative results, fallbacks, and all
   deviations.
7. Release:
   - source and tests;
   - compact configurations;
   - schemas;
   - seeds;
   - environment identities;
   - analysis code;
   - compact result ledgers;
   - exact commands;
   - data-availability statement; and
   - repository URL or DOI.
8. Exclude checkpoints, datasets, raw K/V banks, credentials, and unrelated
   historical project material from release.
9. Repeat collision, citation, bibliography, figure, venue, and accessibility
   audits.
10. Push the final reviewed branch and pull the exact resulting revision into
    the user's local SAVR folder for review.

#### Gate P9

- Every paper number is generated from an authenticated analysis artifact.
- Every citation is used substantively and resolves to the pinned source.
- Claims follow the Section 15 claim ladder and state all scope limitations.
- Public artifacts contain no credentials, datasets, checkpoints, raw caches,
  protected university material, or unrelated historical files.
- Repository, release artifact, and local review folder resolve to the reported
  final revision.
- The paper and repository pass independent reproducibility, visual, and
  accessibility checks.

## 10. Repository and synchronization workflow

### 10.1 Working locations

- Local review repository:
  /Users/veddwivedi/Documents/VLA/SAVR
- TITAN execution repository:
  /home/ved/SAVR
- Remote source of truth:
  the configured GitHub origin

### 10.2 Safe flow

1. Begin every phase with status, branch, revision, and dirty-tree checks.
2. Preserve unrelated user changes, including untracked files.
3. Implement CPU-safe work locally when dependencies permit.
4. Commit only reviewed phase-scoped files on a dedicated branch.
5. Push without force.
6. Pull the exact commit on TITAN inside /home/ved/SAVR.
7. Run TITAN work only from the authenticated commit/config.
8. Preserve compact evidence and reports.
9. Push reviewed compact outputs.
10. Pull the resulting revision into the local Documents/SAVR folder.
11. Report both local and TITAN revisions and whether anything outside the
    project was modified.

Never use reset --hard, force push, destructive cleanup, or broad server
commands.

## 11. Artifact and schema contract

### 11.1 Planned run directories

- results/pair-p3-headroom-v01/
- results/pair-p4-pilot-v01/
- results/pair-p5-labels-v01/
- results/pair-p6-timing-v01/
- results/pair-p7-development-v01/
- results/pair-p7r-onpolicy-v01/ if eligible
- results/pair-p8-confirmatory-v01/

### 11.2 Intervention record

Every record contains at least:

- schema version;
- run/attempt/query/contract/mask IDs;
- suite, task, trajectory hash, original step, and split;
- model/checkpoint/code/config/preprocessing identities;
- current/source observation hashes;
- contract anchor, horizon, remaining step, and abort history;
- complete ledger semantic hash;
- profile, atomic groups, camera/layer/token budgets;
- source age and source-mixture statistics;
- router feature hash;
- dense/intervened/expert action hashes;
- dense expert L1;
- intervened expert L1;
- signed excess regret;
- diagnostics;
- synchronized timing;
- memory;
- exception/status; and
- record semantic hash.

No terminal success field exists in P3-P6 records.

### 11.3 Router artifact

- architecture JSON;
- feature schema/order;
- normalization statistics;
- training split manifest;
- optimizer/hyperparameters;
- seed;
- checkpoint hash;
- calibration object;
- supported input range;
- threshold/fallback rule;
- training/test metrics;
- parent intervention-manifest hash; and
- code revision.

### 11.4 Episode record

P7/P8 episode records add:

- paired condition identity;
- method;
- terminal success;
- steps;
- complete-cycle latency summary;
- reuse/service/fallback summary;
- terminal error/status;
- video hash if retained; and
- semantic hash.

### 11.5 Run summary and immutable evidence

Each completed run must have:

- expected and actual record counts;
- terminal status;
- all resource caps;
- query/episode identity reconciliation;
- semantic hashes;
- environment/hardware identity;
- exact launch command;
- deviation register;
- analyzer version;
- gate-by-gate pass/fail; and
- next-phase eligibility.

Raw evidence is never rewritten. A correction creates a new attempt with a
parent pointer.

## 12. Monitoring and recovery

### 12.1 Long-run monitoring

After launching an owned long run:

- create a thread heartbeat/monitor when appropriate;
- monitor only the frozen technical fields;
- do not inspect protected outcomes early;
- preserve terminal counts and artifact growth;
- publish concise progress without interpreting outcomes; and
- continue until complete or technical stop.

### 12.2 Technical stop

If a runner stops early:

1. inspect only technical log/summary;
2. preserve all evidence;
3. classify integration, resource, environment, or data failure;
4. compare against the frozen recovery allowance;
5. write a technical-stop report;
6. do not reuse incomplete scientific records;
7. change only the minimum technical issue; and
8. request approval if the recovery changes caps, hardware, scientific
   treatment, or population.

### 12.3 Scientific failure

A valid failed gate is final for this method identity under this protocol.
Record it, preserve it, and consult advisors. Do not search thresholds,
subgroups, seeds, or tasks until a favorable result appears.

## 13. Techniques that keep the project on task

1. One authoritative protocol and one method identity.
2. Phase-level approvals and phase-scoped branches.
3. Immutable configurations with semantic hashes.
4. Protected populations opened in order.
5. Outcome-blind timing before labels and success.
6. Trajectory-level splits and inherited mask identity.
7. Predeclared comparator matching.
8. One offline test opening and one confirmatory run.
9. One optional on-policy correction only.
10. No automatic retry.
11. Query, episode, hour, storage, and memory caps.
12. Stop rules that prevent endless method cycling.
13. Evidence ledgers and deviation registers.
14. Independent analyzer reconciliation.
15. Claim ladder tied to gates.
16. Local/GitHub/TITAN revision reconciliation after every phase.
17. Advisor review after any scientific stop rather than immediate redesign.

## 14. Risk register

| Risk | Detection | Control | Stop consequence |
|---|---|---|---|
| No-noops data lost temporal continuity | P0/P1 | Require original step identity or unfiltered/replayed demos | Stop before labels |
| Expert L1 does not predict success | P7 | Keep it surrogate; require closed-loop transfer | Stop or one P7R |
| Dense policy has expert error | P4 | Use cached-minus-dense regret | Report signed baseline |
| Previous action feature uses oracle expert | P2 tests | Carry model-produced contract history | Fail closed |
| Router feature needs skipped compute | P0/P3 | Chronology audit and full cost | Exclude feature |
| Independent masks miss recursive age | P2/P4 | 1/2/4-query contracts | Stop if unsupported |
| Atomic effects do not combine | P4 | Profile contracts and set calibrator | Stop if unpredictable |
| Gripper events are rare | P1/P4 | Stratify/oversample, diagnostic veto | Stop if uncovered |
| Task identity memorization | P5 | No-instruction and rotating task-heldout diagnostics | Narrow claim/stop |
| Labels are numerical noise | P4 | All-fresh/repeat controls | Stop |
| Regret is not predictable | P4/P5 | Held-out correlation/tail gates | Stop |
| Router falls back too often | P6 | Service floor and net timing | Stop |
| No raw timing reserve | P3 | Snet lower-bound gate | Stop |
| Router overhead erases speed | P3/P6 | Measure complete cycle | Stop |
| Multiple K/V banks exceed GPU cap | Every GPU phase | Stream one contract; serialize masks | Technical stop |
| Stored K/V exhausts disk | P4/P5 | Store compact labels only | Technical stop |
| Comparator is unfaithful/unlicensed | P0/P2 | Code/license audit | Narrow claim/stop |
| Baseline chosen after outcomes | P0/P3 | Freeze matching rule | Invalidate run |
| 500 trials underpowered | P7 | Discordance-based P8 power | Increase pre-run or narrow |
| P8 compute is unaffordable | P7 | Estimate before unblinding P8 | Advisor decision/stop |
| One checkpoint/simulator limits scope | Paper | Scope claims explicitly | No universal claim |
| Literature collision appears | P0/P9 | Repeat primary-source search | Reframe before compute/release |
| Valid negative result triggers redesign loop | Every gate | Scientific stop and advisor review | End protocol |

## 15. Claim ladder

| Highest completed gate | Permitted statement |
|---|---|
| P0 | PAIR is a researched, implementable hypothesis |
| P1 | Query-aligned expert data are available |
| P2 | PAIR software semantics pass synthetic correctness |
| P3 | The cache substrate has enough outcome-blind physical headroom |
| P4 | Actual-stale regret is measurable and preliminarily predictable |
| P5 | A locked router improves held-out offline regret |
| P6 | The full router retains measured net speed |
| P7 | The router improves a development success-latency frontier |
| P8 | PAIR supports the targeted positive paper |
| P9 | The claim is reproducible and publication-ready |

No phase may borrow a stronger statement from a later row.

## 16. Paper mapping if P8 passes

### Method section

- exact cache/provenance state;
- deployment chronology;
- actual-stale recursive intervention;
- exact expert L1 regret;
- atomic groups and profile contracts;
- router and empirical calibration;
- selection/fallback;
- physical cost model.

### Experimental section

- base checkpoint and suites;
- demonstration data/splits;
- comparator fidelity;
- phased protected evaluation;
- complete-cycle timing;
- paired non-inferiority;
- mechanistic ablations;
- resource and reproducibility identities.

### Results section

1. physical headroom;
2. intervention-label validity;
3. offline source-aware prediction;
4. net router overhead;
5. development transfer;
6. four-suite confirmation;
7. ablations and failure cases.

### Limitations

- frozen OpenVLA-OFT checkpoint;
- LIBERO simulator;
- known-task demonstration dependence;
- one hardware platform;
- empirical, not formal, risk calibration;
- no collision/constraint safety measurement;
- recent and fast-moving comparator literature; and
- no automatic transfer to flow-matching VLAs.

## 17. Phase-start checklist

Before every phase:

1. read AGENTS.md and this protocol;
2. confirm phase approval;
3. confirm local and TITAN project paths;
4. inspect only project git status/revision;
5. preserve unrelated changes;
6. authenticate prior phase PASS report;
7. freeze run/config/schema identities;
8. confirm outcome-access boundary;
9. confirm transfer, query, episode, time, storage, and memory caps;
10. coordinate GPU if required;
11. confirm no automatic retry;
12. define expected terminal records and summary;
13. define technical-stop handling;
14. run tests/smoke checks;
15. launch only the frozen work.

After every phase:

1. reconcile counts and hashes;
2. run the frozen analyzer;
3. write PASS, FAIL-SCIENTIFIC, STOP-TECHNICAL, or BLOCKED-AUTHORITY;
4. record deviations;
5. preserve immutable evidence;
6. commit only reviewed compact files if authorized;
7. push without force if authorized;
8. pull the exact revision locally;
9. report changes, verification, remaining uncertainty, and server boundary;
10. stop before the next phase.

## 18. Feasibility and final execution decision

### 18.1 Current assessment

| Dimension | Assessment | Decisive unresolved gate |
|---|---|---|
| Organic novelty | Moderate and defensible | P0 collision refresh |
| Integration | High | P2 recursive real-cache tests |
| Dataset semantics | Moderate | P1 unfiltered eight-step chronology |
| Disk | High | P1 unpacked-size cap |
| GPU memory | Moderate-high | Each phase below 23 GiB |
| Router training | High | Small frozen-backbone model |
| Raw physical headroom | Moderate-low | P3 Snet lower bound |
| Label identifiability | Moderate | P4 repeat/noise gate |
| Offline predictability | Moderate | P4/P5 held-out tail gate |
| Closed-loop transfer | Moderate-low | P7 |
| Confirmatory runtime | Moderate-low | P7 power/cost estimate |
| Positive-paper path | Plausible but conditional | P3, P4/P5, P7, then P8 |

### 18.2 Decision

Proceed only one phase at a time, beginning with P0 after explicit approval.

The method is worth testing because it directly addresses the local failure,
uses existing provenance/cache infrastructure, remains distinguishable from
generic perturbation tolerance, and can be falsified before a large
closed-loop campaign.

It is not yet reasonable to predict a positive paper as more likely than not.
The route becomes genuinely promising only after:

1. P3 proves conservative net physical headroom;
2. P4/P5 prove stable, source-aware regret prediction; and
3. P7 proves closed-loop transfer.

Only P8 establishes the positive result.
