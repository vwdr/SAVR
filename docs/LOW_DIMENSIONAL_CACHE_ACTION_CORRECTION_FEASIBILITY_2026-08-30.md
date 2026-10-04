# Low-Dimensional Cache Action Correction: Feasibility and Project Scope

**Date:** 2026-08-30  
**Status:** Research assessment; no implementation or GPU experiment authorized  
**Question:** Can the action-relevant effect of recursively mixed-age VLA K/V
reuse be corrected directly, without reconstructing the full fresh cache?

> **Execution document:** The reviewed phase gates, frozen method family,
> resource limits, and stopping rules are defined in
> `docs/CACHE_ACTION_CORRECTION_EXECUTION_PROTOCOL_V1.md`.

## 1. Bottom-line answer

**Yes, this is feasible enough to justify a staged experiment.** It is not yet
proven likely enough to justify a large execution campaign.

The project has a clear causal chain:

1. SAVR through PAIR established that downstream visual K/V reuse provides real
   physical acceleration but can corrupt actions and terminal task success.
2. Handcrafted thresholds and a learned risk router could not reliably predict
   that harm from compact proxy features.
3. The current projected visual tokens still exist at every query, even when
   selected downstream K/V entries are reused.
4. Instead of predicting whether reuse is safe, the new method directly learns
   the dense-minus-cached action correction from those rich current features,
   the cached action representation, and exact reuse provenance.
5. Because actions affect later observations and cache contents, one
   predeclared on-policy teacher-labeling round is included before final
   confirmation.

This directly addresses the problem exposed by the negative experiments. It is
not a renamed router and does not require training the 7B backbone.

## 2. The scientific contribution can be simple and novel

Novelty does not require inventing residual learning or K/V caching. The
original scientific study is:

> Determine whether selective VLA cache corruption is behaviorally
> compressible: can the action-relevant effect of a high-dimensional,
> recursively mixed-age internal cache be recovered by a small output-space
> corrector using already-available fresh and cached representations?

If successful, the study adds four pieces of new knowledge:

1. whether full fresh-K/V reconstruction is necessary for reliable cached VLA
   control;
2. whether current OpenVLA-OFT projected tokens are sufficient to repair the
   action-relevant error;
3. whether exact tile/layer cache provenance and fresh--stale feature deltas
   materially improve correction; and
4. whether offline correction requires one distribution-matched on-policy
   aggregation stage to survive closed-loop deployment.

This is a legitimate original contribution even though related papers use
residual heads for other problems.

## 3. Evidence supporting feasibility

### 3.1 Our system already exposes the required tensors

The authenticated local OpenVLA-OFT path already provides:

- 512 current projected visual tokens plus one proprioceptive token, each width
  4096;
- 56 cached-path action hidden states, each width 4096;
- an 8x7 cached normalized action chunk;
- exact source query, age, camera, tile, and onset layer for reused entries;
  and
- the dense action from an isolated teacher branch during training collection.

No new image encoder and no 7B-model backpropagation are required.

### 3.2 A physical acceleration budget already exists

BRACE measured a 12.23% complete-cycle reduction for its physically valid
cached profile. PAIR P3R estimated a 17.05% conservative net saving for the
selected profile. The adapter does not need to create acceleration; it needs to
preserve enough of that measured saving while restoring the dense action.

With the local dense path near 1.22 seconds, the observed 12--17% range
corresponds roughly to 146--207 milliseconds of gross headroom. A small adapter
should target far less than this, with a conservative initial overhead budget
of about 25--40 milliseconds. The actual decision must use measured complete-
cycle latency, not this arithmetic estimate.

### 3.3 Existing P4B errors are substantial and concentrated

The completed P4B intervention records provide scalar—but not vector—evidence:

- 776 valid intervention records;
- median mean normalized-coordinate distortion: 0.1877;
- median maximum-coordinate distortion: 0.6094;
- median maximum-to-mean distortion ratio: 2.94;
- 90th-percentile maximum-to-mean ratio: 4.17; and
- median gripper-coordinate error: 0.4143.

This says that reuse does not merely add tiny uniform noise; a smaller subset
of action entries can be much more corrupted than the overall average. That is
consistent with an action-repair problem, but it does **not** prove a low-rank
or learnable residual. Raw 56-dimensional residual vectors were intentionally
not retained, so residual rank and predictability are still open questions.

P4B also contained zero designated gripper-transition contracts among its 240
structured contracts. The new study must deliberately include gripper
transitions; otherwise it would repeat a known coverage gap.

### 3.4 Related research supports the components

- [Latent Bridge](https://arxiv.org/html/2605.02739) demonstrates that current
  visual information and on-policy aggregation can repair stale VLA feature/KV
  paths, while also showing that clean offline training alone leaves a
  deployment gap.
- [A2C2](https://arxiv.org/abs/2509.23224) demonstrates that a small residual
  head using current observations can improve closed-loop performance of stale
  action chunks.
- [Action ControlNet](https://arxiv.org/abs/2606.25985) finds that injecting a
  residual close to the action output can be more robust than perturbing early
  task representations.
- [ViTaR](https://arxiv.org/abs/2608.15816) independently supports bounded
  residual modulation on a frozen VLA when the base policy lacks information
  relevant to execution.
- Frozen visual encoders with small downstream robot controllers have long
  been practical; for example,
  [Radosavovic et al.](https://proceedings.mlr.press/v205/radosavovic23a/radosavovic23a.pdf)
  train compact controllers over a frozen large visual representation.

These papers support feasibility. They also require us to state our narrower
contribution accurately: correction of same-query, selective, recursively
mixed-age internal K/V reuse without reconstructing the full cache.

## 4. Why this is different from earlier failed methods

| Earlier method | What failed | What changes now |
|---|---|---|
| SAVR/VOR/ACR thresholds | Visual/state similarity did not identify all action-critical changes | No threshold is asked to predict task safety |
| BRACE | Efficient reuse existed, but cached action parity failed | Action corruption becomes the supervised target |
| PAIR | Compact handcrafted features could not rank future expert regret reliably on independent trajectories | The model receives rich tile-level projected features and predicts the correction vector, not a scalar risk score |
| PAIR fallback | A gate could only choose cached versus dense; it could not restore missing information | Current features are used to modify the served action directly |
| Offline-only reasoning | Expert trajectories did not represent states induced by an altered policy | One on-policy dense-teacher aggregation stage is part of the method |

The main lesson is not that learning automatically solves the problem. It is
that the previous target and information interface were wrong. PAIR tried to
infer hidden harm from summaries; the new method learns the observable action
effect from substantially richer representations.

## 5. Minimal method

The project should begin with one fixed, physically validated reuse profile and
one fixed maximum source age. It should not simultaneously learn the cache
selector.

At query $q$, define:

- $V_q$: current projected visual features;
- $V_q^{src}$: the projected source features corresponding to reused
  camera/tile entries;
- $H_q^C$: cached-path action hidden states;
- $A_q^C$: cached 8x7 action chunk;
- $s_q$: current proprioception; and
- $p_q$: exact source ages and onset-layer provenance.

The dense teacher provides $A_q^D$. The target is

$$
\Delta A_q^* = A_q^D-A_q^C.
$$

A compact corrector predicts

$$
\widehat A_q
= A_q^C
+ b\odot\tanh R_\phi
  \left(\operatorname{tile}(V_q),
        \operatorname{tile}(V_q-V_q^{src}),
        \operatorname{step}(H_q^C),
        A_q^C,s_q,p_q\right),
$$

where $b$ bounds each output dimension.

### Compact representation

- Pool each 16x16 camera grid into sixteen 4x4 tiles: 32 visual tokens total.
- Align each tile with its source-feature delta and four onset-layer ages.
- Reshape 56 action tokens into eight groups of seven and pool each group into
  one action-step token.
- Project both streams to width 128 or 256.
- Use at most one or two small cross-attention blocks.
- Predict six bounded motion residuals plus a separately treated gripper
  correction for each of eight action steps.

The output layer starts at zero, so the untrained adapter exactly reproduces
the cached policy. Dense queries bypass the adapter completely.

## 6. The first decisive experiment is not a rollout

Before building a closed-loop method, collect a small, balanced tensor set and
answer whether the residual is actually compressible and predictable.

### Required population

- complete trajectories from every LIBERO suite;
- multiple tasks per suite;
- horizons/source ages 1, 2, and 4;
- ordinary motion and deliberately sampled gripper transitions;
- exact-repeat and all-fresh controls;
- whole-trajectory train/calibration separation; and
- no locked-test trajectories.

### Analyses

1. Perform PCA/SVD on the 56-dimensional dense-minus-cached residuals.
2. Report dimensions required for 80%, 90%, 95%, and 99% variance.
3. Fit nested models in order:
   - cached action only;
   - cached action plus provenance;
   - cached action plus cached hidden state;
   - current feature delta only;
   - current feature delta plus cached state;
   - complete candidate.
4. Compare ridge regression, a small MLP, and one small cross-attention model.
5. Evaluate held-out complete trajectories with:
   - weighted action error;
   - first-action error;
   - translation and rotation error;
   - gripper sign/threshold disagreement;
   - 90th/95th-percentile error; and
   - per-suite, task, horizon, and phase breakdowns.

This screen prevents another long project based on a favorable average. If a
small model cannot substantially reduce held-out first-action, gripper, and
tail error, the direction stops before closed-loop work.

## 7. Closed-loop training is necessary but bounded

### R0: offline exact-recursion model

Train the minimal selected adapter on exact recursive cache states generated
along demonstration trajectories. Dense OpenVLA-OFT is the primary teacher.
Expert actions remain a diagnostic rather than an equal target, because dense
and expert actions can be different valid choices.

### R0 pilot

Run a small paired development population comparing:

- dense OpenVLA-OFT;
- the identical uncorrected cache profile; and
- R0-corrected caching.

This is the first direct check of whether offline error recovery survives
closed-loop execution.

### R1: one on-policy aggregation round

If R0 passes the offline screen but shows a clear offline-to-closed-loop gap,
execute R0 and query an isolated dense teacher at the actual states it visits.
Retrain once on a frozen mixture of demonstration and R0-induced states. This
is the cache-specific version of the distribution-matching principle in
[DAgger](https://proceedings.mlr.press/v15/ross11a/ross11a.pdf).

Only one predeclared aggregation round is allowed in the initial project. If
R1 still cannot restore paired success, output-space correction is insufficient
to stabilize the recursive cache and the direction stops.

## 8. What a positive result means

The paper does not need to beat dense OpenVLA-OFT's already high success rate.
A positive reliability--efficiency result is:

1. at the identical cache profile, the corrector significantly improves paired
   task success over uncorrected reuse;
2. corrected reuse is non-inferior to dense OpenVLA-OFT under a predeclared
   success margin;
3. corrected reuse retains a measured complete-cycle latency reduction after
   including all adapter cost;
4. the result confirms on independent trajectories/tasks; and
5. ablations show the gain depends on fresh--stale features and cached context,
   rather than the adapter acting as a separate small policy.

This would support the scientific conclusion that action-relevant repair can
be cheaper than reconstructing the full fresh internal cache.

## 9. Project phases

### Phase C0 — freeze and audit

- freeze one cache profile, reset horizon, data splits, success margin, timing
  method, maximum model size, and stop rules;
- update the literature collision table;
- verify that no locked-test outcome is opened.

### Phase C1 — tensor and systems feasibility

- capture current/source tile features, cached action hidden states, and dense
  residuals;
- verify exact tensor identities and isolated dense/cache branches;
- measure peak memory, extraction overhead, projected storage, and runtime;
- stop if the one-GPU boundary is violated.

### Phase C2 — residual-structure screen

- collect a small balanced development dataset;
- run PCA/SVD and nested simple predictors;
- test gripper, first-action, tail, suite, task, and horizon behavior;
- stop if the complete model does not clearly outperform simple controls on
  held-out trajectories.

### Phase C3 — R0 adapter

- collect the bounded training population;
- train the smallest passing adapter offline;
- freeze one model before calibration evaluation;
- run required cache-specificity ablations.

### Phase C4 — physical integration

- insert the adapter after the cached action representation;
- verify dense bypass, bounded outputs, numerical determinism, memory, and
  complete-cycle latency;
- stop if a conservative net acceleration remains unavailable.

### Phase C5 — R0 paired closed-loop pilot

- compare dense, uncorrected cache, and R0 correction on identical tasks,
  initializations, and seeds;
- inspect the predeclared offline-to-closed-loop transfer gate once.

### Phase C6 — R1 distribution matching

- if authorized by the frozen protocol, collect one R0-induced dataset with a
  shadow dense teacher;
- retrain once and freeze R1;
- no iterative rescue loop.

### Phase C7 — independent confirmation

- evaluate dense, uncorrected cache, R0, and R1 under paired, independently
  frozen conditions;
- report success intervals, per-task results, complete-cycle latency, adapter
  overhead, reuse, and every failure.

### Phase C8 — paper decision

- positive route if reliability and physical efficiency both pass;
- scoped representation finding if only the offline hypothesis passes;
- stop and preserve evidence if closed-loop or latency gates fail.

## 10. Resource feasibility

The P1 audit contains 2,000 trajectories and 41,447 eligible query points.
Approximately 1,400 trajectories are available for training, with separate
calibration and locked-test populations.

A compact record containing 32 pooled current tiles and eight pooled action
hidden groups in BF16 is roughly 320 KiB before metadata. Adding tile-aligned
source deltas could bring a complete record near 0.5--0.6 MiB. A 4,000-query
screen is therefore a few GiB; a larger 25,000--30,000-query training set is on
the order of 13--18 GiB. Both are manageable if data is streamed and full K/V
tensors are not retained.

The official [OpenVLA-OFT resource table](https://openvla-oft.github.io/)
reports 16.2 GB for two-camera/proprioceptive LIBERO inference and 25.7 GB for
batch-one LoRA training. The project therefore uses:

- one forward-only VLA process for feature/teacher collection;
- CPU/disk streaming of compact records;
- a separate adapter-only training process without the 7B model loaded; and
- one-GPU evaluation with the adapter's real overhead included.

This fits the established TITAN operating boundary in principle. Exact peak
memory and wall time remain C1 measurements, not assumptions.

## 11. Main risks and honest mitigations

| Risk | Why it could happen | Mitigation/decision |
|---|---|---|
| Residual is not low-dimensional or predictable | Cache corruption changes attention nonlinearly across layers | C2 PCA plus simple held-out predictors; stop early |
| Repeat of PAIR overfitting | Tasks and trajectories are correlated | Whole-trajectory splits, nested controls, independent calibration |
| Adapter becomes a standalone policy | Fresh features alone may predict actions | Current-only and cached-only ablations are mandatory |
| Offline gains fail closed loop | Policy and cache induce new state distributions | One predeclared R1 dense-teacher aggregation round |
| Internal cache drift continues | Output correction does not update K/V | Fixed short source age and dense reset; stop if R1 cannot stabilize it |
| Gripper failures are hidden | Prior confirmation lacked gripper transitions | Deliberate stratified transition population and separate metric/head |
| Speed disappears | Feature fusion is too large | Small-width architecture and physical timing before rollout scale-up |
| Near-ceiling dense baseline | Improvement over dense may be impossible | Claim non-inferior reliability plus speed, not higher absolute success |
| Literature overlap | Residual VLA adapters and feature bridges exist | Claim the exact mixed-age cache/action-space question and compare directly |
| Repeated rescue cycle | Many flexible redesigns can manufacture optimism | One architecture selection, one R1 round, one independent confirmation |

## 12. Final judgment

This direction is **logically connected to the negative results, technically
implementable with the existing architecture, and scientifically capable of
producing a positive paper**.

The strongest reason to try it is not that residual adapters are fashionable.
It is that our prior experiments isolated a specific mismatch:

- the accelerated path has real compute value;
- the cache can corrupt the action;
- proxy-based avoidance does not generalize; and
- fresh projected evidence is still available before the corrupted downstream
  decision.

Directly learning the low-dimensional behavioral correction is the simplest
remaining intervention that preserves the measured speed substrate while
addressing the observed failure.

The probability of success cannot be responsibly quantified before C2. The
project should proceed only through the small tensor/rank/predictability screen
first. That screen is inexpensive relative to closed-loop evaluation and will
tell us whether the core hypothesis is real before another large experimental
commitment.
