# ACR Version 2 Execution Protocol

Status: **FROZEN BEFORE VERSION 2 IMPLEMENTATION OR OUTCOME COLLECTION**

Freeze date: 2026-08-03

Method: State-Aware Dual-Path Asymmetric Camera Refresh (`SA-DP-ACR`)

Machine-readable freeze: `configs/acr/v2_freeze.json`

## 1. Purpose and amendment boundary

This protocol defines a new positive-paper attempt after ACR Version 1 stopped
negative at its exact Stage 1 gate. It does not alter, relax, or reinterpret
the Version 1 result. A4/A5 outcomes are disclosed exploratory development
inputs. Version 2 must earn new evidence on populations whose ACR outcomes
remain unopened.

The redesign addresses one measured limitation: Version 1 reduced visual CUDA
work but increased total query wall time. It does not claim that the single
conservative failure has a known causal trigger.

Creating this protocol authorizes no implementation, GPU query, simulator
episode, protected-population access, or manuscript change. Each execution
phase requires its own authorization.

## 2. Evidence available at freeze

- A4 upstream FR: Object states `0-9`, seed 0, 97/100 success.
- A5 Stage 1 ACR: Object states `0-2`, seed 0, all three frozen candidates.
- Conservative A5 result: 29/30 success, 26.06% scene reuse, 11.94% visual
  CUDA reduction, and 31.24% query-wall slowdown relative to matched FR.
- Middle/aggressive results: 24/30 and 23/30 success.
- Object states `3-9` have upstream-FR outcomes from A4, but **no ACR or
  SA-DP-ACR outcome**.
- Goal states `0-9` have no ACR outcome and remain independent confirmation.
- All four-suite states `10-49`, seed 7, and reserve seeds 17/27 remain
  protected and unopened.

The canonical diagnosis is `reports/ACR_V2_DIAGNOSIS_REPORT.md`; its immutable
machine record has byte SHA-256
`cf8ea201547030026a204a8d4848ac7d33d3d45d64246a94cdab1d47f3739deb`.

## 3. Frozen research question

Can a training-free dual-path implementation reuse a conservative fixed-scene
camera block while always refreshing the wrist camera, preserve paired task
success within two percentage points, reduce visual CUDA time by at least 10%,
and reduce synchronized policy-query wall time by at least 2%?

For Version 2, a positive result requires all four:

1. paired success non-inferiority under the frozen final analysis;
2. at least 20% realized scene-camera reuse;
3. at least 10% visual backbone/projector CUDA-time reduction;
4. at least 2% synchronized query-wall reduction with a confidence bound
   excluding zero at confirmation/final stages.

The 20%/10% thresholds are deliberately disclosed as A5-informed revisions.
They reflect meaningful measured work—at least one in five scene blocks and
roughly one in ten camera paths avoided—while remaining testable on untouched
confirmation and final evidence. They cannot be changed after this freeze.

## 4. Frozen controller

Use exactly the conservative A4/A5 controller:

| Field | Frozen value |
|---|---:|
| Parent ID | `acr-t25-h2-b30` |
| Scene threshold | 0.2476380718954248 |
| Translation threshold | 0.5479944908411765 |
| Horizon | 2 |
| Hard scene-reuse cap | 0.30 |
| Warm-up | queries 0 and 1 refresh |
| Transition gate | existing gripper-phase veto |
| Wrist policy | always refresh |

Do not add direction reversal, task-specific logic, a mask, a learned router,
another candidate, or a new threshold. The controller's causal decision and
reference-update semantics remain Version 1 exactly.

## 5. Dual-path execution method

### 5.1 Refresh path

When the controller refreshes the scene:

1. call the saved original upstream `_process_vision_features` method once on
   the normal ordered 12-channel scene+wrist pixel tensor;
2. preserve upstream vision-backbone and combined-projector execution exactly;
3. require the returned shape `[1, 512, llm_dim]`;
4. split the projected sequence into the first 256 scene tokens and final 256
   wrist tokens without copying or reordering;
5. store only the detached scene block in the compatible episode cache;
6. return the original combined tensor unchanged to the upstream downstream
   policy path.

This path should be bitwise upstream FR by construction. The correctness gate
must prove it rather than assume it.

### 5.2 Reuse path

When the controller reuses the scene:

1. split out the current wrist six-channel input;
2. invoke the pinned SigLIP and DINOv2 towers for the wrist once;
3. concatenate wrist tower features and invoke the unmodified projector once;
4. load the compatible cached projected scene block;
5. concatenate `[cached_scene, fresh_wrist]`;
6. append current proprioception and execute all downstream policy computation
   unchanged.

The reuse path must produce the same projected sequence and action as Version
1 for identical current inputs and cached scene tokens.

### 5.3 Production hot-path rules

- Install and restore the adapter at episode scope, not by mutating the model
  separately around every query.
- Use one synchronized timing boundary per policy query, identical across FR
  and SA-DP-ACR.
- Structural shape, dtype, device, context, token-order, patch-count, and cache
  identity checks remain online.
- Do not perform redundant CUDA-to-host `.item()` finite scans of intermediate
  projected blocks in production timing mode.
- The bounded correctness mode performs full finite validation.
- Every returned action chunk is checked for finite values before execution;
  failure stops and preserves the attempt.
- No upstream file, checkpoint, weight, prompt, preprocessing rule, or action
  head may change.

### 5.4 Physical and logical accounting

On a refresh, the upstream vision backbone and projector are each one physical
module invocation covering two logical cameras. On reuse, wrist components are
one physical/logical invocation and scene components are zero. Records must
store both physical invocations and logical camera work; neither may be
misreported as the other.

## 6. Required comparisons

1. **Upstream two-view FR:** success and latency oracle.
2. **SA-DP-ACR:** single frozen primary method.
3. **ACR Version 1:** immutable historical development evidence only.
4. After independent confirmation only:
   - Scene-Periodic dual-path ACR at matched scene refresh rate;
   - Scene-Visual dual-path ACR without state/transition gates;
   - current official VLA-Cache compatibility re-audit; execute only if its
     evaluator semantics pass the existing validity requirements.

No baseline may select or retune the primary method.

## 7. Phase V2-A — diagnosis and freeze

Authorized work:

- reconcile all A4/A5 records;
- localize failures without causal overclaiming;
- repeat primary literature/source review;
- freeze this method, machine configuration, splits, gates, and resources;
- correct factual reporting errors.

Exit gate:

- deterministic diagnosis record and tests pass;
- no Version 2 implementation, model query, or rollout occurred;
- protocol/configuration hashes are published and synchronized.

## 8. Phase V2-B — implementation and CPU verification

This phase requires new authorization.

Implement a new adapter without changing Version 1. Minimum tests:

1. original refresh method invoked exactly once;
2. original refresh output returned unmodified;
3. projected scene split/cache ownership and token order;
4. wrist-only reuse component truth table;
5. physical versus logical call accounting;
6. episode-scoped installation and exception restoration;
7. nested/concurrent access rejection;
8. context/shape/dtype/device/patch-count failures stop or refresh safely;
9. production mode contains no intermediate CUDA-to-host synchronization;
10. correctness mode performs full finite checks;
11. action-finite failure preservation;
12. immutable records and recovery identities;
13. all existing 155 repository tests and new tests pass;
14. changed-file Ruff, formatting, mypy, bootstrap, and compilation pass.

Resources: CPU only, no simulator, no model query, no download, 512 MiB new
artifact cap.

## 9. Phase V2-C — bounded correctness and latency

This phase requires separate authorization after V2-B passes.

Use at most one responsibly selected GPU, one model process, 48 real-model
queries, zero simulator episodes, 3,600 seconds, and 512 MiB of artifacts.

### 9.1 Correctness proofs

- refresh projected tokens/actions bitwise equal upstream FR;
- refresh returned tensor is the exact original upstream result;
- reuse projected tokens/actions bitwise equal Version 1 for identical inputs
  and cache;
- scene reuse performs zero scene computation and one fresh wrist path;
- current proprioception and downstream execution remain fresh;
- all structural/cache/error paths fail safely;
- checkpoint and source trees restore exactly.

### 9.2 Paired timing design

- six untimed warm-up queries: two each for FR, DP refresh, and DP reuse;
- 12 timed repetitions per path;
- deterministic counterbalanced path order;
- identical synchronized start/end boundaries;
- report median, mean, all repetitions, CUDA time, wall time, and overhead;
- no outlier deletion.

All must pass:

- DP refresh median wall time no more than 1.05 times FR;
- DP reuse median wall time no more than 0.98 times FR;
- expected weighted wall time using A5's fixed 26.055% reuse no more than 0.98
  times FR;
- all correctness and integrity proofs pass.

Failure stops before any Version 2 rollout.

## 10. Phase V2-D — fresh paired Object development

This phase requires separate authorization after V2-C passes.

Population: Object tasks `0-9`, states `3-9`, seed 0. Run contemporaneous
upstream FR and the single frozen SA-DP-ACR method exactly once: 70 episodes
per policy, 140 attempts total.

- Deterministically alternate policy order by task/state.
- Do not expose success or efficiency summaries until all 140 terminal records
  exist; intermediate checks are technical counts/resources only.
- Scientific failures are never rerun.
- Technical attempts remain immutable and use the next identity only under a
  predeclared recovery rule.

Development passes only if:

- 70/70 terminal episodes per policy;
- SA-DP-ACR success is no more than two episodes below paired FR;
- no task loses more than one success relative to FR;
- scene reuse is at least 20%;
- visual CUDA time falls by at least 10%;
- synchronized query wall time falls by at least 2%;
- all physical/logical call and cache invariants pass;
- zero technical failures occur.

Resources: one GPU/process, 140 attempts, 43,200 seconds, 2 GiB artifacts, no
download. Failure stops; do not change the method or open Goal.

## 11. Independent Goal confirmation

Only after V2-D passes and the user separately authorizes confirmation, run
FR and frozen SA-DP-ACR once on Goal tasks `0-9`, states `0-9`, seed 0: 100
episodes per policy.

Use the same paired/counterbalanced, outcome-blinded execution. Confirmation
passes only with:

- 100 paired terminal episodes;
- success no more than two episodes below FR;
- no task loss above one episode;
- scene reuse at least 20%;
- visual CUDA reduction at least 10%;
- query-wall reduction at least 2%;
- one-sided 97.5% lower confidence bound for latency reduction above zero;
- zero technical/invariant failures.

No tuning is permitted on Goal. Failure stops before baselines, transfer,
power planning, or final evaluation.

## 12. Subsequent positive-paper route

After Goal confirmation:

1. run matched Scene-Periodic and Scene-Visual dual-path baselines on Goal;
2. run paired FR/SA-DP-ACR on LIBERO-10 without tuning;
3. determine final sample size from Goal/LIBERO-10 paired discordance;
4. freeze statistical code, tables, method hash, and final `N`;
5. request separate authorization for the protected four-suite final
   evaluation;
6. run only FR and frozen SA-DP-ACR on the balanced states `10-49`, seed 7;
7. classify the paper positive only if success non-inferiority and all Version
   2 efficiency gates pass.

The final success analysis retains the original two-percentage-point margin,
Newcombe paired interval, stratified paired bootstrap, exact discordance
sensitivity analysis, and suite loss guard. Reserve seeds 17/27 remain locked.

## 13. Stop and on-task controls

Stop immediately for parity failure, latency-gate failure, cache/accounting
failure, source/checkpoint change, resource risk, cap exhaustion, premature
population access, or an unfavorable scientific gate.

Controls:

- one active phase;
- semantic configuration hash before implementation;
- immutable attempts/queries/episodes/manifests;
- outcome-blinded complete-batch analysis;
- no automatic retries or selective exclusions;
- paired contemporaneous latency oracle;
- physical and logical compute accounting;
- exact local/GitHub/TITAN revision reconciliation;
- no manuscript edit until the complete evidence package is approved;
- no write outside `/home/ved/SAVR` on TITAN.

## 14. Frozen interpretation

Version 2 is a credible positive-paper attempt, not a guarantee. A positive
paper requires success preservation and realized acceleration on independent
and protected evidence. Passing only call counts, visual CUDA savings, or a
development sample is insufficient.
