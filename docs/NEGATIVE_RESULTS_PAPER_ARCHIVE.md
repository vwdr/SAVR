# SAVR Negative-Results Evidence Archive

Status: EVIDENCE PRESERVATION DOCUMENT — NOT A MANUSCRIPT CLAIM AUDIT

Last updated: 2026-08-02

## 1. Purpose

This document preserves the evidence needed if the project pivots to an
honest negative-results paper. It consolidates the tested SAVR variants,
predeclared gates, observed outcomes, diagnostic findings, integrity evidence,
and defensible claims. It does not replace the immutable run artifacts or
authorize reinterpretation of a failed gate as a positive result.

The machine-readable companion is
`docs/evidence/negative_results_summary.csv`. Blank CSV cells mean that the relevant
aggregate was not reported in the tracked source report; they are not zeros.

## 2. Scope and frozen reference system

- Base policy: OpenVLA-OFT, continuous L1 action head, eight-action chunks.
- Checkpoint: `openvla-7b-oft-libero-four-suite`.
- Checkpoint revision:
  `638918f3d1c2e43a39a8a20772bdb8b91835e4b7`.
- OpenVLA-OFT revision:
  `e4287e94541f459edc4feabc4e181f537cd569a8`.
- LIBERO revision: `8f1084e3132a39270c3a13ebe37270a43ece2a01`.
- Evaluated suite: LIBERO-Spatial, tasks `0-9`.
- Inputs: third-person image, wrist image, and current eight-dimensional
  proprioception.
- Cached object: projected visual patch embeddings before the current
  proprioception token.
- Final holdout: initial-state IDs `10-49`, seeds `7`, `17`, and `27`; never
  executed or inspected.

## 3. Chronology of tested methods

### 3.1 Full Refresh reference

Full Refresh recomputed both camera representations on every policy query. It
completed 100/100 calibration episodes successfully and produced 1,309 query
traces. This is the performance oracle for the tested split.

### 3.2 SAVR 1.0: whole-prefix adaptive reuse

SAVR 1.0 cached the complete two-camera projected visual prefix. The controller
used a two-camera image-change average, robot-state change, action change, and
a maximum reuse horizon. Nine settings crossed offline skip targets of 25%,
50%, and 75% with horizons 2, 4, and 8.

All 900 settings/episode combinations terminated without infrastructure
failure, but every configuration lost substantially more success than the
frozen two-percentage-point allowance. The least-degrading configuration,
`savr-s25-h2`, achieved 52/100 success at 34.69% online skip.

### 3.3 SAVR 2.0: safety-constrained whole-prefix reuse

SAVR 2.0 replaced camera averaging with independent local camera gates and
added grouped state/action signals, gripper-transition vetoes, warm-up,
stable-fresh history, isolated reuse, and hard prefix budgets.

The Stage 1 result exposed the central trade-off:

| Configuration | Success | Queries | Reuses | Skip | Gate result |
|---|---:|---:|---:|---:|---|
| `savr2-b05` | 30/30 | 388 | 0 | 0.00% | Failed minimum skip |
| `savr2-b10` | 29/30 | 461 | 31 | 6.72% | Failed exact success |
| `savr2-b15` | 27/30 | 473 | 50 | 10.57% | Failed exact success |

Conservative behavior preserved success only when it performed no reuse.
Useful skip rates already coincided with task failures.

### 3.4 SAVR 3.0: post-hoc localized safety redesign

SAVR3 started from `savr2-b15`, vetoed translation direction reversals, and
reduced the wrist-camera threshold to `0.375`. The design was openly labeled
post-hoc and was evaluated once on the policy-specific fresh states `3-9`.

It achieved 69/70 success but only 9/944 full-prefix reuses (0.9534%). It
therefore missed both its exact 70/70 success gate and its 5% minimum skip
gate. The unsuccessful episode was task 9/state 4; it reached the episode
horizon with one reuse among 28 queries and no technical failure.

## 4. Consolidated quantitative evidence

### 4.1 SAVR 1.0 grid

| Configuration | Success | Online skip | Difference from 100/100 FR |
|---|---:|---:|---:|
| `savr-s25-h2` | 52/100 | 34.69% | -48 pp |
| `savr-s25-h4` | 21/100 | 43.29% | -79 pp |
| `savr-s25-h8` | 23/100 | 47.28% | -77 pp |
| `savr-s50-h2` | 12/100 | 56.62% | -88 pp |
| `savr-s50-h4` | 4/100 | 63.26% | -96 pp |
| `savr-s50-h8` | 0/100 | 65.21% | -100 pp |
| `savr-s75-h2` | 3/100 | 64.14% | -97 pp |
| `savr-s75-h4` | 1/100 | 75.00% | -99 pp |
| `savr-s75-h8` | 0/100 | 83.68% | -100 pp |

### 4.2 Cross-version summary

| Method | Population | Best safety/efficiency observation | Interpretation |
|---|---|---|---|
| FR | 100 episodes, states 0-9 | 100/100, 0% skip | Dense reference |
| SAVR1 | 9 x 100 episodes | 52/100 at 34.69% skip | Aggressive whole-prefix reuse is unsafe |
| SAVR2-b05 | 30 episodes, states 0-2 | 30/30 at 0% skip | Safe but computationally identical to FR |
| SAVR2-b10 | 30 episodes, states 0-2 | 29/30 at 6.72% skip | Useful skip missed safety gate |
| SAVR2-b15 | 30 episodes, states 0-2 | 27/30 at 10.57% skip | Higher skip caused larger loss |
| SAVR3 | 70 episodes, states 3-9 | 69/70 at 0.95% skip | Strong recovery of success, insufficient savings |

These populations were used for different predeclared purposes. The table is
descriptive and must not be presented as a single randomized head-to-head
experiment.

## 5. Reproducible diagnostic findings

### 5.1 Closed-loop calibration drift

All nine SAVR1 online settings skipped more often than predicted by replaying
FR traces. For the 25% target family, online overshoot ranged from positive
9.71 to positive 22.30 percentage points. Once reuse changes an action, future observations
and gate signals follow a different trajectory from the FR trace; offline FR
replay is therefore not an online safety oracle.

### 5.2 Camera averaging hid wrist dynamics

Among 783 reuse queries for `savr-s25-h2`, 374 had at least one camera above
the shared image threshold even though the two-camera mean was below it. The
individual exceedances were:

- wrist camera: 372;
- third-person camera: 2.

This directly invalidates simple cross-camera averaging as a conservative
reuse gate. It does not prove that every wrist exceedance caused failure.

### 5.3 First reuse changed the predicted action

Among 87 episodes with comparable FR/SAVR action prefixes, the first action
hash mismatch occurred exactly at the first reuse in all 87. This proves that
whole-prefix reuse changed the policy output immediately. It does not measure
action magnitude or independently establish causal responsibility for task
failure.

### 5.4 Early and consecutive reuse were associated with failure

For `savr-s25-h2`, success rose from 30.4% when first reuse occurred by query 2
to 75.0% when first reuse occurred at query 5 or later. Every failed episode
reached two consecutive reuses, allowing up to two eight-action chunks to be
generated from stale visual features.

These are exploratory associations rather than randomized causal estimates.

### 5.5 Safety constraints eliminated nearly all reuse

SAVR3 generated only nine reuse decisions across 944 queries. Its trigger
totals included 279 wrist-image changes versus only eight full-image changes.
The required stable-fresh condition triggered 899 times, the prefix budget
triggered 424 times, and translation reversal triggered 412 times. These
counts show why the final whole-prefix controller became computationally
equivalent to FR on nearly every query.

## 6. System and efficiency evidence

The 50-episode FR pilot measured, after warm-up:

- mean policy-query CUDA time: `1267.44 ms`;
- mean vision-backbone time: `191.12 ms`;
- mean visual-projector time: `10.08 ms`;
- backbone plus projector: `15.874%` of query CUDA time.

Eliminating all visual work would therefore have an observed query-time
ceiling near 15.87% before cache overhead. This ceiling is modest but
nontrivial. A negative-results paper should report it so readers understand
both the potential value and the fundamental limit of visual-only inference
optimization on this stack.

## 7. Correctness and integrity evidence

- Wrapped FR produced bitwise-identical actions to unmodified OpenVLA-OFT.
- Real reuse invoked zero vision-backbone and projector calls.
- Reuse retained current proprioception and exact downstream execution.
- All reported calibration runs used one selected GPU.
- Technical failures were kept separate from scientific task failures.
- All unsuccessful episodes were retained without rerun or relabeling.
- Protected checkpoint files were restored to their original hashes.
- No final-holdout outcome was inspected.
- No unrelated university-server file, process, service, or allocation was
  modified by the recorded workflows.

## 8. Claims supported by the evidence

The following claims are defensible with careful scope:

1. Whole two-camera visual-prefix reuse can strongly degrade closed-loop
   manipulation success even when adjacent observations appear redundant.
2. FR-trace replay can substantially underestimate online reuse after the
   controller changes the trajectory distribution.
3. Averaging camera-change scores concealed frequent wrist-camera dynamics in
   this two-view OpenVLA-OFT/LIBERO-Spatial setting.
4. The first comparable full-prefix reuse changed the predicted action in all
   87 analyzable episodes.
5. Safety constraints recovered success but reduced full-prefix reuse below a
   meaningful level: SAVR3 achieved 69/70 success at only 0.95% skip.
6. In the tested operating points, no whole-prefix SAVR variant satisfied its
   predeclared success-efficiency gate.

## 9. Claims not supported

Do not claim:

- that visual caching is universally unsafe;
- that SAVR3 is non-inferior to FR;
- that the single SAVR3 failure was caused solely by its reuse;
- that wrist-camera changes are always more important than scene-camera
  changes outside this benchmark/model;
- that all possible thresholds or cache granularities were tested;
- speedup from skipped calls without matched end-to-end measurement;
- statistical generalization to the untouched final holdout;
- superiority or equivalence to VLA-Cache, ADP, or another external method.

## 10. Viable negative-paper framing

### Candidate thesis

Training-free whole-prefix reuse in a chunked VLA exhibits a sharp
success-efficiency frontier: permissive gates alter actions and degrade
closed-loop success, while conservative gates collapse toward dense visual
refresh. Multi-camera aggregation and offline replay obscure this failure
mode.

### Potential contributions

1. A controlled empirical study spanning 100 FR outcomes, 900 SAVR1 outcomes,
   90 staged SAVR2 outcomes, and 70 SAVR3 validation outcomes, all with
   preserved negative evidence.
2. A reproducible demonstration of closed-loop calibration shift between FR
   replay and online reuse.
3. A camera-specific diagnosis showing that cross-camera averaging can hide a
   dynamic wrist view.
4. Correctness instrumentation that proves exactly which visual components
   were skipped while keeping current proprioception.
5. Design guidance: factor cache decisions by sensor or token group rather
   than treating the multi-camera prefix as an indivisible object.

### Suggested paper organization

1. Introduction: efficiency motivation and negative-result question.
2. System and whole-prefix caching formulation.
3. Experimental safeguards and immutable evidence protocol.
4. SAVR1 success-efficiency collapse.
5. Forensic diagnosis of closed-loop drift and camera aggregation.
6. Conservative SAVR2/SAVR3 redesigns and why they collapse toward FR.
7. Lessons for future multi-view VLA caching.
8. Limitations, reproducibility, and ethical reporting of negative results.

## 11. Recommended figures and tables for a negative paper

- Success versus online skip scatter across all SAVR1/SAVR2/SAVR3 settings.
- Offline replay skip versus online skip for the nine SAVR1 settings.
- First-reuse timing versus success for `savr-s25-h2`.
- Camera-specific concealed-threshold counts (wrist 372, scene 2).
- A chronological table of each redesign, its added safeguard, and its gate.
- A component diagram proving where projected visual tokens were cached.
- A trigger-frequency plot for SAVR3 explaining the 0.95% skip rate.

## 12. Evidence locations and hashes

| Evidence | Repository location or TITAN path | Integrity identifier |
|---|---|---|
| Phase 6 report | `reports/PHASE6_CALIBRATION_REPORT.md` | FR trace input `c1724072a9108a77a7c8cec936f4a7e79239dca68aac75a288ca3d4638de9804` |
| Phase 6 raw FR | `/home/ved/SAVR/results/phase6-fr-signals-v1` | Recorded in Phase 6 manifest |
| Phase 6 raw SAVR grid | `/home/ved/SAVR/results/phase6-savr-grid-v1` | Recorded in Phase 6 manifests |
| Phase 6R-A diagnosis | `reports/PHASE6R_A_DIAGNOSIS_REPORT.md` | `9416014d72c661425326317ead68ef2e486fa0f600b147cafe600b2e607c2ccf` |
| Phase 6R-D report | `reports/PHASE6R_D_STAGE1_REPORT.md` | Summary `61a0c9ddfb263ba2123da3dd08500260eba6a454bf335f4830022a81c33a9ebe` |
| Phase 6R-D raw run | `/home/ved/SAVR/results/phase6r-d-stage1-v1` | Manifest `bb451a995796ffe0701f2d01179c6dcd7e0bb93d085041d638758c576864f60b` |
| Phase 6S report | `reports/PHASE6S_D_VALIDATION_REPORT.md` | Summary `bb54eef3620f136c6fcd121b573f58641f4677d92ff02e31ba2f33aba13ad046` |
| Phase 6S raw run | `/home/ved/SAVR/results/phase6s-d-validation-v1` | Manifest `2703dd2939fbd75b5a4ddc08625c2c79451336515abb7098e83ca64cf5553234` |
| Phase 6S analysis | `/home/ved/SAVR/artifacts/phase6s/phase6s_d_analysis.json` | `de570a1b79c7e7e50bf5193f5bf2d2f7048c2336abf10c0dd0b460db51f3e789` |

Large raw results remain on TITAN and are intentionally not committed to Git.
The tracked reports, configurations, code revisions, and hashes provide the
index needed to verify them.
