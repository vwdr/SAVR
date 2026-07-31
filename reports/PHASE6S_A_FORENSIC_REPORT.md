# Phase 6S-A Forensic Localization Report

Status: COMPLETE — DESIGN-SPLIT ANALYSIS ONLY

Date: 2026-07-31

## Scope

This analysis uses only the immutable Phase 6R-D Stage 1 records from
LIBERO-Spatial tasks `0-9`, initial-state IDs `0-2`, and seed `0`. It introduces
no new simulator, model, GPU, or final-holdout outcome. The source run summary
SHA-256 is
`61a0c9ddfb263ba2123da3dd08500260eba6a454bf335f4830022a81c33a9ebe`.

## First unsafe-reuse localization

The first reuse on each unsuccessful trajectory was compared with the matching
`savr2-b05` trajectory. `b05` made no reuse decisions and succeeded on all 30
Stage 1 episodes.

| Failed episode | First action-hash mismatch | First reuse | Translation reversal | Wrist score | Full-image score |
|---|---:|---:|---|---:|---:|
| `savr2-b10_task_08_state_02` | 10 | 10 | yes | 0.35490 | 0.28877 |
| `savr2-b15_task_01_state_00` | 6 | 6 | yes | 0.23585 | 0.17355 |
| `savr2-b15_task_04_state_02` | 0 | 11 | no | 0.51963 | 0.21031 |
| `savr2-b15_task_09_state_01` | 10 | 10 | no | 0.390625 | 0.18562 |

The first unsafe reuse in the first two paths is excluded by a translation
direction-reversal veto. The first unsafe reuse in the other two paths is
excluded by lowering the wrist-camera threshold below their observed scores.

## Disclosed design search

Starting from `savr2-b15`, the analysis vetoed any translation reversal and
tested the fixed wrist-threshold grid below. The denominator is all 473 `b15`
Stage 1 policy queries.

| Wrist cap | Retained reuses | Retrospective skip | Episodes with reuse | Reuses on failed paths |
|---:|---:|---:|---:|---:|
| 0.300 | 20 | 4.23% | 19 | 3 |
| 0.325 | 22 | 4.65% | 21 | 3 |
| 0.350 | 23 | 4.86% | 22 | 3 |
| 0.375 | 24 | 5.07% | 23 | 3 |
| 0.400 | 26 | 5.50% | 24 | 4 |

The frozen selection rule is the smallest 0.025-grid wrist cap retaining at
least 5% retrospective reuse while filtering every first unsafe reuse on the
four failed trajectories. It selects `0.375`.

## SAVR3 design recommendation

Use `savr2-b15` as the base, change only:

1. force refresh when any of the three logged translation dimensions reverses
   direction between the two latest action chunks; and
2. reduce the wrist-camera local-change threshold from
   `0.6124632504463079` to `0.375`.

Retain all other SAVR2 visual, state, action, transition, temporal, cache, and
episode-prefix-budget rules exactly.

## Limitations

- This is a disclosed post-hoc design analysis, not validation evidence.
- The `task_04/state_02` `b15` path already differs from `b05` at query zero;
  nondeterministic paired rollouts limit causal attribution there.
- Three retained reuse records occur later on failed trajectories. They follow
  earlier divergence and do not establish that the frozen SAVR3 rule would
  fail or succeed.
- Filtering the observed first unsafe reuses does not prove causality. SAVR3
  must pass a fresh, predeclared policy-specific validation without tuning.

## Reproducibility

- Script: `scripts/analyze_phase6s_a.py`
- Unit tests: `tests/unit/test_phase6s_a.py`
- Output-file SHA-256:
  `cbce7c66cad74b3421a65bd79f53ee9b9264a3e3cbb7b74cee24703d7ad2ee47`
- Embedded semantic analysis SHA-256:
  `992754e900d5674174c2dc948e7adf2afda2dfe0943d664ea6fb7b704d098132`
