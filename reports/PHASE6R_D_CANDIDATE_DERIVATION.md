# Phase 6R-D Candidate Derivation Report

Status: COMPLETE — STAGE 1 CONFIGURATION FROZEN BEFORE ONLINE ROLLOUT

Report date: 2026-07-31

## Inputs and method

The derivation used all `100` completed Phase 6 FR development episodes and
all `1,309` query traces from LIBERO-Spatial tasks `0-9`, initial states `0-9`,
seed `0`. It used no SAVR 2.0 outcome and executed no simulator or GPU work.

For each required score family, the script constructed adjacent-query
distributions, evaluated the shared linear quantile grid `0.000-1.000` in
steps of `0.001`, applied the frozen `0.90` safety margin, and replayed the
exact SAVR 2.0 warm-up, stable-fresh, transition, isolated-reuse, and hard
episode-prefix-budget rules.

Canonical FR trace-input SHA-256:
`619c7310dff8ada8498c29df641289677a9318bd30aa2e45d04999aeea8b2114`.

## Frozen candidates

| Candidate | Budget | Quantile | Offline reuses / queries | Offline skip | Episodes with reuse | Earliest reuse |
|---|---:|---:|---:|---:|---:|---:|
| `savr2-b05` | 5% | 0.000 | 0 / 1,309 | 0.00% | 0 / 100 | none |
| `savr2-b10` | 10% | 0.999 | 90 / 1,309 | 6.88% | 90 / 100 | query 9 |
| `savr2-b15` | 15% | 1.000 | 128 / 1,309 | 9.78% | 95 / 100 | query 6 |

The 5% prefix cap cannot approve a first reuse before query `19`. None of the
FR development episodes reaches a compatible stable approval under the frozen
rules, so `b05` has zero simulated reuse. It remains in Stage 1 because the
protocol requires every frozen candidate to be screened online. It cannot
advance unless its observed skip reaches 2%, which is not expected.

The high quantiles for `b10` and `b15` do not remove safety rules: each score
threshold is multiplied by `0.90`, either camera can veto, grouped and gripper
checks remain active, two stable fresh queries are required, reuse is isolated,
and the hard prefix budget remains decisive online.

## Provenance and reproducibility

- derivation script: `scripts/derive_phase6r_d_candidates.py`;
- tracked Stage 1 config: `configs/calibration/phase6r_d_stage1.json`;
- semantic canonical config SHA-256:
  `66874e1a2c209ec5809dd1d777de5ce8eeacee63d85e8e4dd1c6f0876bcfc09d`;
- generated artifact SHA-256:
  `4b14588a9dc182dae61bccc4c6b991ae47cb7f8a5d48683c16e48c7b2e0a8b68`;
- a second derivation produced byte-identical output and the same hash;
- `91` repository tests and `9` TITAN subtests passed at the derivation
  implementation checkpoint.

## Stage 1 boundary

The tracked configuration schedules exactly 30 paired episodes per candidate:
tasks `0-9`, initial states `0-2`, seed `0`, for 90 episodes maximum. All raw
downsampled traces are retained. A candidate advances only with `30/30`
successful and reconciled episodes, at least 2% online skip, and zero technical
or invariant errors. No threshold or promotion rule may change after rollout.
