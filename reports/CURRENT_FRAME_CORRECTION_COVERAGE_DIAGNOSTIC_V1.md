# Current-frame correction: offline coverage vs adequacy diagnostic

Recorded 2026-09-21 UTC. Read-only CPU analysis of already-saved evidence. No GPU,
no new data collection, no robot rollouts, and no modification of any frozen
source, summary, weight or evidence file. Script:
`scripts/diagnose_current_frame_correction_coverage.py`.

> **SUPERSEDED.** This report's OOD z-scoring standardized fit and validation
> matrices with their own per-matrix statistics, inflating the coverage gradient.
> Use `reports/CURRENT_FRAME_CORRECTION_DIAGNOSTIC_V2.md`, which corrects the
> z-scoring, adds confound controls (action-only comparison, mechanical bound)
> and tests deploy-available benefit predictors. Keep this file for provenance.

## Question

Why did the trained visual corrector worsen L1 on 260/800 validation observations
(and on 77/144 where uncorrected 384-token compression was already within 0.02 L1
of the dense teacher)? Two predeclared hypotheses:

1. **Coverage**: correction degrades on observations far from the fitting
   distribution (visual/state or instruction space distance to fitted frames).
2. **Adequacy (always-on correction)**: correction alters already-adequate base
   actions in the wrong direction regardless of coverage.

Inputs: all 4,000 saved feature records (3,200 fit, 800 validation) from
`results/current-frame-feature-collection-v01`, decoded from stored BF16 bit
patterns, plus the 800 saved per-arm L1/prediction rows from
`results/current-frame-adapter-fit-v01/validation.json`. Scene vector per record =
mean of the 512 current projected patches (4096-dim) concatenated with the 8-dim
state, z-scored per dimension on the fit records only. OOD score = Euclidean
distance to the nearest fitting neighbor in that scene space.

## Predeclared interpretation (fixed before reading outcomes)

- Coverage dominates if the visual improvement rate (visual L1 < base L1) in the
  highest-scene-OOD quartile is at least 10 percentage points below the lowest
  quartile **and** holds after stratifying by base adequacy.
- Adequacy dominates if base-adequacy stratification shows systematically lower
  visual improvement on small-base-L1 records regardless of OOD, with the OOD
  gradient small or confounded.

## Result

Overall validation: improvement rate 0.675; mean delta L1 (visual minus base)
−0.00184 per observation.

| Scene-OOD quartile (1 = nearest to fit) | n | Visual improvement rate | Mean delta L1 |
|---|---:|---:|---:|
| Q1 (lowest OOD) | 200 | 0.730 | −0.00237 |
| Q2 | 200 | 0.705 | −0.00225 |
| Q3 | 200 | 0.640 | −0.00151 |
| Q4 (highest OOD) | 200 | 0.625 | −0.00122 |

| Base-adequacy quartile (1 = base already closest to teacher) | n | Improvement rate | Mean delta L1 | Mean residual |visual−base| |
|---|---:|---:|---:|---:|
| Q1 (most adequate) | 200 | 0.510 | +0.00007 | 0.009455 |
| Q2 | 200 | 0.615 | −0.00085 | 0.010993 |
| Q3 | 200 | 0.705 | −0.00211 | 0.011893 |
| Q4 (least adequate) | 200 | 0.870 | −0.00446 | 0.014914 |

Near-base band (base L1 < 0.02, n=144): improvement rate 0.465; mean delta L1
**+0.000267** (visual makes these worse on average); mean residual 0.009168.
The corrector perturbs an already-adequate action by roughly half of that band's
error margin, and does so in the wrong direction about half the time.

Spearman correlations (descriptive): delta L1 vs scene-OOD −k1 **+0.105**;
delta L1 vs base L1 **−0.411**; delta L1 vs residual magnitude −0.358;
residual magnitude vs base L1 +0.388.

Stratified improvement rates (rows = OOD quartile, columns = adequacy quartile):

| OOD \ Adequacy | Q1 | Q2 | Q3 | Q4 | across |
|---|---:|---:|---:|---:|---:|
| Q1 (low OOD) | 0.627 | 0.678 | 0.763 | 0.909 | 0.730 |
| Q2 | 0.510 | 0.739 | 0.712 | 0.849 | 0.705 |
| Q3 | 0.419 | 0.566 | 0.673 | 0.878 | 0.640 |
| Q4 (high OOD) | 0.449 | 0.452 | 0.691 | 0.852 | 0.625 |

Per suite: Spatial 0.725 (median OOD 1003), Object 0.640 (1125), Goal 0.720
(1074), LIBERO-10 0.615 (1656). LIBERO-10 is both the most scene-OOD suite and
the suite where the corrector helps least offline.

## Interpretation

Neither pure hypothesis exclusively wins, but the **adequacy association is
stronger and cleaner** than the coverage association:

- The base-adequacy gradient is monotonic and large (0.510 → 0.870 improvement
  rate, i.e., +36 pp across quartiles) and persists in **every** OOD stratum.
- The raw scene-OOD gradient (0.730 → 0.625, 10.5 pp) is at the predeclared
  threshold, and Spearman association is weak (+0.105). Once base adequacy is
  held fixed, the OOD gradient shrinks or inverts inside several cells, so part
  of the raw coverage gradient is adequacy confound, not coverage signal.
- The near-base band (base already within 0.02 of the teacher) is where visual
  correction is net harmful on average and helps only 46.5% of the time.

Conclusion on this offline evidence: the visual corrector's quoted mishaps are
best explained by **alteration of already-adequate base actions** (always-on
correction), with only a weak secondary coverage component. This is descriptive,
not a proven causal mechanism for the robot failures.

## What this does and does not establish

- Does: distinguishes coverage from adequacy on 800 demonstration-derived held-out
  observations; writes per-sample rows and summaries to
  `results/current-frame-correction-coverage-diagnostic-v01/`; modifies nothing.
- Does not: measure the deployment distribution shift. Validation observations
  are held-out demonstration frames, not states reached by the corrected policy,
  which the completed pilot did not save per step. So the coverage question at
  deployment time remains open and would require a new bounded run that records
  per-step corrected actions and observations.

## Effect on the open decision

This strengthens the case that a "never alter an adequate action" or
adequacy-conditional correction rule, rather than longer training or more data of
the same kind, is the natural next control if the correction line is pursued. It
does not, by itself, authorize retraining, threshold selection for deployment, or
a new GPU run; any such step still requires one prospectively frozen protocol.
The alternative line (an honest compression efficiency–accuracy study) remains
unresolved against existing simple-compression literature.