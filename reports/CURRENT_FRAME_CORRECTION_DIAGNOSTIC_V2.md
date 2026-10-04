# Current-frame correction: corrected offline diagnostics V2

Recorded 2026-09-21 UTC. This version supersedes
`reports/CURRENT_FRAME_CORRECTION_COVERAGE_DIAGNOSTIC_V1.md`, which contained an
erroneous OOD z-scoring (fit and validation matrices were standardized with their
own per-matrix statistics). Read-only CPU analysis of already-saved evidence; no
GPU, no new data, no robot rollouts, no modification of frozen source, summary,
weights or evidence. Scripts:
`scripts/diagnose_current_frame_correction_coverage.py` (outputs `...-v02/`) and
`scripts/analyze_correction_benefit_signals.py` (outputs
`results/current-frame-correction-benefit-signals-v01/`).

## 1. Corrected coverage analysis (fit-only z-scoring)

Scene OOD distance = Euclidean distance to nearest fitting neighbor, z-scored per
dimension with fit-only statistics applied to both matrices (bug fixed).

| Scene-OOD quartile | n | Visual improvement rate | Mean delta L1 |
|---|---:|---:|---:|
| Q1 (nearest) | 200 | 0.725 | −0.00233 |
| Q2 | 200 | 0.705 | −0.00230 |
| Q3 | 200 | 0.640 | −0.00151 |
| Q4 (farthest) | 200 | 0.630 | −0.00122 |

Raw coverage gradient: 9.5 percentage points (0.725 → 0.630), below the
predeclared 10 pp threshold. Spearman of delta L1 with OOD distance: +0.104
(weak). The earlier 10.5 pp figure was inflated by the z-scoring bug. Coverage
does not dominate.

## 2. Confound-controlled adequacy analysis

The V1 report's "adequacy dominates" wording was too strong and partly mechanical.
Two controls now address the user's critique:

- **Mechanical bound**: delta = visual L1 − base L1 cannot go below −base L1, and
  sign near zero base error is noise-dominated; improvement rates on small-base
  samples are therefore depressed for any corrector.
- **Action-only control**: the action-only corrector (no visual input) also
  perturbs every action. It "fails" on adequate actions at the same rate as
  visual correction:

| Band (of 800) | Visual improve rate | Action-only improve rate | Mean gain visual | Mean gain action-only |
|---|---:|---:|---:|---:|
| base L1 < 0.02 (n=144) | 0.465 | 0.486 | −0.00027 | −0.00004 |
| base L1 ≥ 0.02 (n=656) | 0.721 | 0.704 | +0.00230 | +0.00193 |

On the near-base band, visual beats action-only on only 48.6% of samples (≈
chance), and absolute gains/losses are tiny (|gain| ≤ 0.008 L1 for both). On the
wide band both correctors help >70% of time with small positive mean gains. So
the near-base "harm" is not visual-specific; it reflects that corrections of
already-adequate actions are bounded and noise-level, for either corrector.

Conclusion: offline data show an association (corrections help where base error
is larger) but this is partly mechanical and shared with action-only. It is NOT
evidence about why the robot failed, whose per-step mechanism was not recorded.

## 3. Can inexpensive deploy-available signals predict correction benefit?

The proposed gating rule (apply correction only when base error is large) cannot
observe base error at query time — that requires the dense teacher. Tested
whether cheap signals available on the compressed path predict "visual correction
helps" (visual L1 < base L1) on **held-out trajectories** (519 train / 281 test,
90 disjoint test trajectories, seed 7, logistic regression on 16 candidate
signals: base action stats, corrector residual magnitude, cross-head agreement
action-only vs visual, instruction/state/projected-patch/action-feature
statistics, corrected scene-OOD distance).

- Held-out AUC: **0.599**, 95% bootstrap CI [0.526, 0.677] — weak predictive
  signal, barely above chance.
- Best single cheap signals: action-feature max |·| (AUC 0.57), corrector
  residual magnitude (AUC 0.56; Spearman with gain +0.27), cross-head agreement
  (AUC 0.54). Base-action statistics are anti-correlated (univariate AUC 0.40,
  useful direction reversed). OOD distance: AUC 0.48, not predictive.
- Gating decision (apply only when logistic p ≥ 0.5) on the test set: mean L1
  0.03481 vs always-apply 0.03471 vs never-apply (plain compression) 0.03640.
  Gating applies to 86% of samples and does **not** beat always-apply; it only
  trims the small baseline compression gap slightly. No deployable gate is
  demonstrated.

## 4. Interpretation and limits

- Offline evidence: correction of already-adequate base actions is a real but
  small, non-visual-specific, partly mechanical phenomenon; scene-OOD coverage is
  a weaker factor; no cheap query-time signal reliably gates correction benefit.
- This does not establish why the robot pilot failed. Per-step corrected actions
  were not saved by the completed pilot, so deployment-state distribution and
  failure mechanisms remain unmeasured.
- No gating rule, retraining, threshold selection or new GPU run is authorized.
  A bounded robot test remains unjustified from these offline results alone;
  the corrector's always-on baseline was already the robot-tested policy.
- The previously verified robot results (dense 120, compression 117, action-only
  118, visual 117 /120; all hashes) are unchanged.