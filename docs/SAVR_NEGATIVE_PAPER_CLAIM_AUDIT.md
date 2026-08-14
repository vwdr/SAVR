# SAVR Negative-Results Paper Claim Audit

**Status:** Complete  
**Audit date:** 2026-08-13  
**Reviewed paper:** `output/pdf/SAVR_Negative_Results_Paper.pdf`

## Scope decision

The preserved repository is sufficient for a bounded negative-results paper about **training-free whole-prefix caching** in one pinned OpenVLA-OFT/LIBERO-Spatial setting. No new experiment is required to support that claim. The evidence does not support a universal rejection of SAVR, visual reuse, or finer-grained caching.

## Claim-to-evidence map

| Manuscript claim | Primary evidence | Permitted wording |
|---|---|---|
| Full Refresh achieved 100/100 on the paired calibration population | `reports/PHASE6_CALIBRATION_REPORT.md` | Exact descriptive count on tasks 0--9, states 0--9, seed 0 |
| All nine SAVR 1.0 settings failed the frozen success gate | `reports/PHASE6_CALIBRATION_REPORT.md`; `docs/evidence/negative_results_summary.csv` | No tested SAVR 1.0 operating point was eligible |
| SAVR 1.0 success ranged from 0--52/100 at 34.69--83.68% online skip | Same as above | Exact configuration-level descriptive results |
| Offline Full Refresh replay underestimated online reuse in all nine settings | `reports/PHASE6R_A_DIAGNOSIS_REPORT.md` | Replay was not an online reuse-budget or reliability oracle in this study |
| Camera averaging concealed 374 threshold exceedances in 783 reuse queries; 372 were wrist-only | `reports/PHASE6R_A_DIAGNOSIS_REPORT.md` | Cross-camera averaging was not conservative for this population |
| First comparable reuse changed the action hash in 87/87 analyzable episodes | `reports/PHASE6R_A_DIAGNOSIS_REPORT.md` | Immediate output divergence; not proof of failure causality or action magnitude |
| SAVR 2.0 recovered 30/30 only at zero reuse; 6.72% and 10.57% skip achieved 29/30 and 27/30 | `reports/PHASE6R_D_STAGE1_REPORT.md` | Descriptive Stage-1 screen; not a paired cross-version comparison |
| SAVR 3.0 achieved 69/70 at 9/944 reuses (0.9534%) and failed both gates | `reports/PHASE6S_D_VALIDATION_REPORT.md` | One frozen post-hoc development validation; not final confirmation |
| The visual backbone and projector were 15.874% of query CUDA time | `reports/PHASE2B_PILOT_REPORT.md` | Measured component fraction and zero-overhead upper bound, not measured SAVR speedup |
| Wrapped Full Refresh was equivalent and real reuse skipped the intended visual calls | `reports/PHASE6R_C_CORRECTNESS_REPORT.md`; `reports/PHASE6S_C_CORRECTNESS_REPORT.md` | Implementation correctness within the tested adapter boundary |
| The primary evaluation summarizes 1,160 terminal episodes | The four reports above | 100 FR + 900 SAVR1 + 90 SAVR2 + 70 SAVR3; the separate timing pilot contains 50 additional episodes |

## Claims intentionally blocked

- SAVR or visual caching is universally unreliable.
- Every reuse, wrist-camera threshold exceedance, or action-hash divergence caused a task failure.
- The tested method is better or worse than Periodic Refresh or Visual-Only Refresh; those baselines were stopped by protocol before rollout.
- Final-holdout non-inferiority, superiority, or population-level generalization; the protected holdout was not run.
- A measured end-to-end latency improvement for SAVR; only component timing and skipped-call counts are available.
- Cross-version statistical comparison between SAVR 1.0, 2.0, and 3.0; their staged populations differ.
- Conclusions about camera-block, token-level, learned, uncertainty-aware, or model-native caching.

## Completeness assessment

The repository contains the controller and adapter code, frozen configurations, correctness tests, stage protocols, machine-readable summaries, immutable-result indexes, provenance revisions, and all numbers needed for the manuscript tables and figures. The meaningful gaps are study limitations, not missing records: one model/suite/hardware setting, seed 0, no eligible matched PR/VOR rollout, no final holdout, and no measured end-to-end SAVR speedup.

## Publication rule

If any manuscript statement exceeds the wording above, narrow it or add evidence before publication. Do not rerun or reinterpret preserved negative trials merely to improve the narrative.
