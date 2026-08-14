# SAVR

Research code and evidence for **A Negative Result for Training-Free Whole-Prefix Visual Caching in VLA Inference**.

SAVR (State-Aware Visual Refresh) is a training-free controller for deciding whether a vision-language-action (VLA) policy should recompute or reuse its complete projected visual prefix. The tested implementation uses image change, robot-state change, recent-action change, and cache-history constraints without modifying the policy weights.

> **Status:** The whole-prefix SAVR study is complete and produced a bounded negative result. No tested configuration preserved Full Refresh task success while also providing a useful visual-skip rate. The repository is private during author review and will require a versioned archival release before publication.

## Main result

The primary study contains **1,160 terminal evaluation episodes**, plus a separate 50-episode Full Refresh timing pilot, using one pinned OpenVLA-OFT checkpoint on LIBERO-Spatial.

| Method | Evaluation population | Task success | Online visual skip | Outcome |
|---|---:|---:|---:|---|
| Full Refresh | 100 episodes | 100/100 | 0% | Reference |
| SAVR 1.0, best tested setting | 100 episodes | 52/100 | 34.69% | Failed success gate |
| SAVR 2.0, 5% cap | 30 episodes | 30/30 | 0% | Failed minimum-skip gate |
| SAVR 2.0, 10% cap | 30 episodes | 29/30 | 6.72% | Failed success gate |
| SAVR 2.0, 15% cap | 30 episodes | 27/30 | 10.57% | Failed success gate |
| SAVR 3.0 | 70 episodes | 69/70 | 0.9534% | Failed success and minimum-skip gates |

The evidence supports a narrow conclusion: **training-free whole-prefix caching did not provide a success-preserving and useful operating point for this model, benchmark, and implementation boundary.** It does not show that all visual caching methods fail. Camera-block, token-level, learned, uncertainty-aware, and model-native caching remain outside this result.

The reviewed paper is available at [output/pdf/SAVR_Negative_Results_Paper.pdf](output/pdf/SAVR_Negative_Results_Paper.pdf).

## Repository contents

- `src/savr/`: refresh controllers, cache adapters, signal calculations, instrumentation, and analysis utilities
- `configs/calibration/`: frozen Full Refresh and SAVR 1.0--3.0 configurations
- `scripts/`: experiment runners, candidate derivation, verification, and analysis commands
- `tests/unit/`: dependency-light tests for the reported SAVR implementation
- `schemas/`: machine-readable run, query, and episode record contracts
- `docs/evidence/negative_results_summary.csv`: compact machine-readable result table
- `reports/PHASE6_CALIBRATION_REPORT.md`: Full Refresh and SAVR 1.0 results
- `reports/PHASE6R_A_DIAGNOSIS_REPORT.md`: replay drift, camera aggregation, and action-divergence analysis
- `reports/PHASE6R_D_STAGE1_REPORT.md`: SAVR 2.0 results
- `reports/PHASE6S_D_VALIDATION_REPORT.md`: SAVR 3.0 results
- `docs/SAVR_NEGATIVE_PAPER_CLAIM_AUDIT.md`: claim-to-evidence boundaries
- `environment/`: pinned software inventory and project-local setup information
- `CITATION.cff`: machine-readable citation metadata

The repository also retains exploratory camera-factorized follow-up code and its development records. Those artifacts are not part of the negative paper's validated claim and should not be interpreted as a positive result.

Large checkpoints, datasets, caches, videos, and most raw rollout directories are intentionally excluded. The tracked reports, frozen configurations, integrity identifiers, and compact result summary preserve the reported evidence boundary. Re-running raw aggregations requires the corresponding project-local rollout records.

## Installation and CPU checks

The core package requires Python 3.10 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[test]"
python -m pytest -q tests/unit
python scripts/validate_bootstrap.py
```

The pinned OpenVLA-OFT/LIBERO environment used for the reported GPU experiments is documented in `environment/README.md`, `environment/locks/`, and `docs/UPSTREAM_PINS.md`. The setup is project-local and should not be treated as a general system installation script.

## Scientific integrity

- All task failures were retained.
- Thresholds and stage gates were frozen before their corresponding online evaluations.
- The final holdout was not opened because no method passed the required development gate.
- Periodic Refresh and Visual-Only Refresh were not run because the predeclared protocol required an eligible SAVR configuration first.
- Skipped visual calls are reported as an efficiency proxy, not as measured end-to-end speedup.
- Cross-stage SAVR 1.0, 2.0, and 3.0 comparisons are descriptive because their evaluation populations differ.

## Citation

Until a venue-specific record or archival DOI is available, cite the author-review preprint as:

```bibtex
@article{dwivedi2026savrnegative,
  title   = {A Negative Result for Training-Free Whole-Prefix Visual Caching in VLA Inference},
  author  = {Dwivedi, Ved and Yang, Cheng and Yuan, Bo},
  year    = {2026},
  note    = {Author-review preprint}
}
```

## License and release status

No open-source license has been selected for this author-review repository. Reuse rights should not be assumed until the authors add a license. Before external release, the repository also needs a permanent archive or DOI and the paper must be converted to the selected venue's format.
