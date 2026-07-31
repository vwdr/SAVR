# Phase 6 Calibration and Power Report

Status: STOPPED — NO ELIGIBLE SAVR CONFIGURATION

## Outcome

The frozen Phase 6 calibration rule was not met. Full Refresh succeeded on all 100 paired LIBERO-Spatial calibration episodes, while every predeclared SAVR setting degraded success by substantially more than the frozen 2-percentage-point margin.

Per the frozen stop rule, thresholds and the margin were not relaxed. No SAVR primary configuration was selected, no matched-budget VOR/PR run was launched, and no final-holdout outcome was executed or inspected.

## Full Refresh calibration oracle

- terminal episodes: 100/100
- successes: 100/100
- query traces: 1309
- trace input hash: `c1724072a9108a77a7c8cec936f4a7e79239dca68aac75a288ca3d4638de9804`

## Frozen SAVR grid results

| Configuration | Offline target skip | Hmax | Success | Paired difference | Online skip | Online refresh |
|---|---:|---:|---:|---:|---:|---:|
| savr-s25-h2 | 25.00% | 2 | 52/100 | -48.0 pp | 34.69% | 65.31% |
| savr-s25-h4 | 25.00% | 4 | 21/100 | -79.0 pp | 43.29% | 56.71% |
| savr-s25-h8 | 25.00% | 8 | 23/100 | -77.0 pp | 47.28% | 52.72% |
| savr-s50-h2 | 50.00% | 2 | 12/100 | -88.0 pp | 56.62% | 43.38% |
| savr-s50-h4 | 50.00% | 4 | 4/100 | -96.0 pp | 63.26% | 36.74% |
| savr-s50-h8 | 50.00% | 8 | 0/100 | -100.0 pp | 65.21% | 34.79% |
| savr-s75-h2 | 75.00% | 2 | 3/100 | -97.0 pp | 64.14% | 35.86% |
| savr-s75-h4 | 75.00% | 4 | 1/100 | -99.0 pp | 75.00% | 25.00% |
| savr-s75-h8 | 75.00% | 8 | 0/100 | -100.0 pp | 83.68% | 16.32% |

The least-degrading setting was `savr-s25-h2` with 52/100 successes, 34.69% online skipped refreshes, and a -48.0-percentage-point paired success difference from FR. It was not eligible.

The offline FR replay target did not transfer safely to closed-loop trajectories: even the conservative target family produced large online success losses. This supports a negative conclusion for the tested operating region, not a claim that all possible SAVR settings must fail.

## Power and frozen configurations

- primary FR configuration: frozen (refresh every query)
- primary SAVR configuration: not frozen; no eligible candidate
- matched VOR/PR configurations: not run by the predeclared stop rule
- paired final sample size: not confirmed because no eligible SAVR operating point exists

Therefore the normal Phase 6 exit gate was not met. Phase 7 final-protocol freezing must not begin without an explicit scientific redesign decision.

## Integrity, resources, and safety

- reconciled GPU-run elapsed time: 24.83 hours
- reconciled GPU-run artifacts: 255.96 MiB
- FR: 100/100 terminal episodes, 0 infrastructure errors
- SAVR grid: 900/900 terminal episodes, 0 infrastructure errors
- every run used one explicitly selected GPU and restored protected checkpoint metadata exactly
- all task failures were retained as scientific outcomes
- no training, model/dataset download, upstream edit, manuscript edit, or final-holdout execution occurred
- no university-server path outside `/home/ved/SAVR` was modified by the Phase 6 workflow

## Primary methodological sources

- OpenVLA-OFT: https://arxiv.org/abs/2502.19645
- LIBERO: https://proceedings.neurips.cc/paper_files/paper/2023/hash/8c3c666820ea055a77726d66fc7d447f-Abstract-Datasets_and_Benchmarks.html
- Matched-pair non-inferiority sample size: https://doi.org/10.1002/bimj.201100231
- McNemar matched-pair sample size: https://doi.org/10.1002/sim.4780110909

## Phase boundary

Phase 6 stops at this negative checkpoint. A follow-up may either end the proposed method as currently formulated or predeclare a new calibration protocol that tests materially more conservative reuse. The current split must not be relabeled as a fresh holdout.
