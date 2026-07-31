# Phase 6R-D Stage 1 Report

Status: COMPLETE — NEGATIVE STOP; NO CANDIDATE ADVANCED

Report date: 2026-07-31

## Frozen matrix

Stage 1 evaluated all three predeclared SAVR 2.0 candidates on
LIBERO-Spatial tasks `0-9`, initial states `0-2`, seed `0`: 30 paired episodes
per candidate and 90 episodes total. It used one GPU and the frozen
`configs/calibration/phase6r_d_stage1.json` configuration.

Advancement required, without exception:

- 30/30 successful and reconciled episodes;
- at least 2% observed skipped visual refreshes;
- zero technical/instrumentation failures;
- complete cache, component, temporal, counter, and prefix-budget invariants.

## Results

| Candidate | Success | Queries | Reuses | Online skip | Technical failures | Advances? |
|---|---:|---:|---:|---:|---:|---|
| `savr2-b05` | 30/30 | 388 | 0 | 0.00% | 0 | No: skip below 2% |
| `savr2-b10` | 29/30 | 461 | 31 | 6.72% | 0 | No: success below 30/30 |
| `savr2-b15` | 27/30 | 473 | 50 | 10.57% | 0 | No: success below 30/30 |

Scientific unsuccessful episodes:

- `savr2-b10_task_08_state_02`;
- `savr2-b15_task_01_state_00`;
- `savr2-b15_task_04_state_02`;
- `savr2-b15_task_09_state_01`.

These episodes are retained as scientific outcomes. They were not deleted,
rerun, or relabeled as technical failures.

## Integrity reconciliation

- terminal episodes: `90/90`;
- query records: `1,322`;
- technical failures: `0`;
- temporal/prefix-budget violations: `0`;
- every query retained its required calibration trace;
- elapsed GPU run time: `5,206.483` seconds (`1.446` hours);
- artifact bytes: `238,869,135`, below the 1-GiB cap;
- checkpoint restoration: passed;
- unexpected checkpoint files: none;
- GPU memory returned to its pre-run aggregate level;
- final holdout: not executed or inspected;
- nothing outside `/home/ved/SAVR` was modified.

Provenance:

- SAVR revision: `5e577debd5161f2dc0303615b87f64cac795f58d`;
- run-summary SHA-256:
  `61a0c9ddfb263ba2123da3dd08500260eba6a454bf335f4830022a81c33a9ebe`;
- manifest SHA-256:
  `bb451a995796ffe0701f2d01179c6dcd7e0bb93d085041d638758c576864f60b`.

## Frozen decision

No candidate satisfies the Stage 1 advancement gate. Phase 6R-D therefore
stops before Stage 2. Phase 6R-E requires an eligible Phase 6R-D candidate and
cannot run. Thresholds, safety margins, populations, and promotion rules were
not relaxed.

This is meaningful negative evidence: conservative reuse can preserve all 30
successes only when it performs no reuse, while the two configurations that
obtain useful skip rates already lose task success in the small safety screen.
It does not support a positive-results SAVR 2.0 paper under the current method.
