# PAIR P3R Outcome-Blind Timing Report

Date: 2026-08-29 EDT  
Completed run: `pair-p3r-vectorized-v02-recovery01`  
Status: **passed**

## Decision

The vectorized actual-source tile scorer solved the overhead failure observed in
P3. All four predeclared P3R timing points passed the complete frozen gate. P3R
therefore establishes positive physical timing headroom for the PAIR mechanism
and supports proceeding to a separately authorized P4 reliability evaluation.

This is an outcome-blind timing result. It does not yet establish action quality,
task success, or closed-loop reliability, and it does not authorize P4.

## Frozen population

- Profiles: `D59_BAL_PT1`, `D62_BAL_PT1`
- Horizons: 2 and 4
- Paired repetitions: 6 per profile/horizon
- Timed blocks: 24/24
- Model queries: 210/210
- Legacy/vectorized recursive equivalence checks: 8/8 exact
- Selected GPU: physical GPU 0
- Peak aggregate GPU memory: 18,261 MiB, below the strict 23,552 MiB limit
- No expert actions, action comparisons, terminal outcomes, simulator use,
  downloads, retry, or outlier deletion

## Results

| Profile | Horizon | Raw saving | One-sided 95% lower bound | Conservative net lower bound | Total overhead | Service | Gate |
|---|---:|---:|---:|---:|---:|---:|---|
| D59_BAL_PT1 | 2 | 20.12% | 19.96% | 13.94% | 0.96% | 100% | Pass |
| D59_BAL_PT1 | 4 | 23.79% | 23.58% | 16.48% | 0.78% | 100% | Pass |
| D62_BAL_PT1 | 2 | 20.46% | 20.20% | 14.12% | 0.94% | 100% | Pass |
| D62_BAL_PT1 | 4 | **24.59%** | **24.40%** | **17.05%** | **0.78%** | **100%** | **Pass** |

The frozen selector chose `D62_BAL_PT1` at horizon 4 because it had the largest
passing conservative net-saving lower bound. Its effective visual reuse fraction
was 48.83%, while source provenance, active-sequence shape, mixed-source
diversity, memory, resource, artifact, and protection invariants all passed.

## Resolution of the P3 failure

For these same profiles and horizons, P3 total overhead was 4.09%--5.22%, above
the 2% gate. P3R reduced it to 0.78%--0.96% by batching all actual-source tile
comparisons across cameras, onset layers, and tiles while preserving exact legacy
mask decisions. The improvement is therefore attributable to the predeclared
implementation redesign rather than a relaxed gate or changed scientific
frontier.

## Gate reconciliation

- At least one headroom point: pass (4/4 points passed)
- Exact legacy/vectorized equivalence: pass
- Sidecar, sequence, and all-fresh controls: pass
- Resource caps: pass
- Protected-population boundary: pass
- Artifact cap: pass
- Shape and source-provenance invariants: pass at all points
- P4 authorization: false

## Evidence

- Analysis: `results/pair-p3r-vectorized-v02-recovery01/analysis.json`
- Worker summary: `results/pair-p3r-vectorized-v02-recovery01/worker_summary.json`
- Paired blocks: `results/pair-p3r-vectorized-v02-recovery01/blocks.jsonl`
- Recovery preflight: `reports/pair_p3r/preflight_v02_recovery01.json`
- Frozen configuration: `configs/pair/p3r_vectorized_v02_recovery01.json`
- Recovery protocol: `docs/PAIR_P3R_TECHNICAL_RECOVERY_01_PROTOCOL.md`

SHA-256 identifiers:

- Analysis: `6f1bad7752403884a880438a24ea77af6702bac48bb75a74c99ceb105907ebda`
- Worker summary: `af5ccea78b3d6393784132e194df396586b39efed367f13040e1b53e6df8a072`
- Blocks: `d024d3ee1a817be7d1fb675b036eaf989b782752409f095387ff0e541548a836`

## Required next checkpoint

P4 must be planned and approved separately. It should test whether the selected
timing point preserves the predeclared reliability criterion under the relevant
evaluation population. The positive-results paper route remains conditional on
that reliability result; P3R alone supports an efficiency claim, not a complete
reliability-efficiency claim.
