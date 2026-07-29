# Phase 2B Full Refresh Pilot Report

Date: 2026-07-29

Status: **COMPLETE — FEASIBILITY GATE PASSED; CHECKPOINT REVIEW PENDING**

## Scope

The approved calibration pilot ran the unmodified Full Refresh policy for exactly
50 LIBERO-Spatial episodes: task IDs `0-9`, initial-state IDs `0-4`, and seed
`0`. It used the pinned combined checkpoint and OpenVLA-OFT revision on one
responsibly selected TITAN RTX.

This is baseline-feasibility evidence. It is not final evaluation and supports
no SAVR performance claim.

## Outcome

- Terminal episodes: `50/50`
- Successful episodes: `49/50` (`98%`)
- Runtime errors: `0`
- Predeclared feasibility threshold: **PASSED**
  - required at least `45/50`: passed
  - required at least one success per task: passed
- Single preserved failure: task `7`, initial state `2`

| Task | Successes |
|---:|---:|
| 0 | 5/5 |
| 1 | 5/5 |
| 2 | 5/5 |
| 3 | 5/5 |
| 4 | 5/5 |
| 5 | 5/5 |
| 6 | 5/5 |
| 7 | 4/5 |
| 8 | 5/5 |
| 9 | 5/5 |

The fixed 50-episode calibration subset shows no gross baseline-success
discrepancy. It is too small and non-independent of later calibration choices
to establish equivalence with the upstream 500-episode result.

## Component timing

The first three queries were predeclared warm-up queries and excluded. The
steady-state summary contains `662` queries.

| Component | Mean | Median | P95 |
|---|---:|---:|---:|
| Policy query wall time | 1267.49 ms | 1266.50 ms | 1287.86 ms |
| Total query CUDA time | 1267.44 ms | 1266.44 ms | 1287.80 ms |
| Vision backbone CUDA time | 191.12 ms | 193.95 ms | 199.84 ms |
| Visual projector CUDA time | 10.08 ms | 9.48 ms | 11.97 ms |
| Backbone + projector CUDA time | 201.19 ms | 203.20 ms | 211.62 ms |
| Action-head CUDA time | 5.03 ms | 5.03 ms | 5.08 ms |
| Residual downstream CUDA time | 1061.21 ms | 1058.79 ms | 1072.55 ms |
| Observation preprocessing wall time | 17.59 ms | 17.67 ms | 21.50 ms |

Backbone plus visual projector accounted for:

- `15.874%` of total query CUDA time
- `15.873%` of synchronized policy-query wall time
- `4.926%` of the complete pilot wall time

If visual computation could be removed from every query with zero cache or
controller overhead, the measured query-time ceiling is approximately a
`15.87%` latency reduction or `1.189×` speedup. Actual SAVR benefit must be
lower because it refreshes on some queries and introduces decision/cache
overhead. This modest but material ceiling supports proceeding to a bounded
implementation and correctness phase; it does not guarantee a useful
success-efficiency trade-off.

## Integrity and resources

- Run ID: `phase2b-fr-spatial-pilot-v1`
- Runner commit: `20daed35f06fe111755ea3b0a4d77388c25f6536`
- OpenVLA-OFT commit: `e4287e94541f459edc4feabc4e181f537cd569a8`
- Checkpoint revision: `638918f3d1c2e43a39a8a20772bdb8b91835e4b7`
- GPU: physical GPU `0`, exposed as the only visible device
- Run wall time: `2703.85 s` (`45.06 min`), below the three-hour cap
- Peak allocated GPU memory: `16,085,880,832` bytes (`14.98 GiB`)
- Peak reserved GPU memory: `16,200,499,200` bytes (`15.09 GiB`)
- Result artifacts: `1,509,206` logical bytes, below the two-GiB cap
- Checkpoint metadata: restored byte-for-byte; before/after hashes matched
- Videos: one success per task plus the single failure
- Post-run GPU audit: GPU `0` returned to `0%` utilization and `6 MiB` used

No process identities were inspected. No unrelated server file, process,
environment, service, permission, or GPU allocation was changed.

## Reproducibility

Raw immutable records remain project-local on TITAN at:

`/home/ved/SAVR/results/phase2b-fr-spatial-pilot-v1`

The reproducible aggregation command is:

```bash
cd /home/ved/SAVR
python3 scripts/analyze_phase2b_pilot.py
```

Its generated aggregate is
`reports/runtime/phase2b_aggregate.json`, which is intentionally ignored
because runtime artifacts remain on TITAN.

## Checkpoint

All Phase 2 exit evidence is now present:

- the checkpoint fits one TITAN RTX
- the bounded all-task run has complete manifests
- no gross baseline-success discrepancy was found
- visual-encoder latency share is quantified

Phase 2 remains `IN_PROGRESS` until this evidence is reviewed and its PR is
explicitly approved and merged. Phase 3 remains unauthorized.
