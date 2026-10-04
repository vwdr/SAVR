# Current-frame robot pilot: frozen readiness record

2026-09-19 UTC. User authorized continuation from the completed offline adapter fit.

## What is ready

- 41 focused original-runtime CPU tests passed: 37 current-frame tests and four
  measured-episode tests. The eight new robot-pilot tests also passed again after
  the worker's CPU weight/input preflight was added.
- Full authenticated predecessor/code/data preflight passed on TITAN with CUDA
  hidden. It resolved and hashed 124 selected initial-state identities, loaded
  both exact final trained checkpoints, and evaluated finite batch-one 8x7 outputs
  on all 16 previously qualified stored feature records. No backbone calls,
  simulator episodes, optimizer updates or new outcome labels were used.
- The frozen schedule contains 480 primary episodes: four policies on 120 matched
  development conditions, all 40 tasks, states 1/2/3, seed 7. Eight native controls
  bring the total to 488. These are development data, not independent holdout data.
- The deployed GPU path must pass its integrated 96-call check before the
  272-call controlled timing block and before any simulator episode. CPU checks
  do not substitute for this qualification.

Configuration: `configs/openvla/current_frame_robot_pilot_v1.json`

SHA-256: `9e3f6728be56821d5da73185f310f1119adda32e2a4fc6e50672cc2e71612c8e`

Worker SHA-256: `9aa9a2d341d2a184eaa3573c909fc77ee45b70618410f9dc98f35f57451ece3f`

Analyzer SHA-256: `de579c675e65998d348f1e4703f82c30bdc95c28342325d7d4cd7541d066fbf2`

## Frozen interpretation

The visual adapter must beat compression alone by at least three net successes,
beat action-only by at least one, lose no more than two successes to dense,
lose no more than two to compression within any suite, and reduce controlled mean
query time by at least 10% versus dense. Dense must reach 108/120 and all eight
native controls must succeed. These are preliminary development criteria, not
statistical confirmation or a paper-ready result. Report every arm and uncertainty.

## Resource and evidence boundaries

One previously coordinated GPU 0, exact UUID in the config, original environment.
Fresh idle check before allocation. Caps: 22,000 backbone calls, 488 episodes,
16 hours, aggregate memory below 23,552 MiB, output below 512 MiB, at least 10 GiB
free disk. Worst-horizon planned calls: 20,952. No retry, training, new data or
model download, other GPU use, GitHub push or unrelated server access.

While active, inspect only owned process health, record counts, bytes, elapsed time
and aggregate GPU-0 telemetry. Do not read outcomes or timing values until the
immutable completed summary exists with exactly 488 episode, 272 timing and 16
qualification rows. On early failure, inspect technical evidence only and stop.
On completion run frozen CPU analysis, reconcile all evidence, copy it locally,
update status, report the result and stop before further research runs.

## Reproduction

From `/home/ved/SAVR`, using `envs/openvla-oft/bin/python`:

```sh
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src \
  envs/openvla-oft/bin/python scripts/run_current_frame_robot_pilot.py \
  --config configs/openvla/current_frame_robot_pilot_v1.json --preflight-only
```

Launch the same worker without `--preflight-only`, with `CUDA_VISIBLE_DEVICES=0`,
into the exclusive result directory `results/current-frame-robot-pilot-v01` and
exclusive terminal log `reports/current-frame-robot-pilot-v01-terminal.log`.
Only after complete immutable evidence, run `scripts/analyze_current_frame_robot_pilot.py`
with CUDA hidden. Read the protocol for population, timing and interpretation details.
