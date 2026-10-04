# Forty-task original-runtime baseline: readiness and launch

Date: 2026-09-10. This is a readiness record, not an experimental result.

The user approved the broader original-policy evaluation. Forty unique tasks
were resolved from previously consumed development conditions, ten per suite,
with initial-state ID 0 and seed 7. Closed-loop evaluation uses original
attention only. Thirty-two offline reference/restoration calls precede it.

## Verified before launch

- Eleven existing CPU tests passed in the original TITAN runtime: attention
  neutrality/intervention/restoration and exact released episode-order behavior.
- Six new population-selection tests passed locally and on TITAN, including
  task count, order, duplicates, seed and exclusion of other populations/states.
- Twelve new completion-analysis tests passed locally and on TITAN, including
  immutable artifact hashes, incomplete/technical-stop rejection, accounting,
  action parity, runtime identity and initial-state hash validation.
- The preflight-only worker authenticated forty source/checkpoint/input files
  and eight consumed HDF5 sources without loading a model.
- GPU 0 had 6 MiB aggregate memory use and 0% utilization. No unrelated process
  details were inspected; no GPU allocation or other job was changed.
- Previous source and evidence files were preserved. New files were copied only
  to matching paths inside /home/ved/SAVR. No installations/downloads.

## Frozen execution

Configuration: `configs/openvla/original_baseline_40task_v1.json`.
SHA-256: `90a65540dfade976ad58924b8318576b38c310ac1d8831db31ceb3d4682daa27`.

Worker: `scripts/run_openvla_original_baseline.py`.
SHA-256: `813ccc08224a4e6751c115e6dde1627d686cb24b7c838ac5372460c28b55b373`.

Analyzer: `scripts/analyze_openvla_original_baseline.py`.
SHA-256: `01224191f80a73df12a780c14ccb51f142cdf65ee3bcdb78240bbcb296da9542`.

Original runtime: `envs/openvla-oft/bin/python`; GPU 0 UUID
`GPU-bb2451d6-2989-a112-5c18-8892943710e4`.
Checkpoint and dependency source identities are inherited from the completed
attention diagnostic and explicitly authenticated in the new configuration.

Run root: `results/openvla-original-baseline-40task-v01`.
Owned runner PID: `982803`; dispatched shortly before 22:57:09 UTC.
The run repeats preflight before creating its immutable launch record.
Technical launch output: `reports/openvla-original-baseline-40task-v01-launch.log`.
No launch-log or partial-outcome inspection is permitted while active.

The detached process uses the original project-local Python with
`CUDA_VISIBLE_DEVICES=0`, `PYTHONDONTWRITEBYTECODE=1`, and `PYTHONPATH=src`.
All run caches/logs remain project-local. Caps: 40 episodes, 1,692 calls,
7,200 seconds, 23,552 MiB aggregate memory (strict upper bound), 512 MiB artifacts.

Heartbeat `monitor-original-openvla-40-task-baseline` is active. Its first
creation request omitted a required destination and was rejected without
creating a task; the corrected thread-local creation succeeded. This was a
monitor-setup issue, not a model-run retry or result.

## Completion requirements

Only inspect outcomes after the complete summary and all forty records exist.
Reconcile hashes, runtime contract, resource limits and exact counts, then
compare the eight repeated conditions with the earlier original arm. Report
every failure without claiming a physical mechanism from terminal counts alone.
Do not pool repeats, compare unmatched populations as paired, or turn a strong
baseline into a positive acceleration claim.

On technical stop, preserve evidence and report without retry. Stop after this
phase; no comparator, caching or training is authorized. Sync non-cache evidence
and final report locally. No GitHub push or manuscript/poster changes.
