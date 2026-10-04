# PAIR-VLA Phase P3 Technical Stop 01

Date: 2026-08-29  
Attempt: `pair-p3-physical-v01`  
Decision: **TECHNICAL STOP; NO SCIENTIFIC RESULT; NO AUTOMATIC RETRY**

## What happened

The authoritative CUDA-hidden preflight passed 84 PAIR/BRACE tests and froze
688 outcome-blind model queries. Aggregate GPU telemetry showed GPU 0 at 6 MiB
used and 0% utilization, so the single authorized worker was launched there.

Before loading the model or consuming any query, importing LIBERO requested an
interactive first-run dataset path because its default `~/.libero/config.yaml`
was absent. Continuing would have written configuration outside the authorized
`/home/ved/SAVR` boundary. The prompt was refused and the worker stopped.

## Preserved evidence

- `reports/pair_p3/preflight.json`
- `results/pair-p3-physical-v01/technical_stop.json`
- Queries used: 0 of 688
- Blocks completed: 0 of 96
- Peak aggregate GPU memory: 6 MiB
- Simulator/outcome access: none
- Expert-action access: none
- Automatic retry: false
- Checkpoint metadata backup remaining: none

The import found `/home/ved/.libero` present but did not create
`/home/ved/.libero/config.yaml`. The directory was not removed because its
pre-attempt ownership/history could not be established safely.

## Corrective design for a separately authorized V02 attempt

The worker now requires `LIBERO_CONFIG_PATH` to resolve to the checked,
project-local directory `configs/pair/libero_runtime`. A complete static
`config.yaml` there points only to authenticated resources under
`/home/ved/SAVR`. The worker refuses any external LIBERO configuration and
therefore cannot trigger the interactive home-directory initialization path.

A V02 attempt must use a new immutable run identity, repeat the complete CPU
preflight against the corrected source, recheck aggregate GPU availability,
and receive explicit user approval before launch. V01 is never deleted,
overwritten, resumed, or treated as timing evidence.
