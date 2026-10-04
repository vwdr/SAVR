# CAC C1H V01 Technical Stop and Recovery Plan

**Date:** 2026-08-31  
**Source run:** `cac-c1h-headroom-v01`  
**Classification:** pre-execution technical integration failure

## What happened

The worker authenticated its inputs and wrote its outcome-free schedules and
manifest, then stopped while importing the simulator utilities. Two LIBERO
environment helpers were imported from `experiments.robot.robot_utils`; in the
pinned upstream checkout they are defined in
`experiments.robot.libero.libero_utils`.

The immutable technical-stop record reports zero episode attempts, zero model
queries, zero progress records, and no partial outcome access. Therefore this
failure contains no scientific result and cannot have informed the repair.

## Bounded repair

Recovery v02 changes only those two import locations. It also adds a regression
test that fails if either helper is again imported from the wrong module. The
recovery preflight must additionally import both helpers from the pinned
upstream environment and rerun all C1H/C1/P3 regressions.

The population, schedule, paired arm order, D62 profile and reset behavior,
Gate H, model checkpoint, resource caps, outcome sealing, and stop-before-C2
boundary are unchanged. The v01 directory remains immutable. Recovery uses the
new output root `results/cac-c1h-headroom-v02-recovery01` and is not authorized
until explicit approval.
