# CAC C1H S6 Technical Stop 04

**Date:** 2026-09-01  
**Run:** `cac-c1h-headroom-s6-v01`  
**Classification:** pre-episode integration stop; not a method result

## Outcome

The single authorized C1H S6 attempt stopped fail-closed before any terminal
episode or task-success outcome. Gate H was not evaluated. This run therefore
provides no positive or negative evidence about corrected D62 closed-loop
reliability or CAC repair opportunity.

Before launch, 95 focused CUDA-hidden tests passed and the authenticated
preflight passed every check. GPU 0 was selected from aggregate telemetry at
6 MiB and 0% utilization. The worker loaded the model and made two pre-episode
model calls, then stopped at the official/custom parity control.

## Exact failure

The corrected custom forward computed the requested CAC capture tensors, but
the C1H `policy_query` wrapper returned only actions, cache, prepared input, and
salience. The parity control then attempted to read `helper["action_hidden"]`
and raised `KeyError: 'action_hidden'`.

This is a narrow return-plumbing omission in the new S6 C1H harness. It is not
an action-semantic mismatch, cache failure, memory failure, simulator failure,
Gate-H failure, or evidence against the scientific method. The earlier shifted
action-readout discrepancy was not reached because the wrapper failed before
the hidden/action comparisons could be completed.

## Integrity

- terminal episode attempts: 0;
- progress/terminal records: 0;
- model queries: 2, both pre-episode controls;
- partial outcomes opened: false;
- peak aggregate GPU-0 memory: 15,699 MiB;
- checkpoint protected bytes, inventory, and backup cleanup: exact/pass;
- restoration error: none;
- automatic retry: false;
- GPU 0 after stop: 6 MiB, 0% utilization;
- C2 started or authorized: false.

Evidence hashes:

- `technical_stop.json`: `a443bfdc64a7482864c3498a0ebe352a1f8c11561be04da17d817b7485997648`
- `technical_traceback.log`: `ea755d9170f77e9652d59ab65b77e0d0c3747d92854ca815058fd7b0c54687b8`
- `worker_manifest.json`: `e873f25754ea5814dd998a164c50e39d9d5ecf0b98f4915be5a7384d3a84d4a7`
- Stage-1 schedule: `94e3f17f10231ff3276335a967b500875f5fc224c464d328adf523ba73832252`
- extension schedule: `271272192479dde582ce701d98868d805de60a3179950d835bd8463491070868`

## Required recovery boundary

No automatic retry is permitted. Any future attempt requires a new version and
explicit authorization. The smallest valid repair is to propagate the already
computed `action_hidden` and `base_action` tensors through `policy_query` when
`capture_cac=True`. Before another simulator attempt, tests must directly
exercise that return contract and an outcome-free GPU qualification must run
the exact four pre-episode calls and prove official hidden, normalized-action,
unnormalized-action, observation-isolation, D62 anchor, and D62 reuse checks.
The population, schedule, D62 values, Gate H, resources, and C2 boundary must
remain unchanged.

