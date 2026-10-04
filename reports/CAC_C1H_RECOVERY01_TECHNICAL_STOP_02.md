# CAC C1H Recovery 01 Technical Stop 02

Date: 2026-08-31  
Run: `cac-c1h-headroom-v02-recovery01`  
Classification: technical integration stop; not a method result

## Outcome boundary

The single authorized Recovery 01 attempt stopped during the pre-episode
official-helper parity control with `KeyError: 'prev_images'`.

- episode attempts: 0
- model queries: 1
- progress records: 0
- partial terminal outcomes opened: false
- peak aggregate selected-GPU memory: 15,275 MiB
- completed worker summary: absent
- automatic retry: false

The incremented model-query counter belongs to the failed pre-episode control.
No dense/D62 terminal episode, task-success result, or Gate-H value exists.

## Integrity and cleanup

Checkpoint metadata and protected bytes were restored exactly, stale loader
backups were removed, and GPU 0 returned to 6 MiB/0% aggregate use. The output
root is preserved immutably at
`results/cac-c1h-headroom-v02-recovery01/`.

- technical-stop file SHA-256:
  `1136ccefc257a08ea7ba13894accef9b3a427cbf7f7044dd5a5637ebf6fca42a`
- technical-stop semantic SHA-256:
  `d5e09ff100a1331f870b9419fdb3f1af74cf2d7f1d8eb774ce9214e6a1fa5c1d`
- frozen configuration semantic SHA-256:
  `f79e0436af87f221e5fb10059baf78605d861f4871b233bd9d67a40d3c35adda`

## Interpretation

This stop does not update CAC's scientific plausibility. The failure occurred
before the first scheduled episode because the pinned action path expected the
cache-specific `prev_images` observation field, while the newly added official
dense parity control supplied the standard prepared observation. That control
was intended to detect integration mismatches and correctly prevented terminal
evaluation from starting, but its own observation contract was incomplete.

There is no automatic retry authorization. C1H remains scientifically
unmeasured and C2 remains unauthorized.
