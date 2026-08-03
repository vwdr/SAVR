# ACR Phase A3 Checkpoint Report

**Date:** 2026-08-02

**Phase:** A3 — bounded real-model correctness

**Disposition:** **COMPLETE — SCIENTIFIC PASS WITH TECHNICAL RECOVERY**

## Outcome

The predeclared 12-query synthetic-input matrix completed all ACR scientific
correctness assertions within the hard 16-query cap:

1. camera-factorized FR projected tokens matched upstream FR bitwise;
2. camera-factorized FR actions matched upstream FR bitwise;
3. changing only the scene camera changed only the scene token block, and
   changing only the wrist camera changed only the wrist block;
4. scene reuse executed zero scene SigLIP, DINOv2, or projector calls;
5. every ACR query executed one fresh wrist SigLIP, DINOv2, and projector path;
6. reuse with current state B produced actions bitwise equal to upstream FR
   with state B and unchanged cameras;
7. injected shape, dtype, device, and context mismatches all failed closed to
   a scene refresh; and
8. both pinned upstream trees remained clean and the checkpoint was restored
   exactly.

This is a positive **correctness** result, not a task-success or efficiency
result. No ACR rollout outcome exists yet.

## Exact evidence

- Upstream and factorized-FR action SHA-256:
  `01284327db30978fd7f56d75858941d648dc8a5337e35143d2d7ec68dfd0869d`
- Reuse/current-state and upstream-current-state action SHA-256:
  `1da8ef14a8d41991cee17a680440f2132809e851abe77c1b31bad166fa3a6898`
- Upstream projected visual-token SHA-256:
  `21821a9554bdf6fdd87e93712488109c33b39c8157336e00eb47b12175330520`
- Reuse scene calls `(SigLIP, DINOv2, projector)`: `(0, 0, 0)`
- Reuse wrist calls `(SigLIP, DINOv2, projector)`: `(1, 1, 1)`
- Visual token shape/dtype/device: `[1, 512, 4096]`, `torch.bfloat16`, `cuda:0`
- Preserved pre-adjudication record count/hash: 14 /
  `e0c66886cdb8abcc130f044e813abb605a79b60dd64e5cdc7bebe5354e7985e4`

## Transparent technical recovery

The original attempt remains immutably marked `failed`. The pinned OpenVLA-OFT
loader temporarily rewrote checkpoint `config.json` and
`modeling_prismatic.py` and created timestamped backups, as it had in prior
validated phases. The A3 runner performed its final checkpoint inventory audit
before restoring those expected loader changes, so it raised:

`Checkpoint inventory mismatch: config.json, modeling_prismatic.py`

This occurred only after all 12 planned queries and every scientific hard-stop
assertion had completed. The committed runner is sequential: any token/action
parity, isolation, reuse, component, current-state, or fail-closed failure
would have raised before reaching the checkpoint audit. A committed CPU-only
adjudicator verified that control flow against runner revision `043853a`,
validated every immutable query record, and independently matched both action
hash pairs and all component counts.

Only the two loader-modified files were restored from the loader-created
backups, using the established Phase 4 recovery procedure. The backups were
verified against prior accepted hashes before restoration. No query was rerun.
The runner now restores these files before its final audit and again in
`finally`.

## Resource and boundary reconciliation

| Item | Result |
|---|---|
| Real-model queries | 12 planned / 16 cap |
| Additional recovery queries | 0 |
| GPU | physical GPU 0, TITAN RTX, UUID `GPU-bb2451d6-2989-a112-5c18-8892943710e4` |
| Aggregate GPU before/after | 6 MiB used, 0% utilization / 6 MiB used, 0% utilization |
| Simulator resets | 0 |
| Rollout episodes | 0 |
| Benchmark populations consumed | none |
| Downloads | none; offline mode enforced |
| A3 artifact bytes after adjudication | 118,511 / 536,870,912 cap |
| Checkpoint inventory | 25 declared files, 15,939,168,050 bytes, all sizes valid |
| Checkpoint config hash | `edd5c5cf...ad8b1` restored |
| Checkpoint modeling hash | `f40ee788...1098fa` restored |
| Upstream trees | exact pinned revisions and clean |

No file outside `/home/ved/SAVR` was modified. No unrelated process, GPU
allocation, service, or university file was inspected or changed.

## Phase-boundary decision

A3 passes after transparent technical recovery. A4 remains unauthorized. A4
would run upstream FR on the 100 fixed LIBERO-Object development episodes,
apply the ≥90/100 and per-task ≥8/10 feasibility gate, then derive exactly the
three frozen candidates twice. No A4 population has been opened.
