# CAC Phase C0 Technical Stop 01

**Date:** 2026-08-30  
**Status:** **PRESERVED TECHNICAL STOP; RECOVERY NOT AUTHORIZED**

## What happened

The first CPU-only C0 manifest-freeze attempt authenticated the approved inputs
and constructed the deterministic schedules in memory. It then raised:

```text
RuntimeError: C2 transition-opportunity minimum failed
```

The stop occurred before any C0 manifest or summary was written. The empty
attempt directory `/home/ved/SAVR/reports/cac_c0` remains preserved.

## Root cause

The protocol requires at least eight transition-centered primary contracts per
task **where available**, with unavailable shortfalls reported before model
execution. The freezer instead required eight for every task.

The permitted train-role audit found exactly two genuine availability
shortfalls:

| Suite | Task | Eligible transition windows, all horizons and permitted train roles |
|---|---|---:|
| LIBERO-Goal | `open_the_middle_drawer_of_the_cabinet` | 0 |
| LIBERO-Goal | `push_the_plate_to_the_front_of_the_stove` | 0 |

For the first task, eligible non-transition windows by
`adapter_fit`/`architecture_selection` were 380/82 at horizon 1, 356/77 at
horizon 2, and 308/67 at horizon 4. For the second they were 406/91, 382/86,
and 334/76. Thus the scheduler had ample valid contracts but no demonstrated
gripper transition to balance for those tasks. This is an implementation of the
gate error, not evidence against CAC.

## Attempt identities

- C0 configuration SHA-256:
  `a6ba82461a382f2d03a450301dbd94c14131ec7e4ad80efae85cc29f988eb587`
- freezer SHA-256:
  `a5e4bb59b2332b570807d8b0fa1ab49fc5378dc4d32a1af3289110ac7820b126`
- governing protocol SHA-256:
  `eb5007eeb52065bc0db873e022bc2b8c522fe658d09ce583dbcff9356a86bba7`

## Protection and resource reconciliation

- GPU calls: 0
- model calls: 0
- simulator episodes/outcomes: 0
- locked-test action values opened: no
- state-ID 10-49 outcomes opened: no
- permitted values opened: train/development expert gripper coordinates only
- new downloads: 0
- files modified outside `/home/ved/SAVR`: none

## Frozen recovery proposal

A recovery may make only these technical changes:

1. preserve the first empty attempt root;
2. write to a new versioned `cac_c0_recovery01` root;
3. enforce eight selected transitions for every task that has at least eight
   eligible transition candidates;
4. retain all valid non-transition contracts for tasks with fewer than eight
   available transitions and emit their exact availability/shortfall table;
5. change no role assignment, contract count, horizon, seed, scientific gate,
   protected boundary, model-call budget, or method field; and
6. rerun the complete CPU-only C0 validation once, only after explicit approval.

C1 remains unauthorized.
