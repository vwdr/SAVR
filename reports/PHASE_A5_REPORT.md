# ACR Phase A5 Checkpoint Report

**Date:** 2026-08-03

**Phase:** A5 — staged ACR development

**Disposition:** **STOPPED NEGATIVE — NO STAGE 1 CANDIDATE ADVANCED**

## Outcome

All three A4-frozen SA-ACR candidates completed the exact 30-episode Stage 1
population with zero technical failures and exact camera-component invariants.
All exceeded the 15% scene-reuse gate, but none preserved the required 30/30
success and 3/3 success on every task. The committed analyzer therefore
returned `STOP_NEGATIVE_NO_STAGE1_CANDIDATE` with an empty advancing set.

Stage 2 was not run. No threshold, horizon, hard cap, gate, population, or
candidate was changed after outcome access. This is a negative ACR method
result under the frozen development protocol; it does not support a positive
paper-level ACR claim.

## Stage 1 results

| Candidate | Success | Scene reuse | Queries | Failed gates |
|---|---:|---:|---:|---|
| `acr-t25-h2-b30` | 29/30 | 142/545 (26.06%) | 545 | success; per-task success |
| `acr-t50-h4-b55` | 24/30 | 328/692 (47.40%) | 692 | success; per-task success |
| `acr-t70-h8-b75` | 23/30 | 351/710 (49.44%) | 710 | success; per-task success |

The conservative candidate missed only Object task 6/state 0. Its per-task
success vector was `3,3,3,3,3,3,2,3,3,3`. The middle candidate's vector was
`3,3,3,2,2,2,3,1,3,2`; the aggressive candidate's vector was
`3,3,3,2,2,1,3,1,3,2`. Scientific failures were retained and never rerun.

## Instrumentation reconciliation

Across the three candidates, all 90 planned attempts produced terminal
episodes. The 1,947 policy queries contained 1,126 scene refreshes and 821
scene reuses. Every query recomputed the wrist SigLIP, DINOv2, and projector
path and invoked downstream execution. Every scene refresh invoked each scene
component exactly once, and every scene reuse invoked each scene component
zero times. There were no technical, cache, counter, timing, schema, or
restoration failures.

The stored steady visual CUDA point estimates were 154.55, 135.90, and
133.24 ms/query from conservative to aggressive. These values are retained as
development evidence only. Because no candidate passed the safety gate, the
conditional full-development efficiency comparison and paper-level selection
were not performed.

## Integrity and resource reconciliation

| Item | Result |
|---|---|
| Stage 1 attempts | 90/90 planned; 90/300 hard cap |
| Terminal episodes | 90/90 |
| Technical failures | 0 |
| Cumulative model-run time | 6,258.59 / 86,400 seconds |
| A5 result storage | 16,265,518 / 2,147,483,648 bytes |
| GPU use | one TITAN RTX, physical GPU 0 |
| Downloads | none |
| Stage 2 episodes | 0 |
| Protected populations | untouched |
| Manuscript changes | none |

The selected GPU UUID was
`GPU-bb2451d6-2989-a112-5c18-8892943710e4`. Each run began at 9 MiB and
0% aggregate utilization. Per-run final snapshots occurred before Python
fully released the model and therefore recorded 261 MiB and, for two runs,
nonzero utilization. After process exit, an aggregate check recorded 6 MiB
and 0%. Checkpoint inventories and metadata hashes matched before and after
every run, and both pinned upstream trees remained clean.

No file outside `/home/ved/SAVR` was modified. No unrelated university file,
directory, process, environment, service, GPU allocation, permission, or
server configuration was changed.

## Reproducibility

The canonical tracked analysis is
`reports/runtime/acr_a5_stage1_analysis.json`. It identifies the frozen A5
configuration SHA-256
`892207dbcac9ee5021a9ce96ce2f7dfb143d82310cd30963a20795a9af786d8a`,
candidate-source SHA-256
`8f1503e4f579df9a0a4026b178492566b4ad206830886dfba2418852574567a4`,
and analysis semantic SHA-256
`d64d27a820ce03b6a845d2555902060ace3f67a9970038b987f99689d19d4d8b`.
The complete immutable query and episode records remain on TITAN under the
three run IDs listed in that analysis.

## Phase-boundary decision

A5 stops negative before Stage 2. A6 is ineligible because no frozen ACR
candidate advanced from development. A6 was not started, A7-A9 were not
started, the final holdout remains unopened, and the manuscript remains
unchanged. Any future method attempt requires a new, explicitly authorized,
predeclared route rather than tuning or rerunning these candidates.
