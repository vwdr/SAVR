# ACR Phase A4 Checkpoint Report

**Date:** 2026-08-03

**Phase:** A4 — development FR and candidate freeze

**Disposition:** **COMPLETE — FEASIBILITY PASS; THREE CANDIDATES FROZEN**

## Outcome

The frozen upstream two-view Full Refresh run completed all 100
LIBERO-Object development episodes. It achieved 97/100 successes, every task
achieved at least 8/10, and all 1,773 query and trace records reconciled with
zero technical failures. The predeclared A4 feasibility gate therefore passed.

This is a positive **feasibility and candidate-derivation** result. It is not
yet a positive ACR method result: A4 executed no ACR rollout and measured no
closed-loop ACR success, reuse, latency, or visual-compute reduction.

## Frozen candidates

| Candidate | Scene threshold | Translation threshold | Horizon | Hard cap | FR-trace replay reuse |
|---|---:|---:|---:|---:|---:|
| `acr-t25-h2-b30` | 0.2476380719 | 0.5479944908 | 2 | 30% | 24.99% |
| `acr-t50-h4-b55` | 0.3004689542 | 0.6859190375 | 4 | 55% | 46.53% |
| `acr-t70-h8-b75` | 0.3004689542 | 0.6859190375 | 8 | 75% | 47.43% |

The canonical candidate record is `configs/acr/candidates.json`. Its byte
SHA-256 is `8f1503e4f579df9a0a4026b178492566b4ad206830886dfba2418852574567a4`;
its declared semantic SHA-256 is
`1cced910aec61f7666c43b90abb12c3da3b804abeb11ca3ea84fb01bc7b279ea`.
The source trace SHA-256 is
`577d683be265af7919deeb58cbbb895c9b6b1975b093c845b39e34a353fb0d69`.

## Exact feasibility evidence

| Item | Result |
|---|---|
| Run ID | `acr-a4-upstream-fr-object-dev00-09-v01` |
| Population | Object tasks 0–9, states 0–9, seed 0 |
| Terminal episodes | 100/100 |
| Successes | 97/100 |
| Per-task successes | 8, 10, 10, 10, 10, 10, 10, 10, 10, 9 |
| Technical failures | 0 |
| Query / compact trace records | 1,773 / 1,773 |
| Elapsed model-run time | 5,789.45 seconds (1.61 hours) |
| Candidate derivations | Four canonical files across two independent analyses, all byte-identical |
| ACR rollouts | 0 |

The run used physical GPU 0, TITAN RTX UUID
`GPU-bb2451d6-2989-a112-5c18-8892943710e4`. It began at 9 MiB and 0%
aggregate utilization. The runner's final snapshot occurred before model
process teardown and recorded 15,677 MiB and 19%; a subsequent aggregate
check after process exit recorded 6 MiB and 0%. The checkpoint inventory and
metadata hashes matched before and after, both pinned upstream trees were
clean, no download occurred, and no protected ACR population was accessed.

## Analysis recovery reconciliation

The first CPU-only analysis outlived a dropped SSH connection and ultimately
completed. Because no completion was visible when the connection dropped, the
precommitted preserve-and-restart rule was used to launch
`analysis-recovery-0002` without rerunning any episode or changing any
derivation rule. The original and recovery analysis records differ only in
their timestamps and corresponding record semantic hashes. Both returned
`PASS_CANDIDATES_FROZEN`, and all four candidate records have the identical
SHA-256 shown above. The immutable attempts remain preserved.

## Resource and boundary reconciliation

| Item | Result |
|---|---|
| GPU count | 1 / 1 cap |
| GPU run time | 1.61 / 8 hours |
| Episode attempts | 100 / 100 |
| Result bytes after both analyses | 35,522,545 / 1,073,741,824 |
| Downloads | none |
| Protected populations | untouched |
| Manuscript changes | none |

No file outside `/home/ved/SAVR` was modified. No unrelated university file,
process, service, environment, GPU allocation, or server configuration was
changed.

## Phase-boundary decision

A4 is complete and the three candidates are frozen. A5 has not started and
requires explicit user authorization. If authorized, A5 Stage 1 will execute
the three frozen candidates under its predeclared safety and reuse gates; no
candidate, threshold, cap, horizon, or selection rule may be changed in
response to A4 outcomes.
