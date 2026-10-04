# PAIR-VLA Phase P0 Report

Date: 2026-08-28  
Phase: P0 — research and freeze  
Outcome access: none  
GPU use: none  
Downloads: none  
Decision: **PASS P0; STOP before P1**

## Executive result

PAIR-VLA remains a defensible, testable research hypothesis after a dated
closest-work, implementation, chronology, objective, resource, and leakage
audit. The method is not declared successful. P0 establishes that the proposed
experiment can distinguish success from failure without outcome-driven method
selection.

The claim was narrowed to the pinned OpenVLA-OFT/LIBERO stack and the strongest
faithfully runnable same-stack cache comparator. Action-JND is the closest
paper, but no official implementation was found and its synthetic
perturbation-tolerance label is not the same as PAIR's actual-source recursive
expert-regret label.

## Gate adjudication

| P0 requirement | Evidence | Decision |
|---|---|---|
| Novelty defensible against dated sources | Primary-source matrix covers VLA-Cache, LAC, Action-JND, Gated VLA-Cache, and project comparators; no exact four-part collision found | PASS, scoped |
| Valid reproducible chronology path | Original sequential LIBERO HDF5 revision and exact four-suite byte set frozen; filtered no-noops RLDS rejected for primary chronology | PASS to P1 validation |
| Exact primary loss | Pinned OpenVLA-OFT source confirms 8x7 continuous masked mean L1, q99 normalization, padding exclusion, and gripper path | PASS |
| Features available before decision and costed | Raw/state/provenance features frozen; projected tiles admitted only because pinned path computes them before downstream reuse; all summary/router time charged | PASS |
| Actual recursive contracts feasible without VLA gradients | Existing SourceLedger records per-layer/token/camera/patch/K/V source; ring capacity supports frozen age four; router is detached | PASS |
| Storage/runtime/23-GiB boundary credible | 33.78-GB source set, 50-GiB P1 cap, ~389.6-GB free; prior peak 18,419 MiB; strict future stop at 23,552 MiB | PASS, not a guarantee |
| Strong comparator runnable or claim narrowed | Official Apache-2.0 VLA-Cache is pinned and previously integrated; unreleased methods remain paper-only | PASS |
| Populations, splits, caps, and stops frozen | Machine-readable config freezes all phases, sampling, statistics, power, memory, and unblinding | PASS |
| No download or GPU authorization requested | P0 used local evidence, primary-source web research, and read-only static server checks only | PASS |

## Material audit findings

1. **Closest-work risk is real but bounded.** Action-JND raises the novelty bar.
   PAIR must earn its distinction in P4 through actual-source and multi-group
   predictive benefit. If those gates fail, the positive route stops.
2. **The existing filtered RLDS data is invalid for primary temporal labels.**
   Original HDF5 demonstrations are required. P1 must verify their exact query
   construction before any labels.
3. **Projected visual features are chronologically valid in the pinned path.**
   Current projected tokens are produced before downstream cache reuse. This
   improves the router's information without hiding a downstream dense pass,
   but their cost remains part of the primary latency metric.
4. **The official-paper software stack and project comparator stack differ.**
   Transformers 4.47.0 is retained for same-stack comparator validity and must
   be disclosed.
5. **The positive-paper path is high-risk but falsifiable.** Prior BRACE timing
   evidence shows physical headroom, while earlier success degradation shows
   why an expert-regret router is necessary. Neither establishes that the new
   router will work.

## Frozen artifacts

- `docs/PAIR_P0_FREEZE.md`
- `configs/pair/p0_freeze_v1.json`
- `schemas/pair/intervention_record.schema.json`
- `schemas/pair/episode_record.schema.json`
- `schemas/pair/router_artifact.schema.json`
- `schemas/pair/run_summary.schema.json`
- `reports/PAIR_P0_SEMANTIC_MANIFEST.json`

## What P0 did not do

- no implementation or tests were added;
- no dataset/model was downloaded;
- no GPU was selected or used;
- no process or GPU allocation was inspected or changed;
- no demonstration regret or simulator outcome was opened;
- no manuscript or reference material was changed;
- no threshold was tuned from results; and
- no commit, push, or synchronization was performed.

## Required next decision

Execution is stopped at the P0/P1 boundary. P1 would require separate explicit
approval for a network transfer capped at 34 GiB and project-local new storage
capped at 50 GiB. P1 uses no GPU and must stop before implementation if the
original HDF5 source cannot authenticate both cameras and exact eight-step
chronology.
