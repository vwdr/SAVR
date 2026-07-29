# Project Status

Last updated: 2026-07-29

## Current phase

Phase 1 — Environment and storage feasibility: **IN PROGRESS**

Phase 0 completed after preparation PR #1 and ledger PR #2 were explicitly approved, merged, and synchronized. Phase 1 proposal PR #3 was subsequently approved and merged as `50eabfac111f65995ce515926aaa291d345c3cf2`.

The bounded Phase 1 installation and CPU-only simulator smoke test passed on branch `agent/phase-1-environment`. Active gate: review and acceptance of the Phase 1 evidence, including the unresolved account-level LIBERO path audit described in `reports/PHASE1_REPORT.md`.

## Research objective

Evaluate whether State-Aware Visual Refresh (SAVR), a training-free wrapper around an existing VLA policy, can reduce repeated visual encoding while preserving task success.

## Proposed decision rule

A visual refresh is triggered when at least one of the following exceeds its configured threshold or limit:

1. image change
2. robot-state change
3. recent action change
4. maximum visual-feature reuse horizon

Otherwise, cached visual features are reused.

## Required comparison policies

- **FR — Full Refresh:** recompute visual features every control step.
- **PR — Periodic Refresh:** recompute on a fixed interval.
- **VOR — Visual-Only Refresh:** use image change plus a maximum horizon.
- **SAVR:** use image, robot-state, recent-action, and maximum-horizon signals.

## Current completion state

Completed:

- SAVR concept and mathematical decision rule
- comparison-policy definitions
- intended metrics
- repository safety and reproducibility rules
- initial experiment plan and machine-readable result contracts
- bounded TITAN hardware/bootstrap diagnostics
- original LaTeX manuscript added unchanged with a recorded SHA-256 checksum
- preparation PR #1 merged with explicit approval
- authoritative `main` synchronized across GitHub, TITAN, and the local review copy
- milestone and decision ledgers prepared for review
- Phase 0 ledger PR #2 merged and synchronized
- official Phase 1 requirements, revisions, and asset sizes verified
- bounded Phase 1 resource proposal prepared
- Phase 1 resource proposal approved by the user
- project-local Micromamba/Python/OpenVLA-OFT/LIBERO environment installed
- exact Conda and pip inventories recorded
- OpenVLA-OFT imports verified without loading a model
- one CPU-only OSMesa LIBERO-Spatial render/action smoke test passed
- actual Phase 1 project storage measured at about `14.70 GiB`, below the `25 GiB` cap

Not completed:

- the manuscript method has not yet been reconciled with OpenVLA-OFT's exact visual-feature boundary
- no base VLA/checkpoint has been formally accepted
- no model checkpoint or dataset has been downloaded
- no policy wrapper or cache implementation exists
- no correctness, latency, success, threshold, or ablation experiment has run
- no empirical claim is supported yet
- possible creation of `/home/ved/.libero` by LIBERO's upstream first-import behavior remains uninspected and unresolved

## Next authorized action

Review the Phase 1 checkpoint PR and decide whether to accept it. Separately decide whether to authorize a narrowly scoped inspection and possible removal of `/home/ved/.libero`; it will remain untouched without explicit authorization. Checkpoint/dataset downloads and GPU work remain unauthorized.

## Candidate initial stack

OpenVLA-OFT with LIBERO remains the leading candidate. Its Python imports and CPU-only LIBERO rendering are now verified in the project-local environment. Model loading, GPU compatibility, baseline success, and the exact visual-feature interception point remain unverified and belong to later gated phases.

## Bootstrap checkpoint

Bootstrap was accepted on 2026-07-29 after:

- the private GitHub repository exists
- the repository is present at `/home/ved/SAVR`
- required context, safety, plan, schema, CI, diagnostic, and next-handoff files validate
- the initial commit is pushed
- no SAVR implementation or large experiment has begun
- missing inputs and risks are explicitly reported
- the final report confirms no modification outside `/home/ved/SAVR` on TITAN
