# Project Status

Last updated: 2026-07-29

## Current phase

Phase 2 — Unmodified Full Refresh reproduction: **IN PROGRESS**

Phase 0 completed after preparation PR #1 and ledger PR #2 were explicitly approved, merged, and synchronized. Phase 1 proposal PR #3 was subsequently approved and merged as `50eabfac111f65995ce515926aaa291d345c3cf2`.

The bounded Phase 1 installation and CPU-only simulator smoke test passed. Phase 2A resource limits and proposal PR #5 were explicitly approved and merged. The pinned combined checkpoint was downloaded and verified within the approved limits. Active GPU gate: the permitted GPU ID must be coordinated without inspecting or inferring other users' allocations.

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
- the empty `/home/ved/.libero` artifact from the interrupted upstream prompt was narrowly inspected, removed with `rmdir`, and verified absent
- Phase 1 accepted and completed
- pinned combined four-suite checkpoint revision and all 25 declared file sizes verified
- Phase 2A checkpoint storage remained within the approved additional-storage cap

Not completed:

- the manuscript method has not yet been reconciled with OpenVLA-OFT's exact visual-feature boundary
- no checkpoint has been loaded and no unmodified Full Refresh episode has run
- no dataset has been downloaded
- no policy wrapper or cache implementation exists
- no correctness, latency, success, threshold, or ablation experiment has run
- no empirical claim is supported yet

## Next authorized action

Obtain a permitted GPU ID from the user or university administrator. Do not inspect shared processes or allocations and do not launch a GPU workload until that ID is explicitly identified.

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
