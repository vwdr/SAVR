# Project Status

Last updated: 2026-07-29

## Current phase

Phase 2 — Unmodified Full Refresh reproduction: **IN PROGRESS**

Phase 0 completed after preparation PR #1 and ledger PR #2 were explicitly approved, merged, and synchronized. Phase 1 proposal PR #3 was subsequently approved and merged as `50eabfac111f65995ce515926aaa291d345c3cf2`.

The bounded Phase 1 installation and CPU-only simulator smoke test passed. The
Phase 2A checkpoint and Full Refresh smoke were accepted and merged in PR #6.
The approved Phase 2B all-task pilot is complete and passed its predeclared
feasibility threshold. Its evidence was accepted and merged in PR #8. All
technical Phase 2 exit criteria are satisfied. Active gate: an explicit
go/no-go decision for the Phase 3 transition.

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
- GPU 0 selected using only user-authorized aggregate utilization/memory evidence
- one unmodified Full Refresh LIBERO-Spatial task 0 / initial-state 0 / seed 0 episode completed successfully
- model fit one TITAN RTX with about `14.98 GiB` peak allocated memory
- checkpoint metadata restored byte-for-byte after the official local loading path
- approved Phase 2B pilot completed all `50/50` planned calibration episodes
- Full Refresh achieved `49/50` successes, with at least `4/5` on every task and no runtime errors
- steady-state visual backbone plus projector time measured `15.874%` of total query CUDA time and `15.873%` of policy-query wall time
- the Phase 2B run, component timing, resource use, and safety audit were reconciled in `reports/PHASE2B_PILOT_REPORT.md`

Not completed:

- the manuscript method has not yet been reconciled with OpenVLA-OFT's exact visual-feature boundary
- no dataset has been downloaded
- no policy wrapper or cache implementation exists
- no SAVR/PR/VOR correctness, latency, success, threshold, or ablation experiment has run
- no empirical SAVR performance claim is supported yet
- the Phase 2-to-Phase 3 status transition has not been authorized

## Next authorized action

Decide whether the measured `15.87%` optimistic query-latency ceiling justifies
authorizing a bounded Phase 3 controller/cache implementation proposal. Do not
begin Phase 3 implementation without explicit approval.

## Candidate initial stack

OpenVLA-OFT with LIBERO remains the leading stack. Model loading, one-GPU
compatibility, and baseline feasibility are verified. The exact cache
interception boundary and output-parity behavior remain Phase 3/4 work.

## Bootstrap checkpoint

Bootstrap was accepted on 2026-07-29 after:

- the private GitHub repository exists
- the repository is present at `/home/ved/SAVR`
- required context, safety, plan, schema, CI, diagnostic, and next-handoff files validate
- the initial commit is pushed
- no SAVR implementation or large experiment has begun
- missing inputs and risks are explicitly reported
- the final report confirms no modification outside `/home/ved/SAVR` on TITAN
