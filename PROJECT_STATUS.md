# Project Status

Last updated: 2026-07-29

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

Not completed:

- the manuscript source is not available in the synced project files
- no base VLA or benchmark has been formally accepted
- no environment has been created
- no dependencies, datasets, or checkpoints have been installed/downloaded
- no policy wrapper or cache implementation exists
- no correctness, latency, success, threshold, or ablation experiment has run
- no empirical claim is supported yet

## Candidate initial stack

OpenVLA-OFT with LIBERO is the leading candidate because its official repository provides LIBERO checkpoints/evaluation code and reports about 16 GB VRAM for LIBERO inference. TITAN exposes 24 GB TITAN RTX GPUs, so nominal capacity is plausible. Compatibility is not yet proven; TITAN RTX compute capability, CUDA/PyTorch compatibility, simulator rendering, storage, model access, and the exact visual-feature interception point all require a controlled smoke test.

## Bootstrap checkpoint

Bootstrap is accepted only when:

- the private GitHub repository exists
- the repository is present at `/home/ved/SAVR`
- required context, safety, plan, schema, CI, diagnostic, and next-handoff files validate
- the initial commit is pushed
- no SAVR implementation or large experiment has begun
- missing inputs and risks are explicitly reported
- the final report confirms no modification outside `/home/ved/SAVR` on TITAN
