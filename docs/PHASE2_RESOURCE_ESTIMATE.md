# Phase 2A Checkpoint and Full-Refresh Smoke Proposal

Prepared: 2026-07-29

Status: proposal only; checkpoint download, model loading, and GPU use are not authorized

## Purpose

Phase 2A tests whether the unmodified OpenVLA-OFT Full Refresh baseline can load and execute one fixed LIBERO episode on one user-selected TITAN RTX. It is a feasibility smoke test, not an SAVR experiment and not evidence for the paper's performance claims.

## Pinned checkpoint

- repository: `moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10`
- revision: `638918f3d1c2e43a39a8a20772bdb8b91835e4b7`
- exact remote size: `15,939,168,050` bytes (`14.84 GiB`)
- access status when verified: public and not gated
- project-local destination: `/home/ved/SAVR/checkpoints/openvla-7b-oft-libero-four-suite`

The combined checkpoint is preferred because one model covers all four target LIBERO suites. No training dataset is required or authorized.

## Requested resource bounds

| Resource | Phase 2A bound |
|---|---:|
| Network transfer | up to `16 GiB` |
| Additional checkpoint storage | up to `18 GiB` |
| Additional logs/results/cache contingency | up to `2 GiB` |
| Additional project storage hard cap | `20 GiB` |
| GPU count | exactly one user-selected GPU |
| GPU smoke wall-time cap | `60 minutes` |

The existing project uses about `14.70 GiB`; the proposed Phase 2A total project allowance is therefore about `34.70 GiB`. The project filesystem had about `439.9 GiB` free after Phase 1, but only project-scoped capacity will be rechecked immediately before an approved download.

## Required user coordination

The user must identify the permitted GPU ID. The agent will not inspect other users' processes or infer GPU availability. Every GPU command will set:

```text
CUDA_VISIBLE_DEVICES=<user-approved GPU ID>
```

Only the selected GPU may be used.

## Bounded execution sequence after approval

1. Recheck the current Git commit, project size, and project-filesystem capacity.
2. Download only the pinned checkpoint revision into the project-local checkpoint path.
3. Verify the downloaded revision and file inventory.
4. Confirm the unmodified upstream evaluation configuration and fixed smoke inputs.
5. Load the checkpoint on the approved GPU.
6. Record load success/failure, peak GPU memory, and dependency/runtime evidence.
7. Run Full Refresh on:
   - suite: `libero_spatial`
   - task ID: `0`
   - initial-state ID: `0`
   - seed: `0`
   - one episode only
8. Record a complete manifest, task outcome, query/action-chunk counts, wall time, and component timing available from the unmodified path.
9. Stop and report before any multi-task pilot, SAVR implementation, calibration, or comparison policy.

## Stop conditions

Stop immediately if:

- the additional project data would exceed `20 GiB`
- the checkpoint revision or inventory cannot be verified
- model loading requires a second GPU, system change, FlashAttention build, or write outside `/home/ved/SAVR`
- the selected GPU is unavailable or the run exceeds `60 minutes`
- an out-of-memory, simulator, dependency, or action-semantics failure occurs
- unmodified upstream inference would require a scientific or behavioral code change

Do not silently change precision, checkpoint, prompt, camera inputs, proprioception, action head, action chunking, crop behavior, or episode semantics to obtain a passing run.

## Explicit exclusions

- no LIBERO training dataset
- no other checkpoint
- no VLA-Cache assets
- no SAVR/PR/VOR implementation
- no multi-task or calibration pilot
- no training or fine-tuning
- no multi-GPU execution
- no unrelated server/GPU/process inspection
- no system-wide change

## Approval requested

Approve:

1. up to `16 GiB` network transfer and `20 GiB` additional project-local storage for the pinned combined checkpoint and bounded smoke artifacts
2. one explicitly identified GPU ID for a maximum `60-minute` unmodified Full Refresh smoke test

Download and GPU execution are separate prerequisites: the checkpoint may be downloaded only after resource approval, and no model workload may start until the GPU ID is supplied.
