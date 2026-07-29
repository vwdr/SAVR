# Initial Stack Assessment: OpenVLA-OFT + LIBERO

Assessment date: 2026-07-29

## Decision

**Proceed to a controlled feasibility smoke test after user approval; do not yet commit to the stack.**

## Evidence supporting the candidate

- The official OpenVLA-OFT repository provides LIBERO evaluation code and pretrained LIBERO checkpoint examples.
- Its documented inference requirement is approximately 16 GB VRAM for LIBERO.
- TITAN reports four NVIDIA TITAN RTX devices with 24,576 MiB each.
- The official LIBERO repository provides benchmark task suites, fixed initial states, headless evaluation examples, and public installation guidance.

Official sources:

- https://github.com/moojink/openvla-oft
- https://github.com/moojink/openvla-oft/blob/main/LIBERO.md
- https://github.com/moojink/openvla-oft/blob/main/SETUP.md
- https://github.com/Lifelong-Robot-Learning/LIBERO

## Unresolved feasibility risks

- TITAN RTX is compute capability 7.5; the exact pinned PyTorch/CUDA/FlashAttention combination must support it.
- The server CUDA driver is visible, but a project-local runtime has not been created.
- MuJoCo/LIBERO headless rendering has not been tested on this host.
- Checkpoint and dataset access/storage have not been approved or tested.
- The exact OpenVLA-OFT visual-feature boundary that can be cached without changing downstream semantics has not been identified.
- OpenVLA-OFT uses multiple images and proprioception in its LIBERO example; SAVR must specify whether all visual streams refresh together and how state remains current when visual features are reused.
- A nominal 16 GB inference estimate leaves limited margin on a 24 GB card; actual peak memory must be measured in the smoke test.

## Next decision gate

Before any implementation:

1. supply and reconcile the manuscript source
2. approve storage/network downloads
3. coordinate one GPU without inspecting or interfering with others' allocations
4. create a project-local environment
5. pin upstream revisions and dependency versions
6. run one FR-only episode and record a complete manifest
