# Phase 1 Resource and Installation Proposal

Prepared: 2026-07-29

Status: proposal only; no installation or asset download has been authorized.

## Recommended Phase 1 scope

Create a fully project-local OpenVLA-OFT/LIBERO environment and test imports plus CPU-only headless rendering. Do not download a VLA checkpoint, the LIBERO training dataset, or VLA-Cache assets. Do not use a GPU.

Proposed project-local locations:

- Micromamba and package cache: `/home/ved/SAVR/.micromamba`
- environment: `/home/ved/SAVR/envs/openvla-oft`
- source checkouts: `/home/ved/SAVR/third_party`
- Hugging Face/cache roots: `/home/ved/SAVR/cache`
- reports and logs: `/home/ved/SAVR/reports`

All of these paths must remain ignored by Git where appropriate.

## Proposed compatibility baseline

- Python `3.10.14`
- OpenVLA-OFT commit `e4287e94541f459edc4feabc4e181f537cd569a8`
- LIBERO commit `8f1084e3132a39270c3a13ebe37270a43ece2a01`
- PyTorch `2.2.0`, torchvision `0.17.0`, torchaudio `2.2.0`
- CUDA `11.8` PyTorch binary family
- custom Transformers fork commit `bc339d9ad707454c0c115970db43c260067c61ab`
- dlimp fork commit `040105d256bd28866cc6620621a3d5f7b6b91b46`
- FlashAttention omitted initially

The official OpenVLA-OFT evaluation documentation reports Python 3.10.14 and PyTorch 2.2.0, and requires its custom Transformers fork. PyTorch officially provides a CUDA 11.8 binary for the 2.2.0 package family. NVIDIA documents the TITAN RTX architecture as compute capability 7.5/Turing. Actual compatibility remains unverified until import and later GPU smoke tests pass.

## Storage and network estimate

Current direct measurements:

- project size: about `1.2 MiB`
- free space on the project filesystem: about `454.6 GiB`

Phase 1 budget:

| Item | Expected transfer | Conservative installed/cache allowance |
|---|---:|---:|
| Micromamba, Python, PyTorch/CUDA runtime, and Python dependencies | `5–10 GiB` | `20 GiB` |
| Pinned source repositories | under `1 GiB` | `1 GiB` |
| Render-test artifacts, reports, and contingency | negligible | `4 GiB` |
| **Phase 1 hard cap** | **up to `11 GiB`** | **`25 GiB`** |

Stop before exceeding the cap. Record actual directory sizes after installation. Do not clear or alter anything outside the project to create space.

Later-phase assets, not authorized by this proposal:

| Asset | Exact remote size | Reserved local allowance |
|---|---:|---:|
| Combined four-suite OpenVLA-OFT checkpoint | `15,939,168,050` bytes (`14.84 GiB`) | `18 GiB` |
| Optional LIBERO RLDS training dataset | `10,230,693,345` bytes (`9.53 GiB`) | not needed; do not download |
| Four separate task-specific checkpoints | about `59.38 GiB` total | not recommended initially |

The official combined checkpoint is the storage-efficient candidate because it supports Spatial, Object, Goal, and LIBERO-10 in one model. Its suitability still requires Phase 2 baseline reproduction.

Conservative lifecycle planning:

- Phase 1 environment/source cap: `25 GiB`
- one combined checkpoint allowance: `18 GiB`
- code, reports, and experiment logs/results reserve: `25 GiB`
- contingency: `2 GiB`
- estimated full-project working allowance: `70 GiB`

This is below the currently observed free space, but shared filesystem availability can change. Recheck only the project filesystem before each approved download stage.

## Planned validation after approval

1. Create the project-local directory layout and ignore rules.
2. Install Micromamba and Python 3.10.14 locally.
3. Fetch only the pinned source revisions.
4. install the pinned OpenVLA-OFT/LIBERO dependency family without `sudo`.
5. save an explicit environment specification and package inventory.
6. verify imports without loading a model.
7. attempt CPU-only headless LIBERO rendering.
8. if rendering requires GPU/EGL access or a system change, stop and request approval.
9. measure actual project storage and write a Phase 1 report.

## Primary evidence

- OpenVLA-OFT setup: https://github.com/moojink/openvla-oft/blob/e4287e94541f459edc4feabc4e181f537cd569a8/SETUP.md
- OpenVLA-OFT LIBERO evaluation guide: https://github.com/moojink/openvla-oft/blob/e4287e94541f459edc4feabc4e181f537cd569a8/LIBERO.md
- OpenVLA-OFT dependency metadata: https://github.com/moojink/openvla-oft/blob/e4287e94541f459edc4feabc4e181f537cd569a8/pyproject.toml
- LIBERO repository: https://github.com/Lifelong-Robot-Learning/LIBERO
- PyTorch 2.2.0 installation matrix: https://pytorch.org/get-started/previous-versions/
- NVIDIA Turing compatibility documentation: https://docs.nvidia.com/cuda/turing-compatibility-guide/
- Combined checkpoint metadata: https://huggingface.co/moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10

## Explicit exclusions

- no checkpoint or dataset download
- no model loading
- no GPU selection or workload
- no calibration or experiment
- no FlashAttention build
- no system package, permission, or server configuration change
- no file or cache outside `/home/ved/SAVR`

## Approval requested

Approve up to `11 GiB` of network transfer and a `25 GiB` project-local disk cap for the Phase 1 environment/source installation and CPU-only simulator smoke test.
