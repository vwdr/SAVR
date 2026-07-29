# Phase 2A Checkpoint Download Report

Date: 2026-07-29

Status: checkpoint verified; GPU execution blocked pending a coordinated GPU ID

## Authorization and scope

The user approved:

- PR #5 and the Phase 2A proposal
- up to `16 GiB` network transfer
- up to `20 GiB` additional project-local storage
- the pinned combined four-suite checkpoint only

No dataset, secondary checkpoint, model workload, or GPU use was authorized or performed during this step.

## Download identity

- repository: `moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10`
- requested revision: `638918f3d1c2e43a39a8a20772bdb8b91835e4b7`
- resolved revision: `638918f3d1c2e43a39a8a20772bdb8b91835e4b7`
- destination: `/home/ved/SAVR/checkpoints/openvla-7b-oft-libero-four-suite`
- download-controller commit: `600afa2c8169c63053b1643f1ae87dfbeea3c1c4`

The official repository metadata resolved to 25 files totaling `15,939,168,050` bytes. Every local file was present and matched its declared remote size.

Selected local metadata hashes:

- `config.json`: `edd5c5cf6d7927e07465cf086ebe41f7b3ec8f3b128a51f71d6db14dad7ad8b1`
- `dataset_statistics.json`: `6ec6ef68d0d5bae4cb5f9fc9acb715a22b9f4545e9e9b300d0d88695cd7afec3`
- `model.safetensors.index.json`: `ca8b53fed8133ee2afcd2fc483de8febf7f5bb0f6bcb09f91189772e59e8f659`

The complete file/size inventory is preserved in the ignored runtime record at `reports/runtime/phase2_checkpoint.json` on TITAN.

## Resource evidence

| Measurement | Result |
|---|---:|
| Declared remote payload | `15,939,168,050` bytes (`14.84 GiB`) |
| Logical checkpoint-path size | `15,939,170,759` bytes |
| Allocated checkpoint directory | `15,565,816 KiB` (about `14.85 GiB`) |
| Total project after download | `30,976,448 KiB` (about `29.54 GiB`) |
| Additional project allocation since Phase 1 | `15,567,408 KiB` (about `14.85 GiB`) |
| Project-filesystem free space after download | `445,696,460 KiB` (about `425.05 GiB`) |

The additional project allocation remained below the approved `20 GiB` cap. The pinned remote payload remained below the approved `16 GiB` transfer cap. Actual network-layer bytes, including protocol overhead, were not independently metered.

## GPU and server safety

- `CUDA_VISIBLE_DEVICES` was set to an empty value by the downloader.
- No model was loaded.
- No GPU command or workload ran.
- No GPU allocation or other users' processes were inspected.
- All checkpoint, cache, log, and report writes remained inside `/home/ved/SAVR`.
- No system-wide configuration, permission, environment, service, or unrelated university file was changed.

## Next gate

The unmodified Full Refresh smoke must not start until the user or university administrator identifies the permitted GPU ID. The agent will not infer availability by inspecting shared allocation or process data.
