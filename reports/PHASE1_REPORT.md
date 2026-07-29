# Phase 1 Environment and Storage Feasibility Report

Date: 2026-07-29

Status: complete; user authorized resolution and merge

## Scope executed

The user approved up to `11 GiB` network transfer and a `25 GiB` project-local disk cap for a Phase 1 environment/source installation and CPU-only simulator smoke test.

Executed:

- project-local Micromamba and Python environment
- pinned OpenVLA-OFT and LIBERO source revisions
- pinned dependency installation without FlashAttention
- dependency consistency check
- OpenVLA-OFT imports without model loading
- one CPU-only OSMesa LIBERO-Spatial render/action step
- environment locks and storage measurement

Not executed:

- checkpoint or dataset download
- VLA model loading
- GPU selection or workload
- SAVR implementation
- calibration, benchmark evaluation, or experiment

## Revisions and environment

- environment implementation baseline: `d609297579265cba4808afc6722d7f2d8183616b`
- OpenVLA-OFT: `e4287e94541f459edc4feabc4e181f537cd569a8`
- LIBERO: `8f1084e3132a39270c3a13ebe37270a43ece2a01`
- Transformers fork: `bc339d9ad707454c0c115970db43c260067c61ab`
- dlimp fork: `040105d256bd28866cc6620621a3d5f7b6b91b46`
- Micromamba: `2.6.2-1`, downloaded binary SHA-256 verified
- Python: `3.10.14`
- PyTorch: `2.2.0+cu118`
- torchvision: `0.17.0+cu118`
- torchaudio: `2.2.0+cu118`
- Transformers: `4.40.1`
- NumPy: `1.26.4`
- robosuite: `1.4.1`
- MuJoCo: `2.3.7`
- Gym: `0.25.2`

The complete resolved package inventories are committed in `environment/locks/`. `python -m pip check` exited successfully with `No broken requirements found`.

- `conda-linux-64-explicit.txt` SHA-256: `203914d3e5a66b5b32bd4679aa3a55c11f20db6caf7f2bd3e58b8ba5bbadddc0`
- `pip-freeze.txt` SHA-256: `0d3e3ca3e2fac46f4ce853d3e936e09abe35e5be4348caf868048881fbb98c55`

## Simulator smoke evidence

The verifier ran with `CUDA_VISIBLE_DEVICES=""`, `MUJOCO_GL=osmesa`, `PYOPENGL_PLATFORM=osmesa`, and a project-local `LIBERO_CONFIG_PATH`.

Observed:

- OpenVLA modules `prismatic` and `experiments.robot.openvla_utils` imported
- suite: `libero_spatial`
- task ID: `0`
- initial-state ID: `0`
- one seven-dimensional zero action executed
- `agentview_image` shape: `[128, 128, 3]`
- `robot0_eye_in_hand_image` shape: `[128, 128, 3]`
- `torch.cuda.is_initialized()`: `false`
- verifier exit status: `0`

Warnings were preserved in the runtime log: upstream Gym is unmaintained, TensorFlow emitted CUDA registration/no-device messages despite CUDA being hidden, and robosuite reported that no optional private macro file exists. None prevented the smoke test.

## Storage evidence

Measured with project-scoped paths after installation:

| Path | KiB | Approximate GiB |
|---|---:|---:|
| `/home/ved/SAVR` total | `15,409,040` | `14.70` |
| `.micromamba` | `1,845,376` | `1.76` |
| `envs` | `9,536,924` | `9.10` |
| `cache` | `3,984,224` | `3.80` |
| `third_party` | `658,380` | `0.63` |
| `reports/runtime` | `160` | `<0.01` |

The project remained below the approved `25 GiB` disk cap. The project filesystem reported `461,263,640 KiB` free after installation.

Exact network-transfer bytes were not directly metered and are therefore `UNVERIFIED`. The installation was limited to the approved package/source families, and no checkpoint or dataset path exists under the project.

## Compatibility corrections

Direct installation of current transitive releases was not sufficient for the pinned upstream code. The reproducible setup now includes:

- LIBERO/robosuite/MuJoCo/Gym compatibility versions
- NumPy `<2` and final pin `1.26.4`
- TensorFlow metadata/protobuf/array-record compatibility versions
- project-local OSMesa rendering
- editable compatibility mode for the pinned LIBERO checkout
- project-local LIBERO configuration to prevent interactive default-path setup

These corrections are implementation evidence, not scientific results.

## Server-boundary audit

No system-wide package, shell configuration, permission, service, process, GPU allocation, checkpoint, or dataset was changed or used.

On the first LIBERO import, upstream code created `/home/ved/.libero` before displaying its configuration prompt. The prompt was interrupted immediately. After the user explicitly authorized narrow inspection and discretionary resolution:

- the exact path `/home/ved/.libero` was inspected to a maximum depth of two
- it was confirmed to be an empty directory owned by `ved`, with its timestamp matching the interrupted import
- `rmdir` removed only that empty directory
- absence of the path was verified afterward

All subsequent LIBERO operations use `/home/ved/SAVR/cache/libero`. No unrelated account or university-server content was inspected or changed.

## Phase conclusion

The technical Phase 1 feasibility criteria passed:

- reproducible setup inputs and resolved inventories exist
- dependency consistency passed
- OpenVLA-OFT imports passed without a model
- CPU-only headless LIBERO rendering passed
- actual project storage is within the approved cap
- no system or GPU change occurred
- the sole account-path side effect from the interrupted upstream prompt was identified and reversed

Phase 1 is complete. Phase 2 checkpoint download and GPU execution remain unauthorized pending the next explicit resource and GPU-selection gate.
