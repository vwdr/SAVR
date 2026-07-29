# Phase 2A Unmodified Full Refresh Smoke Report

Date: 2026-07-29

Status: smoke completed; checkpoint PR review pending

## Scope

The user authorized responsible aggregate GPU inspection and use of an idle GPU for the approved one-episode, maximum 60-minute Full Refresh smoke.

Executed:

- aggregate per-GPU memory/utilization inspection without process identities
- one selected TITAN RTX
- pinned OpenVLA-OFT and combined four-suite checkpoint
- one LIBERO-Spatial task 0, initial-state 0, seed 0 episode
- upstream Full Refresh model/action/environment/task functions
- timing, memory, query-count, manifest, log, and bounded rollout capture

Not executed:

- SAVR, PR, VOR, or VLA-Cache
- more than one episode or task
- calibration or benchmark evaluation
- training, fine-tuning, dataset download, or multi-GPU work

## GPU selection

Only aggregate metrics were inspected. No process list, identity, command, or other user's file was inspected.

GPU 0 was selected after three samples at `21:59:16Z`, `21:59:18Z`, and `21:59:20Z` consistently showed:

- UUID: `GPU-bb2451d6-2989-a112-5c18-8892943710e4`
- model: NVIDIA TITAN RTX
- utilization: `0%`
- memory used: `6 MiB`
- memory free: `24,018 MiB`
- temperature: `35 °C`

An immediate pre-launch sample at `22:01:54Z` showed the same idle state. After the process exited, GPU 0 returned to `0%` utilization and `6 MiB` used.

## Revisions and fixed inputs

- project runner commit: `08518e55d83a07b0c6088e76f82736e939130776`
- OpenVLA-OFT: `e4287e94541f459edc4feabc4e181f537cd569a8`
- checkpoint: `638918f3d1c2e43a39a8a20772bdb8b91835e4b7`
- suite: `libero_spatial`
- task ID: `0`
- initial-state ID: `0`
- seed: `0`
- policy: Full Refresh
- two images, current proprioception, L1 action head, center crop
- eight-action open-loop chunks
- FiLM and diffusion disabled

The project runner called the pinned upstream `initialize_model`, `get_action`, `run_task`, and LIBERO environment path. It restricted the upstream ten-task loop to the one approved task and instrumented calls without changing predicted actions.

## Outcome

| Measurement | Result |
|---|---:|
| Runner status | `SUCCESS` |
| Episodes | `1` |
| Task successes | `1` |
| Task success | `true` |
| Model load time | `5.8574 s` |
| Episode time | `45.1159 s` |
| Policy queries | `10` |
| Actions produced per query | `8` |
| Mean synchronized query time | `1.2815 s` |
| Median synchronized query time | `1.2690 s` |
| Minimum query time | `1.2487 s` |
| Maximum query time | `1.4113 s` |
| Peak GPU memory allocated | `16,085,533,184` bytes (about `14.98 GiB`) |
| Peak GPU memory reserved | `16,200,499,200` bytes (about `15.09 GiB`) |

Run ID: `phase2a-fr-20260729T220204Z`.

The immutable runtime manifest, log, and 46,145-byte rollout are retained under `/home/ved/SAVR/results/phase2a-fr-20260729T220204Z` and intentionally excluded from Git.

## Integrity and limitations

The official local-checkpoint loading path temporarily updates small model-logic metadata files before loading. The project runner preserved the three original files, restored them byte-for-byte in a `finally` block, removed only the two temporary upstream backup files, and verified matching before/after SHA-256 hashes.

This is a single feasibility episode. Its success rate and timing must not be treated as paper evidence, a baseline estimate, or proof of SAVR's performance. TITAN RTX differs from the A100 hardware used in the upstream report. Component-level visual-encoder timing and the all-task Phase 2 pilot remain incomplete.

## Server-safety confirmation

- only GPU 0 was visible to the process
- the run finished in about one minute, below the 60-minute cap
- no other GPU, process identity, job, allocation, or unrelated file was inspected or changed
- all new files remained inside `/home/ved/SAVR`
- no system configuration, permission, service, or environment outside the project changed

## Next gate

Review and merge PR #6. Before the required all-Spatial-task Full Refresh pilot, prepare a separate bounded Phase 2B run-count, runtime, storage, instrumentation, and GPU-use proposal.
