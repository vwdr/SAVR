# Phase 2B All-Spatial Full Refresh Pilot Proposal

Prepared: 2026-07-29

Status: approved and executed on 2026-07-29; see
`reports/PHASE2B_PILOT_REPORT.md`

## Purpose

Phase 2B determines whether the pinned combined OpenVLA-OFT checkpoint reproduces a credible Full Refresh baseline across all ten LIBERO-Spatial tasks on TITAN and quantifies where policy-query time is spent before SAVR implementation begins.

This remains calibration-split baseline validation. It is not final evaluation and cannot support SAVR performance claims.

## Primary reference

The official OpenVLA-OFT paper reports:

- combined four-suite OpenVLA-OFT policy, LIBERO-Spatial success: `97.7%`
- standard OpenVLA LIBERO evaluation: 500 trials per suite (`10 tasks × 50 episodes`)
- two camera images, proprioception, continuous L1 actions, and eight-action chunks

Primary source: https://arxiv.org/abs/2502.19645, LIBERO setup and the appendix table for the single policy trained on all four suites.

The project uses the matching combined checkpoint revision recorded in `docs/UPSTREAM_PINS.md`. The paper's result is a reference, not an assumed local result: TITAN RTX hardware and the 50-episode pilot subset differ from the upstream 500-episode evaluation.

## Fixed pilot design

- policy: unmodified Full Refresh
- suite: `libero_spatial`
- task IDs: `0-9`
- initial-state IDs per task: `0-4`
- seed: `0`
- episodes: `10 tasks × 5 states = 50`
- checkpoint: `638918f3d1c2e43a39a8a20772bdb8b91835e4b7`
- OpenVLA-OFT: `e4287e94541f459edc4feabc4e181f537cd569a8`
- two images and current proprioception
- L1 action head
- FiLM disabled
- diffusion disabled
- center crop enabled
- eight-action open-loop chunks
- upstream LIBERO-Spatial episode limit

These initial states belong to the calibration split and do not overlap the planned final holdout IDs `10-49`.

## Why 50 episodes

Five fixed states per task provide complete task coverage and enough evidence to detect a gross environment/checkpoint reproduction failure while avoiding the full 500-episode baseline before controller correctness and the final protocol are frozen.

At the upstream `97.7%` reference rate, observing at most `44/50` successes has binomial probability below `0.001`. Because this pilot uses a fixed state subset and different hardware, that calculation is only a design diagnostic, not a confirmatory statistical test.

Predeclared review thresholds:

- **pass feasibility review:** at least `45/50` successes (`90%`) and no task with `0/5`
- **gross discrepancy:** fewer than `45/50` successes or any task with `0/5`
- **inconclusive but reviewable:** simulator/runtime interruptions prevent all 50 planned episodes from reaching a terminal status

Crossing a threshold does not authorize changing the checkpoint, task inputs, state set, or success definition. A discrepancy triggers investigation and a user go/no-go decision.

## Timing and instrumentation

Record every policy query, including warm-up queries. Flag the first three queries of the complete run as warm-up and exclude them only from steady-state timing summaries.

Per-query records:

- task ID, initial-state ID, environment step, and query index
- preprocessing wall time
- synchronized total policy-query wall time
- CUDA time for the visual backbone
- CUDA time for the visual projector
- CUDA time for the action head
- residual downstream time
- action-chunk length
- GPU memory allocated/reserved

Technique:

- use CUDA events around the total model call
- attach non-mutating forward hooks to the pinned model's `vision_backbone`, `projector`, and action head
- synchronize only at declared timing boundaries
- verify hook counts equal policy-query counts
- do not change model tensors, return values, precision, or action outputs

The visual compute share will be reported both as:

- visual backbone plus projector CUDA time divided by total query CUDA time
- visual backbone plus projector CUDA time divided by synchronized end-to-end query wall time

These pilot timings establish feasibility and likely upper bounds; final claims still require the frozen timing protocol and controlled repeated evaluation.

## Run integrity

Before launch:

1. verify Git, upstream, checkpoint, and environment revisions
2. verify checkpoint metadata hashes
3. select one idle GPU using only aggregate memory/utilization samples
4. require three consecutive samples with `0%` utilization and at most `128 MiB` used
5. expose only the selected GPU through `CUDA_VISIBLE_DEVICES`

During execution:

- load the model once
- write the run manifest before the first episode
- append an immutable episode record after each terminal episode
- support resume only for missing episodes, never overwrite completed records
- preserve failed and interrupted episodes
- save videos only for failures and at most one success per task

After execution:

- verify exactly 50 terminal episode records
- reconcile task/state pairs against the frozen matrix
- verify query/hook counts
- restore and hash-check checkpoint metadata
- record aggregate GPU state after process exit
- aggregate only from raw records

## Requested resource bounds

Phase 2B requires no additional model or dataset download.

| Resource | Requested bound |
|---|---:|
| Network transfer | none beyond negligible metadata checks |
| GPU count | one responsibly selected idle TITAN RTX |
| GPU wall-time cap | `3 hours` |
| New results/log/video storage | `2 GiB` |
| Episodes | exactly `50` planned |

The measured Phase 2A episode took about `45.12 s`. A simple 50-episode projection is about 38 minutes; the three-hour cap allows unsuccessful episodes to reach the full horizon, environment resets, instrumentation, and safe cleanup.

## Stop conditions

Stop and preserve evidence if:

- no GPU satisfies the declared aggregate idle criteria
- the run would use more than one visible GPU
- model/checkpoint/configuration hashes differ
- checkpoint metadata cannot be restored exactly
- the run exceeds three hours or two GiB of new artifacts
- an out-of-memory, non-finite action, simulator corruption, or repeated environment failure occurs
- instrumentation changes action outputs or hook counts do not reconcile
- any write would leave `/home/ved/SAVR`
- continuing could interfere with shared university work

Do not silently rerun, exclude, replace, or alter a failed episode.

## Explicit exclusions

- no SAVR, PR, VOR, or VLA-Cache execution
- no threshold selection
- no final holdout state
- no dataset or checkpoint download
- no training or model-weight change
- no second GPU
- no process-identity inspection
- no manuscript claim

## Approval requested

Approve:

1. exactly 50 Full Refresh calibration-split episodes
2. one responsibly selected idle GPU for up to three hours
3. up to two GiB of project-local manifests, logs, and bounded videos
4. the predeclared coverage, timing, integrity, discrepancy, and stop rules in this proposal
