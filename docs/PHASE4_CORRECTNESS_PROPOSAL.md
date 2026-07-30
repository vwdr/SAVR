# Phase 4 Correctness and Instrumentation Proposal

Prepared: 2026-07-29

Status: **PROPOSAL ONLY — EXECUTION NOT AUTHORIZED**

## Purpose

Phase 4 determines whether the project-owned controller/cache implementation
is correct at query level and whether wrapped Full Refresh is behaviorally
identical to pinned, unmodified OpenVLA-OFT. It also audits timing, immutable
records, schema validation, and interrupted-run visibility before any
multi-policy simulator smoke or threshold calibration.

This phase is a correctness gate, not an efficiency or success experiment. It
cannot support a SAVR performance claim.

## Frozen software and model inputs

- SAVR base revision: the eventual approved Phase 4 runner commit
- OpenVLA-OFT: `e4287e94541f459edc4feabc4e181f537cd569a8`
- LIBERO: `8f1084e3132a39270c3a13ebe37270a43ece2a01`
- combined checkpoint:
  `638918f3d1c2e43a39a8a20772bdb8b91835e4b7`
- checkpoint directory:
  `/home/ved/SAVR/checkpoints/openvla-7b-oft-libero-four-suite`
- policy configuration: two images, current proprioception, L1 continuous
  action head, eight-action chunks, center crop enabled, FiLM disabled,
  diffusion disabled
- correctness observation: LIBERO-Spatial task `0`, initial state `0`, seed
  `0`

No new checkpoint, model, dataset, environment, or dependency download is
required.

## Execution stages

Stages are sequential. A failure stops later stages and preserves evidence.

### Stage A — dependency-light correctness expansion

Expand CPU tests to cover the complete controller contract:

| Contract | Required check |
|---|---|
| FR | refreshes every query |
| PR | exact periods `1`, `2`, and `3`, measured in policy queries |
| VOR | reacts to each camera and ignores invalid/changing state and action |
| SAVR | separate image-only, state-only, action-only, overlap, and no-trigger truth-table cases |
| Warm-up | empty cache and fewer than two action chunks force refresh |
| Horizon | exactly `H_max` consecutive reuses are allowed |
| Invalid data | missing, non-finite, and shape-incompatible image/state/action data force refresh |
| Context | episode, task, checkpoint, and configuration changes each reset controller and cache |
| Cache | age, context, tensor shape, dtype, and device are enforced |
| Failure | adapter restores the upstream method and invalidates cache |
| Records | query/episode overwrite is rejected and partial records remain visible |

Test thresholds are synthetic fixtures used only to exercise branches. They
are not calibration choices and cannot enter later evaluation configurations.

### Stage B — logging and timing implementation audit

Add project-owned query timing and validation utilities that:

- measure decision wall time
- measure synchronized total policy wall/CUDA time
- measure refresh visual backbone and projector CUDA time
- record zero visual execution and zero visual-hook calls on reuse
- retain downstream wall/CUDA time separately
- exclude warm-up only through an explicit record flag
- preserve both `query_index` and `environment_step`
- validate query, episode, and run records before aggregation

CUDA timing must use events plus explicit synchronization at declared
boundaries. A fake timing backend must test call order on CPU before GPU use.

### Stage C — interrupted-run recovery audit

Use a temporary project-local run directory to simulate:

1. a manifest entering `RUNNING`
2. several immutable query records
3. interruption before terminal completion
4. resume that writes only missing records
5. terminal status `INTERRUPTED` or `COMPLETED`

Existing records must never be overwritten. An interrupted run must remain
enumerable and distinguishable from a completed run.

### Stage D — bounded real-model parity and cache test

Only after Stages A-C pass:

1. verify SAVR, upstream, environment, and checkpoint revisions/hashes
2. select one responsibly idle TITAN RTX from aggregate metrics only
3. expose only that physical GPU through `CUDA_VISIBLE_DEVICES`
4. load the pinned combined checkpoint once
5. create one task-0/state-0 observation and close the environment after use
6. run the fixed parity query matrix below
7. restore and hash-check checkpoint metadata
8. confirm the selected GPU returns to its pre-run aggregate idle state

No rollout or benchmark episode is performed.

## Fixed real-model query matrix

At most six action-chunk queries are permitted:

| Query | Path | Observation purpose | Required result |
|---:|---|---|---|
| 1 | unmodified upstream | reference state A | reference action chunk A |
| 2 | wrapped FR | exact copy of state A/images A | bitwise-identical actions to query 1; one visual call |
| 3 | wrapped FR | second exact copy of A | bitwise-identical actions; second visual call |
| 4 | unmodified upstream | same images, controlled state B | reference action chunk B |
| 5 | wrapped VOR refresh | state A/images A | cache real projected tensor |
| 6 | wrapped VOR reuse | same images, state B | bitwise-identical actions to query 4; zero backbone/projector calls |

State B must be a finite, in-range deterministic perturbation of one
proprioception dimension. It is used only for correctness and is not sent to
the simulator.

The reuse query must additionally prove:

- cached tensor shape equals
  `[1, patches_per_image × 2, language_dimension]`
- cached tensor dtype and device match the refreshed tensor
- the proprioception projector receives normalized state B, not state A
- visual backbone and projector hook counts do not increase
- language-model and action-head paths still execute once
- cache age changes from `0` to `1`
- immutable query record fields and timing fields validate

## Parity acceptance rule

For the deterministic L1 path:

- action shapes must match exactly
- `numpy.array_equal` must be true
- maximum absolute action difference must be exactly `0`
- wrapped FR must refresh on every invocation

Any difference is a correctness failure. Do not relax tolerances, rerun with a
different seed/state, or proceed to Phase 5 without investigation and a new
user decision.

## GPU selection rule

The proposal requests authorization to inspect only aggregate per-GPU
utilization and memory, never process identities. A GPU is eligible only after
three samples two seconds apart show:

- `0%` utilization
- at most `128 MiB` memory used
- at least `22,000 MiB` memory free

Select one eligible GPU, record its physical ID and UUID, and expose only it.
If none qualifies, stop without launching.

## Resource request

| Resource | Requested bound |
|---|---:|
| Network transfer | none |
| New dependency/model/data installation | none |
| GPU count | one responsibly selected idle TITAN RTX |
| GPU/model wall-time cap | `45 minutes` |
| Real-model policy queries | at most `6` |
| Simulator use | one reset/observation only; zero rollout episodes |
| New logs/results/storage | `256 MiB` |
| Training or weight changes | none |

Previous evidence measured about `14.98 GiB` peak allocated memory and roughly
`1.27 s` per query. The requested cap is deliberately conservative for model
load, six queries, instrumentation, checkpoint restoration, and safe cleanup.

## Run identity and records

- proposed run ID: `phase4-correctness-v1`
- write manifest before model load
- write each query record immutably
- record every planned query as `COMPLETED`, `FAILED`, or `INTERRUPTED`
- never reuse the run ID or overwrite a terminal record
- record exact project/upstream/checkpoint revisions, selected GPU, timestamps,
  commands, peak memory, timing configuration, and checkpoint hashes

The Phase 4 runner and record validator must be committed and CPU-tested before
GPU selection.

## Stop conditions

Stop and preserve evidence if:

- any Stage A-C CPU test fails
- the SAVR, upstream, environment, or checkpoint revision/hash differs
- upstream source is dirty
- no GPU meets the aggregate idle rule
- more than one GPU is visible
- model loading or the test exceeds 45 minutes
- new artifacts approach 256 MiB
- peak use risks exhausting the selected GPU
- a non-finite action or tensor appears
- wrapped FR differs from unmodified upstream
- cached tensor metadata is incompatible
- current proprioception is not demonstrably fresh on reuse
- reuse executes the visual backbone or projector
- hook counts, timing synchronization, or records fail to reconcile
- checkpoint metadata cannot be restored byte-for-byte
- any write would leave `/home/ved/SAVR`
- continuing could interfere with university work

Do not silently retry, alter the state perturbation, weaken equality, or
exclude a failed record.

## Explicit exclusions

- no full simulator episode or task-success measurement
- no PR, VOR, or SAVR trajectory evaluation
- no threshold selection or calibration
- no latency/efficiency claim from six correctness queries
- no VLA-Cache work
- no final, ablation, or manuscript work
- no second GPU, training, or model change
- no process-identity inspection

## Exit gate

Phase 4 completes only when:

- every required CPU truth-table/recovery test passes
- real wrapped-FR parity is exact
- real reuse skips visual computation and preserves fresh proprioception
- timing synchronization and record validation pass
- the run has complete terminal records
- checkpoint and upstream source are clean/restored
- the checkpoint report is reviewed and merged

## Approval requested

Approve:

1. Stages A-C CPU implementation and tests
2. aggregate-only selection of one qualifying idle GPU
3. one pinned model load for at most 45 minutes
4. at most six real action-chunk queries and one simulator reset
5. up to 256 MiB of project-local correctness evidence
6. the exact parity, integrity, stop, and no-retry rules above

Approval of this proposal would authorize Phase 4 correctness execution only.
It would not authorize Phase 5, calibration, multi-episode experiments, or
paper claims.
