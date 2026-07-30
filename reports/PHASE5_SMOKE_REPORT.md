# Phase 5 Smoke and External-Baseline Report

Date: 2026-07-29/30  
Status: **TECHNICALLY COMPLETE; AWAITING CHECKPOINT REVIEW**

## Scope and claim boundary

Phase 5 was a bounded structural smoke test, not threshold calibration or a
paper-level method comparison. It ran FR, PR, VOR, and SAVR on
LIBERO-Spatial task 0, initial-state IDs `0-2`, seed `0`, then audited the
pinned official VLA-Cache implementation.

The three-state outcomes cannot establish non-inferiority, comparative success,
or publishable latency improvement. Phase 6 calibration remains unauthorized.

## Pinned execution

- SAVR runner revision:
  `d64088e1a45998bcd7fcc11c07ede3f52df26e3e`
- OpenVLA-OFT:
  `e4287e94541f459edc4feabc4e181f537cd569a8`
- LIBERO:
  `8f1084e3132a39270c3a13ebe37270a43ece2a01`
- Checkpoint:
  `638918f3d1c2e43a39a8a20772bdb8b91835e4b7`
- Core evidence:
  `/home/ved/SAVR/results/phase5-core-smoke-v1`
- Reconciliation:
  `/home/ved/SAVR/results/phase5-analysis-v1/analysis.json`
- GPU: physical ID `0`,
  `GPU-bb2451d6-2989-a112-5c18-8892943710e4`,
  NVIDIA TITAN RTX

The GPU showed `6 MiB` used and `0%` utilization in all three pre-run and all
three post-run aggregate samples. No process identities were inspected.

## Core-policy outcome

All 12 fixed episodes reached a valid terminal record. The runner completed
in `1,038.866 s` (`17.31 min`) and recorded 283 policy queries.

| Policy | Episodes | Successes | Queries | Refreshes | Reuses | Refresh rate |
|---|---:|---:|---:|---:|---:|---:|
| FR | 3 | 3 | 31 | 31 | 0 | 100.00% |
| PR(2) | 3 | 0 | 84 | 42 | 42 | 50.00% |
| VOR diagnostic | 3 | 0 | 84 | 30 | 54 | 35.71% |
| SAVR diagnostic | 3 | 0 | 84 | 30 | 54 | 35.71% |

FR succeeded on all three states. PR, VOR, and SAVR reached the task horizon
without success under deliberately aggressive, uncalibrated diagnostic reuse.
This is useful feasibility evidence: the reuse path is operational, but an
arbitrary high-skip setting is unsafe. It is not evidence that calibrated
SAVR, VOR, or PR will fail.

Diagnostic query-wall medians were `1,270.41 ms` for FR, `1,185.65 ms` for
PR, `1,122.39 ms` for VOR, and `1,123.04 ms` for SAVR. These mixed
refresh/reuse medians are smoke diagnostics only. They are not controlled
steady-state latency estimates and are not approved for a paper claim.

## Correctness and instrumentation audit

The independent analysis passed:

- all 12 planned policy/state pairs in the frozen order
- 283 contiguous immutable query records
- `refresh_count + reuse_count == query_count` for every episode
- exact FR, PR, VOR, and SAVR diagnostic refresh trajectories
- maximum cache age of exactly two reuses
- zero vision-backbone/projector calls on every reuse query
- one vision-backbone/projector call on every refresh query
- one language-model and action-head call on every query
- complete trigger, action-shape, finite-value, timing, and trajectory-digest
  reconciliation
- 133 total visual refresh calls and 150 reuse queries
- exact restoration of protected checkpoint bytes
- no unexpected checkpoint file
- clean pinned OpenVLA-OFT and LIBERO trees after execution

Peak allocated GPU memory was `15,344.36 MiB`; peak reserved memory was
`15,450 MiB`. Logical core-run artifacts occupied `392,019` bytes, well below
the one-GiB cap.

## Official VLA-Cache audit

Pinned:

- VLA-Cache:
  `a4909880573868dee2769343d52e793c0341678b`
- required Transformers fork:
  `9a90a37acacf453433168db8d7769b7ea3c40c06`
- audit evidence:
  `/home/ved/SAVR/results/phase5-vla-cache-compatibility-v1/audit.json`

The isolated CPU environment successfully loaded Transformers `4.47.0`,
tokenizers `0.21.1`, scikit-image `0.25.0`, and the official patch-selection
utility. It did not modify the validated SAVR environment.

The official evaluator is technically excluded from GPU comparison in its
pinned form for two independently verified reasons:

1. It appends the current primary and wrist frames, then assigns
   `prev_images` from those new last entries. Subsequent visual comparisons
   therefore receive the current frame as both the current and previous input.
2. Its episode loop catches runtime exceptions without re-raising them or
   returning an explicit error status, which violates the required terminal
   evidence contract.

Running that path would not constitute a valid published VLA-Cache comparison.
A minimal correction must be explicitly reviewed and labeled before any
external-baseline GPU trajectory evaluation.

## Safety and resource audit

- Only GPU 0 was made visible to the core run.
- No `sudo`, system installation, service/process change, or GPU allocation
  change occurred.
- No new checkpoint or dataset was downloaded.
- The isolated compatibility source, environment, and project-local pip cache
  used about `247 MiB`, below the eight-GiB cap.
- The manuscript was not modified.

One setup deviation occurred and was remediated. The first isolated pip
invocation used pip's default `/home/ved/.cache/pip` before the project-local
cache variable was added. The exact recent cache files written or updated by
that invocation were narrowly inventoried and unlinked; their empty leaf
directories were removed where safe, and a follow-up check found no recent
file remaining there. The setup script now forces
`PIP_CACHE_DIR=/home/ved/SAVR/cache/pip-vla-cache` and disables the
account-level version-check cache. No unrelated content outside this narrowly
scoped pip cache was inspected or changed.

## Exit decision

Phase 5 satisfies its technical exit criteria:

- all four core policies finished;
- no unexplained policy-specific instrumentation difference remains;
- official VLA-Cache has a reproducible technical exclusion; and
- tests, schemas, integrity, resource, and evidence audits pass.

The phase remains administratively open only for user review of the checkpoint
pull request. Do not begin Phase 6 without a new explicit decision.
