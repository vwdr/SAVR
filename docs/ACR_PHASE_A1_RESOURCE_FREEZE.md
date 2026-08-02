# ACR Phase A1 Protocol and Resource Freeze

**Date:** 2026-08-02

**Status:** accepted design freeze; no ACR implementation or outcome

**Machine-readable companion:** `configs/acr/phase_a1_freeze.json`

## 1. Acceptance decision

ACR Protocol V1 is scientifically, statistically, technically, and
operationally adequate to enter CPU-only implementation. Its SHA-256 is:

`aa7153a2acae1826e09aa93fd67cb5f15989f4b4f4346be4e2dcc025196d951a`.

No protocol text was changed during A1. The narrowed novelty boundary from A0,
the unmodified upstream FR oracle, the factorized-FR correctness oracle,
independent development/confirmation/transfer/final populations, deterministic
candidate rules, co-primary gates, negative stops, and protected holdout remain
accepted as written.

This acceptance does not imply that ACR works. The camera-factorized path still
must achieve bitwise projected-token and action parity in A3. A parity failure
is a mandatory stop, not a reason to introduce a tolerance.

## 2. Adequacy review

### 2.1 Scientific adequacy

**PASS.** The protocol:

- asks a falsifiable camera-specific reuse question;
- restricts novelty to the conjunction that survived A0;
- keeps upstream two-view FR as the success and headline-latency oracle;
- prevents Object development outcomes from being called independent evidence;
- permits only one frozen method to reach Goal confirmation;
- prohibits Goal or LIBERO-10 retuning;
- protects states `10-49`, seed `7`, until method, power, analysis, and table
  shells are frozen;
- leaves reserve seeds `17` and `27` unavailable without a separate decision;
- requires both success non-inferiority and realized efficiency for a positive
  method result; and
- mechanically stops after candidate, confirmation, parity, integrity, or
  resource failure.

No method claim, threshold, population, or success margin needs amendment
before A2.

### 2.2 Statistical adequacy

**PASS, subject to the already-required A2 verification tests.** The episode is
the analysis unit, policies are paired by suite/task/state/seed, and query
records remain nested instrumentation rather than independent success samples.
Development is excluded from the final power estimate because it selects the
method. Goal and LIBERO-10 provide the independent paired discordance estimate.
The maximum final sample is capped at 1,600 distinct paired episodes and the
protocol stops rather than inventing independence if that is insufficient.

A2 must implement and independently cross-check the paired Newcombe interval,
Wilson bound, deterministic stratified bootstrap, exact discordance analysis,
and Holm correction before any protected outcome. Exact tie-breaking,
one-sided alpha `0.025`, margin `0.02`, at least 10,000 deterministic bootstrap
resamples, and the task-balanced multiple-of-40 rule remain frozen.

### 2.3 Technical adequacy

**PASS for A2 entry.** The A0 source audit established a project-owned
per-camera boundary without upstream edits. A2 has a complete CPU test matrix,
strict cache identity, fail-closed decisions, immutable records, and explicit
component accounting. Real-model parity remains deliberately unresolved until
A3. No model loading is needed to implement or test A2.

### 2.4 Operational adequacy

**PASS.** Local, private GitHub, and TITAN `main` were identical and clean at
the A1 start revision
`7e57f34a5d0b0ca2c3c84f57a61950246ef8aa61`. Pinned OpenVLA-OFT and LIBERO
trees were clean at their frozen revisions. TITAN reported four 24,576 MiB
TITAN RTX devices; no device was selected or inspected for current allocation,
and no GPU workload ran. The project occupied approximately 30.50 GiB and its
filesystem had approximately 401.46 GiB available.

GPU selection is deferred. Before a future authorized GPU phase, reconnect via
`ssh titan`, verify aggregate device utilization without inspecting process
identities, choose at most one safe device, record its physical ID, and expose
only that device to one model process.

## 3. Frozen revisions and identities

| Item | Frozen value |
|---|---|
| OpenVLA-OFT | `e4287e94541f459edc4feabc4e181f537cd569a8` |
| LIBERO | `8f1084e3132a39270c3a13ebe37270a43ece2a01` |
| Checkpoint | `638918f3d1c2e43a39a8a20772bdb8b91835e4b7` |
| ACR protocol | `aa7153a2acae1826e09aa93fd67cb5f15989f4b4f4346be4e2dcc025196d951a` |
| Query schema | `acr.query.v1` |
| Episode schema | `acr.episode.v1` |
| Run schema | `acr.run.v1` |

The SAVR revision is frozen separately in every future run manifest after the
relevant implementation/configuration checkpoint merges.

## 4. Frozen record schemas

The normative JSON Schemas are:

| Record | Path | SHA-256 |
|---|---|---|
| Query | `schemas/acr_query.schema.json` | `9852af8b40d36c64de6da633991f0a53ecc3d16836c55ea37e88fcf426891704` |
| Episode | `schemas/acr_episode.schema.json` | `ffc3edb8d8c2b3401e10d2ad66566784c595363710904bbc9ba8f27e18018fa4` |
| Run | `schemas/acr_run.schema.json` | `2abc40e7f5d0d7454736f115719e8e41745f4ed31edd55924f68b370aebf3138` |

A2 may implement validation code for these schemas but may not silently change
their required fields or semantics. A discovered defect requires a documented
pre-outcome schema revision and a new hash. Once an affected population opens,
the applicable schema remains immutable.

The schemas require:

- stable run/attempt/query/episode identities;
- explicit completed, failed, and interrupted records;
- per-camera tower/projector calls;
- a fresh wrist-image hash on every query;
- scene decision reasons and cache state;
- current proprioception, action, and context hashes;
- inclusive and steady-state timing fields;
- episode-level success and failure classification;
- resource caps, revisions, configuration hash, and recovery policy in every
  run manifest.

## 5. Frozen run identities

Run IDs use lowercase components and exactly one immutable version suffix:

```text
acr-a<phase>-<policy>-<suite-or-scope>-<purpose>-vNN
```

The normative regular expression is:

```text
^acr-a[3-8]-[a-z0-9]+(?:-[a-z0-9]+)*-v[0-9]{2}$
```

The first execution of a frozen plan uses `v01`. Recovery does not create a
new run version; it creates a new monotonically increasing attempt identity.
A `v02` run requires a documented protocol/configuration revision and user
authorization, never merely an unfavorable or interrupted result.

Frozen run IDs or templates:

| Phase | ID/template |
|---|---|
| A3 | `acr-a3-correctness-none-v01` |
| A4 | `acr-a4-upstream-fr-object-dev00-09-v01` |
| A5 | `acr-a5-sa-acr-object-stage1-<candidate>-v01` |
| A5 | `acr-a5-sa-acr-object-stage2-<candidate>-v01` |
| A6 | `acr-a6-upstream-fr-goal-confirm-v01` |
| A6 | `acr-a6-sa-acr-goal-confirm-v01` |
| A7 | `acr-a7-scene-periodic-acr-goal-baseline-v01` |
| A7 | `acr-a7-scene-visual-acr-goal-baseline-v01` |
| A7 | `acr-a7-upstream-fr-libero10-transfer-v01` |
| A7 | `acr-a7-sa-acr-libero10-transfer-v01` |
| A8 | `acr-a8-upstream-fr-<suite>-final-v01` |
| A8 | `acr-a8-sa-acr-<suite>-final-v01` |

`<candidate>` must be one of the three deterministically derived candidate IDs.
`<suite>` must be `spatial`, `object`, `goal`, or `libero10`.

Attempt and record identities are:

```text
<run_id>/<policy>/<suite>/task-XX/state-XX/seed-X/attempt-XXXX
<attempt_id>/query-XXXXXX
<attempt_id>/episode
```

## 6. Frozen recovery rules

1. Every scheduled episode begins with `attempt-0001`.
2. Records are created without overwrite and remain immutable.
3. A completed scientific failure is terminal and is never rerun.
4. An incomplete or technical attempt remains preserved and counts against the
   phase episode/time/artifact caps.
5. Cache/controller state is never guessed or reconstructed to continue an
   incomplete episode.
6. Only a never-started pairing may resume directly.
7. A permitted technical recovery restarts the full episode under the next
   attempt ID and only under a predeclared recovery rule.
8. The replacement attempt retains a link to its predecessor and the exact
   technical reason.
9. Resource, split, parity, or invariant failures stop the phase before any
   recovery unless the governing protocol explicitly permits it.
10. No artifact may be deleted, excluded, or relabeled to make counters or caps
    pass.

## 7. Runtime and storage basis

Historical measured evidence supplies the planning basis:

- steady-state upstream-FR query wall time: `1.26749 s`;
- 50-episode FR pilot: `2,703.85 s`, or `54.08 s/episode`;
- 90-episode SAVR2 stage: `5,206.483 s`, or `57.85 s/episode`;
- 70-episode SAVR3 stage: `3,784.63 s`, or `54.07 s/episode`;
- broad 1,000-episode Phase 6 aggregate: `24.83 h`, or
  `89.388 s/episode`, including repeated-run overhead;
- largest compact/video artifact rate among the bounded stages:
  `2,654,102 bytes/episode`.

Planning uses the broad maximum runtime and largest artifact rate, each with a
`1.25x` envelope: `111.735 s/episode` and `3,317,628 bytes/episode`.
These are estimates, not permission to exceed the hard caps.

## 8. Bounded phase estimates and hard caps

| Phase | Estimated runtime | Hard time cap | Estimated artifacts | Hard artifact cap | Work cap |
|---|---:|---:|---:|---:|---|
| A1 | CPU only | CPU only | below 512 MiB | 512 MiB | no model/query/episode |
| A2 | CPU only | CPU only | below 512 MiB | 512 MiB | no model/query/episode |
| A3 | 0.5 h including load | 1 h | 64 MiB | 512 MiB | 1 GPU, 16 queries, 0 episodes |
| A4 | 3.60 h | 8 h | 316.4 MiB | 1 GiB | 1 GPU, 100 attempts |
| A5 | 9.81 h | 24 h | 949.2 MiB | 2 GiB | 1 GPU, at most 300 attempts |
| A6 | 6.71 h | 16 h | 632.8 MiB | 2 GiB | 1 GPU, 200 attempts |
| A7 | 13.42 h | 32 h | 1.236 GiB | 3 GiB | 1 GPU, at most 400 attempts |
| A8 | 101.32 h | 200 h | 9.887 GiB | 12 GiB | 1 GPU, at most 3,200 attempts |

The A8 estimate has less artifact headroom than earlier phases. Before A8, the
actual A7 bytes per episode must be applied to the frozen A8 sample size. If
the projected total would reach 12 GiB, stop before opening the final
population and request a storage-plan decision. Do not reduce required records
or delete failed attempts to fit.

All future GPU phases require separate authorization. The caps do not reserve
a GPU, allow concurrent model processes, permit a download, or allow work
outside `/home/ved/SAVR`.

## 9. Artifact policy

- Runtime artifacts live only under `/home/ved/SAVR/results/<run_id>`.
- Compact schemas, configurations, hashes, reports, aggregate tables, and
  analysis code may be committed.
- Raw caches, checkpoints, environments, full image streams, and routine
  videos are not committed.
- Raw images and videos default to disabled; an exception must fit the same
  phase cap and be predeclared.
- Checkpoint copies are prohibited. Only the existing pinned checkpoint may be
  read during an authorized model phase.
- Failed and interrupted attempts count toward all caps.
- Every runner must check projected bytes and elapsed time before scheduling
  the next attempt and stop with headroom rather than cross a cap.

## 10. Population and phase lock

A1 consumed no ACR population. Object states `0-9` seed `0`, Goal states
`0-9` seed `0`, LIBERO-10 states `0-9` seed `0`, all four suites' states
`10-49` seed `7`, and reserve seeds `17/27` remain unopened by ACR.

The only next allowable phase is A2 after explicit user authorization. A2 is
CPU-only implementation and testing. It cannot load the model, use a GPU,
start LIBERO, derive thresholds, run A3 queries, or access any ACR outcome.
