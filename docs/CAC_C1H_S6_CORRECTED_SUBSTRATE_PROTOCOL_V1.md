# CAC C1H S6 Corrected-Substrate Protocol V1

**Phase:** C1H terminal repair-opportunity screen  
**Date:** 2026-09-01  
**Decision boundary:** execute C1H once, then stop before C2

## Purpose

This protocol executes the already-frozen C1H scientific screen using the
semantically corrected OpenVLA-OFT action boundary qualified by S3, S4, and S5.
It does not change the scientific question: determine whether recursive D62
reuse preserves enough closed-loop task success to be a viable substrate for
learned cache-action correction while leaving a measurable dense-versus-D62
repair gap.

The corrected substrate is named `D62_BAL_PT1_S4C_V1`. It reuses the historical
`D62_BAL_PT1` mask proportions, salience ordering, four-interval reuse horizon,
and dense-reset rule. Only the previously invalid action readout and compact
position mapping are corrected. S4 observed no D62 selection change on its
frozen population, and S5 independently reproduced the official released
evaluator exactly while preserving recursive source ages and positive physical
headroom.

## Frozen scientific design

Nothing below may be changed after outcomes are available:

- checkpoint: `openvla-7b-oft-libero-four-suite` at the frozen revision;
- suites: LIBERO Spatial, Object, Goal, and LIBERO-10;
- Stage 1: frozen `headroom_stage1`, state IDs 0--2, 120 paired conditions,
  dense and D62 once per condition, 240 terminal episodes;
- optional extension: frozen `headroom_extension`, state IDs 3--5, another 120
  paired conditions and 240 episodes, opened only by the existing ambiguity
  rule;
- identical initial state, seed, instruction, horizon, preprocessing, action
  chunking, and arm-order schedule within each pair;
- outcome: official LIBERO terminal task success only;
- no adapter training, expert actions, states 10--49, or CAC model selection;
- all original Gate-H thresholds from
  `CACHE_ACTION_CORRECTION_EXECUTION_PROTOCOL_V2.md` remain exact.

Stage 1 proceeds when dense success is at least 75%, D62 success is at least
50%, dense minus D62 is 8--35 percentage points, and at least two suites favor
dense. It stops when dense is below 75%, D62 is below 50%, the gap is at most 2
points, or the gap exceeds 35 points. Otherwise the extension opens. The
cumulative extension proceeds only when dense is at least 75%, D62 at least
50%, the gap is 5--35 points, and at least two suites favor dense.

Passing Gate H means only that a plausible repair opportunity exists and C2
may be proposed. It is not a positive paper result and does not authorize C2.

## Qualified implementation boundary

Before any terminal episode, the worker must:

1. authenticate the S3, S4, and S5 passing evidence and corrected substrate;
2. install the S3-qualified official loader guard over the official
   `third_party/openvla-oft` source tree;
3. initialize exactly one model on exactly one selected idle GPU;
4. create one official simulator observation and keep its source copy isolated;
5. compare official and custom action hidden states, normalized actions, and
   unnormalized actions at tolerance `1e-6`;
6. execute corrected D62 anchor and recursive-reuse controls;
7. fail closed before any episode if any check fails.

The worker must use the same corrected `prepare_query`, `forward_query`, action
readout, compact-position mapping, physical source tracker, and D62 profile
path qualified in S5. No historical shifted-tail action readout is permitted.

## Outcome blinding and integrity

- During a stage, terminal records remain only in worker memory.
- Progress records contain only schedule identity, counts, bytes, elapsed time,
  and query counts; they contain no success or action fields.
- A stage's terminal records and summary are written only after exactly 240
  episodes complete.
- Monitoring may inspect only process health, progress count, artifact bytes,
  elapsed time, and aggregate selected-GPU telemetry until a complete immutable
  stage summary exists.
- The analyzer independently regenerates schedules, verifies every row and
  hash, recomputes summaries, and applies Gate H mechanically.

## Resource and failure boundaries

- one GPU, one model process, no downloads;
- at most 480 episodes and 20,000 model queries;
- at most 10 hours wall time, 1 GiB artifacts, and strictly less than 23,552
  MiB selected-GPU aggregate memory;
- runtime writes only below the new C1H output root;
- checkpoint metadata must be restored byte-for-byte on success or failure;
- any integration, simulator, resource, parity, restoration, or count failure
  produces an immutable technical stop;
- no automatic retry is permitted.

## Completion

After a completed worker summary exists, run the independent C1H analyzer,
reconcile every integrity gate, update the project evidence, sync the repository
to the local review mirror, and stop before C2. If Gate H stops, report the
scientific result without redesigning or reopening the frozen population.

