# Adaptive comparison recovery — 2026-09-25

User authorized diagnosis, the four-method development comparison, then a
prospectively specified confirmation if justified. This is not authorization
to relax parity to obtain a favorable scientific result. No paper claim yet.

## Technical diagnostic (first and separately bounded)

Preserve adaptive-qualification-v01 and v02 in place. Neither reached S1.
V01 stopped at a whole-bridge attention witness; V02 at exact command equality.
The late local command-equality relaxation is not sufficient qualification.

Run scripts/diagnose_adaptive_zero_v1.py on the coordinated GPU 0, only after
fresh identity and idle checks. Same 16 consumed observation inputs, seed 7,
original checkpoint/runtime. Four calls per input: native SDPA, repeated native
SDPA, adaptive with pruning disabled, reference with eager attention forced at
the decoder layers. Total 64 model calls, zero robot episodes, at most 30 min,
23,552 MiB aggregate GPU memory and 64 MiB artifacts. One attempt, no automatic
retry. No new weights, dependencies, dataset downloads, other GPUs or paths
outside /home/ved/SAVR. All runtime caches remain within the result directory.

Record hidden, normalized-action, raw-action and executed-command differences;
record exact repetition, gripper disagreements and continuous-motion error.
Save small action arrays but no full hidden arrays. Record and verify source
hashes before/after, input provenance through the existing reference verifier,
and immutable completion/failure artifacts. No success or latency selection.

Interpretation: adaptive-vs-reference-eager equality isolates the attention
backend from the integration path. Failure of that equality needs debugging.
Even a backend-only difference is not automatically harmless in closed loop.
Do not increase the existing tolerance ceiling to accommodate measured error.
Read all completed diagnostic results before specifying qualification recovery.

## Subsequent stages

Fix verified integration issues, exercise actual runtime paths in CPU tests,
preserve old frozen configurations and use fresh output directories. Reconcile
local/server source hashes. S1 may run only after completed qualification passes
prospectively fixed structural and behavioral requirements. Its four arms and
40-condition development population remain unchanged unless a new protocol is
explicitly documented before measurement. Repeated-image controlled timing is
a microbenchmark, not a natural deployment trace or a cross-paper speed ranking.

One confirmation study is intended, conditional on a defensible development
advantage. Define the exact claim, justified sample size, independent evaluation
conditions, seeds, timing treatment, baselines, uncertainty and stop rule BEFORE
confirmation. Do not reuse exposed development conditions as an untouched test.
Do not begin the manuscript or promise a positive result before evidence permits.
