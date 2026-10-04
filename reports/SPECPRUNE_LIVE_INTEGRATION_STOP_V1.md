# Live comparator integration: reference-reconciliation stop

Date: 2026-09-11. Status: integration NOT accepted. No automatic retry.

The worker executed all 56 fixed offline calls and both scheduled simulator
episodes, totaling 101 model calls. It then failed the final exact comparison
between the new dense episode path and the previously consumed native reference.
This was a reconciliation failure after execution, not a GPU-memory failure or
a crash during a model call. No immutable `worker_summary.json` was produced.

## Verified technical evidence

- Config SHA-256:
  `ae52f5f8d535d5f786a050ce46800dbaff4560f007ee7e88bb0c0dd9bf599de7`.
- Owned PID 1069815, started 18:21:06 UTC, exited with status 1.
- Technical-stop message: `dense bridge changed the consumed reference episode`.
- Traceback: `reconcile`, exact dense tuple check. Expected native tuple is
  success=True, executed steps=78, policy queries=10. The technical stop does
  not report which component differs or its observed value.
- All earlier reconciliation checks precede that exact tuple assertion. Thus
  offline identity/reset tolerances, state persistence, layer checks, counts,
  episode identities and basic episode accounting did not trigger a stop.
- The worker also reverified authenticated source/checkpoint hashes and inventory
  before entering final reconciliation. Existing source and weights were not edited.
- Elapsed 178.92 seconds; peak aggregate GPU memory 16,206 MiB, below its cap.
  After exit, selected GPU 0 reported 6 MiB and 0% utilization.

No outcome analysis was performed. Only the technical stop and traceback were
opened following failure; `records.json` was copied and hash-verified as opaque
evidence. Do not infer a compressed-policy success or failure from this report.
Raw episode data must not be presented as an accepted benchmark.

## Interpretation and next diagnostic boundary

The original native baseline remains the reference. The CPU tests and offline
model checks did not establish exact episode repeatability in this live run.
The cause remains unresolved: the technical error alone cannot distinguish
changed integration semantics from run-to-run numerical/simulator variation.
It does not establish that the scientific correction hypothesis is wrong.

Do not relax the exact gate retrospectively, overwrite this run, tune pruning,
or launch another attempt automatically. The next diagnostic should first
authorize and document narrowly scoped inspection of the dense reference
discrepancy, leaving the compressed outcome unused. Then, if needed, freeze a
same-observation native-versus-dense action trace comparison before another
simulator attempt. This separates action divergence from episode repeatability
instead of changing several components at once. No recovery is launched here.

## Preservation

Six evidence files are synced byte-for-byte locally and on TITAN under
`results/specprune-episode-qualification-v01/`: launch, loaded runtime, count-only
progress, opaque records, technical stop and technical traceback. No runtime
caches were copied. Existing raw results remain immutable. No training,
installation, GitHub push or write outside `/home/ved/SAVR` on TITAN occurred.
No heartbeat was created because the bounded run was monitored in this turn.
