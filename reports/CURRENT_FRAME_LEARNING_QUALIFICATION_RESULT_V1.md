# Current-frame learned correction: integrated qualification passed

Completed 2026-09-15 22:41:43 UTC. Run
`results/current-frame-learning-qualification-v01`, authorized resource-recovery
launcher PID 1544204. The earlier resource-busy launcher performed zero model
calls and its log remains unchanged. No source, configuration, budget or gate
was changed between those dispatches.

## Verified result

- Exactly 80 backbone calls over 16 already-consumed training observations.
- On every observation, the feature-extracting 384-token path and both zero-
  initialized corrected paths returned byte-identical processed commands to the
  qualified plain 384-token path. Query counters advanced exactly once.
- All 16 raw observations remained unchanged. Numeric feature records passed
  lossless BF16-bit-pattern/FP32 serialization and read-back checks.
- Both default-size adapters completed 64 real AdamW optimizer steps on the
  same two qualification observations: 128 updates total. All gradients and
  weights were finite, output gradients were nonzero, and no backbone parameter
  received gradients. Saved final weights reloaded with identical predictions.
- No simulator episodes or research-model evaluation occurred. These diagnostic
  adapters must not be used as the learned-method pilot's final models.

| Same-batch implementation check | Action-only adapter | Current-visual adapter |
|---|---:|---:|
| Parameters | 3,485,959 | 5,071,879 |
| Initial normalized-action mean L1 | 0.01823579 | 0.01823579 |
| Final normalized-action mean L1 | 0.00662592 | 0.00600532 |
| Optimizer steps | 64 | 64 |
| Reported fit/save/reload interval (s) | 0.809 | 1.546 |

The loss reductions establish that the implemented adapters can optimize real
extracted features. Two adjacent observations from one demonstration do not test
generalization, visual dependence, action quality in deployment or task success.
The small difference between these two training losses is not evidence that the
visual adapter is superior. The fit intervals are not complete-query latency.

## Resources and evidence

151.81 seconds inside the worker's timed run, peak aggregate GPU-0 memory
15,343 MiB, 102,703,374 artifact bytes before the summary. All are below the
frozen 30-minute, 23,552-MiB and 512-MiB limits. Source/checkpoint ancestry was
reverified, and no technical stop occurred.

The frozen CPU analyzer passed. Independent local reconciliation checked all
25 summary-listed artifact hashes, all sample/parity/fit/resource conditions,
and 18 progress records. The full directory, including numeric features and
diagnostic adapters, was copied locally. Analysis and launch log also synced.

- Config SHA-256: `b82b4ec88792d9253a7a90bbf7f4bdad90e83b2dcf384d5cdb91df1a84a3c8b5`.
- Summary SHA-256: `a0a1ea06d51883692057862cbaf38e81208486e4d717864a64e62ac5b48a2934`.
- Analysis SHA-256: `4809134cf3e680ac356394f62e612b91be9527243307c10a00aefc36189cd5f5`.

The legacy record field `backbone_sha256` names the authenticated ancestry
configuration, which transitively verifies the checkpoint/runtime. It is not
presented as a digest of a single model-weight file.

## Next step

Generate the 4,000 prospectively selected feature/teacher-label pairs: 3,200 fit
and 800 trajectory-separated validation observations across all 40 tasks.
Use unchanged 384-token compression, one frozen model and sequential dense/
compressed forward passes. New collection code passed 13 targeted CPU tests
(four new collection reconciliation tests plus nine existing sampling/record
tests). Its own source/input/resource preflight is required before launch.

Only after collection verifies may matched research fitting begin under a frozen
optimizer/checkpoint rule. Closed-loop improvement and net timing benefit remain
unmeasured. No positive learned-method or paper claim is supported yet.

All server activity used ssh titan and /home/ved/SAVR. No unrelated university
files, processes, allocations or settings were modified. No GitHub push.
