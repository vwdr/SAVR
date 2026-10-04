# Precision trace: conditional parity passes; historical gate fails

Date: 2026-09-12. Frozen experiment status: NOT ACCEPTED. No retry.

## Observations

The dense-only diagnostic completed 40 model calls over two episodes. Native
controls the first; the precision-aligned bridge controls the second. Each query
evaluates both on the same observation, executing only the designated arm.
Both episodes succeeded in 79 steps with ten queries. The historical reference
expected 78 steps, so final reconciliation failed that unchanged gate. No
completed worker summary exists. Precision alignment did not restore 78 steps
in this test; it must not be claimed to explain the earlier discrepancy.

For all twenty paired queries, head inputs, normalized and raw unnormalized
actions matched exactly: maximum discrepancy zero. Processed float32 command
chunks matched byte-for-byte on each same observation. Every call executed
32 original SDPA layers. This establishes conditional equivalence on the
observations actually visited, not general proof for all possible inputs.

Across independently reset episodes, observation and command hashes matched
for queries 1–8, but differed for 9–10. The separate-episode trace criterion
would therefore also fail. With eight executed actions per query, an identical
command prefix precedes the later observation divergence. This points to
rollout/observation repeatability, not demonstrated differing model outputs on
the same input. Raw simulator-state traces were not stored, so the precise
physics/rendering/observation cause remains unresolved. Do not claim a specific
MuJoCo defect or contact mechanism from this evidence.

## Operational evidence

- Config SHA-256:
  `2fcc0591cb04adeb28aace93745a07411f8b376908bdd79111737dc02f0a693f`.
- PID 1195889, GPU 0, started 18:31:17 UTC; exited status 1.
- Elapsed 137.38 seconds; peak aggregate GPU memory 16,184 MiB.
- Source/checkpoint verification completed before final reconciliation. Existing
  sources, weights and raw results were not changed.
- GPU returned to 6 MiB and 0% utilization after exit.
- Failure: `precision-aligned episode differs from consumed reference`.
- Seven artifacts synced under `results/dense-precision-trace-v01/`.

After the stop, only the authorized dense diagnostic's episode/trace fields
were inspected. The older compressed-policy outcome remains uninspected.
No retry, performance benchmark, training-data collection or model download.

## Decision

Preserve the failed gate and evidence. Do not repeatedly alter the model just
to recover one historical step count. Conditional action equivalence and exact
independent-rollout repetition are different properties.

Before another simulator experiment, define prospective repeatability handling:
contemporaneous native controls, paired initial states, task-level success and
measured variability, retaining command-level identity as an implementation
criterion. Do not silently replace the failed historical gate with a relaxed
threshold. A bounded repeatability measurement may be needed to set future
criteria; this is a methodological decision, not an automatic technical retry.

Independent work progressed: the current-frame corrector, action-only ablation
and spatial selector are implemented and CPU-tested. See
`reports/CURRENT_FRAME_CORRECTOR_FOUNDATION_V1.md`. No robot-data training,
corrector rollout or positive method result has occurred.
