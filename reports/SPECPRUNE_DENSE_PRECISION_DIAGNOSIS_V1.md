# Dense-reference discrepancy: precision-boundary diagnosis

Date: 2026-09-11. User approved the narrowly scoped dense discrepancy diagnosis.
No production patch, GPU run, automatic retry or compressed-outcome inspection.

## What actually differed

The previous native reference and the new dense episode both succeeded, each
using ten policy queries. The prior reference finished after 78 executed control
steps; the new dense bridge finished after 79. The exact gate therefore failed
on step count, not task success or query count. Both used the same authenticated
initial state. The new controller was disabled and discarded no actions.

On the fixed offline observations, the new dense bridge matched the native
model's head inputs, normalized actions and raw unnormalized actions exactly:
maximum recorded discrepancy 0. This does not establish identical commands at
the simulator boundary across all visited observations.

## Confirmed implementation inconsistency

`run_openvla_original_baseline.py` converts the returned action chunk with
`np.asarray(result, dtype=np.float32)` before queuing/executing actions.
`SpecPruneEpisodeQuery` instead uses `np.asarray(...)` without a dtype. The
checkpoint's NumPy action unnormalization combines float32 model outputs with
float64 arrays created from JSON normalization bounds, returning float64.
Consequently the new bridge omitted the explicit float32 rounding used by our
authenticated forty-task baseline. This is a difference from our baseline
wrapper, not a claim that the upstream model itself requires float32 output.

The CPU diagnostic executed only the actual checkpoint's pure NumPy
`_unnormalize_actions` method, using the authenticated fine-tuning
`dataset_statistics.json` and synthetic float32 actions. All four LIBERO suites
reproduced float64 output versus float32 baseline execution. In the fixed
synthetic Spatial example, the maximum rounding difference was approximately
2.92e-8. This number is NOT an observed action difference from the failed run.
Small differences could affect a contact-rich rollout, but causation for the
one-step difference has not been measured.

During diagnostic development, a first CPU probe used config.json's pretraining
statistics. The loader was then checked: it replaces those statistics with
dataset_statistics.json. The final saved diagnostic uses only the actual four
LIBERO statistics. Neither probe ran a model or modified runtime statistics.

## Why the checks missed this

The offline comparison checked native/raw unnormalized actions against the
bridge/raw unnormalized actions. It did not apply the forty-task baseline's
extra execution conversion before comparing them. The new CPU query-wiring
test mocked unnormalization as an identity operation on float32 values, so it
could not reveal the real float64 return type. Shape, finiteness and numerical
tolerance checks alone do not verify exact command dtype and rounding.

This was a gap in our integration tests. The failed exact episode check must
remain preserved; do not reclassify it as a pass because both episodes succeeded.

## Recommended correction and discriminating check

1. Add one explicit execution-boundary conversion matching the verified baseline
   for every future dense/compressed arm. Preserve the raw model action for
   diagnostics. Use a new versioned wrapper/configuration; leave this frozen
   bridge, qualification source and completed evidence unchanged.
2. Add regression tests with actual float64 unnormalization outputs, values not
   exactly representable in float32, gripper processing and the final command
   sent to the simulator. Verify dtype, byte identity after conversion, action
   order and no mutation of the raw chunk. Do not use all-zero float32 mocks.
3. Freeze a dense-only, same-observation trace check on this consumed condition.
   On each queried observation compare native and new dense head/action outputs,
   then compare their final float32, gripper-processed command chunks. Only the
   designated arm controls the simulator. Record observation and command hashes
   to locate the first divergence instead of relying only on episode totals.
4. Repeat with the precision-aligned dense bridge controlling the episode under
   the identical initialization and horizon. Keep arm order, tolerances, call
   caps and no-retry rule fixed before launch. Do not run/tune the compressed
   arm during this diagnostic. If commands match but separate episodes differ,
   investigate simulator/repeatability behavior rather than assuming a new bug
   in the compression method or relaxing gates after the fact.

This diagnosis identifies a concrete contract inconsistency and a test gap. It
does not yet prove that correcting the inconsistency restores 78-step execution.
No additional experiment was launched in this diagnostic step.

## Evidence and scope

Machine-readable result: `reports/SPECPRUNE_DENSE_PRECISION_DIAGNOSIS_V1.json`.
Read-only CPU script: `scripts/diagnose_specprune_dense_precision.py`.
Old records SHA-256:
`fe47fbd39c0263138dc6ce87f9aad98b50fde54367ccdc5740bbbac99b044963`.
Only dense episode fields and offline dense identity values were exposed. The
compressed-policy outcome remains uninspected. No source/checkpoint/results
were rewritten; all TITAN writes stayed inside `/home/ved/SAVR`.
