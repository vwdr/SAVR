# OpenVLA Semantic Parity S3 Recovery 01 Technical Stop

Date: 2026-09-01  
Classification: `TECHNICAL_HARNESS_STOP_NO_SEMANTIC_OR_METHOD_RESULT`  
Status: `ATTEMPT_CONSUMED_NO_AUTOMATIC_RETRY`

## Result boundary

The one authorized S3 Recovery 01 attempt loaded the independently qualified
official checkpoint and executed the frozen four calls for the first offline
observation. It then stopped at the ordered `normalized_proprio` comparison.
Seven earlier preprocessing and feature-boundary comparisons had passed. No
observation completed, no simulator or task outcome was accessed, and no
scientific semantic-parity decision was produced.

The worker then encountered a second error while trying to preserve the failed
comparison, so it wrote no worker summary or comparison manifest. The output
root remained empty until the explicitly labeled post-hoc reconstruction was
added. This limitation is preserved rather than hidden.

## Deterministic diagnosis

This is a harness-shape error, not evidence that the official and corrected
policies differ.

The official boundary hook captures the normalized proprio vector immediately
before `_process_proprio_features` with shape `[8]`. The custom preparation
path reshaped the same eight normalized values early and exposed shape `[1,8]`.
The checkpoint's `_process_proprio_features` method itself reshapes either form
to `[1,8]` before applying the proprio projector. Therefore the compared
pre-reshape forms differed in shape even though their downstream model
semantics are canonicalized identically.

The comparison helper represented shape mismatch as positive infinity. The
evidence writer correctly forbids non-standard JSON numbers with
`allow_nan=false`, so it raised `ValueError: Out of range float values are not
JSON compliant` while serializing the mismatch record. That secondary writer
defect explains why no worker terminal record exists.

## Integrity

- model calls: 4;
- completed observations: 0;
- comparisons reached: 8, with 7 passing before the harness-shape stop;
- outcomes, expert actions, and raw action persistence: none;
- automatic retry: false;
- protected checkpoint hashes match their frozen values;
- no stale loader backup remains; and
- GPU 0 returned to 6 MiB and 0% utilization.

Peak aggregate memory is unavailable because the worker failed while building
its terminal record. It is intentionally recorded as unknown rather than
estimated from the prior loader qualification.

## Required recovery

A valid versioned Recovery 02 must make two narrowly scoped corrections before
another authorization can be considered:

1. compare normalized proprio only after a shared canonical reshape, or retain
   the official `[8]` pre-reshape form on both sides while leaving the model
   computation unchanged; and
2. make every failure record strict-JSON serializable, including shape and
   non-finite mismatches, with tests that prove a terminal record is written.

Recovery 02 must retain the same eight observations, 32-call cap, alternating
schedule, official checkpoint loader, tensor boundaries, `1e-6` gate,
resources, and protected-data rules. It must add adversarial failure-path tests
and a zero-policy-call shape qualification before any new model attempt.

No Recovery 02 attempt is authorized. S4, D62/C1, C1H, C2, simulator work, and
training remain blocked.

Evidence:
`results/openvla-semantic-parity-s3-v02-recovery01/technical_stop_reconstruction.json`.
