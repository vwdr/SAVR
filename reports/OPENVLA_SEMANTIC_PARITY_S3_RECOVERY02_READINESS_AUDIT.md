# OpenVLA Semantic Parity S3 Recovery 02 Readiness Audit

Date: 2026-09-01  
Status: `READY_ONE_ATTEMPT_AUTHORIZED`

Recovery 02 preserves the S3 scientific contract: eight authenticated offline
observations, four calls per observation, a 32-call hard cap, alternating
official/custom order, the same tensor boundaries and `1e-6` tolerance, one
GPU, no simulator or outcomes, and no automatic retry.

Its changes are limited to the two Recovery 01 harness defects:

1. expose the custom normalized-proprio vector in the same `[8]` pre-reshape
   form captured from the official path; the checkpoint still canonicalizes it
   to `[1,8]` internally, so model computation is unchanged; and
2. encode non-finite comparison sentinels as strict-JSON-safe labeled values so
   every semantic mismatch can produce a terminal record.

The independently qualified official loader guard is source-authenticated and
unchanged. The Recovery 01 stop reconstruction, model metadata, source
revisions, frozen inputs, scientific code, execution scripts, resource caps,
and stop-before-S4 boundary are hash-authenticated.

Verification completed before GPU selection:

- local compilation, shell validation, semantic-hash reconciliation, and all
  authenticated-file hashes passed;
- the CUDA-hidden real-source/data/checkpoint preflight passed;
- 33 focused remote tests passed;
- adversarial positive-infinity, negative-infinity, NaN, wrong-cardinality,
  and actual `[8]` versus `[1,8]` mismatch cases passed strict serialization;
- the immutable Recovery 02 output root was absent; and
- all four GPUs were aggregate-idle; GPU 0 reported 6 MiB and 0% utilization.

The user's 2026-09-01 approval permits one Recovery 02 attempt. S4, D62/C1,
C1H, C2, simulator work, training, downloads, and automatic retry remain
unauthorized.
