# ACR V4-A Technical Recovery 2

Status: **FROZEN BEFORE RECOVERY 2**

Date: 2026-08-10

## Preserved recovery-1 stop

Recovery 1 passed V3-D reconciliation, A4 trace loading, both deterministic
candidate replays, and the bootstrap computations in volatile memory. It then
stopped during A5 risk-summary validation before constructing, writing, or
printing the V4-A result. The operator did not inspect candidate values, and
`results/acr-v4a-diagnosis-v01` remained absent.

The cause is another legacy integrity distinction: A5 query/episode records
are validated by frozen JSON schemas, per-episode query hashes, runner
summaries, completion identities, and aggregate record hashes. They do not
carry per-record `semantic_sha256` fields. The V4-A analyzer incorrectly
applied its V3 semantic verifier to them.

## Narrow correction

Delegate A5 validation to the original committed
`scripts/analyze_acr_a5.py::summarize_run` implementation and require its
recomputed `records_sha256` to match the published immutable Stage-1 analysis.
Only after this validation may the V4-A analyzer compute descriptive A5 risk
counts from the same records.

No candidate, output, threshold, replay rule, bootstrap, selection rule,
executor decision, gate, source, resource cap, or result identity changes.
One complete CPU-only recovery-2 invocation is allowed after testing, merge,
and synchronization. Any further failure requires a new preserved decision.
