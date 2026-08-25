# BRACE-B3 v02 Narrow Recovery Freeze

Date: 2026-08-25

Status: Authorized; frozen before corrected GPU selection

## Authorized correction

V02 changes only the pre-model repository guard and immutable run identity.
The guard consumes NUL-delimited Git porcelain-v1 records, so filenames with
spaces remain raw and can be checked against the exact `tmp/` allowlist.

The resolved configuration semantic SHA-256 is
`45dbf458e014bbd6314e12de92256859f57c716a03c25e9091a2286c09f8e925`.
Its recovery overlay is
`configs/brace/b3_physical_v2_recovery.json`.

## Unchanged scientific contract

The BRACE method, P0/P1/P2 profiles, deterministic inputs, pruning layers,
eligibility thresholds, parity tolerances, timing schedule, comparator
dispositions, P4 measurement, 388 planned-query allocation, 420-query hard
cap, 23 GiB memory limit, zero-outcome boundary, and every acceptance gate are
byte-identical to the v01 base configuration.

The known VLA-Pruner upstream-file exclusion remains fixed, and its unused 32
queries cannot be reassigned. No automatic retry follows v02. B4 remains
unauthorized regardless of the result.
