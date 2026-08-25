# BRACE-B3 v01 Pre-Model Technical Stop

Date: 2026-08-25

Decision: **NO SCIENTIFIC RESULT; RECOVERY NOT AUTHORIZED**

## What happened

The aggregate-only selector sealed GPU 0 after three idle samples. The runner
then stopped at its source-tree guard, before starting any worker or loading a
model. Git quoted the preserved untracked `tmp/` paths that contain spaces, but
the guard compared those display strings with an unquoted `tmp/` prefix. It
therefore rejected the otherwise permitted project state.

## Resource and scientific boundary

- model workers, model loads, and model queries: 0;
- simulator outcomes and protected outcome fields: 0;
- downloads and writes outside `/home/ved/SAVR`: 0; and
- scientific result: none.

The selector used only aggregate GPU telemetry and did not inspect process
identities. The sealed launch record remains immutable on TITAN.

## Correction and next boundary

The guard now consumes NUL-delimited porcelain-v1 records, which leaves paths
unquoted for exact `tmp/` allowlist evaluation. This changes no BRACE method,
profile, threshold, timing gate, comparator disposition, query cap, or outcome
boundary.

The frozen no-automatic-retry rule prohibits starting a corrected v02 attempt
without an explicit recovery authorization. B4 remains unauthorized.
