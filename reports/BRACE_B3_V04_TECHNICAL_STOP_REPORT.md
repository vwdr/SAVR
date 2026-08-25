# BRACE-B3 v04 Profile-Gate Technical Stop

Date: 2026-08-25

Decision: **NO COMPLETE BRACE RESULT; V05 NOT AUTHORIZED**

## What happened

V04 confirmed the prior inference-mode correction: aggregate cache-stack
memory peaked at 18,419 MiB, safely below the strict 23 GiB boundary. Core FR
completed 22 queries. The cache worker completed 43 queries through its P0,
sidecar, corrected-cache, and first profile-anchor sequence. The first
patch-change gate then called binary `torch.maximum` with one stacked tensor,
raising `TypeError` before any profile reuse query.

This is a local tensor-API typo, not a BRACE or caching result. Because the
cache worker did not write its terminal artifact, the incomplete evidence is
not scientifically analyzed.

## Resource reconciliation

- completed core-FR queries: 22;
- completed cache queries reconstructed from technical log: 43;
- total reconstructed completions: 65;
- conservative charged queries: 324;
- peak aggregate selected-GPU memory: 18,419 MiB;
- elapsed time: 132.47 seconds; and
- simulator outcomes and protected outcome access: 0.

All immutable evidence is authenticated in the machine-readable companion.
Checkpoint restoration matched the three frozen hashes exactly.

## Correction and boundary

The score now uses proper two-operand `torch.maximum` and `torch.minimum`
calls. A real-tensor regression exercises unchanged input and a single changed
14-by-14 patch, checks the 256-patch output, and requires finite values.

No method, profile, threshold, timing, parity, comparator, resource, outcome,
or acceptance setting changed. V05 requires explicit authorization. B4 remains
unauthorized.
