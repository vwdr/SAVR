# ACR V4-A Technical Recovery

Status: **FROZEN BEFORE RECOVERY**

Date: 2026-08-10

## Preserved stop

The first CPU-only V4-A analyzer invocation stopped while reconciling V3-D
metadata, before A4 trace loading, candidate replay, bootstrap analysis,
selection, or output creation. The analyzer applied the per-record semantic
hash verifier to the V3-D completion manifest. That immutable manifest
intentionally has no `semantic_sha256`; its integrity is instead checked by
the existing V3-D completion identities and aggregate record hash.

Observed error:

`RuntimeError: Semantic hash mismatch: None`

`results/acr-v4a-diagnosis-v01` remained absent. No candidate output, new
scientific result, GPU/model/simulator operation, download, protected outcome,
or manuscript change occurred.

## Narrow correction

Exclude only the V3-D completion manifest from the generic semantic-hash loop.
Continue verifying every V3-D query, episode, and summary record, and retain
the completion status, terminal-count, paired-population, source-record hash,
and immutable input checks. No candidate, threshold, bootstrap, selection,
executor rule, gate, source, resource limit, or output identity changes.

One complete CPU-only recovery invocation is allowed after this correction is
tested, published, merged, and synchronized. Any further failure requires a
new preserved recovery decision.
