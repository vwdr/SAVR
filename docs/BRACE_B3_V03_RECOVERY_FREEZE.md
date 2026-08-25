# BRACE-B3 v03 Narrow Recovery Freeze

Date: 2026-08-25

Status: Authorized; frozen before GPU selection

## Sole correction

The deterministic proprioception fixture now uses the normalization key that
the official evaluator resolves and validates during initialization. For the
pinned checkpoint and object suite, that key is
`libero_object_no_noops`. A regression test covers the exact alias.

The resolved configuration semantic SHA-256 is
`8d6922e797432fcfd079a1c61fa1071ec19b6338dbe8a21178a1b9b6bb701ef9`.

## Additional preflight audit

Before freezing v03, every private model helper used by the cache worker was
reconciled against the pinned VLA-Cache source: vision and proprio projection,
multimodal construction, action masking and hidden-state slicing, action-head
shape, unnormalization, DynamicCache return, and official cache tuple shape.
The same 592-token, 513-visual/proprio, 79-token prompt, and 8-by-7 action
layouts were independently established by the prior V5 real-model harness.

## Unchanged boundary

The method, four profiles, thresholds, sidecar design, timing schedule, parity
tolerances, comparator dispositions, 388 planned and 420 hard query caps,
23 GiB memory limit, zero-outcome rule, and all acceptance gates are unchanged.
There is no automatic retry after v03, and B4 remains unauthorized.
