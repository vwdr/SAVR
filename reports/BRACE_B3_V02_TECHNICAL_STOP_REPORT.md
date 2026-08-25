# BRACE-B3 v02 Pre-Query Technical Stop

Date: 2026-08-25

Decision: **NO SCIENTIFIC RESULT; V03 NOT AUTHORIZED**

## What happened

The v02 repository guard and GPU selection passed. The optimized FR worker
loaded the checkpoint successfully, after which its deterministic input
fixture requested proprioception statistics using `libero_object`. The official
evaluator had already resolved the checkpoint's actual key to
`libero_object_no_noops`, but the fixture did not use that resolved alias. It
raised `KeyError` before entering the model-query loop.

This is a harness/checkpoint-integration defect. It does not test or contradict
BRACE's cache logic, timing hypothesis, action parity, or reliability.

## Resource reconciliation

- completed model queries: 0;
- conservative charged allocation: 22 queries (the entire core-FR block);
- completed methods: 0;
- peak aggregate selected-GPU memory: 15,275 MiB during model loading;
- elapsed time: 25.08 seconds;
- simulator outcomes and protected outcome access: 0; and
- automatic retry: none.

Checkpoint metadata was restored exactly. All three protected hashes match
their previously frozen values, and no loader backup remains.

## Correction and boundary

The deterministic fixture now reads `cfg.unnorm_key`, which the official
evaluator resolves and validates during initialization. A synthetic regression
test covers the exact `_no_noops` alias.

No method, profile, threshold, timing design, comparator disposition, query or
memory cap, outcome boundary, or scientific gate changed. A v03 run would
require explicit recovery authorization. B4 remains unauthorized.
