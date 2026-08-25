# BRACE-B3 v03 Autograd-Memory Technical Stop

Date: 2026-08-25

Decision: **NO BRACE RESULT; V04 NOT AUTHORIZED**

## What happened

V03 passed both prior failure points. Optimized core FR completed its frozen 22
queries. The cache-stack model then loaded, but its first custom dense query
ran without the `torch.no_grad()` protection used by official inference. This
retained autograd activations through the vision and language models. Aggregate
memory reached 24,017 MiB, so the runner correctly terminated it at the strict
23 GiB boundary. The partial query raised CUDA OOM and produced no record.

This memory use is not a valid BRACE or cache-stack measurement: it includes a
training graph that inference does not construct. Consequently, the incomplete
run is not analyzed as a scientific failure or success.

## Resource reconciliation

- completed core-FR queries: 22;
- completed cache queries: 0 (one partial attempt);
- conservative charged queries: 324;
- completed methods: core FR only;
- peak aggregate selected-GPU memory: 24,017 MiB;
- elapsed time: 81.71 seconds; and
- simulator outcomes and protected outcome access: 0.

The immutable core-FR worker, launch, logs, and technical stop are
authenticated in the machine-readable companion. Checkpoint restoration again
matched all three frozen hashes with no loader backup remaining.

## Correction and boundary

The complete custom cache path is now decorated with `torch.inference_mode()`
and asserts that gradients are disabled before preprocessing or forward work.
Official comparator calls already use `torch.no_grad()`.

No method, profile, threshold, sidecar, timing, parity, comparator, resource,
outcome, or acceptance gate changed. The no-retry rule requires explicit
authorization for v04. B4 remains unauthorized.
