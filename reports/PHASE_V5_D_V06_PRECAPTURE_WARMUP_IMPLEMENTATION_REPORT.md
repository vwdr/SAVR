# Phase V5-D v06 Pre-Capture Warm-Up Implementation Report

Status: **PRE-GPU ACCEPTED; STOPPED BEFORE GPU SELECTION**

Date: 2026-08-11

## 1. Decision and evidence basis

V05 remains an immutable zero-query memory technical stop. It retained the
wrist graph before downstream eager warm-up and reached 24,939,331,584 bytes
reserved, 243,269,632 bytes above the unchanged 23 GiB cap. The retained wrist
graph had already added 434,110,464 reserved bytes, which is larger than the
cap excess. This supports one separately identified feasibility hypothesis:
complete both eager warm-ups before retaining either graph.

PyTorch's CUDA semantics require warm-up on a side stream before capture and
its shared-pool example warms both workloads before capturing graph 1 and then
graph 2 with graph 1's pool. Pool sharing remains conditional on capture-order
replay without concurrency. V06 directly implements that documented sequence:

- <https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-graphs>
- <https://docs.pytorch.org/docs/stable/generated/torch.cuda.graph.html>

This is a plausible memory-feasibility correction, not a guaranteed positive
result and not a method or paper result.

## 2. Frozen implementation

- New run ID: `acr-v5d-real-tensor-feasibility-v06`.
- V05 resolved configuration semantic SHA-256:
  `b34c1d70bbc7163419597148906c22daa82cea3b497405aeeb82afcb4802b2cf`.
- V05 technical-stop semantic SHA-256:
  `cb6d9120fc2e6ee69aaa83d677598d21741be8eaf5a3456bc21461d30eb3cc3f`.
- V06 recovery-overlay semantic SHA-256:
  `7f05a5766e40ca1c61b41a6dd9a19eb134cfa817f7e037761e3b595a90eced33`.
- V06 resolved configuration semantic SHA-256:
  `7d0976512b15c6d14486f9e83e5b14513ab7fc919bbf9b55b75c9536b90b92e6`.

The raw backend now creates one explicit stream, runs exactly three wrist
warm-ups followed by three downstream warm-ups, records memory after each,
then captures wrist followed by downstream with one shared private pool. There
is no eager warm-up between captures and no `empty_cache`, allocator change,
retry, graph-body change, cap increase, or gate change. Inherited guards still
enforce stable pointers, one replay stream, wrist-then-downstream replay,
nonconcurrency, invalidation, and exact restoration.

The compiler-first waterfall, V05 transition revalidation, checkpoint, tensor
contract, seven correctness queries, eight schedule warm-ups, 96 paired timed
queries, statistical gates, and 23 GiB cap are unchanged.

## 3. Verification completed

- All 361 local repository tests passed.
- All 8 focused V06 tests passed locally and on TITAN.
- Ruff, focused mypy, shell syntax, and diff checks passed.
- The deterministic CUDA-free preflight passed with semantic SHA-256
  `98b5fbec85f9f96c6de092fac201acb15c2ad97b34d3d124311a0d08f3770ec3`.
- Fake-CUDA tests prove all six eager warm-ups occur before either capture,
  both captures use one stream, the second uses the first graph's pool, replay
  is ordered, failures invalidate, and no capture occurs after warm-up failure.
- PR #86 passed both GitHub validation jobs and merged as
  `025b15d56439ec6d3b427f838da9778ca1c28fa7`.
- TITAN passed the complete CUDA-hidden suite: 361 tests and 9 subtests.
- TITAN's closed-stdin CUDA-hidden import/API preflight passed with CUDA
  uninitialized and semantic SHA-256
  `bfed2ce1b56b826d8187360b3b6c5db7c3f31a8b502cee899dbb5586acd39112`.
- Curated CPU-verification semantic SHA-256:
  `0075b4d598a91b66298f733fedffe572730f854453d7d7381ea934ae2f048793`.

Three earlier full-suite transports returned partial progress without an exit
status because their desktop continuation identifiers were discarded. They
are excluded orchestration attempts, used CUDA-hidden execution, produced no
scientific output, and their temporary pytest files were removed. The
independently monitored final suite passed completely.

## 4. Safety and scientific boundary

This checkpoint performed zero GPU inspection/selection, CUDA initialization,
model load/query, simulator operation, download, task-outcome access, or
manuscript modification. V01-V05 source and evidence hashes remain unchanged.
No V06 run, analysis, or verification result path exists.

## 5. Advancement boundary

V06 is ready only for separate GPU-selection coordination. The next authorized
operation, if approved, is the unchanged aggregate-only selection of one idle
GPU. No GPU inspection, selector, compiler, model, CUDA graph, scientific
query, simulator, or outcome operation is authorized by this report.
