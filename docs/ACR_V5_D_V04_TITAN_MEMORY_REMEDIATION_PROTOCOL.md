# ACR V5-D v04 TITAN Memory-Remediation Protocol

Status: **EXECUTED; TECHNICAL STOP BEFORE RAW MODEL LOAD; NO METHOD RESULT**

Date: 2026-08-11

## 1. Decision and scope

V04 will stay on the existing `ssh titan` host and make one narrow backend
change: the wrist and downstream raw CUDA graphs may share one PyTorch private
graph-memory pool. Everything that determines the scientific result remains
unchanged from V03: selected method, checkpoint, model inputs, controller,
four measured paths, compiler-first/raw-second waterfall, 111-query schedule,
correctness tolerances, timing design, statistical analysis, gates, memory cap,
and claim boundary.

This is a feasibility remediation, not a positive-result claim. V03 produced
zero full model queries. V04 must first pass memory and correctness before any
timing result is interpretable.

## 2. Evidence supporting the change

V03's raw backend used two `torch.cuda.CUDAGraph` objects without a `pool`
argument. PyTorch therefore assigned separate private pools. The run stopped
during the second capture at 24,937,234,432 reserved bytes, 241,172,480 bytes
above the frozen 23 GiB cap, while requesting another 22 MiB.

The pinned PyTorch 2.2 documentation states that multiple captures may share a
private pool when they are always replayed in capture order and never replayed
concurrently. It demonstrates passing `g1.pool()` into the second capture.
The pinned TITAN environment independently confirms that PyTorch
`2.2.0+cu118` exposes `torch.cuda.graph(..., pool=...)`,
`CUDAGraph.pool()`, and `CUDAGraph.capture_begin(pool=...)`.

Primary references:

- PyTorch 2.2 CUDA semantics, "Sharing memory across captures":
  https://docs.pytorch.org/docs/2.2/notes/cuda.html#sharing-memory-across-captures
- PyTorch 2.2 implementation of `torch.cuda.graph` and `CUDAGraph.pool()`:
  https://github.com/pytorch/pytorch/blob/v2.2.0/torch/cuda/graphs.py
- NVIDIA CUDA Graph Programming Guide, graph-memory reuse and ordering:
  https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html

The SAVR runtime meets the semantic precondition: every optimized query is
single-threaded and invokes the wrist graph, the scene-first token copy, and
the downstream graph in that fixed order. Wrist output is an externally owned
static tensor, not a graph-private temporary needed after wrist replay.

## 3. Frozen V04 backend contract

The raw backend must:

1. retain two distinct CUDA graph objects and all existing static input/output
   buffers;
2. allocate one explicit capture stream and use it for both captures;
3. capture wrist first with a new private pool;
4. obtain the opaque token from the wrist graph's `pool()` method;
5. capture downstream second with that exact pool token and capture stream;
6. replay exactly wrist then downstream for every query;
7. require both replays of a pair to use the same CUDA stream;
8. reject downstream-first, wrist-twice, cross-stream, concurrent, unprepared,
   pointer-drift, or post-failure use;
9. record capture-stage allocated and reserved bytes when available; and
10. invalidate fail closed after any replay or ordering failure.

The compiler attempt remains first and unchanged. Its known `sm_75` BF16 PTX
failure may authorize the raw attempt only through the existing mechanical
pre-output fallback. V04 does not skip it based on prior knowledge.

## 4. Why the remediation is credible but uncertain

The required reduction to return to the frozen cap is about 230 MiB. The two
separate graph pools currently prevent temporary allocation reuse across the
sequential captures. Sharing is therefore aimed directly at an observed
duplication mechanism and is the smallest API-supported change.

Success is not guaranteed. The allocator's actual saving depends on the
temporary-lifetime pattern, and the second graph may still require enough
simultaneous live memory to exceed the cap or physical capacity. V04 is
designed to answer that question once without adapting after seeing output.

## 5. Rejected shortcuts

- Do not change controller thresholds, reuse rate, inputs, query counts, model,
  dtype, tensor shapes, correctness tolerances, timing gates, or statistics.
- Do not reduce capture warm-ups or graph count merely to fit memory.
- Do not raise the 23 GiB reservation cap.
- Do not use `PYTORCH_CUDA_ALLOC_CONF` tuning as the primary remedy. V03 used
  fixed shapes and separate graph pools; a generic OOM suggestion is not
  evidence that allocator tuning preserves the intended capture behavior.
- Do not use multiple GPUs, CPU offload, quantization, model sharding, or a
  different checkpoint. Each would change the evaluated system materially.
- Do not test several variants and retain the best result.

## 6. Pre-GPU acceptance checkpoint

Before GPU selection, all of the following must pass locally and on TITAN with
`CUDA_VISIBLE_DEVICES` empty:

- semantic validation of the V04 overlay and its V03 evidence links;
- fake-CUDA tests for one shared pool, one shared capture stream, exact capture
  and replay order, same replay stream, nonconcurrency, pointer identity, and
  invalidation;
- confirmation of the exact pinned PyTorch pool API without CUDA visibility;
- unchanged 7 correctness, 8 warm-up, and 96 timed query identities;
- unchanged scientific sections and 23 GiB cap;
- new immutable V04 paths absent;
- repository, checkpoint, OpenVLA-OFT, and LIBERO integrity checks;
- no model load, CUDA initialization, GPU selection, simulator use, download,
  task outcome, or manuscript edit; and
- synchronized clean `main` on local, private GitHub, and `/home/ved/SAVR`.

## 7. One-attempt execution logic

After explicit GPU coordination, sample aggregate telemetry only and select at
most one eligible TITAN GPU. Freeze its physical index and UUID before model
load. Run V04 once:

1. compiler attempt;
2. exact restoration and fresh-process raw permit if compiler fails before
   output for an allowed reason;
3. shared-pool raw preparation;
4. enforce peak reservation at or below 23 GiB;
5. seven correctness queries;
6. eight untimed warm-ups;
7. ninety-six paired timed queries; and
8. byte-identical analysis plus independent gate verification.

Only a complete 111-query record can advance. Do not inspect or report partial
method outcomes.

## 8. Anticipated failures and predetermined response

- **Pool API mismatch:** stop before capture; preserve technical evidence; no
  alternate API improvisation.
- **Capture-order or replay-stream violation:** invalidate the pair and stop;
  never continue timing.
- **OOM or cap breach:** preserve stage memory trace and stop. Do not change the
  cap, allocator, warm-ups, or tensor contract in the same protocol.
- **Correctness mismatch:** stop before warm-up/timing; shared-pool aliasing is
  not acceptable even if memory fits.
- **Timing-gate failure:** retain it as a valid negative engineering result; do
  not retune on evaluation data.
- **Restoration or integrity uncertainty:** stop, restore exact protected state,
  and prohibit analysis/advancement.
- **No eligible GPU:** do not launch; wait for later coordination.

If V04 fails memory feasibility, a V05 proposal must be a separately researched
system change with a new estimand and explicit paper implications. It must not
be presented as another retry of this frozen gate.

## 9. Advancement boundary

Passing V04 establishes only real-tensor correctness and the predeclared
latency/compute gates. It does not establish task success. V5-E remains a
separate simulator protocol requiring another plan and approval.

## 10. Execution disposition

The authorized launch selected GPU 0 after three samples at 6 MiB and 0%.
The compiler repeated the expected pre-output `sm_75` BF16 failure and passed
exact restoration. The fresh raw process then stopped before model load because
its single immediate revalidation sample read 33% utilization, despite memory
having returned to 6 MiB. The same GPU later read 6 MiB and 0%.

V04 has zero raw preparation launches, full queries, correctness, warm-up,
timing, simulator, or outcome records. It therefore neither tests shared-pool
memory feasibility nor produces a method result. V04 remains immutable and
cannot be retried automatically. Evidence is preserved in
`reports/PHASE_V5_D_V04_TECHNICAL_STOP_REPORT.md` and
`reports/runtime/acr_v5_d_v04_technical_stop.json`.
