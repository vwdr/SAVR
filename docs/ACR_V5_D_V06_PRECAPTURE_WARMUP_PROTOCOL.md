# ACR V5-D v06 Pre-Capture Warm-Up Protocol

Status: **FROZEN FOR PRE-GPU IMPLEMENTATION; GPU EXECUTION REQUIRES SEPARATE COORDINATION**

Date: 2026-08-11

## 1. Decision and scope

V05 remains an immutable pre-correctness memory technical stop. V06 creates a
new run identity and changes only the raw shared-pool backend's preparation
order: warm both graph workloads before retaining either graph. It does not
change the selected method, checkpoint, tensors, graph bodies, shared pool,
capture/replay order, compiler-first waterfall, 111-query schedule,
correctness/timing/statistical gates, 23 GiB cap, or claim boundary.

This is a one-shot environment-feasibility hypothesis. It is not a positive
result and does not authorize threshold tuning or simulator evaluation.

## 2. V05 evidence and mechanism

V05 successfully captured the wrist graph, then OOMed during downstream
warm-up before downstream capture. Its trace was:

- wrist after warm-up: 17,429,342,720 allocated; 17,983,078,400 reserved;
- wrist after capture: 18,077,560,320 allocated; 18,417,188,864 reserved;
- raw peak during downstream warm-up: 24,226,396,160 allocated;
  24,939,331,584 reserved; and
- failure margin: 14 MiB requested with approximately 7.06 MiB free.

Retaining the wrist graph added 648,217,600 allocated and 434,110,464 reserved
bytes before downstream warm-up. The frozen cap excess was 243,269,632 bytes.
The measured retained-graph reservation increment is therefore larger than the
cap gap. This arithmetic motivates, but does not guarantee, feasibility if
downstream warm-up completes before the wrist graph is retained.

## 3. Primary-source basis

PyTorch's CUDA semantics require eager warm-up on a side stream before graph
capture. Its shared-memory example explicitly creates both graphs' static
inputs, runs warm-ups of both workloads, then captures graph 1 followed by
graph 2 using graph 1's pool. PyTorch permits pool sharing only when graphs
always replay in capture order and never concurrently.

Primary references:

- [PyTorch CUDA semantics: warm-up and graph memory sharing](https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-graphs)
- [PyTorch `torch.cuda.graph` pool and stream contract](https://docs.pytorch.org/docs/stable/generated/torch.cuda.graph.html)

V06 implements that documented lifecycle directly. It does not call
`empty_cache`, change allocator configuration, reduce warm-up count, or alter
captured work.

## 4. Frozen raw preparation lifecycle

After V05's exact fresh-process transition revalidation passes, V06 must:

1. allocate the unchanged static buffers;
2. create one non-default capture/warm-up stream;
3. synchronize that stream with the current stream;
4. run exactly 3 wrist eager warm-ups on the shared stream;
5. synchronize and record memory;
6. materialize the unchanged scene-first combined-token buffer;
7. run exactly 3 downstream eager warm-ups on the same stream;
8. synchronize and record memory;
9. capture wrist once on that same stream;
10. obtain the wrist graph's private-pool token;
11. rematerialize the combined-token buffer;
12. capture downstream once on the same stream using the wrist pool; and
13. retain both graph objects and all static buffers.

There are zero eager warm-ups between wrist capture and downstream capture.
Replay remains wrist then downstream, on one stable replay stream, never
concurrently. Every existing pointer, order, stream, invalidation, restoration,
and memory-cap guard remains active.

## 5. Exact exclusions

V06 does not permit:

- `empty_cache` or allocator-environment changes;
- fewer/more warm-ups or an additional preparation attempt;
- different graph bodies, tensors, precision, checkpoint, or model;
- CPU offload, quantization, sharding, or multiple GPUs;
- cap increase or threshold/gate changes;
- backend shopping or automatic retry; or
- simulator, reward, success, task-outcome, or manuscript access.

## 6. Pre-GPU acceptance checkpoint

Before GPU coordination, all of the following must pass with CUDA hidden:

- exact V05 configuration and curated technical-stop links;
- unchanged scientific sections, transition rule, 111-query schedule, and
  23 GiB cap;
- fake-CUDA verification of 3 wrist then 3 downstream warm-ups before either
  capture, one stream, capture order, pool identity, no inter-capture warm-up,
  replay order/nonconcurrency, pointer identity, failure invalidation, and
  memory trace;
- confirmation that no `empty_cache` or allocator mutation exists;
- V01-V05 implementation/evidence hashes unchanged;
- new V06 run/analysis/verification paths absent;
- clean repository/checkpoint/upstream trees; and
- zero GPU inspection/selection, CUDA initialization, model load/query,
  simulator use, download, outcome access, or manuscript modification.

## 7. One-attempt execution logic

After separate explicit GPU coordination:

1. select one eligible GPU using the unchanged three-sample aggregate rule;
2. run the compiler attempt;
3. require exact restoration and an allowed zero-output raw permit;
4. pass the unchanged V05 sustained-idle transition rule;
5. execute the V06 pre-capture warm-up lifecycle once;
6. enforce peak reservation at or below 23 GiB;
7. run 7 correctness, 8 schedule warm-up, and 96 paired timed queries; and
8. run byte-identical analysis plus independent verification only after all
   111 records exist.

## 8. Predetermined responses

- **Pre-capture warm-up OOM/cap breach:** preserve the stage trace and stop;
  no capture, tuning, or retry.
- **Capture OOM/cap breach:** preserve the stage trace and stop; no tuning.
- **Order/stream/pool/pointer failure:** invalidate and stop.
- **Correctness failure:** stop before timing.
- **Timing-gate failure:** retain the valid negative engineering result.
- **Integrity/restoration uncertainty:** restore exact state and prohibit
  analysis or advancement.

Only a complete 111-query record can advance. V5-E remains ineligible until
this gate passes and receives separate authorization.
