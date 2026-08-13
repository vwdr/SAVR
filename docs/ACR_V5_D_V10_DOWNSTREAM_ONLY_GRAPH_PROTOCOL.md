# ACR V5-D V10 Downstream-Only CUDA Graph Protocol

Status: **AUTHORIZED FOR ONE AGGREGATE-SELECTED, FAIL-CLOSED GPU ATTEMPT**

Date: 2026-08-13

Authorization checkpoint: On 2026-08-13, the user explicitly approved V10.
After the pre-GPU checkpoint passed, the user explicitly approved one V10 GPU
attempt. Authorization is limited to one aggregate-only selection and one
frozen execution with at most 111 model queries. It excludes automatic retry,
simulator use, protected outcomes, V5-E, and manuscript changes.

## 1. Decision

V09 remains an immutable technical stop. V10 creates a new run identity and
replaces the failed two-graph optimized reuse backend with a hybrid executor:

- execute the wrist visual core eagerly against its unchanged owned static
  input/output buffers;
- materialize the unchanged scene-first combined-token buffer eagerly; and
- capture and replay only the downstream action core as one CUDA graph.

The selected `v5-a100-b40` IR-SA-ACR policy, scene-cache semantics, model,
checkpoint, input tensors, precision, inference mode, output computation,
correctness tolerances, 111-query schedule, statistical gates, 23 GiB memory
cap, resource rules, and scientific claim boundary remain unchanged.

V10 is a capture-architecture correction, not a new ACR method. Its current
authorization stops before a GPU attempt, simulator run, task-outcome access,
V5-E, or a manuscript change.

## 2. Evidence requiring an architecture change

V08 and V09 independently completed both eager pre-capture warm-ups and the
wrist graph capture, then failed while ending the downstream graph capture.
Their only allocator difference did not change the failure boundary:

| Attempt | Allocator | Peak reserved | Captures completed | Stop |
|---|---|---:|---|---|
| V08 | `expandable_segments:True` | 15.2773 GiB | wrist | downstream capture |
| V09 | default `native` | 15.3848 GiB | wrist | downstream capture |

Both were more than 7.6 GiB below the frozen 23 GiB cap. Missing inference
semantics caused the earlier memory blocker, but neither memory capacity nor
allocator choice explains the surviving failure.

CUDA reports an invalidated capture only when the capture has already
encountered an invalid operation; the generic error can surface at capture
end and does not identify the originating operation. The V08/V09 evidence
therefore leaves two live explanations:

1. the transition from a retained first graph to a second capture is
   incompatible with this pinned runtime/model path; or
2. the downstream core itself contains graph-unsafe work.

Capturing downstream as the first and only graph distinguishes these
explanations without changing its graph body.

## 3. Primary-source basis

PyTorch 2.2 explicitly supports partial-network capture: unsafe or unsuitable
sections may remain eager while capture-safe sections run as CUDA graphs. It
also requires side-stream warm-up, fixed tensor addresses, static shapes, one
active capture per process, and no non-captured CUDA work from another thread
during capture.

NVIDIA documents that an invalid operation invalidates the active capture and
that later capture operations may then report the previous error. NVIDIA's
PyTorch capture guidance likewise explains that the generic `901` error can
mask an earlier unsupported synchronization, allocation, or cross-stream
operation.

Primary references:

- [PyTorch 2.2 CUDA graphs, constraints, warm-up, and partial-network capture](https://docs.pytorch.org/docs/2.2/notes/cuda.html#cuda-graphs)
- [NVIDIA CUDA Programming Guide: stream-capture invalidation](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html#invalidation)
- [NVIDIA CUDA Graph guidance: capture failures](https://docs.nvidia.com/dl-cuda-graph/troubleshooting/capture-failures.html)

The sources establish that the V10 hybrid is supported in principle. They do
not prove that OpenVLA's downstream body is capturable on TITAN; V10 is the
bounded test of that proposition.

## 4. Why this route remains meaningful for a positive-results paper

The earlier V3-C real-model timing record used the same OpenVLA inference
stack and found, across 12 timed reuse queries:

- median reuse total CUDA time: `1151.7416 ms`;
- median wrist visual CUDA time: `75.6681 ms`; and
- derived downstream CUDA time: `1076.0382 ms`, or `93.43%` of total CUDA
  time.

The V5-B selected controller also produced a `35.48%` reuse rate and `17.74%`
logical visual-work reduction on its frozen outcome-blind population.

Thus V10 leaves only the smaller wrist portion eager and targets the dominant
downstream portion for launch-overhead reduction. This gives a strong reason
to test the hybrid while avoiding a claim that it must succeed. A positive
paper route still requires V10 correctness and efficiency, then separately
authorized closed-loop task-success evidence.

## 5. Falsifiable V10 hypothesis

**Primary technical hypothesis:** The V08/V09 failure depends on the
two-capture lifecycle, not on the unchanged downstream graph body. In a fresh
raw process with no retained wrist graph, the downstream core will complete
one capture and replay correctly.

**Efficiency hypothesis:** Because the downstream core represented 93.43% of
prior reuse CUDA time, graphing it alone can still satisfy the unchanged V5-D
end-to-end and paired CUDA efficiency gates.

Predetermined interpretations:

- downstream-only capture failure rejects the primary hypothesis and points
  to downstream-body or process-global capture incompatibility;
- capture success with parity failure rejects semantic validity;
- parity success with efficiency-gate failure establishes a usable but
  insufficient hybrid backend;
- all V5-D gates passing establishes real-tensor feasibility only and advances
  only to preparation of a separately authorized V5-E protocol.

## 6. Frozen executor architecture

### 6.1 Preparation

After the unchanged compiler attempt produces its already-authorized exact
zero-output BF16/PTX failure and permits a fresh raw process, the V10 raw
process must:

1. require the default native allocator and reject any inherited
   `PYTORCH_CUDA_ALLOC_CONF` value;
2. load one model on one selected GPU and enter the complete V08
   `torch.inference_mode()` lifecycle;
3. allocate the unchanged V5-C owned static buffers exactly once;
4. create one non-default preparation/capture stream;
5. synchronize it with the current stream;
6. run exactly three eager wrist warm-ups into the owned wrist-token buffer;
7. materialize the scene-first combined-token buffer;
8. run exactly three eager downstream warm-ups;
9. synchronize and record the pre-capture memory snapshot;
10. create exactly one `torch.cuda.CUDAGraph`;
11. capture the downstream action core exactly once on the side stream with
    `capture_error_mode="global"` and no supplied/shared pool token;
12. instantiate the graph if the pinned API exposes explicit instantiation;
13. retain the downstream graph and all owned buffers; and
14. record the post-capture memory snapshot and architecture provenance.

There is no wrist graph object, wrist capture, second capture, shared graph
pool, pool token, concurrent capture, capture retry, or `empty_cache` call.

### 6.2 Live optimized reuse query

Each completed optimized reuse query must perform exactly this order:

1. copy current wrist pixels, cached scene tokens, prompt embeddings,
   attention mask, and proprioception into their stable owned buffers;
2. execute the unchanged wrist core eagerly once into the owned wrist tokens;
3. concatenate cached scene tokens first and current wrist tokens second into
   the owned combined-token buffer;
4. replay the single downstream graph once on the stable replay stream;
5. read the owned normalized-action output and perform the unchanged
   post-graph host transfer and unnormalization; and
6. observe the controller only after the full query completes successfully.

The work contract remains zero scene-core calls, one wrist-core call, and one
downstream-core call. Pointer drift, stream drift, reentrancy, ordering
violation, or any postlaunch exception invalidates the executor and cache
without retry or controller observation.

## 7. Measurement contract

V10 preserves every existing V5-D query and statistical threshold.

- `wrist_cuda_ms`: CUDA events around the eager wrist call.
- `downstream_cuda_ms`: CUDA events around the single graph replay.
- `sequential_cuda_ms`: `wrist_cuda_ms + downstream_cuda_ms`.
- `visual_cuda_ms`: eager wrist visual time for a reuse query; the cached
  scene core performs no work.
- `total_cuda_ms`: unchanged full device boundary including input copies,
  concatenation, wrist execution, graph replay, and output transfer.
- `wall_ms`: unchanged inclusive controller/preprocessing-to-unnormalized-
  action boundary.

No internal graph event is used to estimate component time. All component
events remain outside graph capture, so timing does not alter the captured
body.

The unchanged schedule is:

- 7 fail-first correctness queries;
- 8 schedule warm-up queries, two per path;
- 96 timed queries covering all 24 lexicographic permutations of Batched-FR,
  V5 refresh, eager reuse, and optimized reuse; and
- at most 111 full model queries total.

The unchanged conjunctive efficiency gates are:

| Gate | Frozen requirement |
|---|---:|
| Optimized reuse wall / Batched-FR median | `<= 0.930988756983` |
| Weighted wall / Batched-FR upper 95% | `<= 0.98` |
| Optimized / eager sequential CUDA upper 95% | `<= 0.96` |
| Weighted total CUDA / Batched-FR upper 95% | `<= 0.98` |
| Weighted visual CUDA reduction lower 95% | `>= 0.10` |
| V5 refresh wall / Batched-FR upper 95% | `<= 1.02` |
| Maximum position-median relative deviation | `<= 0.03` |

Bootstrap unit, 10,000 resamples, seed `20260810`, reuse weight
`0.34180622504322034`, no outlier deletion, and independent byte-identical
verification remain unchanged.

## 8. Correctness, architecture, and resource gates

Before any timed query, all seven existing correctness records must pass:

- Batched-FR oracle shape, finiteness, and work contract;
- eager reuse token/action/gripper parity;
- optimized/eager combined-token parity;
- normalized and unnormalized action parity;
- exact gripper decisions and controller decision parity;
- bitwise repeatability for the repeated input; and
- stable owned-buffer identities with zero scene, one wrist, and one
  downstream call.

V10 adds fail-closed architecture attestations:

- wrist backend exactly `eager-static-buffer`;
- downstream backend exactly `raw-cudagraph`;
- graph objects created/captured/retained exactly `1/1/1`;
- capture label exactly `downstream`;
- no wrist capture and no shared pool API use;
- three wrist then three downstream pre-capture warm-ups;
- inference mode entered and restored;
- default native allocator observed;
- peak reserved memory at or below 23 GiB;
- incremental reserved memory over eager at or below 6 GiB;
- one GPU and at most one concurrent model process;
- no simulator, download, task outcome, or protected-field access; and
- exact checkpoint, method, source, and inference-state restoration.

## 9. Predetermined stop rules

- **Compiler mismatch:** if the compiler outcome differs from the exact
  authorized zero-output compatibility failure, stop before raw fallback.
- **Transition gate failure:** stop before CUDA initialization/model load in
  the raw process; do not switch GPUs or resample.
- **Wrist or downstream warm-up failure:** preserve stage/memory evidence and
  stop before capture.
- **Downstream-only capture failure:** preserve the technical record and stop
  V10 permanently; do not add relaxed capture, change graph body, or retry.
- **Memory-cap breach:** stop before correctness; do not tune allocation.
- **Correctness failure:** stop before schedule warm-ups and timing.
- **Timing or ordering failure:** preserve the complete valid negative
  engineering result; do not threshold-shop.
- **Integrity/restoration uncertainty:** prohibit analysis and advancement
  until exact restoration is proven.
- **Interruption:** preserve the incomplete identity and stop for adjudication;
  no automatic continuation or replacement attempt.

Only one complete 111-query record may reach the frozen analyzer and an
independent verifier. No incomplete run may be analyzed as performance
evidence.

## 10. Pre-GPU implementation acceptance checkpoint

Before requesting GPU approval, implementation must demonstrate with CUDA
hidden and fake-CUDA tests:

- the V09 method, tensors, inference lifecycle, allocator, schedule, gates,
  and restoration boundaries are unchanged;
- the only backend change is two raw graphs to eager wrist plus one downstream
  graph;
- exact warm-up/capture/replay counts and order;
- no wrist graph, pool sharing, second capture, `empty_cache`, relaxed mode,
  or automatic retry;
- pointer/stream/order violations invalidate fail-closed;
- both successful and failing capture paths produce authenticated provenance;
- analyzer and independent verifier reject missing/extra queries, altered
  metrics, thresholds, permutations, or architecture fields;
- focused tests, full repository tests, lint/diff checks, deterministic
  preflight, and pinned TITAN CUDA-hidden verification pass;
- prior V01-V09 evidence hashes remain unchanged; and
- no GPU inspection/selection, CUDA initialization, model query, simulator,
  download, outcome access, or manuscript change occurs.

Implementation completion stops at this checkpoint. A V10 GPU selection and
single execution require separate explicit approval.

## 11. Alternatives rejected or deferred

- **Single combined wrist-plus-downstream graph:** deferred because it removes
  independently measured component timing and changes a larger execution
  boundary than necessary.
- **Capture downstream first, then wrist as a second graph:** rejected because
  it preserves the proven two-capture risk.
- **Thread-local or relaxed capture:** rejected because it weakens failure
  detection without isolating the graph architecture.
- **Graph-body edits, disabled cache/hidden states, alternate action head:**
  deferred because they change model computation and require new eager oracles.
- **Two-GPU sharding, quantization, offload, newer GPU, newer PyTorch, or
  higher memory cap:** rejected for V10 as broader environment/method changes.
- **Eager-only executor:** retained as the correctness reference, not the
  optimized candidate, because it cannot test launch-overhead recovery.

## 12. Scientific progression after V10

Passing V10 does not by itself create a positive-results paper. It would
support only three claims: the hybrid backend is real-tensor correct, memory
feasible, and faster under the frozen isolated-query design. The next steps
would still be:

1. freeze V5-E before any simulator outcome is opened;
2. confirm closed-loop task success against matched Batched-FR and relevant
   reuse baselines;
3. measure realized refresh rate, latency, task success, and failure modes;
4. conduct predeclared ablations and independent confirmation; and
5. write manuscript claims only after all paper-level gates are reconciled.

Until then, V5-E and every protected population remain sealed.
