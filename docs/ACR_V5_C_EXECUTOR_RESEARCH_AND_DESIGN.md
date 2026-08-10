# V5-C Executor Research and Design Record

Status: **COMPLETE BEFORE V5-C IMPLEMENTATION**

Date: 2026-08-10

## 1. Question

What executor boundary can plausibly convert the selected IR-SA-ACR reuse rate
into measured wall-time improvement without changing controller decisions,
model weights, visual tokens, action outputs, or failure semantics?

## 2. Quantitative requirement

V5-B selected `v5-a100-b40` with replay reuse fraction
$r=0.3547659334$ and lower 95% bound $0.3418062250$. The prior real-model
V3-C refresh/Batched-Full-Refresh wall ratio was $f=1.005452$. To achieve a
future weighted wall ratio $w\le0.98$, the reuse path must satisfy

\[
u \le \frac{w-(1-r)f}{r}.
\]

This gives:

- $u\le0.933709$ at the V5-B reuse point estimate; and
- $u\le0.930989$ at the reuse lower bound.

The current V3 reuse/BFR ratio was approximately `0.9750`, so visual omission
alone is unlikely to provide the required margin. V5 therefore needs a
separately attributable executor optimization. These calculations use prior
development timing only and are feasibility targets, not new measurements.

## 3. Pinned implementation findings

### Existing reusable boundary

`BatchedDualPathOpenVLAAdapter` already separates:

- refresh: batched scene+wrist SigLIP, DINOv2, and projector;
- reuse: cached scene tokens plus fresh wrist SigLIP, DINOv2, and projector;
- downstream: fresh language-model and action-head computation; and
- controller: CPU scene/state/action-transition decision before the model
  query.

The wrist visual core has fixed tensor ranks and, within a compatible context,
fixed shapes, dtype, device, camera order, patch count, and module identities.

### Why wrist-only capture is insufficient as the sole design

V3-C measured approximately 75.7 ms of visual CUDA work inside a roughly
1,161.6 ms reuse query. Reducing launch overhead only inside that visual slice
cannot be assumed to recover the additional roughly 4-5 percentage points of
end-to-end wall margin required by V5. Wrist-only optimization remains useful,
but the design must also cover the fixed-shape downstream GPU core if feasible.

### Why the complete Python query is not capture-safe

The pinned OpenVLA-OFT action path performs host-visible work, including:

- Python prompt/action-mask construction;
- tensor creation and control flow;
- conversion of normalized actions through `.cpu().detach().numpy()`; and
- NumPy action unnormalization.

CUDA stream capture cannot include synchronous host transfers or arbitrary
dynamic CPU work. The complete `predict_action` function is therefore rejected
as a raw capture boundary.

### Selected split boundary

The proposed executor has two fixed-shape GPU cores with host orchestration
outside capture:

1. **Wrist visual core:** current wrist pixels → SigLIP/DINOv2 → feature concat
   → visual projector → current wrist tokens.
2. **Downstream action core:** current combined scene/wrist tokens, prompt
   embeddings/masks, and proprioceptive input → unchanged language model and
   L1 action head → normalized action tensor.

Scene-cache load, static-buffer copies, controller logic, compatibility checks,
failure handling, action transfer to CPU, and NumPy unnormalization remain
outside captured/compiled cores. Refresh queries remain on the existing eager
batched Full-Refresh path.

This split captures the dominant GPU work without pretending that host logic
is graph-safe.

## 4. Primary technical guidance

- [PyTorch CUDA semantics](https://docs.pytorch.org/docs/main/notes/cuda.html#cuda-graphs)
  states that graph replay uses the same kernels and arguments, including the
  same pointer addresses; new data must be copied into long-lived input memory.
- [PyTorch `torch.compile`](https://docs.pytorch.org/docs/stable/generated/torch.compile.html)
  describes `reduce-overhead` as a CUDA-graph-oriented mode for reducing Python
  overhead, but warns that applicability is not guaranteed and can use more
  memory.
- [NVIDIA CUDA Graphs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html)
  separates definition, instantiation, and repeated execution, and prohibits
  synchronization or status queries during active capture.
- [NVIDIA capture constraints](https://docs.nvidia.com/dl-cuda-graph/latest/cuda-graph-basics/constraints.html)
  emphasize fixed memory/lifetime behavior, self-contained stream dependencies,
  and capture restrictions on synchronous allocation and transfer APIs.
- [PyTorch 2.2 release notes](https://docs.pytorch.org/blog/pytorch2-2/)
  document compiler/CUDA-graph improvements relevant to the project's pinned
  `torch==2.2.0+cu118` environment.

These sources support testing a static-buffer executor. They do not establish
that the pinned OpenVLA stack will compile, capture, fit in memory, preserve
numerics, or meet latency gates.

## 5. Static-buffer principles

For each prepared shape/context, the executor must own stable device buffers
for every captured input and output. New query values are copied into those
buffers before replay. The graph/compiled callable may not retain pointers to
short-lived caller tensors.

The compatibility key includes at least:

- checkpoint and upstream revision;
- executor and controller identities;
- instruction/prompt shape;
- preprocessing and action-head identity;
- dtype and device;
- image height/width and patch count;
- projected-token and embedding dimensions;
- proprioception presence/shape; and
- model training/evaluation state.

Any mismatch prevents replay. There is no automatic cross-context reuse.

## 6. Lifecycle and failure design

The executor lifecycle is:

`UNPREPARED → PREPARED → ACTIVE → INVALIDATED`

- `prepare` validates identities/shapes, allocates owned buffers, and constructs
  the backend outside a timed query.
- `run_reuse` copies current inputs, invokes each prepared core once, and
  returns the current normalized action tensor/output.
- `reset` clears episode-scoped bindings without changing model weights.
- `invalidate` prevents further optimized execution after any mismatch or
  partial failure.

If readiness/compatibility fails before optimized GPU work begins, the adapter
may force a normal refresh with an explicit executor-unavailable reason. If a
failure occurs after an optimized core begins, the cache and executor are
invalidated, the controller is not observed, and the query stops. It is unsafe
to silently rerun a partially executed action query.

## 7. Compilation/capture waterfall

The future GPU phase will test one predeclared waterfall, not select whichever
result looks best:

1. pinned PyTorch `torch.compile` with `fullgraph=True`, static shapes, and
   `mode="reduce-overhead"` for each isolated GPU core;
2. if compilation is technically unsupported before timing, a project-owned
   raw `torch.cuda.CUDAGraph` implementation of the same cores; and
3. if neither passes capture, memory, and numerical gates, stop before online
   rollout.

The eager executor is always retained as the correctness oracle, not as a
post-failure way to claim optimized performance.

## 8. Alternatives rejected

- **Capture the whole `predict_action`:** includes CPU/NumPy transfers and
  dynamic host logic.
- **Wrist-only optimization as the complete method:** unlikely to meet the
  calculated end-to-end margin by itself.
- **Change controller thresholds while optimizing:** confounds V5-B selection
  with executor effects.
- **Capture refresh and reuse in one graph:** their camera work and shapes
  differ; refresh remains the oracle path.
- **Reuse actions or downstream hidden states:** changes the method and
  reactivity boundary.
- **Patch upstream source files:** unnecessary and increases restoration risk;
  V5 uses project-owned wrappers.
- **Silently fall back after partial execution:** risks double execution and
  corrupted temporal/cache state.

## 9. Evidence boundary

This audit supports freezing and CPU-testing the executor contract. It does not
authorize or support a GPU speed, memory, compilation, capture, numerical, task-
success, or paper claim. Those require V5-D after V5-C implementation passes.
