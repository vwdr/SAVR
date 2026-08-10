# V5-D Real-Tensor Executor Research and Measurement Design

Status: **COMPLETE BEFORE V5-D PROTOCOL FREEZE**

Date: 2026-08-10

## 1. Research question

Can the V5-C static executor be realized on the pinned OpenVLA-OFT stack with
real CUDA tensors so that it preserves the selected IR-SA-ACR computation and
has enough measured end-to-end margin to justify any simulator rollout?

V5-D is deliberately a feasibility gate, not a task experiment. It must answer
four questions in order:

1. Can both frozen cores be compiled or captured without graph breaks,
   dynamic fallback, stale pointers, or excess memory?
2. Do real wrist tokens, scene-first combined tokens, normalized actions, and
   final actions match the eager reference within frozen tolerances?
3. Does the optimized reuse path meet the wall, CUDA, visual, and ordering
   margins required by the V5-B reuse lower bound?
4. Does every failure leave the checkpoint, source, model methods, controller,
   cache, and GPU state recoverable and auditable?

No task-success field is needed or permitted to answer these questions.

## 2. Immutable scientific basis

The selected controller remains `v5-a100-b40`:

- scene threshold `0.30046895424836606`;
- translation threshold `0.685919037527938`;
- horizon `1`;
- hard prefix reuse cap `0.40`;
- warm-up queries `0` and `1`;
- mandatory refresh after each completed reuse; and
- cache-age/latch agreement.

V5-B measured an output-blind reuse point of `0.3547659334461365` and a
lower 95% bound of `0.34180622504322034`. With the prior V3 refresh/Batched-FR
wall ratio `1.005452`, a weighted wall ratio of at most `0.98` requires the
reuse path to reach approximately `0.930988756983` of Batched FR at the reuse
lower bound. V5-D freezes both this conservative direct reuse target and the
more fundamental measured weighted ratio gate.

## 3. Pinned source audit

The audit used only repository evidence and read-only files under
`/home/ved/SAVR`.

### Software and model

- OpenVLA-OFT: `e4287e94541f459edc4feabc4e181f537cd569a8`;
- checkpoint: `638918f3d1c2e43a39a8a20772bdb8b91835e4b7`;
- PyTorch: `2.2.0+cu118`;
- Triton: `2.2.0`;
- Transformers fork: `bc339d9ad707454c0c115970db43c260067c61ab`;
- inference dtype: `torch.bfloat16`;
- fused SigLIP/DINOv2 vision with `256` patches per camera;
- projected/language dimension: `4096`;
- L1 regression action head with shape `[1,8,7]` before host reshape; and
- proprioception enabled, FiLM disabled, diffusion disabled.

The checkpoint image processor fixes two `224×224` fused inputs. The frozen
prompt tokenizes to `[1,21]`, receives the expected empty token, 56 action
placeholders, and one stop token, producing `[1,79]` input IDs/mask and
`[1,79,4096]` embeddings for the downstream core.

### Exact host/core boundary

The pinned `predict_action` path constructs tokens, labels, masks, and input
embeddings before vision. It then:

1. computes projected scene/wrist visual tokens;
2. projects current proprioception and appends it to the visual sequence;
3. builds multimodal embeddings and attention;
4. runs the full language model;
5. slices action hidden states and runs the L1 action head;
6. reshapes and transfers normalized actions to CPU/NumPy; and
7. unnormalizes actions with dataset statistics.

Therefore:

- the wrist visual core ends at fresh `[1,256,4096]` wrist tokens;
- the downstream core begins with current combined tokens, prompt
  embeddings/mask, and proprioception and ends at `[1,8,7]` normalized action
  tensors;
- prompt construction, preprocessing, controller work, static-buffer copies,
  CPU transfer, reshape, and NumPy unnormalization remain outside the cores;
  and
- inclusive wall timing must include those necessary host operations for each
  path instead of timing only graph replay.

This keeps the optimized path honest: it receives credit for omitted scene
model preprocessing/work on reuse, but not for omitting prompt or action work
that a real query still requires.

## 4. Official compiler and CUDA-graph guidance

The design relies on primary technical guidance:

- [PyTorch `torch.compile`](https://docs.pytorch.org/docs/stable/generated/torch.compile.html)
  documents `fullgraph`, static specialization, and `reduce-overhead`. It also
  warns that CUDA-graph overhead reduction is not guaranteed and may increase
  memory.
- [PyTorch CUDA Graphs](https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-graphs)
  requires long-lived input/output memory and new values copied into the same
  addresses before replay.
- [PyTorch `CUDAGraph`](https://docs.pytorch.org/docs/stable/generated/torch.cuda.CUDAGraph.html)
  documents capture, instantiation, replay, reset, capture error modes, and
  graph memory pools.
- [NVIDIA CUDA Graph constraints](https://docs.nvidia.com/dl-cuda-graph/latest/cuda-graph-basics/constraints.html)
  require static topology, shapes, parameters, and memory addresses and forbid
  host-device synchronization inside capture.
- [NVIDIA PyTorch integration guidance](https://docs.nvidia.com/dl-cuda-graph/latest/torch-cuda-graph/torch-integration.html)
  assigns fixed graph-input copies and lifetime correctness to the user.
- [PyTorch numerical accuracy](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html)
  explains why mathematically equivalent floating-point paths are not assumed
  bitwise identical.

These sources justify the experiment, not its result. The pinned 2.2 stack may
still fail compilation, capture, memory, correctness, or timing.

## 5. Backend waterfall without result shopping

The backend order is frozen:

1. compile each core with pinned `torch.compile`, `backend="inductor"`,
   `fullgraph=True`, `dynamic=False`, and `mode="reduce-overhead"`;
2. only if compilation is technically unsupported before any correctness or
   timed record, start a fresh process and capture both cores with project-owned
   `torch.cuda.CUDAGraph` objects; and
3. stop if raw capture is technically unsupported.

Raw graphs are not a second statistical candidate. They are allowed only when
the compiler cannot produce a stable executable before measurements. A compile
backend that reaches correctness and then fails parity or timing terminates
V5-D; its unfavorable result cannot be replaced by raw-graph results. Mixed
compiler/raw cores are prohibited.

Compiler technical unavailability is limited to construction/first-call
failure, full-graph failure, static-key recompilation, verified eager fallback,
or preparation OOM before correctness. The record must contain no correctness
or timing output when this classification is made.

## 6. Four measured paths

Every timed repetition executes all four paths:

1. `batched-fr`: current scene and wrist through the established Batched-FR
   oracle, with no controller;
2. `v5-refresh`: selected controller plus current scene/wrist Batched-FR,
   cache store, downstream action, CPU conversion, and unnormalization;
3. `eager-reuse`: selected controller plus compatible cached scene, fresh
   wrist, eager downstream action, CPU conversion, and unnormalization; and
4. `optimized-reuse`: identical reuse inputs/host work through the selected
   static backend.

Each path starts from a freshly constructed deterministic episode state. This
prevents earlier timed paths from changing later controller/cache state. Reuse
paths are primed to an eligible query; refresh is primed with the mandatory
post-reuse latch. Priming is untimed, uses no model call, and is verified before
every measured query.

The two deterministic V3-C image/state patterns are reused exactly. No
simulator, task state, reward, or success label is opened.

## 7. Counterbalancing and uncertainty

The timed schedule uses every permutation of the four paths exactly once: 24
repetition blocks and 96 timed queries. Thus every path appears six times in
each ordinal position. Two untimed warm-ups per path precede timing.

Analysis uses the repetition block as the resampling unit. A deterministic
10,000-resample paired bootstrap reports 95% percentile intervals for direct
and weighted ratios. No outlier deletion, optional trimming, alternative
summary, or post-output backend choice is permitted.

Material ordering bias is defined before execution: for each path/metric, the
largest absolute relative deviation of a position-specific median from the
overall median must not exceed `0.03`.

## 8. Frozen positive feasibility margins

All gates are conjunctive:

- optimized reuse median wall/Batched-FR ratio at most
  `0.930988756983`;
- upper 95% weighted wall/Batched-FR ratio at most `0.98`, using reuse weight
  `0.34180622504322034`;
- upper 95% optimized/eager sequential-CUDA ratio at most `0.96`;
- upper 95% weighted total-CUDA/Batched-FR ratio at most `0.98`;
- lower 95% weighted visual-CUDA reduction at least `0.10`;
- upper 95% V5-refresh/Batched-FR wall ratio at most `1.02`; and
- maximum ordering-bias deviation at most `0.03`.

These are feasibility gates for the next phase, not paper-level speed claims.
They are intentionally stronger than merely showing fewer logical scene calls.

## 9. Memory and capture safety

The prior pinned run peaked at `16,100,537,856` allocated bytes and
`16,238,247,936` reserved bytes on a 24 GiB TITAN RTX. V5-D permits:

- at most `23 GiB` peak reserved memory;
- at most `6 GiB` incremental reserved memory above the eager baseline;
- one model process and one visible GPU;
- no shared graph pool unless the frozen implementation proves sequential
  lifetime safety before capture; and
- no allocation, host sync, filesystem write, audit hash, or value scan inside
  either captured/timed core.

Compiler and Triton caches must be redirected under the immutable V5-D result
directory. Nothing may write to a user-global cache or outside
`/home/ved/SAVR`.

## 10. Evidence boundary

Passing V5-D would establish bounded synthetic-input real-model correctness and
a credible timing margin on one pinned GPU. It would not establish online task
success, robustness, transfer, independent confirmation, or a positive paper
result. Failure stops before simulator work. No threshold, backend, tolerance,
repetition count, or gate may be changed after output.
