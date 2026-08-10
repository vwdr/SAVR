# ACR V5-D Bounded One-GPU Feasibility Protocol

Status: **FROZEN BEFORE BACKEND IMPLEMENTATION, GPU SELECTION, OR EXECUTION**

Authorization date: 2026-08-10

Machine freeze: `configs/acr/v5_d_gpu_feasibility_freeze.json`

Research basis: `docs/ACR_V5_D_RESEARCH_AND_MEASUREMENT_DESIGN.md`

## 1. Current authorization boundary

The user approved preparation of this protocol only. This document does not
authorize:

- GPU inspection or selection;
- backend implementation;
- model loading or querying;
- CUDA compilation/capture/timing;
- environment or dependency changes;
- simulator reset/episode execution;
- any success/reward/outcome access; or
- manuscript modification.

After this checkpoint merges and synchronizes, V5-D requires a separate user
authorization. Before any device is selected, the agent must explicitly report
that it is entering the one-GPU phase and coordinate under `AGENTS.md`.

## 2. Immutable method and evidence

V5-D preserves without change:

- selected method `v5-a100-b40`;
- controller `acr-isolated-controller-v1`;
- executor contract `acr-reuse-executor-static-v1`;
- integration `ir-sa-acr-static-executor-v1`;
- V5-B semantic SHA-256
  `8a9f15b818b58ed2868d4b1123a222a4c062507161ab7de911d8d233f3b1efec`;
  and
- V5-C semantic SHA-256
  `f7a8d11d4574add57caa630c03463375421d9482984478be769f497b1c9d0b66`.

No controller signal, threshold, reason, decision, reuse cap, cache rule,
compatibility field, or CPU evidence may change.

## 3. Pinned stack

- SAVR protocol parent: `b730754c93ccc4706b90b01c772389374a593acd`;
- OpenVLA-OFT: `e4287e94541f459edc4feabc4e181f537cd569a8`;
- LIBERO: `8f1084e3132a39270c3a13ebe37270a43ece2a01`;
- checkpoint: `638918f3d1c2e43a39a8a20772bdb8b91835e4b7`;
- PyTorch: `2.2.0+cu118`;
- CUDA package family: `11.8`;
- Triton: `2.2.0`;
- Transformers fork: `bc339d9ad707454c0c115970db43c260067c61ab`;
- model mode: evaluation plus `torch.inference_mode()`;
- dtype: `torch.bfloat16`;
- L1 regression and proprioception enabled; and
- FiLM and diffusion disabled.

The launch must verify the exact lock, source, checkpoint, V5-C source, and
configuration hashes before loading the model. Any mismatch stops V5-D.

## 4. Pre-GPU implementation gate

After separate V5-D authorization, implement only a project-owned PyTorch
backend for the frozen V5-C interfaces. Upstream and checkpoint source may not
be edited. Before selecting a GPU, CPU/fake-backend tests must prove:

1. exact backend identities and waterfall order;
2. static buffer allocation and pointer stability;
3. current A/B/A input copies update all owned values;
4. fixed core call order and zero scene calls on reuse;
5. compiler technical-failure classification cannot observe correctness or
   timing results;
6. raw fallback is impossible after correctness begins;
7. no mixed backend can be selected;
8. timing labels/permutations/query budgets are exact;
9. paired-bootstrap and ordering-bias analysis is deterministic;
10. memory/resource stops are fail closed;
11. checkpoint/model method restoration is exception safe; and
12. complete repository regression passes.

The proposed implementation, tests, and exact execution command must be
committed, pushed, reviewed, merged, and synchronized before GPU selection.
The merged execution commit is recorded in the launch manifest.

## 5. User-coordinated GPU selection

Only after the implementation gate and explicit user coordination:

1. reconnect through `ssh titan` and stay inside `/home/ved/SAVR`;
2. inspect only aggregate GPU index, UUID, name, memory total/used, and
   utilization—never process identities or commands;
3. take three aggregate samples ten seconds apart;
4. a device is eligible only if every sample has utilization at most `5%` and
   memory use at most `512 MiB`;
5. select the lowest-index eligible device, or stop if none is eligible;
6. record physical ID, UUID, all three samples, driver, and device name in an
   immutable launch manifest before model load;
7. expose only that physical device with `CUDA_VISIBLE_DEVICES` and require
   exactly one logical `cuda:0`; and
8. abort if the launch snapshot no longer satisfies the eligibility limits.

The selected physical ID cannot be known or frozen during protocol preparation
without violating the mandatory pause. It becomes immutable in the launch
manifest after the required coordination and before any model/GPU query.

## 6. Project-local execution environment

Before Python starts, set project-local cache roots for Hugging Face, Torch,
TorchInductor, and Triton under the immutable V5-D result directory. Require
offline modes and disable user-site, network logging, tokenizer parallelism,
and downloads. The runner must refuse a project root other than
`/home/ved/SAVR`.

No `sudo`, package install, daemon, service, permission change, environment
mutation, or write outside the project is allowed.

## 7. Backend waterfall

### Attempt 1: compiler backend

Compile both isolated cores with exactly:

```python
torch.compile(
    core,
    backend="inductor",
    fullgraph=True,
    dynamic=False,
    mode="reduce-overhead",
)
```

Both cores must use the V5-C owned buffers and frozen static shapes. Record
compiler counters/logs needed to prove one stable compiled graph per core, no
graph break, no static-key recompile, and no verified eager fallback.

### Permitted raw fallback

Raw fallback is allowed only before the first correctness record and only for:

- compiler construction or first-call error;
- full-graph capture error;
- recompile on unchanged compatibility keys;
- mechanically verified eager fallback; or
- preparation OOM.

Preserve the compiler attempt. End its process. Before raw capture, require a
fresh process, exact source/checkpoint hashes, restored methods, and an eligible
aggregate GPU snapshot. Cumulative wall, artifact, and preparation-launch caps
still apply.

Capture each core with a dedicated non-default stream and project-owned
`torch.cuda.CUDAGraph`. Warm up each core three times on the side stream,
synchronize only outside capture, capture with `capture_error_mode="global"`,
instantiate before use, and retain all referenced buffers for graph lifetime.

### Prohibited fallback

If the compiler backend emits any correctness or timed record, parity, memory,
or timing failure stops V5-D. Raw graphs may not replace it. Raw capture failure
also stops. Mixed compiler/raw cores and eager performance substitution are
prohibited.

## 8. Deterministic inputs and tensor contract

Use the exact V3-C deterministic A/B images, state midpoint, and instruction:

```text
move the robot safely to the target
```

The raw images are `256×256×3` uint8 patterns with the frozen content hashes.
Expected processed/static tensors are:

| Tensor | Shape | Type/device |
|---|---|---|
| wrist pixels | `[1,6,224,224]` | bfloat16 / selected CUDA |
| cached scene tokens | `[1,256,4096]` | bfloat16 / selected CUDA |
| fresh wrist tokens | `[1,256,4096]` | bfloat16 / selected CUDA |
| combined tokens | `[1,512,4096]` | bfloat16 / selected CUDA |
| prepared input IDs | `[1,79]` | int64 / selected CUDA |
| prompt embeddings | `[1,79,4096]` | bfloat16 / selected CUDA |
| attention mask | `[1,79]` | integer/bool pinned dtype / selected CUDA |
| proprioception | `[1,8]` | bfloat16 / selected CUDA |
| normalized actions | `[1,8,7]` | floating / selected CUDA |

Every compatibility identity and shape must be verified before a core launch.
A/B/A optimized execution must prove that the second A reproduces the first A
and that B differs where the eager oracle differs.

## 9. Correctness schedule and gates

Before timing, execute exactly seven labeled queries:

1. A Batched-FR oracle;
2. B Batched-FR oracle;
3. A eager-reuse reference;
4. B eager-reuse reference;
5. A optimized reuse;
6. B optimized reuse; and
7. A optimized repeat after B.

Required gates:

- exact tensor shape, dtype, device, camera order, prompt IDs, and masks;
- wrist/combined token `rtol=0.016`, `atol=1e-5` versus eager reuse;
- normalized-action `rtol=0.001`, `atol=1e-4` versus eager reuse;
- final unnormalized action `rtol=1e-5`, `atol=1e-6`;
- exact action shape and identical gripper-open/closed decisions;
- all values finite in correctness mode;
- optimized A-repeat bitwise identical to optimized A;
- one wrist and downstream call and zero scene calls on optimized reuse;
- stable owned pointers across A/B/A;
- compatible cache identity and no cross-query contamination;
- no OOM, graph break, recompile, eager fallback, or lifecycle failure; and
- reference/optimized controller decisions and reasons identical.

Any failure stops before warm-up or timing and does not permit another backend.

## 10. Timing schedule

Paths, in frozen order:

1. `batched-fr`;
2. `v5-refresh`;
3. `eager-reuse`;
4. `optimized-reuse`.

Run two untimed warm-ups per path. Then run all 24 lexicographic permutations
of the four paths, one permutation per repetition block. Inputs alternate A/B
by block. This yields 96 timed queries and exact ordinal balance.

For every query record:

- inclusive wall time from controller/preprocessing start through current
  CPU/NumPy unnormalized actions;
- total CUDA event time including required device copies and action transfer;
- wrist-visual CUDA time;
- downstream CUDA time;
- sequential CUDA time defined as wrist plus downstream for reuse;
- component calls and backend launch/replay count;
- cache event and controller reasons;
- current input/output hashes outside the timed interval; and
- allocated/reserved memory outside the timed interval.

CUDA events and synchronization bracket measurement but remain outside both
captured cores. No profiler, tensor hash, full finite scan, serialization,
filesystem write, memory query, or device synchronization is allowed inside a
timed core. No outlier may be removed.

## 11. Frozen analysis

Use 10,000 deterministic paired bootstrap resamples of the 24 repetition
blocks with seed `20260810`. Report medians and 95% percentile intervals.

Use fixed reuse weight `0.34180622504322034`. For every bootstrap resample:

```text
weighted_metric = (1-r) * median(v5-refresh) + r * median(optimized-reuse)
weighted_ratio = weighted_metric / median(batched-fr)
```

All gates must pass:

- optimized-reuse median wall/Batched-FR `≤ 0.930988756983`;
- weighted wall/Batched-FR upper 95% `≤ 0.98`;
- optimized/eager sequential-CUDA upper 95% `≤ 0.96`;
- weighted total-CUDA/Batched-FR upper 95% `≤ 0.98`;
- weighted visual-CUDA reduction lower 95% `≥ 0.10`;
- V5-refresh/Batched-FR wall upper 95% `≤ 1.02`;
- maximum position-median deviation `≤ 0.03`; and
- every correctness, memory, work, lifecycle, restoration, and resource gate
  passes.

The committed analyzer must be tested and merged before execution. Run it twice
and require byte-identical semantic output. A separate verifier must recompute
all gates without trusting the analyzer's booleans.

## 12. Budgets

- GPUs: at most `1` selected device;
- model processes: at most `1` at a time;
- full model queries: exactly `111` if timing completes (`7` correctness, `8`
  warm-up, `96` timed), hard cap `111`;
- backend-preparation core launches: hard cap `24` cumulatively across the
  permitted waterfall;
- simulator episodes/resets: `0`;
- downloads: `0`;
- task outcomes: `0`;
- wall time: `7,200 seconds` cumulative;
- new artifacts: `1 GiB`;
- peak reserved GPU memory: at most `23 GiB`; and
- incremental reserved memory over eager baseline: at most `6 GiB`.

Every attempted call/launch counts even if it fails. Compiler-internal kernel
autotuning is recorded separately and cannot relax the logical call caps.

## 13. Immutable records and failure handling

Run ID: `acr-v5d-real-tensor-feasibility-v01`.

Write once under `results/<run_id>`:

- protocol/config and execution revision hashes;
- launch manifest with selected GPU samples;
- backend-attempt records;
- every full-query and preparation-core label;
- compiler/capture logs;
- correctness comparisons;
- timing records;
- memory/resource snapshots;
- technical summary on any stop;
- analyzer output and independent verification; and
- terminal run summary.

No automatic retry is allowed. The sole compiler-to-raw transition is the
frozen pre-correctness technical waterfall, not a retry. An interrupted run is
preserved and stops for adjudication; no label is silently reused.

After any error:

1. invalidate executor and cache;
2. leave controller unobserved for an incomplete query;
3. restore all patched methods;
4. restore checkpoint metadata byte-for-byte and remove only loader backups
   created by this attempt;
5. verify SAVR/OpenVLA/LIBERO trees and checkpoint hashes;
6. synchronize/release only the selected device through normal process exit;
7. record aggregate selected-GPU telemetry; and
8. preserve all evidence.

## 14. Stop and claim boundary

Any correctness, fallback, memory, timing, bias, invariant, restoration,
resource, or verification failure stops V5-D before simulator work.

Passing V5-D permits only preparation of the next online-development protocol.
It does not authorize an episode, establish task success, validate transfer,
open Goal/final populations, or support a positive-results paper claim.
