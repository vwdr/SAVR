# ACR V5-D v05 Transition-Revalidation Recovery Protocol

Status: **EXECUTED; SHARED-POOL MEMORY TECHNICAL STOP; NO METHOD RESULT**

Date: 2026-08-11

## 1. Decision and scope

V04 remains an immutable zero-query technical stop. V05 creates a new run
identity and changes only the fresh raw process's aggregate GPU revalidation
after an authorized compiler failure. The selected method, checkpoint, tensor
contract, shared-pool raw backend, compiler-first waterfall, query identities,
111-query schedule, correctness tolerances, timing/statistical analysis, 23 GiB
cap, and claim boundary remain unchanged.

This recovery does not weaken resource protection. It replaces one immediate
sample with a fixed, output-independent stabilization rule that retains the
same 5% utilization and 512 MiB limits and requires every sample to pass.

## 2. Observed failure and primary-source basis

V04 selected physical GPU 0 after three samples at 6 MiB and 0%. Its compiler
process then failed as anticipated and restored exactly. Four seconds later,
the fresh raw process observed 6 MiB but 33% utilization and stopped before
model load. A later aggregate-only snapshot was 6 MiB and 0%.

NVIDIA defines `utilization.gpu` as the percentage of time during a recent
sample period in which one or more kernels executed. The sample period can be
between one sixth of a second and one second, depending on product. Therefore,
an immediate aggregate reading is a measurement over a recent window, not an
instantaneous ownership signal.

Primary reference:

- [NVIDIA NVML `nvmlUtilization_t` reference](https://docs.nvidia.com/deploy/nvml-api/structnvmlUtilization__t.html)

The V05 rule discards two seconds—strictly longer than NVIDIA's documented
maximum sample window—then requires three new samples separated by five
seconds. This is a conservative sustained-idle check, not a favorable-sample
search.

## 3. Frozen transition rule

The rule applies only in a fresh `raw-cudagraph` process after the compiler
attempt emitted a valid raw-transition permit and exact restoration passed.
Before importing PyTorch, initializing CUDA, or loading the model, the process
must:

1. verify the frozen physical index and UUID from the immutable launch record;
2. wait exactly 2 seconds without reading GPU telemetry;
3. collect exactly 3 aggregate-only snapshots of that selected GPU;
4. wait exactly 5 seconds between snapshots;
5. require every snapshot to retain the same index and UUID;
6. require every snapshot to use at most 512 MiB and at most 5% utilization;
7. write one create-once semantic record containing all three snapshots; and
8. continue using the final passing snapshot as the runner's pre-model gate.

If sampling, identity, memory, utilization, or evidence creation fails, stop
before model load. There is no additional sampling window, threshold increase,
GPU switch, automatic retry, process inspection, or allocation inspection.

## 4. Unchanged scientific experiment

V05 preserves byte-semantically from V04:

- `v5-a100-b40` and all controller/cache semantics;
- OpenVLA-OFT/LIBERO/checkpoint revisions and hashes;
- deterministic inputs and real tensor shapes/dtypes;
- shared wrist-then-downstream CUDA-graph pool and stream/order constraints;
- compiler first, raw second only after an allowed pre-output failure;
- 7 correctness, 8 warm-up, and 96 paired timed queries;
- action/token tolerances and lifecycle/restoration checks;
- paired statistics and every advancement gate;
- the 23 GiB peak-reservation cap; and
- zero simulator, task-success, reward, or final-population access.

The transition record is technical evidence and is excluded from latency
measurement.

## 5. Pre-GPU acceptance checkpoint

Before GPU coordination, all of the following must pass with CUDA hidden:

- V04 configuration and technical-stop semantic links;
- V04 scientific/backend sections unchanged in the resolved V05 config;
- exact frozen cooldown/sample count/interval/threshold validation;
- fake-snapshot tests for pass, utilization failure, memory failure, identity
  drift, create-once evidence, no extra sampling, and cached second access;
- unchanged 111 query identities and 23 GiB cap;
- new V05 immutable run/analysis/verification paths absent;
- repository, checkpoint, OpenVLA-OFT, and LIBERO integrity;
- no GPU inspection/selection, CUDA initialization, model load/query,
  simulator, download, outcome access, or manuscript modification; and
- clean synchronized `main` on local, private GitHub, and TITAN.

## 6. One-attempt execution logic

After separate user coordination, run the standard three-sample aggregate GPU
selector and freeze one physical index/UUID. Execute V05 once:

1. compiler attempt;
2. exact restoration and fresh-process permit for an allowed pre-output error;
3. the frozen transition-revalidation rule;
4. V04 shared-pool raw preparation and 23 GiB enforcement;
5. seven correctness queries;
6. eight warm-ups;
7. ninety-six paired timed queries; and
8. byte-identical analysis plus independent verification.

Only a complete 111-query record can advance. Partial method outcomes remain
sealed and unreported.

## 7. Predetermined failure responses

- **Transition sample fails:** stop before model load; no additional wait,
  sample window, GPU switch, raw fallback, or automatic retry.
- **Shared-pool API/order/stream failure:** invalidate and stop.
- **OOM or cap breach:** preserve stage memory trace and stop; do not tune.
- **Correctness failure:** stop before timing.
- **Timing-gate failure:** retain as a valid negative engineering result.
- **Restoration/integrity uncertainty:** restore exact state and prohibit
  analysis or advancement.

## 8. Advancement boundary

Passing V05 establishes only the frozen real-tensor correctness and efficiency
gates. It does not establish task success. V5-E remains separately planned and
requires new approval. No manuscript change is authorized by this protocol.

## 9. Execution disposition

The authorized run passed the new transition window with three samples at
6 MiB and 0%. The compiler failed as anticipated and restored exactly. The raw
backend completed wrist warm-up and wrist capture, then OOMed during downstream
warm-up before downstream capture or correctness. Peak reservation was
23.2266 GiB, 243,269,632 bytes above the frozen 23 GiB cap.

V05 has zero correctness, schedule warm-up, timing, simulator, or outcome
records and is not a method result. It remains immutable and cannot be retried.
Evidence is preserved in `reports/PHASE_V5_D_V05_TECHNICAL_STOP_REPORT.md`
and `reports/runtime/acr_v5_d_v05_technical_stop.json`.
