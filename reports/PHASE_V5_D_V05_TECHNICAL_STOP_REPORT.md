# Phase V5-D v05 Technical-Stop Report

Status: **TECHNICAL STOP BEFORE CORRECTNESS — NO METHOD-PERFORMANCE RESULT**

Date: 2026-08-11

## 1. Execution outcome

After explicit authorization, three aggregate samples selected physical GPU 0
at 6 MiB and 0% utilization. The compiler repeated the anticipated first-call
BF16/PTX failure on `sm_75`, restored the checkpoint exactly, and authorized a
fresh raw process.

The V05 transition correction passed all three sustained-idle samples: each
showed the same index/UUID, 6 MiB, and 0%. The raw process loaded the model and
entered the V04 shared-pool backend.

Wrist warm-up and wrist capture completed. Downstream warm-up then failed while
requesting 14 MiB, before downstream capture or the first correctness query.
Peak allocation was 22.5626 GiB and peak reservation was 23.2266 GiB, exceeding
the unchanged 23 GiB cap by 243,269,632 bytes.

## 2. Scientific interpretation

The transition-revalidation correction worked, but the shared-pool change did
not make the frozen backend feasible on TITAN RTX. The memory trace shows why:
the wrist graph was retained successfully, but downstream warm-up exhausted
memory before a second graph could be captured into the shared pool.

V05 contains zero full model queries, correctness records, schedule warm-ups,
timed records, simulator operations, rewards, success fields, or task outcomes.
The analyzer and finalizer were not run. This is not positive or negative
method-performance evidence.

## 3. Integrity and resource audit

- Execution revision:
  `bcedc0722872b1d57e61d87af3e65c0e67c07072`.
- Selected GPU: physical 0, UUID
  `GPU-bb2451d6-2989-a112-5c18-8892943710e4`, compute capability 7.5.
- Compiler preparation: one launch; 80.2589 seconds.
- Raw preparation: eight launches; 7.5767 seconds.
- Maximum concurrent model processes: one.
- Post-stop selected-GPU telemetry: 6 MiB used, 0% utilization.
- No simulator, download, task outcome, unrelated process, or allocation was
  accessed.

Both attempts restored the checkpoint exactly. No loader backup remains. SAVR,
OpenVLA-OFT, and LIBERO are clean.

## 4. Immutable evidence

- Launch semantic SHA-256:
  `d24d4e068b03fa12031d844cfa2f978c18aff39006422899408b250d0f2a80cf`.
- LIBERO-config semantic SHA-256:
  `f6906a8a3bcbd746d98533f79dd2698e0e359abbb401de12b235050a6c69d5b3`.
- Compiler-attempt semantic SHA-256:
  `04742bda7d484dff35dfac028fab54fe07d7e86b5b47610c2035228c36ddb202`.
- Raw-transition semantic SHA-256:
  `999dbb3e04976e589802a5d04f3a0d8dc6b979e53308413e3a0ad4a026aae7c0`.
- Transition-revalidation semantic SHA-256:
  `8335fc00a97272ef292c5be6c4dd66f85db6d9293af370f7fb904ec060327ede`.
- Raw-attempt semantic SHA-256:
  `db33d74710c0f4a1fa2f5d5c85d2378a080f31b53fa8a2655f18a175f2a77b07`.
- Curated technical-stop semantic SHA-256:
  `cb6d9120fc2e6ee69aaa83d677598d21741be8eaf5a3456bc21461d30eb3cc3f`.
- Curated record: `reports/runtime/acr_v5_d_v05_technical_stop.json`.

The complete create-once evidence remains on TITAN under
`results/acr-v5d-real-tensor-feasibility-v05/`.

## 5. Disposition

`STOP_NO_RETRY_RESEARCH_SEPARATELY_FROZEN_PRECAPTURE_WARMUP_OR_COMPATIBLE_HARDWARE`

V05 remains immutable and V5-E remains ineligible. A future same-hardware route
would need a separately researched system change, such as completing both
cores' allocation-generating warm-ups before retaining either graph, with a new
estimand/protocol and unchanged anti-shopping protections. Compatible
higher-memory hardware remains the lower-risk engineering alternative.
