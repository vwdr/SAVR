# Phase V5-D v06 Technical-Stop Report

Status: **TECHNICAL STOP BEFORE CORRECTNESS — NO METHOD-PERFORMANCE RESULT**

Date: 2026-08-11

## 1. Execution outcome

After explicit authorization, the frozen aggregate selector chose physical GPU
0. All three selection samples reported 6 MiB used and 0% utilization. The
compiler repeated the anticipated BF16/PTX failure on `sm_75`, produced zero
correctness or timing records, restored the checkpoint exactly, and authorized
one fresh raw process.

The sustained transition gate then passed three more samples at the same UUID,
6 MiB, and 0%. The raw process loaded the pinned model and entered V06's new
pre-capture lifecycle. All wrist warm-ups completed. Downstream warm-up then
failed while requesting 22 MiB, before either graph was captured and before
the first correctness query.

The resource monitor observed peak allocation of 22.5932 GiB and peak
reservation of 23.2285 GiB. Reservation exceeded the unchanged 23 GiB cap by
245,366,784 bytes.

## 2. Scientific interpretation

V06 tested the specific hypothesis motivated by V05: avoiding the retained
wrist graph during downstream warm-up might recover enough memory. The
hypothesis did not hold on TITAN RTX. Downstream warm-up again reached
24,939,331,584 trace-reserved bytes—the same trace reservation observed in
V05—and OOMed before any capture. Moving both warm-ups before capture therefore
did not reduce the dominant eager preparation footprint enough to fit.

V06 contains zero full model queries, correctness records, schedule warm-ups,
timed records, simulator operations, rewards, success fields, or task outcomes.
The analyzer and finalizer were not run. This is a negative environment-
feasibility result, not positive or negative method-performance evidence.

## 3. Integrity and resource audit

- Execution revision:
  `535a4a6b80be114b2cc03bfbfce49a6dbc4ccf05`.
- Selected GPU: physical 0, UUID
  `GPU-bb2451d6-2989-a112-5c18-8892943710e4`, compute capability 7.5.
- Compiler preparation: one core launch; 81.3003 seconds.
- Raw preparation: eight recorded core launches; 7.8152 seconds.
- Completed V06 pre-capture warm-up stages: wrist only.
- Completed graph captures: none.
- Maximum concurrent model processes: one.
- Post-stop selected-GPU telemetry: 6 MiB used, 0% utilization.
- No simulator, download, task outcome, unrelated process, or allocation was
  accessed.

Both attempts restored the protected checkpoint files byte-for-byte, removed
their own loader backups, and left SAVR, OpenVLA-OFT, and LIBERO clean.

## 4. Immutable evidence

- Launch semantic SHA-256:
  `e686d3497e5a92d75924246b3d0566bd9ac59aaab6dcc665495b32364652c699`.
- LIBERO-config semantic SHA-256:
  `3cb72e7cad7e4539969f82de6ce23d464b42c46f834aa040f1833f4dc592b56f`.
- Compiler-attempt semantic SHA-256:
  `0f458015aa87d9cd85a8321074b426d0b367e5a4be5be50d1c1a28d92090cc7f`.
- Raw-transition semantic SHA-256:
  `eb9436a92a750d152f19d23a639427940fce2094aff6585b9cee5636f87fbcb4`.
- Transition-revalidation semantic SHA-256:
  `d21f85d98b7e6fbc036a6f24dddf438f1b83777c7646cd10c431a5ed3b0aaaa3`.
- Raw-attempt semantic SHA-256:
  `ab6c3407efd4d9b3b593b7878c3b65441a1ba380a1fb98dd60698bc4283235d8`.
- Curated technical-stop semantic SHA-256:
  `0588f628a118a2f467215c2337bc23452f3b8e98d0b5865c37be0d2892a18edb`.
- Curated record: `reports/runtime/acr_v5_d_v06_technical_stop.json`.

The complete create-once raw evidence remains on TITAN under
`results/acr-v5d-real-tensor-feasibility-v06/`.

## 5. Disposition

`STOP_NO_RETRY_V06_MEMORY_INFEASIBLE_ON_TITAN_RTX_USE_COMPATIBLE_HIGHER_MEMORY_HARDWARE_OR_SEPARATELY_RESEARCHED_SYSTEM_CHANGE`

V06 is immutable and V5-E remains ineligible. Retrying, changing allocator
settings, reducing warm-ups, raising the cap, or shopping backends would violate
the frozen design. The exact V5-D method now needs compatible higher-memory
hardware or a separately researched and frozen system change before real-
tensor correctness and latency can be evaluated.
