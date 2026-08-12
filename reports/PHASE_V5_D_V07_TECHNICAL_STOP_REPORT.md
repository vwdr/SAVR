# Phase V5-D v07 Technical-Stop Report

Status: **TECHNICAL STOP BEFORE CORRECTNESS — NO METHOD-PERFORMANCE RESULT**

Date: 2026-08-11

## 1. Execution outcome

The frozen selector chose physical GPU 0 after three aggregate samples at
6 MiB used and 0% utilization. The compiler repeated the anticipated BF16/PTX
failure on `sm_75`, produced no correctness or timing records, restored the
checkpoint exactly, and authorized one fresh raw process. The sustained
transition gate passed three more samples at the same UUID, 6 MiB, and 0%.

The raw process attested the exact frozen allocator setting
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` before PyTorch import. It
loaded the model, completed all wrist pre-capture warm-ups, and then OOMed on a
22 MiB downstream allocation before either graph capture and before the first
correctness query.

Peak allocation was 23.0152 GiB and peak reservation was 23.2246 GiB. The
unchanged 23 GiB cap was exceeded by 241,172,480 bytes (230 MiB).

## 2. V07 hypothesis result

V07 tested whether PyTorch's documented expandable-segments allocator could
recover V06's reserved-but-unallocated memory. The allocator was active and
the peak reserved-minus-allocated indicator fell from 682,160,640 bytes in V06
to 224,799,232 bytes in V07, a reduction of 457,361,408 bytes. The OOM message
likewise reported about 203.64 MiB reserved but unallocated rather than V06's
approximately 648.56 MiB.

However, peak allocated memory increased by 453,167,104 bytes and peak reserved
memory fell by only 4,194,304 bytes. Therefore the allocator changed memory
layout as intended but did not reduce total preparation memory enough to fit
on the TITAN RTX. This mechanism-specific recovery hypothesis failed.

## 3. Scientific boundary

V07 contains zero full model queries, correctness records, schedule warm-ups,
timed records, graph captures, simulator operations, rewards, success fields,
or task outcomes. The analyzer and finalizer were not run. The result concerns
environment feasibility only; it is neither positive nor negative evidence
about ACR task performance.

## 4. Integrity and resource audit

- Execution revision: `1b9a479b88700e91b3605988d08bfcc9971e69ee`.
- Selected GPU: physical 0, UUID
  `GPU-bb2451d6-2989-a112-5c18-8892943710e4`, compute capability 7.5.
- Compiler preparation: one core launch; 80.2643 seconds.
- Raw preparation: eight recorded core launches; 7.9667 seconds.
- Completed pre-capture warm-up stages: wrist only.
- Completed graph captures: none.
- Maximum concurrent model processes: one.
- Post-stop GPU telemetry: 6 MiB used and 0% utilization.
- No simulator, download, outcome, unrelated process, or unrelated allocation
  was accessed.

Both attempts restored all protected checkpoint files byte-for-byte, removed
their own loader backups, and left the repository clean.

## 5. Immutable evidence

- Launch semantic SHA-256:
  `899c68ef65ff0a8d78d886c6460db9ebbf1ab93e2d4c3bfb1cf0bc30b30ecf7d`.
- LIBERO-config semantic SHA-256:
  `4e1432938ae57348553e467062462fac72af155ea470c6953fe67762d14a1c8e`.
- Compiler-attempt semantic SHA-256:
  `b0484df40cffa599f4861ede85c137d7e4d22b91032631c556d06b3fed541108`.
- Raw-transition semantic SHA-256:
  `2aa7d058eef6d3cb667a7ed5e0503757abe615dda0dc7df095ca055fee4313d0`.
- Transition-revalidation semantic SHA-256:
  `0f573732f4e9cb07b2e2611556194d0bb2b71afb0aa963a7c13d50f871a96f75`.
- Raw-attempt semantic SHA-256:
  `ef8c0eecadf5c251dae9fbb104a4a5ec25688504ce69020c6b0455552514bd19`.
- Curated technical-stop semantic SHA-256:
  `17c6c68ed075f6848768d81eb158ae1d522b2b670df37d0c1db3ab54439bc8c1`.
- Curated record: `reports/runtime/acr_v5_d_v07_technical_stop.json`.

The complete create-once raw evidence remains on TITAN under
`results/acr-v5d-real-tensor-feasibility-v07/`.

## 6. Disposition

`STOP_NO_RETRY_V07_ALLOCATOR_IMPROVED_FRAGMENTATION_BUT_MEMORY_INFEASIBLE_ON_TITAN_RTX`

V07 is immutable and V5-E remains ineligible. Do not retry or tune this
allocator experiment. A future same-hardware route must be separately
researched, justified, frozen, and designed to reduce the active preparation
footprint—not merely allocator fragmentation—while preserving the scientific
method and timing validity.
