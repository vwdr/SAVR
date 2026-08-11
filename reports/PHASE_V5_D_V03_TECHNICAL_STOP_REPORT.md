# Phase V5-D v03 Technical-Stop Report

Status: **TECHNICAL STOP BEFORE CORRECTNESS — NO METHOD-PERFORMANCE RESULT**

Date: 2026-08-11

## 1. Execution outcome

After explicit authorization, the frozen aggregate selector chose physical GPU
0. Its three samples each showed 6 MiB used and 0% utilization. V03 then
executed the complete predeclared backend waterfall.

TorchInductor failed on the first compiler preparation call because its BF16
PTX requires `sm_80` or newer, while TITAN RTX is `sm_75`. The v03 restoration
correction worked: both protected files were restored, the two exact
`.back.<timestamp>` artifacts were removed, the inventory gate passed, and a
fresh-process raw transition was mechanically authorized.

The raw-CUDA-graph process loaded the model and completed eight preparation
launches. It then ran out of memory before the first correctness record while
requesting another 22 MiB. Peak allocation was 22.5233 GiB and peak reservation
was 23.2246 GiB. This exceeded the frozen 23 GiB reservation cap by
241,172,480 bytes and exhausted the usable capacity of the 24 GB device.

## 2. Scientific interpretation

V03 contains zero full model queries, correctness records, schedule warm-ups,
timed records, simulator operations, rewards, success fields, or task outcomes.
The analyzer and finalizer were therefore not run.

This is not positive or negative evidence about the selected state-aware
visual-refresh method's correctness, speed, or task performance. It is a
decisive environment-feasibility result: neither frozen optimized backend can
be evaluated on the current TITAN RTX node. Compiler execution requires a
newer architecture, while raw capture exceeds the available memory and the
frozen cap.

## 3. Integrity and resource audit

- Execution revision:
  `db3ec5e94e77256fdc56297cd5da0720f322a1aa`.
- Selected GPU: physical 0, UUID
  `GPU-bb2451d6-2989-a112-5c18-8892943710e4`, compute capability 7.5.
- Compiler preparation: one launch; 80.1089 seconds.
- Raw preparation: eight launches; 7.7560 seconds.
- Maximum concurrent model processes: one.
- Post-stop selected-GPU telemetry: 6 MiB used, 0% utilization.
- No download, simulator, task outcome, unrelated process, or allocation was
  accessed.

Both backend attempts independently passed exact checkpoint restoration. No
loader backup remains. The protected hashes and SAVR, OpenVLA-OFT, and LIBERO
source trees are clean.

## 4. Immutable evidence

- Launch semantic SHA-256:
  `9182c0ae93b5fbedf972bb4959fee6070edeb712a08b8c0169267710a45c48b7`.
- Compiler-attempt semantic SHA-256:
  `0fd183117bd786f684d17ca971ab6ca23f207908ad6b91334b69313f34257ed8`.
- Raw-transition semantic SHA-256:
  `647eba1d4ebb3d25f94ec2d94681d20a471d2e686631337cc0c9f1d452643d1b`.
- Raw-attempt semantic SHA-256:
  `addf0d2d9cf5f259ec13c249fbd6c8a36a599b055dc6d463a18144fd6430550e`.
- Curated technical-stop semantic SHA-256:
  `1016569f642b21266e8f0b75b5906716200055f5d37385c5501b6711f9a6bd54`.
- Curated record: `reports/runtime/acr_v5_d_v03_technical_stop.json`.

The complete immutable run evidence remains on TITAN under
`results/acr-v5d-real-tensor-feasibility-v03/`.

## 5. Disposition

`STOP_NO_RETRY_PREPARE_SEPARATELY_AUTHORIZED_V5D_V04_ENVIRONMENT_AMENDMENT`

V03 will not be changed or retried. V5-E remains ineligible. The next rational
route is to freeze the same experiment for a compatible, higher-memory GPU
environment; it is not to alter thresholds or select a favorable backend after
seeing results.
