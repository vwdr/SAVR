# Phase V5-D v02 Technical-Stop Report

Status: **TECHNICAL STOP BEFORE CORRECTNESS — NO METHOD RESULT**

Date: 2026-08-10

## 1. What happened

After explicit user authorization, the selector took the frozen three
aggregate samples. All four TITAN RTX devices were eligible; physical GPU 0
was selected by the predeclared lowest-index rule. It showed 6 MiB used and 0%
utilization in every sample.

The canonical run-local LIBERO configuration passed, and the pinned OpenVLA
model loaded. The first TorchInductor preparation call then failed before any
correctness record. Triton generated BF16 PTX, while the selected TITAN RTX is
compute capability 7.5 and `ptxas` requires target `sm_80` or newer for those
instructions. The attempt was correctly classified as a pre-output compiler
technical failure.

The frozen waterfall would ordinarily permit a fresh-process raw-CUDA-graph
attempt. The fail-closed restoration guard instead stopped the wrapper because
the upstream loader created two files named `.back.<timestamp>`, while the
cleanup allowlist recognized `.bak` and names containing `backup`, but not
`.back`. No raw attempt was started.

## 2. Scientific interpretation

V02 produced no method result. The model loaded, but there were zero full model
queries, correctness records, warm-ups, timed records, simulator operations,
or task outcomes. Neither rewards nor success fields were accessed. Therefore
this run cannot support or oppose the positive-results paper route.

The compiler error is hardware/backend compatibility evidence, not evidence
against state-aware visual refresh. The raw backend remains empirically
untested.

## 3. Resource and integrity audit

- Selected device: physical GPU 0, UUID
  `GPU-bb2451d6-2989-a112-5c18-8892943710e4`.
- Peak allocated memory: `15,768,091,136` bytes.
- Peak reserved memory: `16,076,767,232` bytes.
- Compiler-attempt wall time: `81.28880628105253` seconds.
- Post-stop selected-GPU telemetry: 6 MiB used, 0% utilization.
- No simulator, download, new outcome, unrelated process, or allocation was
  accessed.

The restoration routine had already restored all three protected checkpoint
files byte-for-byte. Two duplicate loader backups remained. Each duplicate was
hashed and confirmed identical to its protected original, then only those two
exact timestamped files were removed. The checkpoint inventory and all three
protected hashes now pass the pinned validator. The SAVR, OpenVLA-OFT, and
LIBERO source trees are clean.

## 4. Immutable evidence

- Launch semantic SHA-256:
  `0c723dd5ff93c0dfe4544dd2f50b6e7ff91409fb3ba47059508701fa8081cec8`.
- LIBERO-config attestation semantic SHA-256:
  `40869c46d250099836a0ddf0385d421b4ff5cea9ca5c9110b301a5b5211cf14d`.
- Compiler-attempt semantic SHA-256:
  `2eb417293ece405c8c161ab275926318766698c600e9d31a1ea89ad56934ec68`.
- Curated technical-stop semantic SHA-256:
  `0a30bd847bf2e1549c376200e559a23c670b33c0b01215926c90a15704487661`.
- Curated record: `reports/runtime/acr_v5_d_v02_technical_stop.json`.

The full run-local launch, configuration, and compiler-attempt records remain
immutable on TITAN under
`results/acr-v5d-real-tensor-feasibility-v02/`.

## 5. Disposition

`STOP_NO_RETRY_PREPARE_SEPARATELY_AUTHORIZED_V5D_V03`

V02 will not be altered or retried. The next eligible work is the separately
reviewed v03 recovery correction. This report authorizes neither v03
implementation nor GPU execution.
