# Phase V5-D V10 Technical-Stop Report

Status: **TWO-CAPTURE HYPOTHESIS REJECTED; TECHNICAL STOP BEFORE CORRECTNESS**

Date: 2026-08-13

## 1. Execution outcome

The frozen aggregate selector chose physical GPU 0 after every sample met the
eligibility limits. The compiler produced the anticipated zero-query BF16/PTX
failure on `sm_75`, restored the checkpoint exactly, and permitted one fresh
raw process. The transition gate then passed three samples at the same UUID,
6 MiB used, and 0% utilization.

The raw process authenticated V10's default native allocator and complete
inference-mode lifecycle. It completed exactly three eager wrist warm-ups and
three eager downstream warm-ups. With no wrist graph, no retained prior graph,
and no shared pool, it attempted the downstream core as the first and only
CUDA graph. CUDA invalidated that capture before any graph was retained.

## 2. V10 hypothesis result

V10 rejects the hypothesis that V08/V09 failed because downstream capture
followed a retained wrist graph. Removing both the wrist capture and the
second-capture transition did not make the unchanged downstream body
capturable. The surviving causal class is an operation in the downstream body
or a broader process/runtime CUDA-graph incompatibility on the pinned stack.
The generic CUDA error does not identify the originating kernel, so no narrower
cause is claimed.

Peak reservation was 16,519,266,304 bytes (15.3848 GiB), exactly equal to V09
and 8,176,795,648 bytes (7.6152 GiB) below the unchanged 23 GiB cap. Memory
capacity was therefore not the stop condition.

## 3. Scientific boundary

V10 contains zero full model queries, correctness records, schedule warm-ups,
timed records, simulator operations, rewards, success fields, or task outcomes.
No final record exists, and neither the performance analyzer nor independent
performance verifier ran. This is a negative technical result about the V10
capture-architecture hypothesis—not a result about ACR correctness,
efficiency, or task success.

## 4. Integrity and resource audit

- Execution revision: `bd926130bd7a34001f8c5ab0808c212562dda898`.
- Selected GPU: physical 0, UUID
  `GPU-bb2451d6-2989-a112-5c18-8892943710e4`, compute capability 7.5.
- Compiler preparation: one core launch; 79.9636 seconds.
- Raw preparation: seven authenticated launches; 9.5422 seconds.
- Completed warm-up order: wrist, downstream.
- Capture attempt order: downstream; completed capture order: empty.
- Graph objects created/retained: `1/0`; wrist captures: `0`.
- Maximum concurrent model processes: one.
- Post-stop selected-GPU telemetry: 6 MiB used and 0% utilization.
- No simulator, download, outcome, unrelated process identity, or unrelated
  allocation was accessed.

Both processes restored every protected checkpoint file byte-for-byte,
removed their own loader backups, and left the repository clean. The raw
process also restored the prior inference thread state.

## 5. Immutable evidence

- Launch semantic SHA-256:
  `0db68b839368670e8f87ddf0700adfdbba9a43b33b951a450db48881c8dfde4f`.
- LIBERO-config semantic SHA-256:
  `23bf3b639f103a52d429342571aa44ae0f77bc90a4cd2eb9081919cad197b83d`.
- Compiler-attempt semantic SHA-256:
  `a9d8fd1581ca97f50ecc072b9304f8aad2aad102f8152c096db0c76f306c40d1`.
- Raw-transition semantic SHA-256:
  `b46af765d14a9ed77da0069e5be90c45c4fc8ebe46dc82a42948f87fa8687c41`.
- Transition-revalidation semantic SHA-256:
  `5e3e3329df070ecf81e321582b79bd6d9973a609bd6a09ab42e5350b2adf2402`.
- Raw-attempt semantic SHA-256:
  `318a0621e4072dbabbd938cc4bfc1f7d1d97cc2ef6b151b4983582f027ad7f84`.
- Inference-lifecycle semantic SHA-256:
  `ad00224ad6c71ba00c2a9a0f5aa919de16b64fe3c02744a3749688c9900ca703`.
- Curated technical-stop semantic SHA-256:
  `fd72f7bed0869820e885d707264a12e4bbf3a4d97e89b8de2c3eed0a84a856d0`.
- Curated record: `reports/runtime/acr_v5_d_v10_technical_stop.json`.

The complete create-once evidence remains on TITAN under
`results/acr-v5d-real-tensor-feasibility-v10/`.

## 6. Disposition

`STOP_NO_RETRY_V10_TWO_CAPTURE_HYPOTHESIS_REJECTED_DOWNSTREAM_ONLY_CAPTURE_TECHNICAL_FAILURE`

V10 is immutable and must not be retried. V5-E remains ineligible. Any future
route must be separately researched and frozen; it cannot assume that merely
changing graph count, capture order, allocator, or available memory will make
the unchanged downstream body capturable.
