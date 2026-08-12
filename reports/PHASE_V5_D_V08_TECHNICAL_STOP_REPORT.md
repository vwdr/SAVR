# Phase V5-D V08 Technical-Stop Report

Status: **MEMORY RECOVERY CONFIRMED; TECHNICAL STOP BEFORE CORRECTNESS**

Date: 2026-08-11

## 1. Execution outcome

The frozen selector chose physical GPU 0 after every aggregate sample showed
6 MiB used and 0% utilization. The compiler produced the anticipated BF16/PTX
failure on `sm_75`, emitted no correctness or timing record, restored the
checkpoint exactly, and permitted one fresh raw process. The raw transition
gate then passed three more aggregate samples at the same UUID, 6 MiB, and 0%.

The raw process attested both `expandable_segments:True` and the exact V08
inference state: gradient tracking disabled and inference mode enabled. It
completed all wrist and downstream pre-capture warm-ups and captured the wrist
graph. The downstream graph capture then failed because CUDA reported an
operation failure caused by a previous error during capture. The immutable
error does not identify the originating kernel or operation, so a narrower
root cause must not be claimed from this attempt.

## 2. V08 hypothesis result

V08's specific mechanism was strongly confirmed. Peak reservation fell from
V07's 24,937,234,432 bytes (23.2246 GiB) to 16,403,922,944 bytes
(15.2773 GiB), a reduction of 8,533,311,488 bytes. Peak allocation fell by
8,649,790,464 bytes. The raw attempt was 8,292,139,008 bytes (7.7227 GiB)
below the unchanged 23 GiB cap.

Therefore, the missing inference semantics—not insufficient TITAN memory—were
the active cause of the prior preparation-memory blocker. V08 also restored
the prior thread-local inference state exactly. However, V08 did not establish
a usable two-graph backend because the second capture failed.

## 3. Scientific boundary

V08 contains zero full model queries, correctness records, schedule warm-ups,
timed records, simulator operations, rewards, success fields, or task outcomes.
No final record exists, and neither analyzer nor independent verifier ran.
This is positive mechanism evidence about memory recovery, but it is not a
positive or negative result about ACR correctness, efficiency, or task success.

## 4. Integrity and resource audit

- Execution revision: `a91917cc4be5ad65e227f022dfb63408b1f9a8cb`.
- Selected GPU: physical 0, UUID
  `GPU-bb2451d6-2989-a112-5c18-8892943710e4`, compute capability 7.5.
- Compiler preparation: one core launch; 79.9668 seconds.
- Raw preparation: eight recorded core launches; 9.7444 seconds.
- Completed pre-capture warm-up order: wrist, downstream.
- Completed graph-capture order: wrist only.
- Maximum concurrent model processes: one.
- Post-stop selected-GPU telemetry: 6 MiB used and 0% utilization.
- No simulator, download, outcome, unrelated process, or unrelated allocation
  was accessed.

Both attempts restored every protected checkpoint file byte-for-byte, removed
their own loader backups, and left the repository clean. The inference context
also restored the prior thread state.

## 5. Immutable evidence

- Launch semantic SHA-256:
  `ef59ec64cc1b07b972b199dcd83ef1c6a88faef7798b26c3bc7c8651edecdd72`.
- LIBERO-config semantic SHA-256:
  `3c496f5f889a0d2753a0a85ff92d9ebad3d1c7ee1c3dffc5c3587dc82eae5ad8`.
- Compiler-attempt semantic SHA-256:
  `a976ee0775472e0f4e9d1a6dc631f7397271af9d26d9d99fcbd5ee568ac0ee4b`.
- Raw-transition semantic SHA-256:
  `f1d7e611cf4986507c9054307c570e1e802dce8434286b27e3eac52b8323ef23`.
- Transition-revalidation semantic SHA-256:
  `2cc9fe48ebd68b704f33d7e49e93c93fe0da8ea03216655b26ce6b977868bcf5`.
- Raw-attempt semantic SHA-256:
  `876502f1ee281ad2fd425fab856e9108594e46511da3e081b9bd32624175aa9e`.
- Inference-lifecycle semantic SHA-256:
  `41fc9c80c1645eb8d65f6c152baedc824a47980e6388d9840dc00dff6735d892`.
- Curated technical-stop semantic SHA-256:
  `3572abf107ad1b0ef10557e27c66b3d5ad1d967a5f82b633c572bef907d16d98`.
- Curated record: `reports/runtime/acr_v5_d_v08_technical_stop.json`.

The complete create-once raw evidence remains on TITAN under
`results/acr-v5d-real-tensor-feasibility-v08/`.

## 6. Disposition

`STOP_NO_RETRY_V08_MEMORY_RECOVERED_SECOND_GRAPH_CAPTURE_TECHNICAL_FAILURE`

V08 is immutable and V5-E remains ineligible. Do not retry V08. Any further
same-hardware route must separately research the capture error, define a
narrow falsifiable correction, freeze a new identity, and pass CUDA-hidden
verification before another GPU is selected.
