# Phase V5-D V09 Technical-Stop Report

Status: **ALLOCATOR HYPOTHESIS REJECTED; TECHNICAL STOP BEFORE CORRECTNESS**

Date: 2026-08-12

## 1. Execution outcome

The frozen selector chose physical GPU 0 after every aggregate sample showed
6 MiB used and 0% utilization. The compiler produced the anticipated BF16/PTX
failure on `sm_75`, emitted no correctness or timing record, restored the
checkpoint exactly, and permitted one fresh raw process. The raw transition
gate then passed three more aggregate samples at the same UUID, 6 MiB, and 0%.

The raw process authenticated the V09 correction: the allocator environment
was absent and PyTorch reported the default `native` backend. It also entered
the frozen inference state with gradient tracking disabled and inference mode
enabled. Both pre-capture warm-ups and the wrist graph capture completed. The
downstream graph capture then failed at the same boundary as V08 because CUDA
reported an operation failure caused by a previous error during capture.

## 2. V09 hypothesis result

V09 rejects its narrow allocator hypothesis on this pinned stack. Replacing
`expandable_segments:True` with the default native allocator did not make the
two-graph backend usable. Peak reservation was 16,519,266,304 bytes
(15.3848 GiB), 115,343,360 bytes higher than V08, while remaining
8,176,795,648 bytes (7.6152 GiB) below the unchanged 23 GiB cap. Peak
allocation was 17,257,472 bytes higher than V08.

Thus memory capacity remains solved, but allocator choice was not the cause of
the second-capture failure. The surviving problem is multi-graph CUDA capture
compatibility on the pinned `sm_75`/PyTorch stack. The immutable CUDA message
still does not isolate the originating operation, so no narrower kernel-level
cause is claimed.

## 3. Scientific boundary

V09 contains zero full model queries, correctness records, schedule warm-ups,
timed records, simulator operations, rewards, success fields, or task outcomes.
No final record exists, and neither analyzer nor independent verifier ran.
This is a negative technical result about one recovery hypothesis—not a
positive or negative result about ACR correctness, efficiency, or task success.

## 4. Integrity and resource audit

- Execution revision: `ffc8146e58301b6bc5c6688c42fdec9f85b0e517`.
- Selected GPU: physical 0, UUID
  `GPU-bb2451d6-2989-a112-5c18-8892943710e4`, compute capability 7.5.
- Compiler preparation: one core launch; 80.2046 seconds.
- Raw preparation: eight recorded core launches; 9.7542 seconds.
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
  `e64b73fe9620f189327deeff597c882bdb29667da872351b475d5ab45c19cb6c`.
- LIBERO-config semantic SHA-256:
  `c96a9412a165aaf444e0e6aaa5e50907916563df0917923bc4ff618acaf4d27d`.
- Compiler-attempt semantic SHA-256:
  `b71099f70c9f38a50ed80d573f237018d45e4130a9a882745d74e6389dfa157b`.
- Raw-transition semantic SHA-256:
  `147b46abdd67e5a1bb4f502aa401f56d1f1cd488c5434d69fa81ff021a7a931d`.
- Transition-revalidation semantic SHA-256:
  `eb63a74f017f5ff2fb07b8c0f0339179078d82a24fa6c38ea57f659b70aa7949`.
- Raw-attempt semantic SHA-256:
  `e2d5058732ed93fc9a4c10b327279af14de5e8c62dcda003f3bb48d9ba01214a`.
- Inference-lifecycle semantic SHA-256:
  `7f3127be2cf818aec659996e291c73e45d9d1e1cd25eaf818fa0cc17570a25cf`.
- Curated technical-stop semantic SHA-256:
  `2113acaad46550b26da8bbfcfe25de4e78312e55e7047974d5b555dd88316209`.
- Curated record: `reports/runtime/acr_v5_d_v09_technical_stop.json`.

The complete create-once raw evidence remains on TITAN under
`results/acr-v5d-real-tensor-feasibility-v09/`.

## 6. Disposition

`STOP_NO_RETRY_V09_DEFAULT_ALLOCATOR_HYPOTHESIS_REJECTED_SECOND_GRAPH_CAPTURE_TECHNICAL_FAILURE`

V09 is immutable and V5-E remains ineligible. Do not retry V09. Any further
same-hardware route must separately research and freeze a capture-architecture
change rather than another memory setting, then pass CUDA-hidden verification
before a new GPU selection.
