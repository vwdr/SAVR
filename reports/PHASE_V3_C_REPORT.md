# ACR Version 3 Phase V3-C Report

Status: **COMPLETE — POSITIVE CORRECTNESS AND LATENCY GATE**

Date: 2026-08-04

## Decision

SA-BDP-ACR passed every frozen V3-C correctness and latency condition. This is
the first predeclared positive method result in the ACR redesign program. It
supports proceeding to a separately authorized V3-D task-success evaluation;
it is not yet a task-success or paper-level result.

## Execution identity and resources

- Run: `acr-v3c-correctness-latency-v01`.
- SAVR revision: `77f41954e91f3416fe0b7c550305901ee4db4e4d`.
- GPU: physical ID `3`, UUID
  `GPU-1a087319-2f38-30c6-ac06-d362f0643af1`, NVIDIA TITAN RTX.
- Pre-run aggregate state: 9 MiB used, 0% utilization.
- Post-process aggregate state: 6 MiB used, 0% utilization.
- Model processes: one; model queries: exactly `64/64`.
- Simulator resets/episodes, rollouts, downloads, training, and new benchmark
  outcomes: `0`.
- Run wall time: `83.6411` seconds; peak allocated/reserved GPU memory:
  `16,100,537,856` / `16,238,247,936` bytes.
- Artifact bytes: `87,602`, below the 512 MiB cap.
- Post-result repository verification: `216` tests plus `9` subtests pass in
  TITAN's pinned CPU environment; the independent analyzer and 85-file
  bootstrap validation pass.

## Correctness

All eight correctness queries passed:

- BFR and V3 refresh returned the exact upstream projected tokens on both
  deterministic inputs (maximum absolute difference `0.0`, stronger than the
  frozen `rtol=0.016`, `atol=1e-5` allowance).
- BFR and V3 refresh actions were bitwise identical to sequential FR.
- V3 reuse tokens and actions were bitwise identical to V2 reuse for the same
  current input and owned scene cache.
- V3 reuse actions were also bitwise identical to the sequential-FR oracle.
- Actual physical calls reconciled: sequential FR `2/2/1`; BFR, V3 refresh,
  and V3 reuse `1/1/1` for SigLIP/DINOv2/projector.
- All 56 warm-up/timed calls returned the same action hash.
- Scene/wrist order, cache ownership, logical work, source identities, finite
  outputs, and episode restoration passed.

## Latency result

Every path retains all 12 timed repetitions; no outlier was deleted.

| Path | Median wall (ms) | Mean wall (ms) | Median visual CUDA (ms) | Mean visual CUDA (ms) |
|---|---:|---:|---:|---:|
| Sequential FR | 1229.4910 | 1229.4771 | 152.5992 | 152.5714 |
| Batched FR | 1191.3517 | 1191.1877 | 115.0153 | 115.0161 |
| V3 refresh | 1197.8476 | 1197.6186 | 114.8876 | 114.9153 |
| V3 reuse | 1161.6012 | 1161.8354 | 75.6681 | 75.6478 |

At the frozen reuse weight `0.26055045871559634`, V3 weighted wall time is
`1188.4036` ms and weighted visual CUDA time is `104.6690` ms.

| Frozen gate | Observed | Required | Result |
|---|---:|---:|---|
| BFR / sequential-FR wall ratio | 0.968980 | ≤ 0.98 | PASS |
| V3 refresh / BFR wall ratio | 1.005452 | ≤ 1.02 | PASS |
| V3 reuse / BFR wall ratio | 0.975028 | ≤ 0.98 | PASS |
| V3 weighted / sequential-FR wall ratio | 0.966582 | ≤ 0.98 | PASS |
| V3 weighted / BFR wall ratio | 0.997525 | ≤ 1.00 | PASS |
| V3 weighted visual CUDA reduction | 31.4092% | ≥ 10% | PASS |

Thus BFR reduced sequential-FR median wall time by `3.1020%`. The complete V3
method reduced weighted wall time by `3.3418%` versus sequential FR, remained
`0.2475%` faster than BFR, and reduced weighted visual CUDA work by `31.4092%`.

## Integrity and preservation

- Exactly 64 unique query identities reconcile: 8 correctness, 8 warm-up, and
  48 timed queries, with 12 timed repetitions per path.
- Result semantic SHA-256:
  `3f77171fbf42015fb0f6e74c0f5d49c8f58890a64355b2de4348407cef79ab02`.
- Complete-result file SHA-256:
  `a419edd2fbccc0239169d3fa11a2c84e5202b0c0ca6237ddecaf4003e353682d`.
- Manifest file SHA-256:
  `71ab1d0bf19f38b7e9f7909ec08f5e47f4fdc0c539e9472646780f35c7aae69a`.
- Loader-created checkpoint backups were removed and the three protected
  checkpoint files restored to their accepted hashes.
- SAVR, OpenVLA-OFT, LIBERO, and VLA-Cache worktrees are clean.
- No unrelated university file, process, environment, service, allocation, or
  server configuration was modified.

The complete per-query record and manifest are preserved in
`reports/runtime/acr_v3_c.json` and
`reports/runtime/acr_v3_c_manifest.json`. The independent committed analyzer
reproduces the positive decision.

## Evidence boundary and next gate

V3-C establishes real-model numerical correctness and bounded synthetic-input
latency on one GPU. It does not establish closed-loop LIBERO success,
generalization, statistical task non-inferiority, or final paper claims.

V3-C is complete. V3-D remains unauthorized and requires a separate user
decision before any simulator episode or Object state `3-9` outcome.
