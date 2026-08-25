# BRACE-B3 Pre-v05 Technical Audit Report

Date: 2026-08-25  
Disposition: **COMPLETE; ACCEPTED**

## Outcome

Every B3 path that can be exercised without a model load or GPU passed the
frozen pre-v05 audit. This substantially reduces the chance of another simple
integration failure, but it is not a BRACE result and does not guarantee that a
future full-model attempt will complete.

## Corrections and added coverage

- Corrected patch-change range computation to call binary `torch.maximum` and
  `torch.minimum` with two tensor operands.
- Removed hardcoded `cuda:0` allocation from sidecar salience and ordered-token
  outputs; both now inherit the live tensor device.
- Added real-PyTorch tests for patch scoring, all four profiles, P2 wrist-source
  rotation, all three horizons, all 72 timed cycles, deterministic source
  provenance, SDPA capture/salience/output parity, SDPA call-count failure,
  candidate insufficiency, and complete query accounting.
- Added a complete synthetic 388-query analyzer population that exercises
  every conjunctive B3 reporting gate through immutable analysis output.
- Added a server audit of worker, runner, analyzer, checkpoint-restoration,
  pinned private-interface, backend, checkpoint-alias, repository-revision,
  source-authentication, and authorization-boundary contracts.

## Verification

- Local complete repository suite: **423 passed, 8 skipped**. The skipped tests
  require PyTorch packages absent from the local lightweight environment.
- Focused TITAN real-PyTorch suite with CUDA hidden: **8 passed**.
- Immutable TITAN BRACE suite with CUDA hidden: **43 passed**.
- Audit status: `accepted`.
- CUDA initialized: `false`.
- Model loads / model queries / simulator outcomes: `0 / 0 / 0`.
- Protected-outcome access: `false`.
- All worker, runner/analyzer, backend, and pinned-repository gates: `true`.
- Audited source revision:
  `20a57adcbd0c3f22d2e08659eba8cc65ba9ba103`.
- Audit semantic SHA-256:
  `35cb1448b3c2ec80f02bcbfa3fc9c09b927dd0d435fe021d43c1b070f55da273`.
- Local and TITAN audit-file SHA-256:
  `67dd44561bafa8dcdec68c73fa6d353f536750802552c047f331106df5c0c0c8`.

The machine-readable record is
`reports/runtime/brace_b3_pre_v05_audit.json`.

## Interpretation and next boundary

The v04 failure class and the remaining CPU-testable execution/reporting paths
are now covered. Residual risks necessarily require the real model: exact live
checkpoint tensors, cache mutations across a complete physical cycle, GPU
memory behavior, comparator imports under their own environments, and measured
latency. Those risks cannot be proven away by a CPU audit.

No v05 identity, selector, launch record, model load, or GPU attempt was
created. A v05 proposal must freeze only the audited correction, preserve every
scientific method/profile/gate/resource invariant, and obtain separate explicit
authorization. B4 remains unauthorized.
