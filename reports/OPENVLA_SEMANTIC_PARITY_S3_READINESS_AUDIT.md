# OpenVLA Semantic Parity S3 Readiness Audit

Date: 2026-08-31  
Decision: `READY_FOR_ONE_OUTCOME_FREE_S3_ATTEMPT`

## Frozen purpose

S3 tests whether the corrected custom dense path is the same pinned OpenVLA-OFT
policy as the released evaluator before any renewed D62, CAC C1, C1H, training,
or simulator work. It is a semantic integration gate, not a method result.

## Population and calls

- Eight immutable offline observations: two from each of LIBERO-Spatial,
  LIBERO-Object, LIBERO-Goal, and LIBERO-10.
- Observation step zero only; no expert actions, rewards, success fields,
  simulator states, or locked outcomes.
- Exactly four full policy calls per observation: released official evaluator,
  corrected `use_cache=None`, corrected `use_cache=True`, and corrected
  `use_cache=True` with the reversible sidecar.
- Exactly 32 calls total, with official/custom order alternated by observation.

## Independent comparisons

The official path is captured by temporary reversible hooks. The diagnostic
compares token IDs, embeddings, action masks, pixels, language features, vision
features, normalized proprioception, projected features, masked inputs,
multimodal tensors/masks, the exact action-head input, normalized actions, and
unnormalized actions. Numeric tolerance is at most `1e-6`; deterministic
custom/cache/sidecar comparisons are exact. Only hashes, shapes, maxima, and
the first mismatch are persisted.

## Failure containment

- One explicitly selected GPU and one model process.
- Strict peak aggregate memory below 23,552 MiB.
- Thirty-minute wall cap, 256-MiB artifact cap, offline caches only, no
  downloads, and no automatic retry.
- New immutable output root; source/config/code/checkpoint hash drift fails
  before model execution.
- Model configuration and checkpoint inventory are restored; technical
  tracebacks are sealed.
- Any mismatch stops S3. S4, C1H, C2, training, and simulator use remain
  unauthorized.

## Verification

- 22 dedicated semantic/package tests pass on TITAN.
- Full suite: 536 passes and 9 subtest passes.
- The sole failure is the pre-existing unrelated ACR-V10 test that expects
  completed historical V10 result directories not to exist.
- Real-data/checkpoint static preflight passes.
- Python compilation, shell syntax, whitespace, source-drift, output-reuse,
  timeout, memory, call arithmetic, no-outcome, and no-legacy-tail checks pass.

## Authorization boundary

The user approved continuation on 2026-08-31. This authorizes one bounded S3
attempt only. A pass stops before S4; a mismatch or technical stop cannot
trigger an automatic retry.
