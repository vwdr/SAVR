# ACR Phase A5 Preflight

**Authorization:** user approved Phase A5 on 2026-08-03.

**Outcome access at freeze:** A4 upstream-FR development evidence only; no ACR
rollout outcome exists.

## Frozen Stage 1 scope

- Policy: State-Aware Asymmetric Camera Refresh Version 1.
- Candidates: exactly `acr-t25-h2-b30`, `acr-t50-h4-b55`, and
  `acr-t70-h8-b75` from the canonical A4 record.
- Population: LIBERO-Object tasks `0-9`, states `0-2`, seed `0`.
- Planned attempts: 30 per candidate and 90 total.
- First attempt identity: `attempt-0001`.
- Stage 2 population: unopened and unavailable until the Stage 1 analyzer
  mechanically identifies an advancing candidate.

Every query must recompute one wrist SigLIP, DINOv2, and projector path. A
scene reuse must execute zero scene-path components, while a scene refresh
must execute exactly one of each. Current proprioception and downstream
execution remain fresh. Records include per-camera synchronized CUDA timing,
wall timing, exact decisions, inputs, actions, component counts, cache age,
configuration/provenance hashes, and episode reconciliation.

## Mechanical Stage 1 gate

A candidate advances only with exactly 30 reconciled terminal episodes, 30/30
success, 3/3 success for every task, at least 15% aggregate scene reuse, wrist
refresh count equal to query count, exact scene-component savings on every
reuse, and zero technical, cache, counter, timing, or schema failures.

Failure of all candidates stops A5. Thresholds, candidates, horizons, caps,
gates, and populations cannot be changed. Stage 2 executes only for candidates
listed as advancing by the committed analyzer.

## Conditional Stage 2 scope

For each advancing candidate, run Object tasks `0-9`, states `3-9`, seed `0`
once. Combine these 70 episodes with its fixed 30 Stage 1 episodes. Development
eligibility requires success no more than two episodes below the A4 FR result,
no task more than one success below FR, at least 40% scene reuse, at least 10%
visual CUDA-time reduction, exact invariants, and zero technical failures.
Selection follows Protocol V1 without discretionary interpretation.
Visual-CUDA and synchronized query-latency point values are the sum of their
stored steady-state per-query measurements divided by the corresponding query
count. Reduction is `1 - ACR / upstream-FR` using the matching fixed Object
population. These definitions are frozen before ACR outcome access.

## Hard resources and recovery

- one explicitly selected GPU and one model process;
- 86,400 cumulative seconds;
- at most 300 attempted episodes, including technical attempts;
- 2,147,483,648 cumulative A5 result bytes;
- no downloads and no writes outside `/home/ved/SAVR`.

Scientific failures are terminal and never rerun. No episode is retried
automatically. A technical attempt remains immutable and stops the active run;
any recovery requires a separately documented, predeclared rule and uses the
next monotonically increasing attempt identity.

## Publication boundary

This configuration, runner, analyzer, tests, and preflight must merge and be
synchronized to TITAN before the first A5 episode. A5 outcomes cannot modify
this checkpoint. Phase A6 is not authorized.
