# BRACE-B2 Baseline, Cache, and Provenance Correctness Report

Date: 2026-08-25

Accepted run: `brace-b2-correctness-v02`

Decision: **ACCEPTED WITH COMPARATOR DISPOSITIONS**

## Scope

B2 tested CPU/synthetic implementation correctness only. It authenticated the
corrected VLA-Cache evaluator, two isolated DynamicCache stacks, BRACE cache
and provenance mechanics, adversarial tests, and pinned comparator sources.
CUDA was hidden and uninitialized. No model, checkpoint, dataset, GPU,
simulator, policy outcome, latency measurement, or task-performance result was
used. B3 did not start.

## Immutable v01 technical stop

`brace-b2-correctness-v01` ended before a run summary because two checker
oracles were wrong: restoration was compared with an aliased tensor that the
test later mutated, and the core Transformers 4.40.1 source was incorrectly
required to contain the separate VLA-Cache 4.47.0 position-update
customization. The attempt made zero model queries, policy observations,
simulator steps, or GPU operations. Its evidence was preserved rather than
rewritten.

- Manifest SHA-256:
  `793e733cafbef8b79db3f2a802119c573d7f822b5345a532f5050ca1c1b6b402`
- Technical-stop semantic SHA-256:
  `1d8d6ff69a63d77774d885f598cf7b0b9c43587d99ed36d79115598bcff20a53`
- Technical-stop file SHA-256:
  `34b0072184225361617ac7189e2f18aea0835f72f9a24cbd5d3388939bcc2cdb`

V02 corrected only those two test oracles. It did not change the method,
profiles, comparator set, scientific gates, resource caps, or B3 proposal.

## Frozen v02 identity

- Source revision: `b06ab06c98e8f958ee58aa254c92a7c9d25618ce`
- Configuration semantic SHA-256:
  `04370384312f4474fd5488fa3b4dad1559d3d7916564501c802fdec8214906bd`
- Summary semantic SHA-256:
  `368a86377cbe1507445f0d06cd2ed48395e34c8e770d0f51c0dc9adaca95b2a9`
- Manifest file SHA-256:
  `8668e1d273fa10ca66f199bb75f7b4d73e2333868854ab5508e27d556c6c6e7c`
- Summary file SHA-256:
  `10f08e83aec504205064849142142a5c6c47079e6be4ba0e674d67c9829e4eb0`

Independent reconciliation reproduced the summary semantic hash and confirmed
13/13 true gates.

## Results

- Both pinned stacks passed independent clone, failure-safe transaction
  restoration, and BRACE absolute-position update checks: Transformers 4.40.1
  and 4.47.0 on PyTorch 2.2.0+cu118, with CUDA uninitialized.
- The corrected evaluator uses the true previous cache-source frame and
  propagates episode exceptions without changing the cache algorithm or its
  configuration.
- The server adversarial suite passed: 23 tests in 1.40 seconds.
- Runtime sequence mapping, camera-token change scores, dense sidecar
  attention, profile nesting, contract expiry/abort, P0--P4 execution
  identities, per-token provenance, live-source retention, immutable intent
  records, and cross-arm isolation passed their frozen checks.
- Pinned comparator source totaled 132,931,536 bytes, below the 1 GiB cap.
- The complete run took 5.15 seconds and remained below all resource caps.

## Comparator dispositions

- VLA-ADP source and configuration passed preflight, but its released package
  requires an isolated overlay on the core 4.40.1 stack.
- VLA-Pruner source and configuration passed preflight, but its customized
  4.47.0 stack must remain isolated.
- SpecPrune-VLA source/configuration checks passed, but the repository lacks
  the advertised top-level license; method-specific execution remains blocked
  pending license clarification.
- Gated VLA-Cache (arXiv:2608.10824) had no discoverable official code at the
  freeze. Any later implementation must be labeled a matched reproduction,
  not official code.

## Interpretation and boundary

B2 establishes that the frozen BRACE software substrate is internally
consistent under synthetic CPU tests and that known comparator limitations are
explicit. It does not establish real-model parity, memory feasibility, speed,
closed-loop reliability, competitiveness, or a positive-paper result.

B3 is the next protocol-eligible phase, bounded by at most 480 balanced
real-model queries, one separately authorized GPU, zero simulator outcomes,
and 23 GiB peak memory. It requires separate authorization. No work outside
`/home/ved/SAVR` was modified on TITAN.
