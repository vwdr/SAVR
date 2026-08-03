# ACR Phase A2 Checkpoint Report

**Date:** 2026-08-02

**Phase:** A2 — implementation and CPU verification

**Disposition:** **COMPLETE — ALL A2 EXIT GATES PASS**

## Outcome

ACR Version 1 is implemented as a separate project-owned package. It does not
modify SAVR1-3 or either pinned upstream tree. The implementation includes:

- a camera-factorized OpenVLA-OFT adapter that assembles projected tokens only
  as `[scene][wrist]` and preserves the unchanged downstream query;
- deterministic SA-ACR, Scene-Visual, Scene-Periodic, and factorized-FR
  controller semantics;
- a strict one-entry projected scene-token cache with complete context,
  shape, dtype, device, and patch-count compatibility;
- deterministic scene, normalized EEF translation, transition, and audit
  signals;
- per-camera SigLIP, DINOv2, and projector call counts plus synchronized
  component timing;
- write-once attempt/query/episode identities, preserve-and-restart recovery,
  exact lossless compact float encoding, frozen-schema validation, and record
  reconciliation;
- deterministic derivation of exactly the three frozen ACR templates; and
- Wilson, Newcombe paired method 10, exact McNemar, deterministic stratified
  paired bootstrap, Holm adjustment, and the frozen sample-size gate.

No real checkpoint/model was loaded. No GPU was selected or used. No LIBERO
process, rollout episode, ACR population, or outcome was accessed. No
data-derived ACR threshold was produced. The manuscript was not modified.

## CPU verification

| Check | Result |
|---|---|
| Complete repository test suite | PASS — 133 tests |
| New ACR-specific suite | PASS — 31 tests |
| Source and test Ruff checks | PASS |
| Source mypy checks | PASS — 22 source files |
| Bootstrap validation | PASS — 68 required files |
| Wheel/package build | PASS |
| Whitespace/diff validation | PASS |

The ACR tests cover scene-first/wrist-second token order, block shapes,
camera isolation, cache ownership and every mutable context identity,
always-fresh wrist behavior, zero scene work on reuse, current proprioception,
every controller reason and exact boundary, warm-up, transition, horizon,
hard prefix cap, fail-closed metadata, exception restoration, interrupted-run
recovery, immutable schema/reconciliation failures, candidate byte identity
and tie-breaking, synchronized timing, and statistical reference values.

The paired confidence interval implementation reproduces Newcombe's published
method-10 example for cells `(e,f,g,h)=(36,12,2,0)`: `[0.0569, 0.3404]` to
four decimal places.

## Phase-boundary audit

1. **What was authorized?** Phase A2 implementation and CPU verification only.
2. **What actually ran?** Local source editing, deterministic mocked CPU tests,
   static checks, schema validation, and package construction.
3. **Which populations were consumed?** None.
4. **Did records and counters reconcile?** Yes in all positive fixtures, and
   every tested mismatch failed closed.
5. **Did any gate fail?** No A2 gate failed. Test-discovered implementation
   defects were corrected before the final full pass.
6. **Were rules changed after outcomes?** No ACR outcome exists and Protocol
   V1 was not changed.
7. **Is the next population untouched?** Yes. A3 uses no population and A4-A8
   populations remain unopened.
8. **Were resources within the A2 cap?** Yes: CPU only, zero GPU time, compact
   source/test artifacts, and no model or dataset download.
9. **Was the university server affected?** No A2 implementation command ran on
   TITAN before the finalized repository synchronization; synchronization is
   limited to `/home/ved/SAVR` and CPU verification only.

## Preserved uncertainty

Mathematical camera separability is implemented but real-model bitwise parity
is not claimed. Phase A3 must still prove exact projected-token and action
parity within at most 16 policy queries and zero rollout episodes. Any parity
failure remains a hard stop.

## Stop point

Phase A2 stops here. Phase A3 requires explicit user authorization. A2 does
not authorize a real-model load, GPU use, simulator rollout, development FR
trace, candidate outcome, or manuscript edit.
