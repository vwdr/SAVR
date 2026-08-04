# ACR Version 2 Diagnosis Report

**Date:** 2026-08-03

**Scope:** Existing A4/A5 evidence and primary-source research only

**New GPU queries or rollout episodes:** 0

## Decision summary

Pursue a new **State-Aware Dual-Path Asymmetric Camera Refresh**
(`SA-DP-ACR`) route. Preserve the conservative Version 1 controller exactly,
but replace its execution hot path:

- a scene-refresh query uses the original upstream two-view vision/projector
  path once, then splits and caches the projected scene block;
- a scene-reuse query executes only the fresh wrist vision/projector path and
  concatenates it with the cached scene block;
- redundant per-camera host synchronizations are removed from production
  timing while correctness remains fail-stop and fully audited.

This redesign targets a measured implementation bottleneck. It does not add a
post-hoc signal veto or claim that the single conservative failure has a known
cause.

## Reconciled A5 evidence

The machine-readable diagnosis is the immutable TITAN record
`results/acr-v2-diagnosis-v01/record.json`. Its byte SHA-256 is
`cf8ea201547030026a204a8d4848ac7d33d3d45d64246a94cdab1d47f3739deb`
and semantic SHA-256 is
`52b7cc20b7b934c597ee43fa529ad4f3635f85fe2a87569b50df3cc5dc1f809c`.

### Safety/compute frontier

| Policy | Success | Scene reuse | Visual CUDA change vs FR | Query-wall change vs FR |
|---|---:|---:|---:|---:|
| Upstream FR | 30/30 | 0% | reference | reference |
| `acr-t25-h2-b30` | 29/30 | 26.06% | **11.94% reduction** | **31.24% slower** |
| `acr-t50-h4-b55` | 24/30 | 47.40% | 22.57% reduction | 30.05% slower |
| `acr-t70-h8-b75` | 23/30 | 49.44% | 24.08% reduction | 29.98% slower |

The exact FR Stage 1 point values were 1,238.629 ms/query wall and 175.511
ms/query visual CUDA over 514 steady queries. The conservative candidate used
1,625.611 ms/query wall and 154.552 ms/query visual CUDA. Therefore Version 1
demonstrated real visual compute savings but not realized end-to-end
acceleration.

### Conservative failure localization

The conservative failure was Object **task 6/state 0** (pick up the butter and
place it in the basket), not state 2. Upstream FR and both more aggressive ACR
candidates succeeded on that exact task/state. The first three conservative
action hashes matched FR; the first mismatch occurred at query 3, exactly the
first scene reuse.

That first reuse had a logged translation-direction reversal. This is not a
unique failure signature:

- 12 successful conservative episodes also had a direction reversal at their
  first reuse;
- 10 successful conservative episodes had the exact combination of first
  reuse at query 3 plus a direction reversal;
- the two more aggressive policies succeeded on task 6/state 0 despite more
  scene reuse.

Adding a direction-reversal veto would therefore be an outcome-driven patch
without causal support and would eliminate many successful reuses. Version 2
does not add it.

### Aggressive-candidate consistency

Six of the middle candidate's six failures were also failures for the
aggressive candidate: task/state pairs `(3,2)`, `(4,1)`, `(5,1)`, `(7,1)`,
`(7,2)`, and `(9,1)`. The aggressive candidate added one failure at `(5,2)`.
This stable overlap supports the broad conclusion that the high-reuse region
is unsafe, while the single conservative miss is insufficient to identify a
new signal rule.

## Root-cause boundary

The diagnosis establishes two facts and one non-fact:

1. **Fact:** ACR Version 1 saves camera-specific visual CUDA work.
2. **Fact:** its execution path adds enough wall overhead to erase those
   savings.
3. **Not established:** any one logged signal caused the conservative
   scientific failure.

Source inspection explains a plausible latency mechanism. Upstream FR calls
the normal two-view vision boundary and one projector over the combined patch
sequence. Version 1 manually invokes camera components, projects camera blocks
separately, performs repeated structural/finite checks, and reconstructs the
combined sequence. Even though CUDA visual work falls on reuse queries, this
fragmented hot path introduces CPU/GPU dispatch and synchronization gaps.

Version 2 must prove the mechanism with a bounded paired microbenchmark. The
source explanation is a hypothesis until that gate passes.

## Research reconciliation

Primary sources were rechecked on 2026-08-03:

- [Selective Perception for Robot](https://arxiv.org/abs/2602.15543) keeps the
  wrist camera permanently active and uses it as the anchor for routing other
  views. This independently supports the asymmetric information hierarchy,
  but its trained router and feature attenuation differ from temporal
  camera-block reuse.
- [VLA-Cache](https://arxiv.org/abs/2502.02175) reports that moderate adaptive
  token reuse can preserve OpenVLA-OFT success and lower CUDA latency. It also
  shows that static visual similarity alone can reduce success. Its
  attention/layer KV caching differs from encoder/projector camera-block
  caching.
- [Learning to Accelerate VLAs through Adaptive Visual Token
  Caching](https://arxiv.org/abs/2602.00686) explicitly reports that methods
  with lower theoretical FLOPs can become slower when selection overhead is
  high. This directly supports a mandatory realized-latency gate.
- [Static-Dynamic Disentanglement for Efficient Multi-Frame
  VLAs](https://arxiv.org/abs/2602.03983) learns persistent token groups and a
  recache gate. Its trained multi-frame architecture differs from this
  training-free, fixed-camera block cache, but reinforces the need to separate
  static and dynamic visual information.

The Version 2 novelty boundary remains narrow: training-free temporal reuse of
one complete fixed-camera encoder/projector block, an always-fresh wrist block,
unchanged token order and policy weights, and a dual physical execution path
that preserves the upstream refresh path.

## Why the conservative controller is retained

`acr-t25-h2-b30` is the only defensible parent:

- it had the highest A5 success (29/30);
- it still delivered 26.06% scene reuse and 11.94% visual CUDA reduction;
- the two higher-reuse settings produced six and seven failures;
- no logged feature supports replacing or locally tuning its thresholds.

This choice is explicitly development-informed. A5 cannot support a Version 2
claim. All Version 2 evidence begins with correctness/latency microbenchmarks
and then the unopened ACR outcomes for Object states `3-9`.

## Falsification conditions

Stop the positive-paper route if any of these occurs:

- the dual refresh path is not bitwise identical to upstream FR;
- the reuse path is not bitwise identical to Version 1 for the same tensors,
  cache, and state;
- the optimized reuse path does not beat upstream FR wall latency by at least
  2% in the paired microbenchmark;
- fresh development loses more than two successes versus paired FR;
- realized scene reuse is below 20%, visual CUDA reduction below 10%, or
  query-wall reduction below 2%;
- any cache, component, provenance, or technical invariant fails.

These rules prioritize a credible positive result over merely producing a
positive-looking number.
