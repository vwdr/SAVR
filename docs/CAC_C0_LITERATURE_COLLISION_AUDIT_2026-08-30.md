# CAC C0 Literature-Collision and Comparator Audit

**Search date:** 2026-08-30  
**Scope:** primary paper/project records available through the search date  
**Decision:** **NO DIRECT COLLISION FOUND; NARROW CAC NOVELTY SURVIVES**

## Question audited

The audit asked whether prior work already combines all of the following on a
materially identical problem:

1. correction after same-query, selective layer/tile, recursively mixed-age
   visual K/V reuse;
2. exact physical source provenance for every reused group;
3. direct prediction of the dense action effect rather than a reuse decision or
   full K/V reconstruction;
4. current visual tokens already computed by the accelerated path; and
5. one controlled demonstration-to-on-policy dense-teacher aggregation round.

Residual action correction, frozen-backbone adaptation, learned cache routing,
and latent-delta prediction are each established ideas. CAC's possible novelty
is only their cache-specific combination and evaluation above. The project must
not claim that residual correction itself is new.

## Primary-source comparison

| Work | Mechanism and target | Relationship to CAC | Collision? |
|---|---|---|---|
| [OpenVLA](https://proceedings.mlr.press/v270/kim25c.html) / [OpenVLA-OFT](https://arxiv.org/abs/2502.19645) | Open VLA backbone; OFT adds continuous action decoding and efficiency-oriented fine-tuning. | Frozen base policy and action representation used by CAC. Neither repairs selective stale K/V. | No |
| [LIBERO](https://proceedings.neurips.cc/paper_files/paper/2023/hash/8c3c666820ea055a77726d66fc7d447f-Abstract-Datasets_and_Benchmarks.html) | Lifelong robot-learning benchmark and task suites. | Evaluation substrate, not a cache method. | No |
| [VLA-Cache](https://arxiv.org/abs/2502.02175) | Selects visually stable tokens and reuses their K/V state across control queries. | Direct substrate and essential baseline. It selects reuse but does not correct the resulting action using exact recursive provenance. | No |
| [LAC](https://arxiv.org/abs/2602.00686) | Learns layer-adaptive cache behavior for VLA acceleration. | Learned compression/allocation rather than post-cache dense-action-effect correction. | No |
| [AC2-VLA](https://arxiv.org/abs/2601.19634) | Joint action-context routing of cognition reuse, token pruning, and component execution, with self-distillation. | Action-aware learned acceleration, but not a corrector for one fixed mixed-age K/V substrate. | No |
| [Action-JND](https://arxiv.org/abs/2608.21247) | Learns action-conditioned tolerance scores for compression, including stale-K/V reuse. | Highly relevant evidence that action-space supervision improves cache decisions. It chooses what may be compressed; it does not repair an already corrupted action from exact source provenance. | No |
| [Gated VLA-Cache](https://arxiv.org/abs/2608.10824) | Uses action-token confidence to invalidate unsafe cache reuse and refresh. | Relevant reliability comparator. It falls back to refresh rather than correcting the cached action. | No |
| [Latent Bridge](https://arxiv.org/abs/2605.02739) | Predicts feature or K/V deltas between slow-backbone calls in dual-system VLAs; includes a DAgger pipeline. | Closest on learned latent repair and on-policy aggregation. It predicts a replacement latent/K/V state under periodic slow calls, not the dense action effect after same-query layer/tile mixed-age reuse. | No, but closest latent-space neighbor |
| [CloudEdgeVLA](https://arxiv.org/abs/2608.00569) | Combines stale high-level/cloud information with fresh edge-side information. | Closest on fresh/stale fusion, but uses a cloud-edge architecture rather than exact mixed-age provenance and direct dense-action residuals. | No |
| [A2C2 / Leave No Observation Behind](https://arxiv.org/abs/2509.23224) | A lightweight head uses the latest observation and base action to correct stale actions within an executed action chunk. | Closest on the direct residual-action target. Its error source is within-chunk observation staleness; it does not consume or repair selectively mixed-age K/V. | No, but closest action-space neighbor |
| [Action ControlNet](https://arxiv.org/abs/2606.25985) | Adds learned action-conditioning/control to a frozen VLA-style policy. | Supports the feasibility of compact frozen-backbone action adaptation, but is not cache-error correction. | No |
| [ActionCache](https://arxiv.org/abs/2607.06370) | Retrieves intermediate actions to warm-start flow-based VLA generation. | Caches action-generation state on different backbones; it does not correct OpenVLA-OFT visual K/V reuse. | No |
| [ViTaR](https://arxiv.org/abs/2608.15816) | Efficient VLA adaptation/routing method. | Relevant efficient adaptation context, not mixed-age K/V action correction. | No |
| [FiberTune](https://arxiv.org/abs/2606.08653) | Parameter-efficient VLA adaptation. | Relevant frozen/mostly frozen adaptation context, not cache repair. | No |
| [VLA-Corrector](https://arxiv.org/abs/2607.01804) | Detects visual-evolution deviations, truncates stale action chunks, and triggers guided replanning. | A learned corrective wrapper, but it detects and replans for open-loop chunk drift rather than correcting the action output of selective K/V reuse. | No |
| [OxyGen](https://arxiv.org/abs/2603.14371) | Manages and shares VLA K/V caches across tasks and time for parallel serving. | Systems-level K/V management; not action-level repair of stale visual state. | No |

## Compatibility and empirical-comparator decision

- **VLA-Cache:** official code is available and already forms the authenticated
  OpenVLA-OFT/LIBERO substrate in this repository.
- **AC2-VLA:** [official code](https://github.com/SunnyYWD/AC-2-VLA) is public,
  but it targets CogACT/Prismatic, Bridge/OXE training, SIMPLER evaluation, and
  recommends multi-GPU training. It is not a like-for-like one-GPU
  OpenVLA-OFT/LIBERO comparator under the frozen project boundary.
- **Latent Bridge:** [official code](https://github.com/1999Lyd/Latent-Bridge)
  is public, but its supported bridge paths are GR00T-N1.6 and pi0.5 and require
  a different dual-system model/training pipeline.
- **Gated VLA-Cache, Action-JND, and LAC:** no official implementation was
  located that was both runnable on the pinned OpenVLA-OFT checkpoint and
  compatible with the one-GPU/no-new-large-download boundary as of the search
  date.

Therefore C0 does not authorize an unreliable reimplementation. CAC will report
these methods conceptually and cite their published results, but will make no
empirical-superiority claim over an unreproduced method. Dense OpenVLA-OFT,
uncorrected D62, and the frozen matched learned controls remain the executable
comparators.

## Novelty boundary after audit

No located source contains all five audited elements. The surviving research
question is consequently narrow but organic:

> Can a compact learned action-space corrector, conditioned on current visual
> evidence and exact recursive mixed-age cache provenance, recover the
> reliability lost by a fixed selective K/V reuse policy without reconstructing
> fresh K/V or issuing a hidden dense call?

The result would still be scientifically useful if negative, but C0 does not
assert that the method will work. The protocol's headroom, predictability,
physical-timing, development, and locked-confirmation gates are required before
any positive claim.

## Search limitations and watch rule

The field is moving rapidly and several records above are recent 2026
preprints. Search/index latency, unlinked code releases, or a later revision may
change the collision assessment. Repeat this audit before manuscript submission
and before any novelty claim. A later source combining all five elements stops
the claim for advisor review; it does not justify relabeling CAC after results.
