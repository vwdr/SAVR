# ACR Version 5 Research Audit: Correct Isolated Reuse

Status: **COMPLETE BEFORE V5 CPU IMPLEMENTATION**

Date: 2026-08-10

## Research question

How should State-Aware Visual Refresh guarantee that whole-scene camera
features are never reused on two consecutive completed queries, without
abandoning training-free, state-aware, asymmetric camera refresh?

## Exact project finding

ACR Version 1 interprets the horizon with this condition:

```text
refresh when cache_available and cache_age >= horizon
```

The cache age is zero immediately after a refresh and increments after each
reuse. Consequently, horizon 2 permits reuse at ages 0 and 1, then refreshes at
age 2. It therefore permits a maximum reuse streak of two. V4-A's prose called
horizon 2 “no consecutive reuse,” but its separately frozen gate required a
maximum streak of one. The gate correctly stopped all six candidates.

The repository already contains a safer precedent in SAVR 2.0: an explicit
`maximum_consecutive_reuses=1` rule plus a fresh-query stability requirement.
That implementation is separate from ACR and cannot silently repair the
frozen V3/V4 controller, but it confirms that explicit state-machine semantics
are practical in this codebase.

## Primary-source findings

| Primary source | Relevant finding | Boundary for this project |
|---|---|---|
| [VLA-Cache](https://arxiv.org/abs/2502.02175) | Adjacent robotic observations contain exploitable temporal redundancy, but visually static task-relevant regions must still be recomputed; naive static reuse substantially reduced success in its ablation. | Token-level decoder caching differs from ACR's whole scene-camera block. It motivates fail-closed semantic safeguards but does not validate our latch. |
| [FlashVLA](https://arxiv.org/abs/2505.21200) | Training-free reuse benefits from combining visual-token stability and action consistency; removing either signal degraded the reported efficiency/success trade-off. | It reuses actions and prunes tokens, whereas SAVR keeps the current action head and wrist path. Its thresholds cannot be transferred. |
| [AC2-VLA](https://arxiv.org/abs/2601.19634) | Adaptive VLA computation should include previous action context rather than rely only on visual similarity. | AC2-VLA trains an adaptive policy and changes more computation dimensions, so it is evidence for the design principle, not a compatible baseline or implementation. |
| [VLA-Corrector](https://arxiv.org/abs/2607.01804) | Long open-loop intervals reduce reactivity; event-triggered corrective replanning shortens the horizon when observed dynamics deviate. | It addresses action-chunk execution with a learned monitor, not visual-feature reuse. It supports limiting stale intervals, not a claim that one-step reuse is sufficient. |
| [VLASH](https://arxiv.org/abs/2512.01031) | Temporal misalignment between observation and execution can destabilize actions in asynchronous VLA inference. | SAVR remains synchronous; the relevance is the general need to bound staleness. |
| [Event-triggered control with designable minimum inter-event time](https://arxiv.org/abs/2002.00058) | Explicit temporal separation can be encoded as controller state rather than inferred indirectly from a threshold. | Its stability theorems do not apply to OpenVLA-OFT or establish SAVR safety. This is an architectural analogy only. |
| [ActionCache](https://arxiv.org/abs/2607.06370) | A different training-free cache can use multimodal keys to reuse intermediate actions in flow-based VLAs. | It targets flow-matching action heads, not OpenVLA-OFT scene features, and does not justify cross-episode or action reuse here. |
| [Reducing Temporal Redundancy for Efficient VLA Inference](https://arxiv.org/abs/2607.12287) | Incremental visual updates can exploit dynamic versus static regions. | It combines perception changes with trained diffusion-step compression and is outside the frozen training-free camera-block scope. |

## Synthesis

The literature consistently supports three principles relevant to the
correction:

1. visual redundancy exists, but visual similarity alone is insufficient;
2. action/state context and current task-relevant perception matter; and
3. stale/open-loop intervals should be explicitly bounded rather than assumed
   from a loosely named horizon.

No reviewed primary source establishes that exactly one scene-feature reuse is
safe for OpenVLA-OFT on LIBERO. The one-reuse limit is therefore a conservative
project hypothesis derived from prior negative development evidence, not a
published guarantee.

## Corrected mechanism

The corrected controller is **Isolated-Reuse State-Aware ACR** (`IR-SA-ACR`).
It preserves the V3 asymmetric computation boundary: the scene camera may be
cached, while the wrist camera, current proprioception, and downstream action
head remain fresh on every query.

IR-SA-ACR adds a controller-owned latch:

1. a completed reuse sets `refresh_required_after_reuse=true`;
2. while set, every decision is forced to refresh with an explicit
   `post-reuse-refresh` reason;
3. only a successfully completed refresh clears the latch;
4. a failed query does not clear it;
5. `horizon=1` remains mandatory as defense in depth; and
6. cache age must agree with the latch (`0` after refresh, `1` after reuse), or
   the controller forces refresh with `isolation-state-mismatch`.

This creates the auditable temporal language `R* (U R)*`: after warm-up, a
reuse `U` can occur only between refreshes `R`. No execution trace accepted by
the controller can contain `U U`.

## Rejected corrections

- Merely changing the V4 configuration from horizon 2 to horizon 1: too easy
  to misdocument or bypass through inconsistent external cache age.
- Relaxing the V4 maximum-streak gate to two: post-hoc and contrary to the
  safety intent.
- Adding the direction-reversal veto as the sole correction: V4-A showed that
  the frozen variants still produced streaks of two and missed efficiency
  targets.
- Reusing prior V4 thresholds as a selected V5 method: would convert observed
  development outputs into an undeclared selection rule.
- Reusing actions, pruning tokens, training a monitor, or changing weights:
  materially expands the method beyond the requested correction.

## Claim boundary

The research supports implementing and CPU-verifying the latch. It does not
support a task-success, reuse-rate, latency, CUDA-Graph, or positive-paper
claim. Threshold selection and all model/simulator measurements require a new
output-blind protocol after this correction checkpoint.
