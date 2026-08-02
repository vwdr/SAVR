# ACR Phase A0 Novelty Audit

**Audit date:** 2026-08-02

**Protocol:** `docs/ACR_EXECUTION_PROTOCOL_V1.md`

**Scope:** literature and available-code comparison only; no ACR implementation or outcome collection

## 1. Exit conclusion

The ACR novelty distinction **survives, narrowly and conditionally**.

No identified method contains the complete proposed contribution:

1. temporal reuse of one complete scene-camera projected-token block;
2. a freshly encoded and projected wrist-camera block on every policy query;
3. continued presence of both blocks in the original scene-first, wrist-second positions;
4. a deterministic, training-free, fail-closed controller using scene change,
   scene-relative end-effector translation, a fixed gripper-transition veto,
   and a finite horizon;
5. skipping scene encoder and projector work rather than only pruning tokens,
   attenuating views, or reusing language-model KV entries; and
6. direct closed-loop evaluation in the two-view, proprioceptive, eight-action
   OpenVLA-OFT LIBERO policy.

This is not a defensible claim to generic adaptive perception, multi-view
importance, temporal visual caching, sensor-rate decoupling, or state-aware
efficient VLA inference. Those broader ideas are established. Any eventual
paper must state the contribution at the camera-block and controller level.

The closest conceptual precedents are:

- **BFA++**, which learns manipulation-phase-dependent inter-view importance
  and token pruning, including a head/wrist asymmetry;
- **Selective Perception**, which learns a router that uses wrist observations
  and task context to attenuate other cameras and modalities;
- **DySta**, which learns static/dynamic token groups and recache gates for
  temporal reuse; and
- **DAM-VLA**, which maintains modality-specific latent buffers at different
  sensor rates.

They substantially narrow the positioning, but they do not establish the full
ACR construction above.

## 2. Frozen comparison dimensions

| Dimension | ACR Version 1 requirement |
|---|---|
| Reuse unit | entire projected scene-camera patch-token block |
| Temporal operation | reuse the last valid scene block across policy queries |
| Wrist semantics | encode and project the current wrist image on every query |
| Token layout | retain scene block, then wrist block, then current proprioception |
| Controller | deterministic, training-free, fail-closed |
| Signals | scene image, scene-relative EEF translation, fixed gripper transition, age/cap |
| Saved work | scene vision towers and scene projector invocation |
| Base policy | unmodified pinned two-view OpenVLA-OFT inference downstream of the boundary |

## 3. Full comparison matrix

Legend: **Yes** means the paper/code explicitly contains the property; **No**
means it does not; **Partial** means an adjacent but materially different
mechanism.

| Work | Temporal visual reuse | Complete camera block | Wrist always freshly encoded | Both views retained in original positions | Training-free deterministic controller | Skips camera encoder + projector | Relation to ACR |
|---|---:|---:|---:|---:|---:|---:|---|
| OpenVLA | No | No | N/A | N/A | N/A | No | Base VLA; no temporal cache |
| OpenVLA-OFT | No | No | Yes under FR | Yes | N/A | No | Exact frozen base system and factorization source |
| VLA-Cache | Yes | No; patch/token KV entries | No; both primary and wrist patches may be reused | Partial | Yes, heuristic | Primarily LLM KV/token computation, not a whole scene tower/projector path | Closest executable cache baseline; different granularity and wrist semantics |
| Learnable Adaptive Caching (LAC) | Yes | No; selected tokens | No stated camera asymmetry | Partial | No; selector and ratio predictor are trained | Token/KV reuse | Learned task-driven caching; not ACR |
| VLA-ADP | No temporal cache | No; token pruning | No | No; tokens are removed | Yes, training-free gate, but pruning is action-aware | Reduces downstream token work | Important state/action-aware pruning precedent |
| VLA-IAP | No cached camera block | No; token pruning | No | No; tokens are removed | Yes, training-free | Reduces downstream token work | Interaction-aligned visual compression, not temporal camera reuse |
| Selective Perception | No temporal reuse | No; learned feature weighting | Wrist is an ungated input but not an ACR freshness contract | Partial; low-weight features are attenuated | No; router is trained with VLM labels | Claims task-proportional computation, not cached scene encoder/projector reuse | Closest camera-routing precedent |
| Multi-camera-to-single-camera distillation | No | No | No | No; deploys a single-view student | No; policy is trained | Replaces multi-view deployment | Training/data method, not inference reuse |
| Multi-Camera View Scaling | No | No | No | Optional parallel per-view action aggregation | No; policies are trained | No temporal saving | Data/view scaling and action aggregation, not caching |
| BFA | No temporal reuse | No; view-weighted fusion | No | Partial | No; importance model/training | Token/fusion reduction | Direct inter-view importance precedent |
| BFA++ | No temporal reuse | No; hierarchical token pruning | No; wrist importance changes with phase | No; tokens are pruned | No; supervised predictors and post-training | Reduces retained tokens, not cached camera tower/projector blocks | Closest phase-dependent head/wrist compute allocation |
| DySta | Yes | No; learned static token groups | No camera-specific freshness contract | Partial | No; disentanglement and recache gates are learned | Reuses static-token KV cache | Closest learned temporal reuse; LIBERO acceleration uses the third-person-only OFT setting |
| FUTURE-VLA | Partial; compresses history | No | No | Multi-view history is compressed | No; trained architecture | Reduces fixed-budget history processing | Long-history compression, not per-query camera refresh |
| DAM-VLA | Yes, modality latent buffers | No camera-level scene/wrist factorization | No | Modalities update at native rates | No; trained gated architecture | Avoids synchronous modality recomputation | Closest asynchronous sensor-rate precedent |
| UniFS | Yes, layer groups at different rates | No | No | N/A | No; trained hierarchy | Reuses slow layer representations | Multi-frequency backbone, not camera-block caching |
| ActionCache | No visual reuse | No | No | N/A | Yes, training-free | Reuses intermediate actions | Different bottleneck and cache object |
| MVPruner / ST-Prune | Partial temporal/view compression | No | No | No; tokens are pruned | Mixed | Token/FLOP reduction | Autonomous-driving VLMs, outside closed-loop manipulation VLA setting |
| **SA-ACR (proposed)** | **Yes** | **Yes: scene** | **Yes** | **Yes** | **Yes** | **Yes: scene path** | Exact claimed conjunction |

## 4. Available-code audit

| Work | Official code status checked on 2026-08-02 | Audit result |
|---|---|---|
| OpenVLA | Public official repository | No temporal or camera-selective refresh in the base method |
| OpenVLA-OFT | Public official repository; pinned local revision `e4287e94541f459edc4feabc4e181f537cd569a8` | Two images are concatenated scene-first/wrist-second, split into independent six-channel image groups in the vision backbone, then concatenated as patch blocks |
| VLA-Cache | Public official repository; pinned local revision `a4909880573868dee2769343d52e793c0341678b` | Separately identifies stable primary and wrist patches, then combines both sets into reusable language-model cache indices; wrist tokens are eligible for reuse |
| LAC | Official GitHub link resolves to `JiahanFan/LAC` | Repository contained only a README at audit time; paper method was audited in full text |
| VLA-ADP | Public official repository `chen7086/VLA-ADP` | Action-aware visual-token pruning, not temporal camera-block caching |
| BFA++, Selective Perception, DySta, VLA-IAP | No implementation link was present in the audited paper/project material | Full paper method audited; absence of a link is not treated as proof that code does not exist elsewhere |
| Multi-Camera View Scaling | Official project page available | Page describes training-time pseudo-demonstrations and optional action aggregation; no temporal caching mechanism |

## 5. Primary-source evidence supporting distinctions

- OpenVLA paper: https://arxiv.org/abs/2406.09246
- OpenVLA-OFT paper: https://arxiv.org/abs/2502.19645
- OpenVLA-OFT official code: https://github.com/moojink/openvla-oft
- VLA-Cache paper: https://arxiv.org/abs/2502.02175
- VLA-Cache official code: https://github.com/siyuhsu/vla-cache
- LAC paper: https://arxiv.org/abs/2602.00686
- LAC official repository: https://github.com/JiahanFan/LAC
- VLA-ADP: https://arxiv.org/abs/2509.22093
- VLA-ADP official code: https://github.com/chen7086/VLA-ADP
- VLA-IAP: https://arxiv.org/abs/2603.22991
- Selective Perception: https://arxiv.org/abs/2602.15543
- Multi-camera distillation: https://arxiv.org/abs/2303.07026
- Multi-Camera View Scaling: https://arxiv.org/abs/2604.00557
- BFA: https://arxiv.org/abs/2502.11161
- BFA++: https://arxiv.org/abs/2602.20566
- DySta: https://arxiv.org/abs/2602.03983
- FUTURE-VLA: https://arxiv.org/abs/2602.15882
- DAM-VLA: https://arxiv.org/abs/2606.12105
- UniFS: https://arxiv.org/abs/2606.22794
- ActionCache: https://arxiv.org/abs/2607.06370
- MVPruner: https://arxiv.org/abs/2606.27660
- ST-Prune: https://arxiv.org/abs/2604.19145

## 6. Search log

**Databases/surfaces:** arXiv web index/full-text HTML and official GitHub or
official project pages linked by papers. Secondary curated lists were used only
to discover candidates; all inclusion decisions use primary papers or official
code.

**Execution date:** 2026-08-02.

Exact query families repeated during Phase A0:

```text
site:arxiv.org "multi-view" "vision-language-action" camera routing caching
site:arxiv.org camera "vision-language-action" cache view selection robot
site:arxiv.org asymmetric camera refresh VLA robot
site:arxiv.org temporal multi-camera visual feature reuse robot policy
site:arxiv.org/abs/260* VLA efficient multi-view camera selective perception robot
site:arxiv.org/abs/25* VLA multi-camera adaptive computation view
site:arxiv.org vision-language-action multi-camera view pruning routing inference
site:arxiv.org robot policy camera feature caching wrist scene temporal
site:arxiv.org "camera" "cache" "vision-language-action" robot
site:arxiv.org "multi-view" VLA "token pruning" camera
site:arxiv.org "wrist camera" "scene camera" VLA efficiency
site:arxiv.org "asymmetric" camera refresh robot policy
site:arxiv.org "Efficient Long-Horizon Vision-Language-Action Models via Static-Dynamic Disentanglement"
site:arxiv.org "FUTURE-VLA" temporal adaptive compression multi-view
site:arxiv.org "camera-level" caching VLA multi-view
site:arxiv.org VLA "view-level" caching
site:arxiv.org/abs/2607 "efficient" "vision-language-action" camera
site:arxiv.org/abs/2606 "vision-language-action" token cache robot
site:arxiv.org/abs/2605 VLA temporal cache visual camera
site:arxiv.org/abs/2604 VLA multi-view efficient camera routing
site:arxiv.org "DAM-VLA" Decoupled Asynchronous Multimodal
site:arxiv.org "per-modality latent buffers" VLA
site:arxiv.org VLA asynchronous camera views refresh scene wrist
site:arxiv.org VLA per-camera update frequency wrist scene
site:arxiv.org robot policy multi-camera asynchronous inference cache
site:arxiv.org "wrist" "cache" "OpenVLA-OFT"
```

Code-discovery queries included the exact paper titles plus `GitHub`, and
official repository/project links embedded in the papers were followed.

## 7. Inclusion and exclusion decisions

Included works satisfy at least one of: VLA temporal reuse, multi-view camera
routing, token pruning/caching, asynchronous modality processing, state/action
aware compute allocation, or the exact OpenVLA-OFT base system.

Excluded from the main matrix:

- generic image/video token pruning with no robotics or VLA evaluation;
- multi-camera perception for autonomous driving unless it added a directly
  relevant temporal/view-compute mechanism (MVPruner and ST-Prune are retained
  only as boundary cases);
- action-decoding acceleration with no visual or sensor reuse, except the very
  recent ActionCache boundary case;
- memory methods whose purpose is task memory rather than computation reuse;
- robustness or data-augmentation methods without adaptive visual computation.

## 8. Required paper positioning

If ACR eventually passes all experimental gates, use wording equivalent to:

> We study deterministic camera-block temporal reuse for a two-view,
> chunked OpenVLA-OFT policy: the fixed scene-camera projected-token block may
> be reused, while the wrist-camera block and proprioception are always current.

Do not use wording equivalent to:

- “the first adaptive multi-view VLA”;
- “the first state-aware efficient VLA”;
- “the first temporally cached VLA”;
- “the first asynchronous multimodal VLA”; or
- “the first method to recognize phase-dependent wrist-camera importance.”

## 9. Phase A0 novelty gate

**Decision: PASS, with narrowed claim.**

The distinction is sufficiently concrete to justify implementation and
correctness testing. It must be rechecked immediately before manuscript
submission because the area is moving rapidly. This decision authorizes no
performance claim and provides no evidence that ACR will succeed.
