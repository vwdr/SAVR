# Phase 6R-B Research and Design Review

Status: COMPLETE

Review date: 2026-07-31

## Research question

What training-free refresh rule can address the observed SAVR 1.0 failures
without changing OpenVLA-OFT weights, action decoding, or the validated
projected-visual-feature cache boundary?

## Primary-source findings

### VLA-Cache

VLA-Cache demonstrates that temporal visual redundancy is real, but also shows
that raw visual similarity is not sufficient. Its NeurIPS 2025 paper reports
that naively reusing visually static tokens reduced success, and recovers much
of that loss by forcing task-relevant regions to remain fresh. It uses
patch-level comparisons rather than one global image average.

Design consequence for SAVR 2.0:

- replace two-camera global averaging with independent, local patch-change
  checks;
- treat either camera as capable of vetoing reuse;
- preserve full refresh for locally changing regions because SAVR's validated
  cache is all-or-nothing rather than token-selective.

Source: https://arxiv.org/abs/2502.02175, especially Sections 3.2-3.3.

The current official VLA-Cache repository still resolves to revision
`a4909880573868dee2769343d52e793c0341678b`, the same revision already audited
in Phase 5. Its evaluator validity problems therefore remain unresolved. No
official VLA-Cache GPU comparison is reinstated by this review.

Source: https://github.com/siyuhsu/vla-cache

### Adaptive Action Chunking

Adaptive Action Chunking reports that long fixed chunks reduce responsiveness
to new observations and that smaller chunks are favored near grasping and
manipulation phases. OpenVLA-OFT executes eight-action chunks, so one stale
visual reuse already influences a substantial open-loop interval.

Design consequence for SAVR 2.0:

- allow only one isolated reused query;
- require fresh-query recovery before another reuse;
- veto reuse around gripper transitions and unstable action context.

Source: https://arxiv.org/abs/2604.04161, especially Sections 3.2-3.3 and the
phase analysis in Section 5.1.4.

### AC2-VLA

AC2-VLA conditions computation on previous actions and multimodal context,
supporting the relevance of action context to adaptive VLA computation. Its
router requires action-guided self-distillation, so it is evidence for the
design motivation but not an implementation candidate for this training-free
project.

Design consequence for SAVR 2.0:

- retain action context;
- use deterministic grouped action and gripper-transition safeguards rather
  than a learned router.

Source: https://arxiv.org/abs/2601.19634.

### DeeR-VLA

DeeR-VLA shows that dynamic computation should be evaluated under explicit
average and peak resource constraints. It changes architecture and requires a
multi-exit model, so it is out of scope.

Design consequence for SAVR 2.0:

- keep an explicit online skip cap and report both average and worst-case
  refresh behavior;
- do not treat an average offline target as an online guarantee.

Source: https://arxiv.org/abs/2411.02359.

### OpenVLA-OFT

OpenVLA-OFT combines parallel continuous-action decoding with action chunking.
The pinned project checkpoint emits eight-action chunks, verified directly in
earlier SAVR phases. The redesign must preserve this action path exactly.

Source: https://arxiv.org/abs/2502.19645 and pinned OpenVLA-OFT revision
`e4287e94541f459edc4feabc4e181f537cd569a8`.

## Options considered

### Selected: safety-constrained full-feature reuse

- all-or-nothing projected visual-feature reuse;
- local per-camera change detection;
- grouped state/action checks;
- gripper transition veto;
- temporal stability, isolated reuse, and online budget enforcement.

This option directly addresses the Phase 6R-A findings while preserving the
validated integration and training-free claim.

### Rejected: learned routing

Learned routing may offer greater acceleration, but it changes the scientific
claim, requires training data/objectives, and introduces a new model component.

### Rejected: token-level KV caching

Token-level caching could save more downstream compute but would replace the
paper's cache boundary and substantially alter OpenVLA-OFT internals. The
official comparison remains technically invalid under the Phase 5 audit.

### Rejected: post-hoc task-specific thresholds

Task-specific tuning would exploit known calibration failures and weaken
generalization. SAVR 2.0 uses one configuration across all Spatial tasks.

## Design conclusion

SAVR 2.0 is a conservative safety controller, not merely a lower-threshold
version of SAVR 1.0. It changes the signal aggregation and adds temporal and
budget safeguards while leaving the policy, cache tensor, and action head
unchanged. Exact semantics and evaluation gates are frozen in
`docs/PHASE6R_PROTOCOL_V1.md`.
