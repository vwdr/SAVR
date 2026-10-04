# Backend-preserving pruning: contribution audit

Date: 2026-09-26. Research and read-only code/evidence inspection; no GPU run.
User authorized pursuing the proposed novelty-check → comparison → confirmation
route. This report records the outcome of its FIRST gate, before spending more
compute or drafting unsupported method claims.

## Decision

**The proposed general method claim does not pass the novelty check.** Computing
pruning scores separately while preserving an optimized attention output path
is established. More importantly, VLA-Pruner v1 explicitly points to SparseVLM's
FlashAttention adaptation in its implementation details. Our native-SDPA port
cannot therefore be called a newly invented backend-preserving pruning method.

This does not prove no VLA-specific empirical study could be original. It means
we have not identified or validated such a contribution, and an additional
speed comparison alone would not establish one. Do not automatically launch
the proposed large confirmation or write a positive-method manuscript.

## Sources actually inspected

1. **VLA-Pruner v1**, implementation details §4(c), paragraph preceding Table 1:
   explicitly proposes applying the FlashAttention adaptation from SparseVLM.
   [Original paper version](https://arxiv.org/html/2511.16449v1).
   Its revised v5 was also inspected; a literal FlashAttention search did not
   find that wording in v5. Absence from a later version does not erase the
   earlier disclosure. [Revised version](https://arxiv.org/html/2511.16449v5).

2. **SparseVLM**, FlashAttention compatibility appendix: describes a first pass
   preserving the normal fast-attention hidden-state computation, plus an
   auxiliary operation to derive attention statistics for pruning.
   [Paper v4](https://arxiv.org/html/2410.04417v4),
   [ICML 2025 publication record](https://proceedings.mlr.press/v267/zhang25s.html).
   The OpenReview PDF request redirected to a verification page and the PMLR
   PDF request failed; the arXiv full text was successfully read instead.

3. **Balanced Token Pruning**, NeurIPS 2025, implementation details on PDF page8:
   separately computes required scores at designated pruning layers while
   retaining FlashAttention compatibility. Direct prior art for the general
   separation of score extraction and optimized attention.
   [Published paper](https://papers.neurips.cc/paper_files/paper/2025/file/5aab3631d0d3131281fb88265db69480-Paper-Conference.pdf).

4. **ZipCache**, NeurIPS 2024, §4.3/Figure4: extracts saliency using selected
   probe-token attention while retaining fast attention for the remaining
   computation. Contextual precedent, not the same VLA method or exact arithmetic.
   [Published paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/7e57131fdeb815764434b65162c88895-Paper-Conference.pdf).

5. **ETA-VLA**, preprint, §III-C.2: uses eager attention only at selected sparse
   layers, retaining SDPA/FlashAttention elsewhere. This is related VLA-family
   execution engineering for autonomous driving, not identical to preserving
   SDPA at every layer. Not treated as a matched OpenVLA-OFT/LIBERO baseline.
   [Preprint](https://arxiv.org/html/2603.25766).

6. **Pinned VLA-Pruner source**, commit
   `84d4b7192c77abf1585610e2f12393319b7ebff9`: `modeling_llama.py` has
   `LlamaSdpaAttention.forward` eager fallback for requested attention weights
   (lines561–576 in the fetched source), and the pruning forward forces
   `output_attentions=True` (line1162). This confirms a difference between this
   released path and our adaptation, not novelty of the adaptation.
   [Pinned source](https://raw.githubusercontent.com/MINT-SJTU/VLA-Pruner/84d4b7192c77abf1585610e2f12393319b7ebff9/src/openvla-oft/transformers/src/transformers/models/llama/modeling_llama.py).
   The earlier temporary local upstream-source directory no longer exists.
   This audit read the pinned raw source online, without installing it or
   claiming to rerun the historical upstream parity tests.

## What our implementation actually does

Inspected `src/savr/openvla/adaptive_sdpa.py` and `specprune.py`'s
`NativeAttentionScores`. At selection/history layers3/15, our code separately
forms QK-transpose scores, softmaxes in float32, casts to query dtype, and
returns the original SDPA result for the forward output. Other layers retain
native SDPA. The existing VLA-Pruner-derived selector, history and budgets stay
in use; this is not a newly designed selection rule.

In notation, main output is O = SDPA(Q,K,V), with auxiliary
A = softmax(QK^T/sqrt(d)) used for selection. Separating these two computations
explains why querying scores need not force the main output onto an eager
implementation. Prior art already describes this general design. Our exact
BF16 rounding, bidirectional OFT layout, positional handling and regression
tests are implementation details, not by themselves a new scientific claim.

The helper currently materializes a full auxiliary score matrix at two layers.
Do not claim a new memory-linear score algorithm, a novel fused kernel, universal
numerical equivalence, or direct improvement over every published implementation.

## What the measurements establish and do not establish

- The 64-call diagnostic showed adaptive-zero and reference-eager matching on
  16 inputs, with both differing from native SDPA. Thus the discrepancy was
  reproduced by the backend switch alone on these inputs. It is not evidence
  that the released eager implementation is mathematically incorrect or that
  its robot success is worse. Exact bitwise equality was our qualification
  requirement, not a universal validity rule for floating-point implementations.
- Native-preserving qualification passed 176 calls and16checks with exact
  zero-pruning actions. This establishes the tested integration property.
- Four-arm development screen: dense39/40; fixed38439/40; adaptive38438/40;
  adaptive25639/40. Adaptive256 controlled mean query time was15.58% below dense.
  That speedup includes token removal. It does NOT isolate the benefit of the
  backend adaptation versus a pruning-enabled eager implementation.
- Independent verifier was rerun in this audit and passed. All immutable
  evidence remains unchanged. No inference about unmeasured eager closed-loop
  success, unseen conditions or population equivalence is warranted.

## Why a new comparison is not automatically the next step

A matched comparison could estimate an engineering benefit of this local port.
It cannot transform an already disclosed adaptation into a new algorithm. For
a VLA-specific empirical contribution, first state the previously unanswered
question and why resolving it matters. One plausible question is whether backend
changes confound closed-loop compression evaluations, but the current evidence
only demonstrates action differences, not a causal robot-performance effect.
This question is an unvalidated research option, not a replacement promise of
positive results. Absence from this targeted literature search is not proof
of novelty or a high probability of a favorable result.

If that narrower empirical study is deliberately selected later, it requires
at least a backend × pruning comparison: native dense, eager dense, native
pruned and eager pruned, same checkpoint/inputs/precision/hardware and full
timing. Include a strong selective-score implementation, not only an all-layer
eager baseline, to avoid an artificially weak comparator. Record selector
indices: changing backend can also change later token ranks. Distinguish
fixed-input numerical/timing effects from closed-loop trajectory effects.
Do not call each differing command a failure or a safety violation. Freeze
the claim, meaningful effect/precision, exposure-audited population and resource
budget before opening new outcomes. This is a DESIGN REQUIREMENT, not a frozen
protocol or experiment dispatch.

## Current stop and required research decision

The generic backend-preserving-method branch is closed as a novelty claim.
No new comparator run, confirmation, training, server access, manuscript,
email or GitHub push was performed in this audit. The existing engineering fix
and positive compression measurements are retained, not discarded.

Before more GPU work, decide whether to pursue a clearly scoped empirical
systems contribution with the advisor, or undertake genuine new method
development with additional time. Neither path is a completed positive-results
paper today. Do not silently relabel replication as novelty or keep inventing
replacement methods whenever an audit rules out the previous claim.

Correction to earlier assistant guidance: the backend-preserving idea should
have been checked against these sources before being presented as the leading
paper route. The audit is a reason to stop this claim, not to weaken its standard.
