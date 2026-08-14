# SAVR Citation Relevance Audit

Date: 2026-08-13

## Purpose

This audit applies a claim-level standard: a source remains in the manuscript only when it directly supports a statement that the paper needs. The bibliography is not padded to reach a target count. Sources are not cited merely because they are prominent or adjacent to the topic.

## Retained sources and their exact roles

| Key | Source | Manuscript role | Why it is necessary |
|---|---|---|---|
| `rt1` | Brohan et al., RT-1 | Establishes scalable language-conditioned robot control in the VLA lineage. | Direct foundation for the policy class discussed in the introduction and related work. |
| `rt2` | Zitkovich et al., RT-2 | Supports the claim that vision-language knowledge was incorporated into robotic action generation. | Direct foundation for the VLA lineage. |
| `openvla` | Kim et al., OpenVLA | Supports the OpenVLA architecture, scale, and open-model lineage. | The evaluated OpenVLA-OFT checkpoint derives from this model. |
| `openvlaoft` | Kim, Finn, and Liang, OpenVLA-OFT | Supports the continuous-action, parallel-decoding, action-chunked policy stack used in the experiments. | This is the evaluated implementation family. |
| `efficientvla` | Yang et al., EfficientVLA | Supports the description of task-aware visual-token selection and other model-path VLA acceleration. | Direct comparison showing that efficiency can be pursued at a finer granularity than whole-prefix reuse. |
| `vlacache` | Xu et al., VLA-Cache | Supports adjacent-input, token-selective VLA visual caching. | Closest published point of comparison to SAVR. |
| `depthcache` | Li et al., DepthCache | Supports depth-guided and motion-adaptive visual-token compression. | Directly relevant training-free visual-computation method. |
| `learnedcache` | Wei et al., adaptive visual-token caching | Supports learned cached-token selection and reuse-ratio prediction. | Directly relevant learned alternative to SAVR's hand-designed controller. |
| `gatedcache` | Wu, Kawaharazuka, and Okada, neural introspection gating | Supports model-confidence gating for VLA cache reuse. | Directly relevant recent alternative; cited only as an August 2026 preprint. |
| `actioncache` | Oi et al., ActionCache | Supports caching in iterative action generation rather than at the complete visual prefix. | Direct comparison of a distinct cache boundary. |
| `dagger` | Ross, Gordon, and Bagnell, DAgger | Supports the closed-loop distribution-shift and compounding-error argument. | Section 2.3 makes this methodological claim and therefore needs prior-work support. |
| `libero` | Liu et al., LIBERO | Defines the benchmark and LIBERO-Spatial suite used in evaluation. | Primary benchmark citation. |
| `clopperpearson` | Clopper and Pearson | Defines the exact binomial confidence intervals reported for terminal success. | Statistical-method citation for Tables 3 and 4. |

## Removed sources

| Source | Reason for removal |
|---|---|
| Open X-Embodiment | Broad dataset/ecosystem context was not needed to explain the tested policy or result. |
| Octo | A broad VLA survey-style comparison was not needed for the paper's bounded whole-prefix-caching claim. |
| $\pi_0$ | The manuscript does not evaluate or make a claim that depends on this model. |
| TinyVLA | General compact-model context did not support a necessary experimental or methodological claim. |
| SmolVLA | General compact-model context did not support a necessary experimental or methodological claim. |
| DynamicViT | Generic token-pruning precedent was more remote than the retained VLA-specific efficiency sources. |
| Token Merging | Generic vision-token merging precedent was unnecessary once direct VLA-specific methods were cited. |

## Recent-preprint control

`gatedcache` was revalidated on 2026-08-13 as arXiv:2608.10824v1, posted 2026-08-11, by Zhijie Wu, Kento Kawaharazuka, and Kei Okada. It remains explicitly labeled a preprint and is used only to support the existence of action-token-confidence gating. Its bibliographic identity and publication status must be checked again immediately before submission.

## Submission rule

Before submission, every citation must still satisfy all three checks:

1. the cited source directly supports the sentence containing it;
2. the bibliographic identity and publication status match an official record; and
3. removing the source would leave a real claim unsupported, not merely reduce the reference count.

