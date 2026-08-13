# SAVR Negative-Results Paper Full Audit

**Audit date:** 2026-08-13

**Status:** Complete; no known claim, citation, compilation, or layout defect remains in the project draft

**Title:** *Safety-Efficiency Limits of Training-Free Whole-Prefix Visual Caching for Vision-Language-Action Inference*

## Audit scope and outcome

| Area | Outcome |
|---|---|
| Scientific scope | Bounded to training-free whole-prefix caching on one pinned OpenVLA-OFT/LIBERO-Spatial system; no universal caching claim |
| Claims and numbers | Reconciled against the frozen phase reports and machine summary; no unsupported headline claim found |
| Experimental design | Stage populations, gates, stopping decisions, untouched holdout, and unrun matched baselines are stated explicitly |
| Interpretation | Descriptive cross-stage comparisons are distinguished from paired or confirmatory inference |
| Efficiency | Reports measured visual-component share and skipped calls; does not misstate either as measured end-to-end speedup |
| References | Expanded from 6 to 18 directly relevant works; all 18 are cited and no citation is unresolved |
| Figures and tables | Every final page inspected; Figure 2 and the evidence-index table were redesigned for clarity |
| Build | Stable 11-page US Letter PDF; no undefined references/citations, overfull boxes, or LaTeX warnings |
| Reproducibility | Source, final PDF, evidence records, immutable-result indexes, hashes, and audit files are tracked |

## Reference coverage

The 18 references are intentionally distributed across five needs:

1. VLA foundations and generalist policies: RT-1, RT-2, Open X-Embodiment, Octo, $\pi_0$, and OpenVLA.
2. The exact experimental stack and efficient VLA models: OpenVLA-OFT, TinyVLA, and SmolVLA.
3. General visual-token efficiency: DynamicViT and Token Merging.
4. Direct VLA acceleration and caching: EfficientVLA, VLA-Cache, DepthCache, adaptive visual-token caching, neural-introspection KV-cache gating, and ActionCache.
5. Evaluation benchmark: LIBERO.

Six references were not enough to position the paper against the modern VLA-efficiency and caching literature. Eighteen is adequate for this focused project draft without padding the bibliography with tangential work. Venue-specific related-work expectations should still be checked at submission time.

## Figure 2 correction

The previous figure could be read as though the high-success panel contained a separate experiment or as though overlapping points were missing. The corrected figure:

- labels panel (a) as the full operating region and boxes the region expanded in panel (b);
- states that no point is moved in the expansion;
- uses one legend for Full Refresh, SAVR1, SAVR2, and SAVR3;
- labels every high-success point with its exact count;
- states that Full Refresh and SAVR2-$b05$ coincide at 0% skip and 100% success;
- states that SAVR2-$b05$ made no reuse decision; and
- warns that the stages used different populations and gates, so the frontier is descriptive rather than randomized.

## Title decision

The original method-oriented title implied a general efficient-inference contribution. The revised title identifies the actual contribution and boundary: a measured safety-efficiency limit for training-free, whole-prefix visual caching in VLA inference. It is informative without claiming that all SAVR-like or visual-caching approaches fail.

## Remaining submission work

No additional experiment is required for the bounded negative result already reported. Before external submission, the authors still need to choose a venue, apply its template and page limits, confirm author order/affiliations, and perform the venue's final disclosure and artifact checks.

## Final integrity identifiers

- LaTeX SHA-256: `5110060fb9581bace1a9d73031860b739d17b63eba1c97f297a2bcd30a84ce5b`
- PDF SHA-256: `48024cd1b9caab0fe9d5ace1b2ef1369176fed6eb98e0e53076c5682ae16130d`
