# SAVR Negative-Results Paper Full Audit

**Audit date:** 2026-08-13

**Status:** Complete for author review; submission-format and public-artifact work remains

**Title:** *A Negative Result for Training-Free Whole-Prefix Visual Caching in VLA Inference*

## Audit scope and outcome

| Area | Outcome |
|---|---|
| Scientific scope | Bounded to training-free whole-prefix caching on one pinned OpenVLA-OFT/LIBERO-Spatial system; no universal caching claim |
| Claims and numbers | Reconciled against the frozen phase reports and machine summary; no unsupported headline claim found |
| Experimental design | Stage populations, gates, stopping decisions, untouched holdout, and unrun matched baselines are stated explicitly |
| Interpretation | Descriptive cross-stage comparisons are distinguished from paired or confirmatory inference |
| Efficiency | Reports measured visual-component share and skipped calls; does not misstate either as measured end-to-end speedup |
| References | 13 claim-relevant works; all 13 are cited, none is unresolved, and seven tangential entries were removed |
| Figures and tables | Every final page inspected; Figure 2, confidence-interval tables, evidence index, and threshold tables were checked for clarity |
| Build | Stable 12-page tagged US Letter PDF; no undefined references/citations or overfull boxes; one non-fatal accessibility-package destination warning remains |
| Reproducibility | Exact controller thresholds, source/PDF hashes, repository status, DOI status, and artifact-release work are stated |

## Reference coverage

The bibliography follows a claim-level relevance rule, not a target count. The 13 retained entries support four necessary areas: the evaluated VLA lineage, directly related VLA efficiency/caching methods, closed-loop distribution shift, and the benchmark/statistical method. Open X-Embodiment, Octo, $\pi_0$, TinyVLA, SmolVLA, DynamicViT, and Token Merging were removed because no necessary manuscript claim depended on them. The full mapping is recorded in `docs/SAVR_CITATION_RELEVANCE_AUDIT.md`.

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

The original method-oriented title implied a general efficient-inference contribution, while the intermediate safety-efficiency title overstated the measured outcome. The final title identifies both the negative-result contribution and its narrow boundary without implying collision safety or a universal limit.

## Remaining submission work

No additional experiment is required for the bounded negative result already reported. Before external submission, the authors still need to choose a venue, apply its template and page limits, confirm author order/affiliations, create a public versioned repository/archive with a persistent identifier, revalidate the August 2026 preprint, and perform the venue's final accessibility, disclosure, and artifact checks.

## Final integrity identifiers

- LaTeX SHA-256: `b3ff0704add279411293b1a8acef79d0a91b18c8ed2a923119b4ece2480fd8e9`
- PDF SHA-256: `bb191bcf472183aebbe3eb1041710bc22cb3b1a06a873edad3c86ff05d13659c`
