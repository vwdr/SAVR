# SAVR Negative-Results Paper Completion Report

**Completed:** 2026-08-13  
**Status:** Scientifically complete author-review preprint; venue conversion and public artifact release remain required before external submission

## Deliverables

- LaTeX source: `manuscript/State-Aware Visual Refresh for Efficient VLA Inference.tex`
- Reviewed PDF: `output/pdf/SAVR_Negative_Results_Paper.pdf`
- Claim audit: `docs/SAVR_NEGATIVE_PAPER_CLAIM_AUDIT.md`
- Full manuscript audit: `docs/SAVR_NEGATIVE_PAPER_FULL_AUDIT.md`
- Citation relevance audit: `docs/SAVR_CITATION_RELEVANCE_AUDIT.md`
- Evidence archive: `docs/NEGATIVE_RESULTS_PAPER_ARCHIVE.md`
- Machine-readable result summary: `docs/evidence/negative_results_summary.csv`

## Integrity

- LaTeX SHA-256: `3e2364621b4d3e9f183fe8645a52cd67157fbb718243bc4262947174a7292a8b`
- PDF SHA-256: `c0a3a342bb87b65e812e28f862d9e2e2957368fe83cdd29d71d75e423c5babcb`
- PDF size: 269,865 bytes
- PDF pages: 12, US Letter

## Verification

- Compiled through three stable pdfLaTeX passes on TITAN inside `/home/ved/SAVR/tmp/pdfs`.
- Final build had no undefined citation or cross-reference warnings, overfull boxes, or PDF text-extraction syntax warnings. TITAN's older accessibility package produced malformed marked-content warnings, so it was removed rather than used to make a false tagging claim; a tagged/PDF-UA venue build remains submission work.
- All twelve final pages were rendered and visually inspected. Figure 2 uses an explicitly marked expansion region, a shared legend, exact high-success labels, and a caption that discloses point overlap and the different staged populations. Tables remain in narrative order, and the widest threshold table is split into readable panels.
- All 13 citation keys resolve to 13 used bibliography entries. Every retained source has a documented claim-level role; seven broad or tangential sources were removed rather than kept for reference count.
- Tables 3 and 4 report two-sided 95% exact Clopper--Pearson intervals for every terminal-success count.
- Source/document `git diff --check` passed (the generated PDF is excluded from whitespace checks).
- The manuscript/bootstrap tests passed 17/17. The repository suite passed 390 tests plus 9 subtests when the pre-execution-only V10 preflight test was excluded; that one legacy test now correctly detects that immutable V10 result directories exist after the completed V10 attempt.

## Scientific boundary

The paper reports a bounded negative result for whole-prefix SAVR on one pinned OpenVLA-OFT/LIBERO-Spatial system. It does not claim that visual caching is universally unreliable, does not use the protected final holdout, and does not claim a measured end-to-end SAVR speedup. The reported count is 1,160 primary evaluation episodes plus a separate 50-episode timing pilot. Matched Periodic Refresh and Visual-Only Refresh rollouts were not eligible under the frozen stopping protocol and are stated as a limitation.
