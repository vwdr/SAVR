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

- LaTeX SHA-256: `b3ff0704add279411293b1a8acef79d0a91b18c8ed2a923119b4ece2480fd8e9`
- PDF SHA-256: `bb191bcf472183aebbe3eb1041710bc22cb3b1a06a873edad3c86ff05d13659c`
- PDF size: 605,594 bytes
- PDF pages: 12, US Letter

## Verification

- Compiled through three stable pdfLaTeX passes on TITAN inside `/home/ved/SAVR/tmp/pdfs`.
- Final build had no undefined citation or cross-reference warnings and no overfull boxes. The older accessibility package emits one non-fatal destination warning while still producing a readable 12-page PDF that reports `Tagged: yes`; formal PDF/UA conformance is not claimed.
- All twelve final pages were rendered and visually inspected. Figure 2 uses an explicitly marked expansion region, a shared legend, exact high-success labels, and a caption that discloses point overlap and the different staged populations. Tables remain in narrative order, and the widest threshold table is split into readable panels.
- All 13 citation keys resolve to 13 used bibliography entries. Every retained source has a documented claim-level role; seven broad or tangential sources were removed rather than kept for reference count.
- Tables 3 and 4 report two-sided 95% exact Clopper--Pearson intervals for every terminal-success count.
- Source/document `git diff --check` passed (the generated PDF is excluded from whitespace checks).
- The manuscript/bootstrap tests passed 17/17. The repository suite passed 390 tests plus 9 subtests when the pre-execution-only V10 preflight test was excluded; that one legacy test now correctly detects that immutable V10 result directories exist after the completed V10 attempt.

## Scientific boundary

The paper reports a bounded negative result for whole-prefix SAVR on one pinned OpenVLA-OFT/LIBERO-Spatial system. It does not claim that visual caching is universally unreliable, does not use the protected final holdout, and does not claim a measured end-to-end SAVR speedup. The reported count is 1,160 primary evaluation episodes plus a separate 50-episode timing pilot. Matched Periodic Refresh and Visual-Only Refresh rollouts were not eligible under the frozen stopping protocol and are stated as a limitation.
