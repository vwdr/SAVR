# SAVR Negative-Results Paper Completion Report

**Completed:** 2026-08-13  
**Status:** Publication-ready project draft; author review still required before external submission

## Deliverables

- LaTeX source: `manuscript/State-Aware Visual Refresh for Efficient VLA Inference.tex`
- Reviewed PDF: `output/pdf/SAVR_Negative_Results_Paper.pdf`
- Claim audit: `docs/SAVR_NEGATIVE_PAPER_CLAIM_AUDIT.md`
- Full manuscript audit: `docs/SAVR_NEGATIVE_PAPER_FULL_AUDIT.md`
- Evidence archive: `docs/NEGATIVE_RESULTS_PAPER_ARCHIVE.md`
- Machine-readable result summary: `docs/evidence/negative_results_summary.csv`

## Integrity

- LaTeX SHA-256: `5110060fb9581bace1a9d73031860b739d17b63eba1c97f297a2bcd30a84ce5b`
- PDF SHA-256: `48024cd1b9caab0fe9d5ace1b2ef1369176fed6eb98e0e53076c5682ae16130d`
- PDF size: 239,006 bytes
- PDF pages: 11, US Letter

## Verification

- Compiled twice to stable references with `latexmk`/pdfLaTeX on TITAN inside `/home/ved/SAVR/tmp/pdfs/savr-negative-paper`.
- Final build had no undefined citation or cross-reference warnings and no overfull boxes.
- All eleven final pages were rendered and visually inspected. Figure 2 now uses an explicitly marked expansion region, a shared legend, exact high-success labels, and a caption that discloses point overlap and the different staged populations.
- All 18 citation keys resolve to 18 used bibliography entries; the bibliography covers the VLA lineage, efficient VLA architectures, general visual-token efficiency, directly related VLA caching, and LIBERO.
- `git diff --check` passed.
- Repository tests passed after using the repository source path: 391 passed.

## Scientific boundary

The paper reports a bounded negative result for whole-prefix SAVR on one pinned OpenVLA-OFT/LIBERO-Spatial system. It does not claim that visual caching is universally unsafe, does not use the protected final holdout, and does not claim a measured end-to-end SAVR speedup. Matched Periodic Refresh and Visual-Only Refresh rollouts were not eligible under the frozen stopping protocol and are stated as a limitation.
