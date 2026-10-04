# LaTeX source package

This package contains the complete source for:

**A Negative Result for Training-Free Temporal Visual Caching in OpenVLA-OFT**

## Files

- `paper.tex` - complete manuscript and all TikZ/PGFPlots figures
- `references.bib` - bibliography database
- `IEEEtran.cls` - IEEE journal document class
- `IEEEtran.bst` - IEEE bibliography style

## Compile

Run the following commands from this directory:

```text
pdflatex paper.tex
bibtex paper
pdflatex paper.tex
pdflatex paper.tex
```

The resulting manuscript is `paper.pdf`.
