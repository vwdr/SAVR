# MIT URTC SAVR Poster: Traditional Layout Specification

**Phase:** 2 - Information hierarchy and spatial plan  
**Revision:** 4  
**Status:** Complete  
**Canvas:** 24 x 36 inches, portrait

## 1. Scientific order

The poster will follow this traditional sequence:

1. Introduction
2. Methods
3. Results
4. Discussion and Conclusion
5. Limitations

There will be no finding banner before the Introduction and no separate research-question box. The research objective will be stated naturally within the Introduction.

Discussion and Conclusion will share one section. A separate Conclusion section would repeat the same interpretation and reduce space available for the evidence and limitations.

## 2. Visual structure

```text
┌──────────────────────────────────────────────────────────────┐
│ FULL TITLE + NUMBERED AUTHORS                 RUTGERS LOGO    │
│ NUMBERED AFFILIATIONS                                        │
├──────────────────────────────────────────────────────────────┤
│ INTRODUCTION          │ METHODS                              │
│ 2-3 concise bullets   │ experiment figure + compact setup    │
├──────────────────────────────────────────────────────────────┤
│ RESULTS                                                      │
│ Large complete-prefix figure with short caption              │
│ ┌──────────────────────────┬───────────────────────────────┐ │
│ │ Scene-camera figure      │ Fine-grained reuse figure     │ │
│ │ + caption                │ + caption                     │ │
│ └──────────────────────────┴───────────────────────────────┘ │
├───────────────────────────────────┬──────────────────────────┤
│ DISCUSSION & CONCLUSION           │ LIMITATIONS              │
│ graphical feedback explanation   │ evidence boundary        │
│ + final interpretation            │                          │
├───────────────────────────────────┴──────────────────────────┤
│ REFERENCES | ACKNOWLEDGMENTS | CONTACT | GITHUB QR           │
└──────────────────────────────────────────────────────────────┘
```

## 3. Page allocation

| Region | Top | Bottom | Purpose |
|---|---:|---:|---|
| Header | 0.35 in | 3.70 in | Title, authors, affiliations, Rutgers logo |
| Introduction and Methods | 4.00 in | 12.20 in | Side-by-side context and self-contained experiment explanation |
| Results | 12.50 in | 27.30 in | Three figure-led result panels |
| Discussion/Conclusion and Limitations | 27.60 in | 33.20 in | Interpretation and evidence boundary |
| Footer | 33.50 in | 35.65 in | Four properly sized footer areas |

## 4. Header requirements

- Every line of the full title uses the same typeface, weight, and size.
- Authors use numbered affiliations:
  - Ved Dwivedi: affiliations 1 and 2
  - Cheng Yang: affiliation 2
  - Bo Yuan: affiliation 2
- Affiliation 1: John P. Stevens High School, Edison, New Jersey
- Affiliation 2: Department of Electrical and Computer Engineering, Rutgers University-New Brunswick, Piscataway, New Jersey
- Rutgers logo remains in the upper-right.
- A dark-red rule separates the header from the scientific content.

## 5. Introduction

The Introduction occupies the left side of the upper scientific band. It will contain two or three concise bullets:

- VLA policies repeatedly process similar camera observations during robot control.
- Reusing earlier visual information may reduce inference work, but stale information may change robot actions.
- This study tests whether training-free temporal reuse can reduce inference cost without reducing closed-loop task success.

There is no separate research-question panel. The objective will be included in the bullets.

## 6. Methods

Methods occupies the wider right side of the upper scientific band. It will include a large experiment-overview figure and a compact setup strip. The figure does not need to carry every detail by itself; together, the figure, labels, and setup strip must make the experiment self-explanatory.

The figure must show:

- scene camera, wrist camera, language instruction, and robot state
- OpenVLA-OFT producing an action chunk
- dense inference as the reference
- complete visual-prefix reuse
- scene-camera reuse while refreshing the wrist camera
- fine-grained reuse of internal visual states
- LIBERO closed-loop evaluation
- task success and complete query time as the main outcomes

The setup strip will identify the model, benchmark, inputs, measurements, and hardware. Supporting text should remain short and directly tied to the figure.

## 7. Results

Results will be primarily figure based.

### Figure 1 - Complete visual-prefix reuse

- Largest result figure
- Success versus skipped visual refresh
- Dense reference and conservative setting clearly identified
- Three large callouts: 100/100, 34.69-83.68% reuse with 0-52% success, and 69/70 with only 0.95% reuse
- Caption states the direct interpretation

### Figure 2 - Scene-camera reuse

- Success and latency shown as separate measures
- Dense and scene reuse both show 67/70 observed successes
- 25.24% scene reuse
- 1,188.28 versus 1,190.97 ms/query
- Caption states that success count was maintained but inference was not faster

### Figure 3 - Fine-grained internal reuse

- Controlled timing result and closed-loop result visually separated
- 22.60% median query-time reduction in the systems test
- 68/120 dense successes versus 0/120 reuse successes
- Caption prevents the timing measurement from being mistaken for a successful closed-loop speedup

## 8. Discussion and Conclusion

This combined section will use a graphical closed-loop feedback sequence:

```text
older representation -> changed action -> changed next observation
        ^                                      |
        └──── later reuse on a new trajectory ─┘
```

The final interpretation will state that none of the tested training-free strategies achieved both reliable closed-loop performance and meaningful end-to-end efficiency.

## 9. Limitations

Limitations receive a larger independent panel. It must include:

- one OpenVLA-OFT checkpoint
- one simulator benchmark family
- one GPU class
- different populations across the three experiment families
- terminal success rather than formal robot safety
- no claim that all VLA caching methods fail

## 10. Color and typography

Only these colors are permitted:

- White: background
- Black: text and data labels
- Deep red `#A6192E`: section backgrounds and visual emphasis
- Dark red `#6D001A`: header-content separator

All content-panel, figure-placeholder, footer-divider, logo-placeholder, and QR-placeholder boundaries will be black. There will be no blue, green, gray, or additional accent color in the poster template.

Figures will use a neutral-first visual system so they remain distinct from the red poster structure. Black and dark gray will carry axes, labels, baselines, and most data. Red will be reserved for one scientifically meaningful highlight or comparison when needed. White space, direct labels, and shape or line-style differences must preserve readability without relying on color alone.

Typography:

- One sans-serif family throughout
- Title: 48-54 pt, consistent across every title line
- Section headings: 40-46 pt, white on red
- Body: 32-36 pt
- Figure labels and captions: at least 28 pt

## 11. Phase 2 revision decision

- Full title styling unified: pass
- Author affiliations corrected: pass
- Red/white/black palette locked: pass
- Introduction placed in the upper-left without a separate question box: pass
- Methods placed in the wider upper-right with a figure and compact setup: pass
- Results specified as figure-led: pass
- Discussion and Conclusion combined: pass
- Limitations enlarged: pass
- Footer expanded and divided into four usable areas: pass

Phase 3 must implement this revised structure before any final scientific text or figures are inserted.
