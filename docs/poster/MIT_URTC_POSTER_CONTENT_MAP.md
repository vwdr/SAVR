# MIT URTC SAVR Poster: Verified Content Map

**Phase:** 1 - Scientific story and source verification  
**Status:** Complete  
**Intended subject area:** Technology of Automation  
**Poster format:** 24 x 36 inches, portrait

## 1. Poster purpose

The poster must explain one question:

> Can OpenVLA-OFT reuse visual representations from earlier robot-control steps to reduce inference time without reducing closed-loop task success?

The answer supported by the completed experiments is:

> None of the three tested training-free reuse strategies achieved both reliable closed-loop performance and a meaningful end-to-end efficiency improvement.

This is a bounded result for one OpenVLA-OFT checkpoint, the LIBERO simulator, and the tested reuse mechanisms. It is not a claim that all visual caching or all VLA acceleration methods fail.

## 2. Working poster title

**Training-Free Temporal Visual Caching in OpenVLA-OFT: A Closed-Loop Evaluation**

This title is more accessible for a poster than an emphasis on the phrase “negative result.” The evidence and conclusion will still be presented directly.

## 3. One-sentence audience takeaway

> Reusing older visual representations saved computation in some tests, but either reduced robot task success or failed to improve actual inference time.

This sentence should appear near the top of the poster and remain understandable without reading the methods.

## 4. Scientific narrative

The poster should follow this sequence:

1. VLA policies repeatedly process similar camera images.
2. Reusing earlier visual representations may reduce computation.
3. Older representations may omit small but important changes in the robot, object, or wrist-camera view.
4. We evaluated three increasingly fine reuse strategies under closed-loop control.
5. Complete visual-prefix reuse reduced task success whenever reuse became substantial.
6. Reusing only the scene camera preserved the observed success count, but did not improve end-to-end latency relative to an optimized dense baseline.
7. Fine-grained internal reuse reduced query time in a controlled systems test, but failed closed-loop control.
8. Effective acceleration must preserve information about the current control state, not only exploit visual similarity.

## 5. Required poster content

### Motivation

- OpenVLA-OFT processes scene-camera and wrist-camera images at every policy query.
- Consecutive observations often share backgrounds and objects.
- Temporal redundancy suggests an opportunity to reuse previously computed visual information.
- Closed-loop control makes reuse risky because a changed action changes later observations.

### Research objective

Communicate the objective visually rather than through a long equation:

> Reduce visual processing while maintaining task success and lowering complete inference time.

All three conditions matter. Logical reuse alone is not a successful result.

### Model and evaluation

- Model: released `openvla-7b-oft-libero-four-suite` checkpoint
- Benchmark: LIBERO simulated robot-manipulation tasks
- Inputs: language instruction, scene camera, wrist camera, and robot state
- Primary reliability measure: terminal task success
- Primary efficiency measure: complete query time relative to dense inference
- Hardware: one NVIDIA TITAN RTX

### Evaluated strategies

Use public, descriptive names only:

1. **Complete visual-prefix reuse**  
   Reuse the projected visual representation of both cameras and refresh it using image, robot-state, action, and cache-age signals.

2. **Scene-camera reuse**  
   Keep the wrist-camera representation current and reuse only the more stable scene-camera representation.

3. **Fine-grained internal reuse**  
   Reuse selected visual key-value states across transformer layers and image regions while limiting their age.

## 6. Verified headline results

### Result A - Complete visual-prefix reuse

- Evaluation population: 1,160 primary closed-loop episodes on LIBERO-Spatial
- Separate timing pilot: 50 episodes and 662 steady policy queries
- Dense reference: 100/100 successful episodes
- Nine permissive reuse settings:
  - visual refresh skipped on 34.69-83.68% of policy queries
  - task success ranged from 52/100 to 0/100
- Most conservative final setting:
  - 69/70 successes
  - only 9 reuses across 944 policy queries
  - 0.9534% skipped visual refreshes

**Poster interpretation:** Meaningful complete-prefix reuse reduced task success. High success was recovered only when reuse became negligible.

### Result B - Scene-camera reuse

- Evaluation population: 70 dense and 70 reuse episodes on paired LIBERO-Object states
- Batched dense inference: 67/70 successes
- Scene-camera reuse: 67/70 successes
- Scene representation reused on 25.2434% of policy queries
- Batched dense wall time: 1,188.28 ms/query
- Scene-reuse wall time: 1,190.97 ms/query
- Difference: 2.69 ms/query, corresponding to a 0.23% slowdown
- Visual CUDA time decreased by 8.46%, but controller and execution overhead removed the component-level benefit.

**Poster interpretation:** Scene-camera reuse matched the observed dense success count but did not improve end-to-end latency.

### Result C - Fine-grained internal reuse

- Controlled systems test: 97 model calls
- Median complete-query time reduction: 22.5969%
- Closed-loop population: 120 paired initial conditions across four LIBERO suites
- Dense inference: 68/120 successes
- Fine-grained reuse: 0/120 successes
- Per-suite reuse results: 0/30 on Spatial, Object, Goal, and LIBERO-10

**Poster interpretation:** The cache reduced measured query time in a controlled test but did not preserve a functional closed-loop policy.

## 7. Main scientific explanation

The poster should explain the failure through a simple feedback chain:

> Older visual state -> changed action -> changed robot state -> changed next image -> later reuse decisions occur on a different trajectory

Two supporting observations may appear beside this diagram:

- In 87 examined complete-prefix episodes, the first reused query changed the predicted action.
- Of 374 reuse decisions where at least one camera crossed its change threshold, 372 were caused only by the wrist camera.

These observations help explain why global visual similarity and camera averaging were insufficient.

## 8. Required conclusions

The conclusion area should contain three short points:

1. **Complete-prefix reuse:** useful reuse rates reduced task success.
2. **Scene-camera reuse:** task success was maintained, but actual inference was not faster.
3. **Fine-grained reuse:** measured computation decreased, but closed-loop control failed.

Final conclusion:

> Training-free temporal reuse at the tested representation boundaries did not provide a reliable success-latency tradeoff. Future approaches should preserve current task-relevant information or explicitly train the policy to tolerate reused representations.

The second sentence is a research implication, not a demonstrated positive result.

## 9. Limitations that must remain visible

- One OpenVLA-OFT checkpoint
- One simulator benchmark family
- One GPU class
- Different method families used different evaluation populations
- The combined success-reuse frontier is descriptive, not a randomized head-to-head comparison
- The final dense four-suite success rate was 56.67%, below the published OpenVLA-OFT average
- Terminal task success was measured; collisions, constraint violations, and formal robot safety were not
- Results should not be generalized to all VLA architectures or all caching methods

## 10. Terminology rules

### Use

- dense inference
- complete visual-prefix reuse
- scene-camera reuse
- wrist-camera refresh
- fine-grained internal reuse
- transformer key-value states
- terminal task success
- end-to-end or complete-query latency
- closed-loop evaluation

### Do not use

- internal method or phase names such as SAVR1, SAVR2, SAVR3, ACR, BRACE, PAIR, CAC, C1H, or D62
- gate names, recovery identifiers, run labels, or internal configuration names
- “safety” as a synonym for task success
- “speedup” for the scene-camera experiment
- “equivalent performance” for the 67/70 observed count match
- “22.6% end-to-end speedup” for the controlled fine-grained systems result
- invalidated exploratory action-quality or risk-routing results
- technical debugging history

## 11. Figure content priorities

The design phase should create four visual elements:

1. **System and reuse diagram** - current inputs, OpenVLA-OFT, robot action, and the three reuse boundaries
2. **Main success-reuse plot** - complete-prefix operating points with dense and conservative references
3. **Two-part improvement comparison** - scene-camera success/latency result and fine-grained timing/success result, kept visually separate
4. **Closed-loop feedback diagram** - why a small reuse error can propagate through later observations

The results area must receive the most space. Detailed tables from the paper should not be copied onto the poster.

## 12. Content priority and space budget

### Essential

- research question
- three reuse strategies
- three headline results
- closed-loop explanation
- bounded conclusion

### Supporting

- model and benchmark
- sample sizes
- definitions of task success and latency
- key limitations

### Footer only

- short references
- acknowledgments
- GitHub QR code
- contact information

### Exclude

- literature-review paragraphs
- full mathematical derivations
- threshold tables
- implementation-recovery history
- artifact hashes
- detailed confidence-interval tables
- proposed methods that were not trained or evaluated

## 13. Phase 1 verification record

The content map was reconciled against:

- `output/tex/temporal_reuse_tro/paper.tex`
- `reports/PHASE6_CALIBRATION_REPORT.md`
- `reports/PHASE6R_D_STAGE1_REPORT.md`
- `reports/PHASE6S_D_VALIDATION_REPORT.md`
- `reports/PHASE_V3_D_REPORT.md`
- `reports/CAC_C1_S5_REQUALIFICATION_PASS.md`
- `reports/CAC_C1H_S6_RECOVERY01_GATE_H_STOP.md`
- `docs/evidence/negative_results_summary.csv`

The three reported result families are scientifically distinct and must remain labeled with their own populations and measurement types.

## 14. Phase 1 exit decision

Phase 1 is complete. The poster has a verified central question, narrative, result set, terminology boundary, limitation set, and figure data plan.

Phase 2 may begin with the reading sequence and spatial hierarchy. No final prose or figures should be produced until that hierarchy is approved.
