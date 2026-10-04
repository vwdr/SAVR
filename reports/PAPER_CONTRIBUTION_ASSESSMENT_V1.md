# Paper contribution assessment: current-frame compression + residual correction in OpenVLA-OFT

Date: 2026-09-21. Prepared before any manuscript draft. No new experiments, no GPU,
no training. All numbers below come from the frozen, independently verified
results (`reports/CURRENT_FRAME_ROBOT_PILOT_RESULT_V1.md`,
`CURRENT_FRAME_ADAPTER_FIT_RESULT_V1.md`, `analysis.json`, fit `metrics`), and from
the published literature cited inline.

## 1. What the verified evidence supports

Robot-development pilot (120 conditions: 40 tasks × 3 states, seed 7; four
policies on identical starting conditions):

| Policy | Success | Controlled mean query time | vs dense |
|---|---:|---:|---:|
| Dense 512 tokens | 120/120 (100%) | 1,201.22 ms | — |
| Compression 384/512 (75% retention) | 117/120 (97.5%) | 1,017.60 ms | −15.29% |
| Compression + action-only corrector | 118/120 (98.33%) | 1,018.95 ms | −15.17% |
| Compression + visual corrector | 117/120 (97.5%) | 1,019.80 ms | −15.10% |

Paired 95% bootstrap intervals (10k draws, seed 7): compression−dense **−2.50 pp
[−5.00, 0.00]**; visual−compression **0.00 pp [−3.33, 2.50]**; visual−action-only
**−0.83 pp [−4.17, 2.50]**.

Offline adapter fitting (800 validation records, 20/task, 40 tasks, fitting-disjoint
trajectories; single training seed; all records from the historical training pool):

- Visual corrector mean action L1 **0.03305615** vs compression 0.03489551 →
  **−5.271%** relative.
- Action-only 0.03332298 → −4.506%; visual vs action-only → **−0.801%**.
- Shuffled-label control 0.16091179 (confirms the visual corrector is not fitting
  label noise); visual gripper-sign disagreement rose +0.094 pp vs compression.

Supported statements (and only these):

1. At a fixed 75%-retention operating point, compression gives a real, measured
   **15.29% controlled mean query-time reduction** at **−2.50 pp success**
   in a near-ceiling single-seed pilot. Acceleration is attributable to
   compression; the corrector adds ~2.2 ms (0.216%), i.e. it does not accelerate.
2. The learned residual corrector **improves offline action error**
   (−5.27% vs compression; −0.80% vs action-only) but **does not improve
   closed-loop success** (net 0 vs compression, −1 vs action-only, in 120
   matched conditions). This offline→closed-loop transfer gap is a genuine,
   honestly reportable finding — a negative result, not a negative claim about
   the method we were hoping to validate.
3. Qualitative protocol strengths exist and are publishable as methodology:
   matched starting conditions, frozen predeclared gates, native/shadow controls,
   ceiling analysis, and explicit exposure/one-seed/timing-boundary accounting.

## 2. Closest published work and where we stand against it

### 2.1 Current-frame visual-token compression (directly on our model family)

- **VLA-Pruner** (Liu et al., arXiv 2511.16449): training-free, attention-guided
  adaptive token pruning for VLA inference, evaluated on **OpenVLA and OpenVLA-OFT
  on LIBERO** at 50%/25%/12.5% token retention using the standard protocol.
  On OpenVLA-OFT at **50% retention they report 1.46× speedup and ~101% relative
  success (slightly above vanilla)**; up to ~1.9–2.0× at 25–12.5% retention.
- **EfficientVLA** (Yang et al., NeurIPS 2025): task-aware visual-token selection +
  layer pruning + diffusion-head caching on CogACT/SIMPLER; 1.93× speedup, FLOPs
  reduced to 28.9%, 0.6% success drop.
- **VLA-Cache** (Xu et al., arXiv 2502.02175): adaptive temporal KV caching across
  frames; on OpenVLA-OFT at 50% reuse, ~1.34× speedup at ~99% relative success
  (as re-measured by VLA-Pruner).
- **FLASHVLA** (arXiv 2505.21200), **Token Expand–Merge** (arXiv 2512.09927),
  **FAST** (arXiv 2501.09747): training-free token compression / action-reuse /
  tokenization for VLA inference.

Assessment: our fixed stratified 384/512 (75%) spatial retention is a static,
hand-specified subsampler — a simpler cousin of FastV-style fixed retention at a
**more conservative operating point than already-published adaptive methods on the
same model and benchmark**. The published adaptive methods operate at 25–50%
retention with equal-or-better success; our 75%-retention point delivers a smaller
time reduction (−15.29% vs their −31–50%) at a small success cost. **As a
compression method, our contribution is plainly not novel or competitive.** We must
not frame it as one, and no reviewer will accept a "method beats baseline" claim.

The one thing our timing adds beyond those papers is an *end-to-end controlled
latency measurement that includes the released action head, corrector inference,
feature extraction, copies and output conversion* for an 8×7 parallel-decoding
OpenVLA-OFT deployment. That is a methodology/measurement contribution, not a method
one, and it is not directly comparable to published ms/action-chunk numbers on
different hardware.

### 2.2 Learned action correction on a frozen VLA

- **Residual policy adaptation** (preprint, figshare 31329868): thin task-specific
  residual heads on a frozen VLA backbone — the same family as our corrector.
- Residual-RL / residual-action lines (Xiao et al.; general residual RL literature).

Our corrector does not beat the premise of that line; its distinguishing content is
the **empirical transfer verdict**: residual heads improve offline L1 on a
compressed base but not closed-loop success. To our knowledge this specific
measurement (offline gain + closed-loop null on a compressed OpenVLA-OFT, with
paired conditions and a shuffled control) has not been reported, and it is timely
because the field routinely uses offline action error as a proxy for policy quality.

## 3. What is missing before a peer-reviewed submission

Ranked by how much they gate reviewability (all are new experiments; none launched):

1. **Evaluation scale and protocol match (critical).** Our 30 conditions/suite,
   single-seed pilot cannot be compared to the standard LIBERO protocol used by
   VLA-Pruner/OpenVLA family (≈500 episodes/suite). At the current scale, ±2–3
   success differences are inside noise. Minimum: run the four policies under the
   standard protocol on an **untouched state/task split** with **≥3 seeds**.
   Without this, no success-rate claim survives review.
2. **Head-to-head comparator on identical hardware/protocol.** Neither
   VLA-Pruner/FLASHVLA/VLA-Cache nor a static 25%-retention baseline was run in
   our harness. Publishable comparison requires at least one of these (VLA-Pruner
   releases code) under matched checkpoint, GPU, and protocol.
3. **FLOPs and memory accounting.** Published papers report FLOPs(T) and latency
   together; we currently have latency only.
4. **Corrector robustness.** One training seed. Need ≥3 seeds to establish both the
   offline gain direction and the closed-loop null as stable rather than incidental.
5. **Generalization evidence.** All evaluation states were historically exposed
   development conditions. Any generalization sentence requires a held-out set.

## 4. Verdict and possible contributions

**Can we proceed to a peer-reviewed submission now? No — not for the claims one
would normally want (a better compression method, or correction that improves
closed-loop success). Neither is supported.**

Two defensible papers are visible from the verified evidence, in increasing order
of additional work:

- **(A) Workshop / empirical-study paper (current evidence, minimal added runs):**
  *"Fixed current-frame visual-token retention in OpenVLA-OFT: a controlled
  measurement, and why offline action-error gains did not transfer to closed-loop
  success."* Contributions: (1) an evaluation/measurement contribution — controlled
  end-to-end latency with full overhead accounting at a fixed conservative
  operating point; (2) a documented negative/transfer finding about residual
  correctors (offline −5.27%/−0.80% L1, closed-loop null, shuffled control pass);
  (3) reproducibility practices (frozen gates, paired conditions, ceiling and
  exposure analysis). This is an honest-study framing whose novelty is the
  evaluation and the negative result, not a method.
- **(B) Full paper (requires the §3 gaps closed):** the study above + standard
  multi-seed protocol + ≥1 matched comparator + FLOPs/memory. This targets a
  main-line venue and expands claims from "measured tradeoff at one point" to a
  defensible efficiency–accuracy characterization.

Recommendation: pursue (A) as the immediate deliverable (advisor-facing draft and
email per your instruction), state the §3 gaps explicitly, and treat (B) as the
outcome if the advisor confirms value — noting that its load-bearing cost is the
standard-protocol multi-seed robot evaluation plus one comparator run.

## 5. Explicit non-claims (to keep out of any draft)

- No claim that visual correction improves overall robot success (it does not).
- No claim of equivalent success, faster task completion, or confirmed
  generalization.
- No "method beats baseline" claim versus published adaptive compression.
- No significance claim from single-seed 120-condition counts; intervals are
  descriptive.
- No comparison of our latency to published ms numbers without matched hardware and
  protocol.