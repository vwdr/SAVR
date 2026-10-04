# Cache-Aware Learning Research Audit

**Date:** 2026-08-30  
**Status:** Initial research conclusion; refined by the dedicated deep dive  
**Purpose:** Determine whether relaxing the training-free constraint creates a
scientifically defensible positive-results direction after SAVR, ACR, BRACE,
and PAIR.

> **Refinement notice:** The later
> [Fresh-Grounded Cache Correction deep dive](FRESH_GROUNDED_CACHE_CORRECTION_DEEP_DIVE_2026-08-30.md)
> identified A2C2 as an additional direct collision and refined the proposal's
> architecture, teacher objective, and mandatory on-policy training stage. Use
> that document for the current recommendation.

## 1. Executive conclusion

The literature supports the broad premise that learning can make VLA
acceleration substantially more reliable than handcrafted reuse rules. It does
**not** support treating “a learned cache selector” as a new contribution.
Learned selection, action-aware distillation, action-tolerance estimation,
confidence invalidation, stale-feature prediction, and fresh-vision/stale-
semantics fusion have all appeared in recent work.

The project evidence also rules out a second simple idea: another learned
reuse-risk router. PAIR used actual recursive cache provenance and expert-regret
supervision, yet its favorable 40-contract pilot did not generalize to the
independent 240-contract confirmation. The larger study obtained Spearman
0.2001, only 3.14% matched-service CVaR90 improvement, and a negative lower
bound. The central failure was not lack of router capacity alone; the cheap
pre-decision features did not reliably expose the task- and trajectory-specific
information destroyed by reuse.

The strongest remaining organic direction is therefore a **fresh-grounded
cache adaptation** method:

> Keep the expensive VLA backbone frozen, reuse downstream visual K/V state,
> but train a small action-side adapter to combine the recursively cached
> action representation with the *current visual features that the existing
> visual encoder and projector already compute*. Train the adapter on exact
> mixed-age cache states to recover the full-refresh action, rather than asking
> a router to predict whether an unseen error will occur.

This is a provisional research direction, not yet a frozen method. Its novelty
is **moderate and scoped**, and its positive-result plausibility is
**meaningfully better than another gate**, because it supplies missing current
information instead of trying to infer cache damage from proxies. It still
requires a small, decisive feasibility study before a full execution protocol.

## 2. What our experiments require the next method to solve

| Established finding | Design requirement |
|---|---|
| Whole-prefix and aggressive K/V reuse can materially reduce task success | The method must change how cached representations are consumed, not just retune a threshold. |
| BRACE measured a real speed substrate: P2-D50 reduced complete-cycle time by 12.23% | Preserve downstream K/V reuse as the acceleration mechanism. |
| BRACE action parity failed for every timed cached profile | Train against action-relevant targets; visual similarity is insufficient. |
| PAIR P3R found 17.05% conservative net saving for the selected raw profile | A lightweight learned module has a plausible latency budget. |
| PAIR P4B failed independent confirmation despite exact provenance and recursive contracts | Do not make the paper depend on predicting reuse harm from the same class of cheap summaries. |
| PAIR error was horizon-, task-, and trajectory-dependent | Training must expose the module to recursively realized mixed-age cache states, not isolated clean perturbations. |
| Exact-repeat and all-fresh controls passed | The negative result is scientific rather than numerical noise; the redesign must be substantive. |

## 3. Primary-literature map and novelty collisions

Recent 2026 papers are treated as primary-source method records, not as
independently reproduced facts. Several report strong results but provide
incomplete or unavailable code; their reported numbers cannot be assumed to
transfer to this repository.

| Work | What it establishes | Consequence for this project |
|---|---|---|
| [VLA-Cache](https://openreview.net/pdf?id=QZYZ0Xm58q) | Training-free, token-wise temporal K/V reuse can provide physical acceleration. | Reuse itself is not novel; it remains a baseline/substrate. |
| [LAC](https://arxiv.org/html/2602.00686) | Learns both token selection and cache ratio with task gradients; reports 1.76x speedup and +1.9 pp LIBERO success. | A learned selector, learned ratio, optical-flow policy, or generic cache-aware loss is already occupied. Its linked repository contained only a README at audit time. |
| [AC2-VLA](https://arxiv.org/html/2601.19634) | Jointly learns reuse, pruning, and layer execution with action-guided self-distillation. | Action-conditioned adaptive computation and self-distilled gates are not sufficient novelty. |
| [LightVLA](https://arxiv.org/abs/2509.12594) | Uses differentiable token pruning and reports simultaneous latency and success gains. | Gumbel/straight-through token selection is not a new mechanism. |
| [Action-JND](https://arxiv.org/html/2608.21247) | Learns token-wise action tolerance from feature perturbations and applies it to K/V reuse and pruning, including OpenVLA-OFT. | Action-consistency learning or token sensitivity alone is occupied. |
| [Gated VLA-Cache](https://arxiv.org/abs/2608.10824) | Uses model-output confidence to invalidate cache reuse. | Confidence-triggered full refresh is not novel and does not repair missing information. |
| [The Gate, Not the Cache](https://arxiv.org/html/2608.00391) | Shows self-harvested gates can fail silently; unconditional dense refresh during actuation slack restores reliability on its hardware. | Gate provenance must be controlled. The local 1.220 s dense path did not fit our roughly 0.4 s action window, so this is not presently the local solution. |
| [Latent Bridge](https://arxiv.org/html/2605.02739) | Predicts stale-feature/KV deltas and uses DAgger to close recursive deployment shift; reports 1.65–1.73x episode speedup. | A generic feature-delta predictor or DAgger-trained bridge is a direct novelty collision. It also shows why clean-pair training alone is inadequate. |
| [CloudEdgeVLA](https://arxiv.org/html/2608.00569) | Fuses stale semantic features with current lightweight vision and trains fresh/stale paths toward the current action. Its ablation reports a large benefit from current vision. | A generic “stale semantics + fresh vision” action head is already occupied. Any continuation must target the different mixed-age, internal K/V reuse problem and avoid an extra vision encoder. |
| [VLAConf](https://arxiv.org/html/2605.29605) | Learns calibrated task-success confidence from frozen VLA representations and demonstrates selective assistance. | Success-confidence estimation and abstention are not standalone novelty. They also require outcome-labeled rollouts. |
| [SelectiveNet](https://proceedings.mlr.press/v97/geifman19a.html) and [Conformal Risk Control](https://arxiv.org/abs/2208.02814) | Establish risk–coverage optimization and post-hoc risk control under their assumptions. | Calibration can govern fallback, but it cannot create information absent from the cached path or guarantee closed-loop success under arbitrary distribution shift. |
| [DAgger](https://proceedings.mlr.press/v15/ross11a/ross11a.pdf) | Shows why training on states visited by the learned policy is needed to control sequential covariate shift. | At least one predeclared on-policy aggregation stage is scientifically justified if the initial adapter passes offline tests. |
| [HarmoniCa](https://openreview.net/pdf?id=d1191766ad52130ac64033d012c6beebf23a6346.pdf) | In diffusion caching, explicitly identifies prior-step disregard and objective mismatch as training/inference failures. | The VLA adapter must train on a recursively evolving cache and an action-level objective, not independent one-step cache pairs. |
| [Randomized cache certificates](https://arxiv.org/abs/2607.21475) | Randomization can restore attribution for discarded information, but the paper explicitly finds attribution is not general failure prediction. | Randomized probing remains a useful diagnostic/ablation, not the primary positive mechanism. |

## 4. Why the generic cache-aware proposal is insufficient

The initial proposal bundled three ideas: train for selective reuse, preserve
actions under reuse, and abstain when uncertain. Each component is reasonable,
but the bundle is not yet a contribution:

1. **Selective reuse learning** overlaps directly with LAC and AC2-VLA.
2. **Action preservation** overlaps with Action-JND and action-guided
   distillation.
3. **Abstention/fallback** overlaps with Gated VLA-Cache, VLAConf, and general
   selective-prediction work.
4. **A learned risk router over actual stale contracts** is essentially the
   PAIR hypothesis, which failed independent confirmation.
5. **A feature-delta corrector** overlaps directly with Latent Bridge.
6. **A stale-feature action head with fresh vision** overlaps conceptually with
   CloudEdgeVLA.

A publishable continuation therefore needs a narrower mechanism tied to the
specific information flow and measurements of this repository.

## 5. Strongest candidate: fresh-grounded mixed-age cache adaptation

### 5.1 Research question

Can a small adapter recover full-refresh OpenVLA-OFT actions from a recursively
mixed-age downstream visual K/V cache by grounding the cached action
representation with current projected visual tokens that are already computed
before the reuse decision?

### 5.2 Information flow

At query $t$, the existing stack already computes current visual encoder and
projector outputs $V_t$. The accelerator then reuses selected downstream
visual K/V entries, producing a cached action representation
$H_t^{cache}$. The proposed adapter $C_\phi$ receives:

- current projected visual tokens $V_t$;
- cached action-token hidden states $H_t^{cache}$;
- proprioception and previous executed action state;
- instruction context already available in the action representation; and
- compact exact cache provenance/age descriptors.

It predicts a bounded residual on the action representation or action chunk:

\[
\hat a_t^{adapt}
= a_t^{cache}
+ C_\phi\!\left(V_t,H_t^{cache},s_t,a_{t-1},p_t\right).
\]

The dense VLA remains the teacher and the fail-closed reference. The initial
study should freeze one or two physically validated reuse profiles; learning a
new selector at the same time would confound whether correction itself works.

### 5.3 Training target

Training examples must be produced by the real recursive cache implementation,
not by independently replacing one clean tensor. For a valid expert action
$a_t^*$, dense action $a_t^{dense}$, and adapted cached action
$a_t^{adapt}$, a suitable research objective is:

\[
\mathcal L
= \lambda_{demo}\,\|a_t^{adapt}-a_t^*\|_1
+ \lambda_{teacher}\,\|a_t^{adapt}-a_t^{dense}\|_1
+ \lambda_{id}\,\|C_\phi(V_t,H_t^{dense},\ldots)\|_1.
\]

The identity term prevents the adapter from degrading fresh/dense inputs. A
later version may add a prespecified tail-sensitive term, but the method should
not be rescued by tuning a CVaR objective after seeing held-out results.

### 5.4 What is actually new

The provisional contribution is **not** fresh vision, stale-feature learning,
an action adapter, DAgger, or caching individually. It is their scoped
instantiation for:

1. internally mixed-age, token- and layer-resolved K/V state rather than a
   uniformly delayed backbone feature;
2. reuse of the VLA's already-computed current projected tokens rather than a
   second edge vision encoder;
3. correction at the single-system OpenVLA-OFT action interface while keeping
   the 7B policy frozen; and
4. training/evaluation on the exact recursive cache contracts whose failure
   was measured by SAVR through PAIR.

This is **provisional, moderate novelty**. A refreshed collision audit and
advisor judgment are required before claiming it as a paper contribution.

## 6. Why this has a credible positive mechanism

1. **It supplies missing information.** PAIR tried to predict damage from
   summaries. This adapter receives current task-rich visual tokens and can
   correct the served action directly.
2. **The current visual path is already paid for.** The project measured the
   visual backbone/projector at about 15.9% of full-refresh time and the
   downstream path at about 83.7%, leaving meaningful downstream reuse headroom.
3. **The speed substrate is measured locally.** BRACE and PAIR P3R showed
   physical downstream savings before learning.
4. **External evidence supports current grounding.** CloudEdgeVLA's ablation
   found that current vision, not merely a larger stale representation, was the
   decisive factor under delay.
5. **The module can be trained without updating the 7B backbone.** Cached
   dense/stale features can be generated in forward-only collection, then the
   small adapter can be trained offline.
6. **Recursive training addresses the observed distribution shift.** DAgger,
   Latent Bridge, and HarmoniCa independently support training on deployment-
   induced histories rather than clean one-step pairs.

These are reasons to test the method, not evidence that it will pass closed-
loop evaluation.

## 7. Feasibility under current resources

The official OpenVLA-OFT documentation reports approximately 15.9 GB for
LIBERO inference but at least 25.7 GB for its minimum two-camera/proprioceptive
LoRA training configuration. A standard backbone LoRA run therefore does not
fit one 24 GB TITAN GPU under the project's strict memory boundary.

The recommended direction avoids that requirement:

- all 7B VLA passes are forward-only and use the already validated inference
  envelope;
- only a small adapter is optimized;
- feature collection and adapter training are separated;
- adapter training can run on stored, bounded tensors without the VLA loaded;
- evaluation remains on one GPU with every adapter cost included in complete-
  cycle latency; and
- multi-GPU sharding, system changes, and a new timing methodology are not
  required for the first feasibility test.

Storage volume, adapter width, token pooling, and exact tensor tap points must
be measured before freezing a protocol. Raw full-layer tensors should not be
logged indiscriminately; the data design must retain only the representations
needed by the candidate adapter.

## 8. Main failure modes

| Risk | Why it matters | Earliest honest test |
|---|---|---|
| Current projected tokens plus cached hidden state still lack enough information | Then no small action adapter can reconstruct dense behavior. | Held-out offline dense-action recovery on exact recursive cache states. |
| Adapter memorizes tasks or trajectory phase | LIBERO has repeated layouts and instructions. | Complete-trajectory splits, task-held-out/no-instruction ablations, and suite-balanced reporting. |
| Offline recovery does not transfer closed loop | Small action errors can change future states. | One frozen on-policy development pilot, followed by at most one predeclared aggregation round. |
| Adapter latency consumes the reuse gain | A large cross-attention module can erase the measured 12–17% headroom. | CUDA-synchronized physical microbenchmark before closed-loop outcomes. |
| Identity path degrades dense behavior | A corrector may change actions even when no correction is needed. | Exact fresh-control and dense non-inferiority checks. |
| Recursive cache drift exceeds adapter range | Correcting actions does not refresh internal K/V. | Fixed maximum source age plus mandatory dense reset; horizon-stratified recovery. |
| Tail failures remain despite lower mean error | Mean L1 can hide rare contact-critical errors. | Prespecified high-quantile/CVaR diagnostics and closed-loop paired success; diagnostics cannot replace success. |
| Closest-work collision | CloudEdgeVLA and Latent Bridge are close conceptual neighbors. | Repeat primary-source search and write an explicit contribution comparison before implementation freeze. |
| Strong recent baselines lack runnable code | LAC and Action-JND cannot currently be compared faithfully. | Use official VLA-Cache and project baselines; narrow claims and never present a reimplementation as official. |
| High dense baseline leaves little success headroom | A positive paper may need non-inferior success plus speed rather than higher absolute success. | Freeze a reliability–efficiency frontier claim, not a guaranteed success-rate increase. |

## 9. Candidate-family decision

Scores are 1 (weak) to 5 (strong) and are comparative judgments, not success
probabilities.

| Candidate | Novelty | Fit to our failure | Positive mechanism | Local feasibility | Decision |
|---|---:|---:|---:|---:|---|
| Reimplement LAC/Action-JND | 1 | 4 | 4 | 3 | Reproduction, not central contribution |
| Another PAIR-style risk router | 2 | 5 | 1 | 5 | Reject after P4B |
| Generic feature/KV delta predictor | 1 | 5 | 5 | 2 | Reject as Latent Bridge collision |
| Full cache-robust VLA LoRA | 2 | 5 | 4 | 1 | Reject locally; standard training exceeds one-GPU memory |
| Confidence/conformal fallback only | 1 | 3 | 2 | 4 | Secondary safeguard, not solution |
| Randomized cache attribution | 4 | 4 | 2 | 2 | Diagnostic research, not strongest positive route |
| Actuation-slack dense refresh | 1 | 5 | 5 | 1 | Externally validated but does not fit measured local window |
| **Fresh-grounded mixed-age cache adapter** | **3** | **5** | **4** | **4** | **Best candidate; conditional feasibility study** |

## 10. Research-gated next step

Do **not** write a full multi-phase execution protocol yet. First create a
small design-and-feasibility protocol that answers three questions without
opening a large outcome-search loop:

1. Does a small adapter materially reduce held-out dense-action error on exact
   recursive cache states relative to uncorrected reuse and an MLP-only control?
2. Does its measured inference overhead preserve a conservative net speed gain?
3. Does one frozen small closed-loop pilot preserve or improve paired task
   success at that measured speed point?

Failure of any question ends this direction. Passing all three would justify a
full protocol with independent confirmation, strong baselines, sealed outcomes,
and a manuscript-level method specification.

## 11. Bottom-line judgment

- **Is cache-aware learning worth pursuing?** Yes, but not as a generic learned
  gate.
- **Is the strongest candidate guaranteed to produce a positive paper?** No.
- **Is it more plausible than continuing PAIR?** Yes, because it changes the
  information available to the accelerated action path rather than trying to
  predict invisible damage from the same weak proxies.
- **Is it feasible on TITAN?** A small frozen-backbone adapter is plausible;
  standard OpenVLA-OFT LoRA training is not within one 24 GB GPU.
- **Is it novel?** Provisionally and moderately, only under the exact mixed-age
  K/V, already-computed-fresh-token, frozen-single-system formulation. The
  novelty claim must remain narrow.

The appropriate next artifact is a bounded feasibility protocol for this
candidate—not another broad positive-results roadmap and not immediate GPU
execution.
