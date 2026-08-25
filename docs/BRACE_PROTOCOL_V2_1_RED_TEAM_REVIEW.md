# Red-Team Review of BRACE Execution Protocol V2.1

**Date:** 2026-08-25

**Reviewed artifact:** `docs/BRACE_EXECUTION_PROTOCOL_V2_1.md`

**Review scope:** causal validity, treatment/deployment alignment, simulator
replay, cache provenance, baseline fidelity, novelty, router calibration,
statistics, population leakage, timing, memory, cost, server safety, and likely
reviewer objections

**Evidence boundary:** No BRACE implementation, model run, GPU operation,
policy outcome, or new simulator population was produced

**Formal-method audit:** `docs/BRACE_FORMAL_METHOD_AUDIT_V1.md`

## 1. Final verdict

V2.1 is coherent enough to authorize B1 only. The red-team pass found and
corrected material issues that V2 missed: an invalid released-baseline evaluator,
the exact-stack VLA-ADP competitor, unsupported pointwise risk language,
multi-contract overlap, brittle one-profile selection, sampling-weight bias,
vague duplicate controls, comparator inconsistency, and incomplete power/cost
rules. The final consistency pass additionally fixed the action-disagreement
baseline's deployment boundary and sealed-analysis verification.

No remaining defect found in this review invalidates B1. The expensive BRACE
hypothesis remains high risk. A strong protocol improves the credibility of any
result; it does not make terminal cache harm easier to predict.

Revised judgment ranges:

| Event | Plausibility after V2.1 |
|---|---:|
| B1 replay acceptance | 80--90% |
| B2 faithful baseline/cache/provenance integration | 45--60% |
| B3 at least one clean profile with 8% net cycle speed | 30--45% |
| B4 affordable usable profile/harm population | 20--35% |
| B5 useful held-out empirical risk--coverage | 15--30% |
| Any favorable BRACE method result | 25--40% |
| Competitive positive paper from the complete program | **15--25%** |
| Competitive paper conditional on B1--B5 all passing | **40--55%** |

These are calibrated research judgments, not statistical probabilities. The
previous 25--40% paper estimate is superseded.

## 2. Exhaustive defect-resolution matrix

| Domain | Defect or attack | V2.1 disposition | Residual status |
|---|---|---|---|
| Treatment | Single cached query did not match cumulative deployment | Contracts test complete 1/2/4-query bursts | Resolved |
| Gate provenance | Accelerated attention could recursively define the gate | Deployable gates come only from dense anchors | Resolved, but anchor staleness remains empirical |
| Cache state | Scalar age hid mixed token sources | Per-layer/per-token source ledger and source-image ring buffer | Resolved technically; B2 must prove mapping |
| Decoder mechanics | Arbitrary per-layer masks ignored that removed hidden states cannot return | Nested reuse sets and nondecreasing layer budgets | Resolved in formal design; B2 unproven |
| Token layout | Fixed prompt/text spans could silently target wrong positions | Runtime-derived and digested sequence map | Resolved in formal design |
| Context dependence | Static pixels were implicitly treated as representation-static despite bidirectional attention | Pre-outcome proprio/action envelopes, context perturbation tests, terminal supervision | Mitigated; central empirical risk remains |
| Source identity | Image/source age omitted the nonvisual context that produced a cached K/V | Immutable multimodal source records and exact-source drift checks | Resolved formally; B2 unproven |
| Anchor attention | Requesting attention may force SDPA to eager execution | Detached sidecar Q/K map requires dense action/backend/timing parity | Resolved as a gate |
| VLA-Cache fidelity | Released evaluator aliases current/previous frames and suppresses errors | Minimal pinned correction, explicit naming, source/hash/parity audit | Resolved in protocol; implementation unproven |
| Comparator | V2 omitted released exact-stack VLA-ADP, VLA-Pruner, and SpecPrune-VLA | Mandatory executable baselines | Resolved |
| Comparator rule | “And” versus “or” created an inconsistent success criterion | One explicit Pareto/non-domination hierarchy | Resolved |
| Profile search | One speed-only profile risked missing the viable frontier | At most six outcome-blind profiles; development-only elimination | Resolved with bounded multiplicity/cost |
| Risk claim | Per-state upper probability bound was not supportable | Empirical grouped selective-risk control only | Resolved |
| Multi-arm identification | Profile assignment overlap/propensity was absent | Known nonzero assignment and arm-order probabilities | Resolved |
| Joint selection | Separately calibrated contracts do not calibrate fastest-accepted selection | Calibrate and evaluate the full deployed router | Resolved statistically; sample size remains empirical |
| Abort estimand | Immediate re-routing after an experimental abort changes the assigned treatment | FR fills the remaining fixed horizon; repeated routing is tested on policy | Resolved |
| Enrichment | Enriched harmful samples could bias natural risk | Separate representative stratum and design-consistent weighting | Resolved |
| Controls | “Negligible” duplicate discordance was undefined | Any unexplained terminal discordance stops collection | Resolved |
| Distribution shift | FR-continuation labels may drift under BRACE | One versioned aggregation round and explicit drift gate | Mitigated, not eliminated |
| Future motion | Router cannot observe motion after contract entry | ITT includes aborts; horizons bounded; unpredictable harm stops method | Irreducible empirical risk |
| Features | Current dense-VLA signals would erase acceleration | Explicit pre-inference allowlist and dense-feature prohibition | Resolved |
| Action comparator | “Action-disagreement router” could imply running FR during deployment | Disagreement supplies development labels only; deployed comparator uses the same timed pre-inference allowlist | Resolved |
| Model selection | One optimization run could make the router result seed-sensitive | Bounded validation-only search, at least five fixed seeds, frozen selection/ensemble rule | Resolved |
| Timing | Accelerated-query timing hid anchors/router/expiry | Amortized contract-cycle and complete overhead required | Resolved |
| Slack baseline | P4 timing could be simulated incorrectly | Exact same-frame, one-chunk-old pipeline with spillover/total work | Resolved |
| Memory | Cache/provenance estimates could miss fork/runtime overhead | 23 GiB measured cap and B3 authority | Resolved as a gate |
| Dependency drift | Transformers forks differ materially | Isolated runtime, parity tests, no proven-env overwrite | Resolved as a gate |
| Statistics | Fixed 500 episodes ignored discordance and suites | Paired power with clustering, multiplicity, attrition, true effect | Resolved |
| Analysis integrity | A frozen plan alone did not verify the implementation | Synthetic known-answer tests and independent primary-result recomputation before claims | Resolved |
| Multiplicity | Profiles/contracts/suites allowed optimistic selection | Development freeze plus hierarchical family-wise testing | Resolved |
| Unit of analysis | Branch rows from one episode are dependent | Complete episode grouping for every split/interval | Resolved |
| Protected data | Profile/threshold tuning could consume final groups | Explicit development/final separation and freeze | Resolved |
| Cost | Harmful-class target could exceed 40 hours | Sequential 20/40/80 checkpoints and B4a-measured scaling | Resolved as a stop rule |
| Novelty | Adaptive/action-aware routing already exists | Narrow terminal-intervention/cache-contract claim | Provisionally distinct, moving field |
| Scope | Simulation could be overstated as safety/general robotics | Checkpoint/LIBERO/simulator-only claim boundary | Resolved |
| Shared server | Research could affect unrelated university work | Project-only paths, one coordinated GPU, no process inspection | Resolved |

## 3. Causal and label validity

The intervention is defensible if B1 passes. Fresh reset plus complete action
prefix replay establishes the same published initial condition and behavioral
history without pretending that MuJoCo coordinates are a full system snapshot.
Running both treatment arms from reconstructed dense anchors exposes both
potential outcomes in the deterministic simulator.

The formal reconstruction requires replay to the historical anchor, a valid
anchor cache and action, execution of that common anchor chunk, and branching
at the following query. Creating a new anchor at the treatment observation
would test a different intervention.

The contract, not one query, is the treatment. Known contract assignment and
arm-order probabilities prevent profile selection from being confounded with
task or apparent difficulty. Common versioned continuation makes the effect
policy-specific but interpretable.

Remaining limits:

- terminal success is discontinuous and task-specific;
- labels under FR continuation need not equal labels under BRACE continuation;
- deterministic simulation does not establish physical-robot reliability;
- a contract may become harmful only because of future observations unavailable
  when it was selected.

These are now explicit hypotheses/stop conditions rather than hidden
assumptions.

## 4. Physical-speed feasibility

For accelerated-query fractional reduction `d` and horizon `h`, the ideal
anchor-cycle reduction before router overhead is `d*h/(h+1)`. Reaching 8%
therefore requires at least:

| Horizon | Minimum accelerated-query reduction before overhead |
|---|---:|
| 1 | 16% |
| 2 | 12% |
| 4 | 10% |

Actual requirements are higher after feature extraction, routing, aborts, and
transfers. Horizon 1 is unlikely to pass unless cache acceleration is strong.
Horizons 2/4 have more speed headroom but more gate/KV staleness.

Historical FR inference near 1.267 seconds on TITAN makes lightweight routing
relatively cheap, but also makes a dense pass unlikely to fit inside the roughly
400 ms actuation window used by modern-H100 reports. P4 may therefore spill and
lose its off-path advantage on TITAN. Only B3 measurement can decide.

Memory remains plausible, not proven: historical eager peak was approximately
15.38 GiB under the 23 GiB cap, while visual K/V and provenance add bounded but
implementation-dependent storage.

## 5. Baseline and novelty audit

The comparison field is unusually strong:

- VLA-Cache reports up to 1.7x CUDA acceleration;
- Gated VLA-Cache reports nearly lossless caching on OpenVLA-OFT at its tested
  settings;
- actuation-slack refresh repairs self-harvested collapse while preserving
  sparse serve speed on faster hardware;
- VLA-ADP is published, released, OpenVLA-OFT-compatible, and reports up to
  1.35x speedup with competitive LIBERO success;
- VLA-Pruner and SpecPrune-VLA have released OpenVLA-OFT code and report strong
  token-reduction frontiers;
- LAC learns token selection and cache ratios and reports 1.76x wall-clock
  speedup with improved average LIBERO success;
- SP-VLA reports lossless LIBERO acceleration using action-aware scheduling and
  token pruning;
- AC2-VLA already combines action context, reuse, pruning, and selective layer
  execution.

BRACE cannot win by presenting learned action-aware routing or cache reuse as
new. Its remaining distinction is the paired terminal-intervention target,
cache-induced trajectory distribution, bounded clean-provenance contract, and
empirical selective-risk deployment rule as one system.

That distinction is narrow but defensible today. It becomes publication-worthy
only if outcome supervision beats VLA-ADP-style motion, image-change, confidence,
and static clean-profile baselines. A novelty search is mandatory again before
B2, B7, and submission.

## 6. Data and statistical feasibility

The primary data risk is rare harm. To collect 190 harmful examples:

| Natural harmful prevalence | Paired contracts before controls/attrition |
|---|---:|
| 2% | 9,500 |
| 5% | 3,800 |
| 10% | 1,900 |
| 15% | 1,267 |

Historical episode throughput does not convert exactly to branch-pair cost.
Each pair has two continuations, dense-anchor reconstruction, prefix replay,
controls, and possible attrition. B4a measurement is therefore the only valid
scaling basis.

For a two-point paired non-inferiority margin with true difference zero, a
one-sided 95% normal approximation requires roughly:

| Paired discordance | Episodes before multiplicity/attrition |
|---|---:|
| 2% | 136 |
| 5% | 339 |
| 7.5% | 508 |
| 10% | 677 |
| 20% | 1,353 |

If BRACE is truly 0.5 points below FR, the remaining 1.5-point margin needs
about 1,203 episodes at 10% discordance. Per-suite claims and group dependence
can require more. V2.1 appropriately refuses to assume 500 is sufficient.

## 7. Failure scenarios that can still end BRACE

### 7.1 Technically implementable but not fast

Clean provenance may require enough anchors and recomputation that corrected
VLA-Cache or VLA-ADP remains faster. This likely stops at B3.

### 7.2 Fast and already reliable

Moderate clean caching may preserve success so well that harmful examples are
too rare. A static clean profile could then be a positive engineering result,
but not evidence for the BRACE router.

### 7.3 Harmful but unpredictable

Longer contracts may create labels, while cheap current-state features cannot
foresee future camera/contact dynamics. This is the most important B5 failure.

### 7.4 Predictable but no net frontier gain

The router may refresh often enough that corrected VLA-Cache, Gated VLA-Cache,
VLA-ADP, or P4 dominates its success/latency/work point.

### 7.5 Development success fails confirmation

Profile, task, or suite heterogeneity may make the selected operating point
unstable. Grouped sealed confirmation and multiplicity control prevent this
from becoming an optimistic paper claim.

### 7.6 Novelty preemption

The area is moving weekly. A new paper could combine outcome supervision and
cache routing before submission even if the current search is clear.

## 8. Likely reviewer objections and required evidence

1. **Why not VLA-ADP, VLA-Pruner, SpecPrune-VLA, or Gated VLA-Cache?** Run them
   faithfully on the same stack and show the complete frontier.
2. **Why terminal rollouts instead of action similarity?** Demonstrate superior
   held-out risk--coverage and show examples where action similarity misses
   terminal harm.
3. **Is the VLA-Cache comparison official?** No. Disclose the two evaluator
   corrections, preserve the patch, and call it faithfully corrected.
4. **Does the risk score guarantee safety?** No. Report an empirical grouped
   population bound and reject any pointwise/physical-safety wording.
5. **Was the best profile selected after seeing the test set?** Preserve the
   development grid and untouched sealed population.
6. **Is enrichment hiding natural prevalence?** Report representative results
   separately and use logged inclusion probabilities.
7. **Why one checkpoint and simulator?** State the scope limitation or add
   another compatible checkpoint/benchmark only under a new resource plan.
8. **Is labeling cost practical?** Report GPU-hours per harmful example and
   total branch-pair cost.

## 9. Final go/no-go

- **GO:** B1 CPU replay harness. It is low cost and useful even if BRACE fails.
- **CONDITIONAL GO:** B2 only after B1 acceptance and a frozen baseline patch/
  dependency plan.
- **NO-GO WITHOUT SEPARATE AUTHORIZATION:** downloads, B3 GPU/model work, B4+
  policy outcomes, or protected populations.

This review found no additional protocol correction required before B1. It
cannot prove that no future implementation bug, empirical failure, or newly
published collision will appear. Those uncertainties are precisely why V2.1
uses phase gates rather than promising a positive outcome.

## 10. Primary sources rechecked

1. VLA-Cache: <https://openreview.net/forum?id=QZYZ0Xm58q>
2. The Gate, Not the Cache: <https://arxiv.org/abs/2608.00391>
3. Gated VLA-Cache: <https://arxiv.org/abs/2608.10824>
4. VLA-ADP: <https://arxiv.org/abs/2509.22093>
5. VLA-ADP code: <https://github.com/chen7086/VLA-ADP>
6. LAC: <https://arxiv.org/abs/2602.00686>
7. SP-VLA: <https://arxiv.org/abs/2506.12723>
8. SP-VLA code: <https://github.com/ChildTang/SP-VLA>
9. AC2-VLA: <https://arxiv.org/abs/2601.19634>
10. Action-JND: <https://arxiv.org/abs/2608.21247>
11. FlashVLA: <https://arxiv.org/abs/2505.21200>
12. EfficientVLA: <https://arxiv.org/abs/2506.10100>
13. VLA-Pruner: <https://arxiv.org/abs/2511.16449>
14. VLA-Pruner code: <https://github.com/MINT-SJTU/VLA-Pruner>
15. SpecPrune-VLA: <https://arxiv.org/abs/2509.05614>
16. SpecPrune-VLA code: <https://github.com/alexwhz-sjtu/SpecPrune-VLA>

Recent preprints and repositories remain provisional and require revalidation
at every scientific freeze.
