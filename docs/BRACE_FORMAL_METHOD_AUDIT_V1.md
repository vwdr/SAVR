# Red-Team Audit of BRACE Formal Method Specification V1

**Date:** 2026-08-25

**Reviewed artifact:** `docs/BRACE_FORMAL_METHOD_SPECIFICATION_V1.md`

**Evidence boundary:** Design and primary-source review only; no BRACE code,
model, GPU, branch, simulator outcome, or positive result was produced

## 1. Verdict

The formal specification is detailed enough to guide B1 and, conditionally,
later implementation without inventing cache mechanics during coding. It does
not raise the current 15--25% pre-gate probability of a competitive positive
paper. Its value is that several previously hidden ways to obtain a false
positive or an unimplementable design are now explicit stop conditions.

No unresolved issue below invalidates CPU-only B1. Several can end BRACE at B2
through B5.

B1 verifies simulator replay through a scripted anchor-action boundary only.
It cannot verify a real VLA anchor or K/V cache without a model/GPU; that parity
is intentionally deferred to B3.

## 2. Problems found and corrections made

| Problem discovered during formalization | Why the earlier idea was flawed | Formal correction |
|---|---|---|
| Arbitrary layerwise reuse masks | The corrected VLA-Cache forward drops hidden states; a dropped token cannot return deeper in the decoder | Reuse sets must be nested and budgets nondecreasing |
| Layer-local eligibility | A token accepted early could be required for recomputation at a deeper layer after its hidden state was already removed | Require suffix eligibility across every deeper pruning layer |
| Fixed token spans | Official utilities assume fixed visual/text positions that can change with prompt/checkpoint | Derive and digest the complete runtime sequence map |
| “Dense attention is free” | Requesting attention may switch backend, change numerics, or add large anchor cost | Attention gate requires dense output/backend/timing parity or is removed |
| Static pixels imply valid K/V | OpenVLA-OFT uses bidirectional multimodal attention, so deeper visual states can depend on current proprio/action context | Outcome-blind robot/action drift envelopes plus terminal-effect supervision; isolated-context tests required |
| Image-only provenance | A reused K/V entry was produced from an entire multimodal source context, not only its image | Immutable source records and drift checks against every live K/V source |
| Re-running an anchor at treatment time | It changes the K/V source and shifts the intervention | Reconstruct the historical anchor, execute its chunk, then branch at the next query |
| Re-routing after an experimental abort | The assigned treatment would silently become a different dynamic policy | Fill the remaining assigned horizon with FR; test repeated routing separately on policy |
| One cache age | Different layers/tokens retain different sources after progressive reuse | Layer/token source ledger and source-image ring buffer |
| Independent contract calibration | Choosing the fastest among multiple accepted contracts introduces selection bias | Calibrate the complete joint routing rule |
| Harm score interpreted as probability | Class weighting, enrichment, and propensity clipping invalidate naive probability meaning | Treat output as ranking; estimate risk on representative groups |
| Large image/router network | Harm labels are likely scarce and routing overhead can erase speed | <=128 engineered features, <10k-parameter MLP, logistic challenger only |
| Assumed monotonic risk | More image change/age/reuse need not be pointwise harmful; beneficial branches exist | No monotonic architectural constraint; trends are diagnostics |
| Cheap image difference treated as exact | Low-resolution or CPU patch comparison can misalign tokens and hide latency | Exact preprocessed patch map for execution; approximations only as router features |
| Undefined zero-patch cosine | Constant or zero-valued patches can produce NaN or arbitrary eligibility | Frozen epsilon/zero convention, clamping, and nonfinite fail-closed tests |
| Per-contract label treated as fully observed | Each sampled state exposes only one randomized contract outcome | Assignment/inclusion propensities and overlap diagnostics |
| Unnormalized IPW population formula | Dividing a weighted inclusion sample by its observed row count does not generally recover the target population | Primary normalized Hájek estimates; HT only with known frame size |
| Rare-event bootstrap treated as a guarantee | Zero observed harm can collapse a resampling interval even when uncertainty is large | Freeze one policy, collect direct independent calibration branches, and use an exact binomial upper limit |
| Outcome-supervised router assumed useful | Terminal harm can be rare, discontinuous, or caused by future motion absent from current features | Sequential prevalence/predictability gates remain terminal stop rules |

## 3. Architecture feasibility

The cache substrate is implementable in principle because the released
VLA-Cache fork already removes selected hidden states and preserves their K/V at
original positions. The formal design follows this mechanism rather than
inventing arbitrary per-token skipping.

The most serious B2 risks are:

1. the project's Transformers 4.40.1 fork and VLA-Cache's 4.47.0 fork differ;
2. `DynamicCache` mutates in place and can contaminate branch arms;
3. attention capture may require a different kernel/backend;
4. visual K/V may change under nonvisual context even for identical pixels;
5. progressive token removal can break masks or action-position extraction;
6. two-camera indices can be confused; and
7. mixed-source eligibility requires exact layerwise and multimodal-source
   provenance; and
8. experimental abort handling must not start a different treatment inside the
   assigned horizon.

These are testable. None should be “fixed” by accepting approximate P0 parity
after outcomes.

## 4. Learning feasibility

The router architecture is deliberately smaller than modern learned pruning
controllers because the expected branch dataset is much smaller. This improves
sample efficiency and deployment cost but may reduce predictive power.

The terminal target has one genuine advantage: it directly labels silent
closed-loop harm that action distance can miss. It also has three major
disadvantages:

- high labeling cost;
- rare and discontinuous positive labels; and
- dependence on future motion and the continuation policy.

Therefore, terminal supervision is not assumed superior. It must beat matched
metadata, image-motion, previous-action, and action-disagreement supervision.
If it does not, the claimed BRACE contribution fails even if the router itself
looks accurate.

## 5. Causal validity

Paired reset-prefix-anchor reconstruction is valid only if both arms reach the
same pre-treatment state and cache history. “Same MuJoCo state” is insufficient.
The required equality covers controller goals, observable clocks/caches,
wrappers, action queues, cameras, dense action, sequence positions, K/V cache,
and provenance.

A cache-contract abort is part of the assigned treatment. The paired branch
uses FR for every remaining query in that fixed treatment window rather than
re-routing. Ordinary deployment may route again from the new anchor; the B6/B7
closed-loop phases test that repeated-decision policy and expose any mismatch
from the local branch estimand.

The causal effect is local to:

- the sampled behavior-mixture distribution;
- the assigned bounded contract;
- the frozen continuation policy; and
- the deterministic simulator/version.

It is not a general causal claim about caching, real robots, or a future BRACE
version. The one aggregation round measures rather than assumes stability under
cache-induced state distribution shift.

## 6. Calibration and statistical risks

Propensity-aware evaluation is necessary but may have prohibitive variance.
The joint policy `pi_lambda` can select a contract whose randomized assignment
probability was small. Required diagnostics are:

- minimum propensity and overlap by contract/feature stratum;
- maximum and distribution of weights;
- effective sample size;
- clipped versus unclipped sensitivity;
- representative versus enriched estimates; and
- task/episode cluster influence.

The one-point development risk budget is intentionally stricter than the final
two-point task-success margin, but the units are different: local branch harm
does not add linearly to episode success loss under repeated dependent routing.
It may yield zero useful coverage and cannot be advertised as allocating half
of the final margin. B7/B8 supply the episode-level test. Relaxing the local
budget after seeing results would still invalidate the positive-paper route.

A 1% one-sided upper bound is also data-hungry. Even a direct zero-event sample
needs 299 independent episode groups for a one-sided 95% exact binomial limit.
The corrected design therefore freezes one policy on validation and evaluates
that policy prospectively on one query boundary per independent calibration
episode. This avoids the much larger importance-weighted multi-policy sample,
but B5 must still stop if the 40-GPU-hour cap cannot support it.

The formal deployable-set cap of three contracts is important. With as many as
18 screened profile/horizon combinations, randomized assignment would leave too
few outcome matches for a joint router and produce extreme inverse-propensity
variance. Screening may study the larger grid, but the learned deployed policy
must remain small.

Architecture selection and threshold calibration require disjoint grouped
populations. This increases the B5 harmful-class target and may make the
40-GPU-hour cap infeasible. Reusing validation or held-out outcomes would create
a more flattering but invalid risk--coverage curve; cost infeasibility is a
stop, not a reason to merge these roles after seeing results.

The empirical upper bound is not advertised as conformal, distribution-free,
or pointwise because weighted clustered adaptive-policy selection violates the
simple exchangeability setting of standard conformal risk control. Final paired
closed-loop non-inferiority remains necessary.

The rare-event bootstrap and Thomas--Theocharous--Ghavamzadeh HCOPE calculation
remain development sensitivity analyses, not the acceptance certificate.
Random assignment across contracts makes a 1% simultaneous OPE bound extremely
sample-inefficient. The frozen single-policy calibration instead uses direct
paired Bernoulli outcomes and an exact Clopper--Pearson upper limit. If it fails,
the method stops; it cannot select the next-most-favorable validation policy on
the same calibration population.

## 7. Physical speed and memory

The router itself should be cheap, but exact source-difference maps,
provenance checks, DynamicCache updates, and dense anchors may erase the gain.
The relevant quantity is the complete anchor-plus-contract cycle, including
aborts—not an accelerated kernel in isolation.

Nested reuse can create physical speed because it reduces Q/K/V, attention-
query, output, and MLP work for removed visual hidden states. However:

- small token reductions may remain memory/launch bound;
- position-preserving cache updates may synchronize;
- mixed-source bookkeeping may increase memory traffic;
- horizon one needs especially strong per-query reduction; and
- attention-gate collection may make anchors slower than FR.

The bidirectional-attention finding is a central scientific risk: image-static
tokens are not necessarily representation-static. The robot/action envelope
can reject large context changes, but only the paired terminal experiment can
show whether the remaining reuse is reliable. If the envelope must reject most
queries, the router cannot deliver a positive efficiency result.

B3 must measure these effects on the actual TITAN GPU. No FLOP equation or
published RTX 4090/H100 result substitutes for that evidence.

## 8. Novelty and competitor pressure

BRACE cannot claim adaptive token selection, action-aware routing, temporal
reuse, selective prediction, or learned cache ratios. VLA-ADP, VLA-Pruner,
SpecPrune-VLA, SP-VLA, LAC, AC2-VLA, Gated VLA-Cache, and VLA-Cache already
occupy those areas.

The remaining potentially novel unit is narrow:

> selecting bounded, clean-provenance visual-KV contracts from replay-verified
> paired terminal intervention effects on cache-induced trajectories.

That claim becomes meaningful only if terminal supervision creates a better
measured reliability--efficiency frontier than cheaper supervision and the
strong exact-stack methods. A method that merely reproduces VLA-ADP's action
gate with more expensive labels is not a positive BRACE paper.

## 9. Likely failure modes by phase

| Phase | Failure | Interpretation |
|---|---|---|
| B1 | Replay/anchor state differs | Counterfactual labels are invalid; stop |
| B2 | P0 or sequence/cache parity fails | Substrate is not faithful; stop or redesign before outcomes |
| B2 | Attention gate changes dense path | Remove attention profiles or stop if none remain meaningful |
| B3 | Nested clean profiles are not fast | No physical premise; stop |
| B4 | Harm is nearly absent | Static clean cache may be enough; router claim unsupported |
| B4 | Harm is common | Contracts are not reliable enough for useful coverage |
| B5 | Cheap features cannot rank harm | Future-dependent failures are not pre-forward predictable; stop |
| B5 | Joint-rule weights collapse | Offline calibration is unidentifiable; collect under better overlap or stop |
| B6 | Effect labels shift under BRACE | FR-continuation supervision does not transfer; stop |
| B7/B8 | Strong baseline dominates | No competitive positive BRACE paper |

## 10. What would constitute a flawed positive result

Reject any apparent positive result that relies on:

- comparing only with FR;
- using the uncorrected VLA-Cache evaluator;
- reporting FLOPs or served-only latency instead of complete ITT cycles;
- using current dense logits/attention to route without timing them;
- tuning profiles or thresholds on confirmation outcomes;
- ignoring aborted/failed assignments;
- treating enriched prevalence as natural prevalence;
- calibrating contracts separately but deploying a joint argmin router;
- calling nonsignificance equivalence;
- hiding a failed exact-stack competitor integration;
- calling empirical simulator reliability “safety”; or
- claiming novelty for action-aware caching alone.

## 11. Final assessment

The method is now formal enough to code without unresolved mathematical intent.
It is still an explicitly high-risk research direction. The current estimates
remain:

- B1 acceptance: 80--90%;
- competitive positive paper before gates: **15--25%**; and
- competitive paper conditional on B1--B5 all passing: **40--55%**.

These are judgments, not measured probabilities. The formalization improves
validity and stop-fast efficiency; it cannot guarantee favorable empirical
behavior.

## 12. Rechecked primary sources

1. OpenVLA-OFT: <https://www.roboticsproceedings.org/rss21/p017.html>
2. VLA-Cache: <https://openreview.net/forum?id=QZYZ0Xm58q>
3. VLA-Cache code: <https://github.com/siyuhsu/vla-cache>
4. Gated VLA-Cache: <https://arxiv.org/abs/2608.10824>
5. VLA-ADP: <https://openreview.net/forum?id=ea6j8k8Rnw>
6. VLA-ADP code: <https://github.com/chen7086/VLA-ADP>
7. VLA-Pruner: <https://github.com/MINT-SJTU/VLA-Pruner>
8. SpecPrune-VLA: <https://github.com/alexwhz-sjtu/SpecPrune-VLA>
9. SP-VLA: <https://openreview.net/forum?id=RwdGIIjPlC>
10. LAC: <https://arxiv.org/abs/2602.00686>
11. AC2-VLA: <https://arxiv.org/abs/2601.19634>
12. Selective classification: <https://papers.neurips.cc/paper_files/paper/2017/hash/4a8423d5e91fda00bb7e46540e2b0cf1-Abstract.html>
13. Off-policy evaluation: <https://proceedings.mlr.press/v70/wang17a.html>
14. DAgger: <https://proceedings.mlr.press/v15/ross11a.html>
15. High-confidence off-policy evaluation:
    <https://ojs.aaai.org/index.php/AAAI/article/view/9541>

No further method correction is currently required before B1. Re-audit is
mandatory after B1 reveals actual simulator replay semantics and after B2
reveals the integrated cache behavior.
