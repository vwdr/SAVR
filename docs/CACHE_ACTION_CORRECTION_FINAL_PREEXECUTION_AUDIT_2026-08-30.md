# Cache Action Correction Final Pre-Execution Audit

**Date:** 2026-08-30  
**Scope:** scientific logic, architecture, data, statistics, systems, novelty,
reproducibility, and paper viability  
**Disposition:** CAC remains a plausible bounded study; Protocol V1 is
superseded by `CACHE_ACTION_CORRECTION_EXECUTION_PROTOCOL_V2.md`

## 1. Audit standard

This review did not ask whether CAC was merely implementable. It asked whether
the proposed experiment could distinguish a real cache-specific correction
effect from generic imitation, leakage, favorable development noise, technical
artifacts, or an underpowered final comparison.

The audit re-read the complete V1 protocol, project status, population ledger,
PAIR P1/P3R/P4B evidence, actual OpenVLA-OFT action-head implementation, and the
physical source tracker. It also repeated the primary-source novelty search.
No model, GPU, simulator, or protected outcome was used.

## 2. Final findings and V2 resolutions

| Area | V1 problem | Why it matters | V2 resolution |
|---|---|---|---|
| Action representation | Pooled seven action-token states into one step token | The frozen action head concatenates all seven coordinate-token states; pooling destroys ordered information | Hook the existing post-`layer_norm2`, pre-`fc2` `8 x 4096` action-head representation |
| Source identity | One apparent source token/delta per tile | A tile can have different physical sources at onset layers 2/6/9/11 | Use 128 layer-resolved source-delta tokens and exact per-group provenance |
| Instruction context | Current-only comparator was undefined and could omit language | A task-agnostic controller without instruction is structurally disadvantaged | Every learned model receives the same instruction embedding; generic comparator removes only cache-specific inputs |
| Comparator fairness | Ablations could be interpreted as inference-time masking | Masking a trained full model is distribution shift and not a fair comparator | Train every comparator/ablation independently with matched budget and learned null tokens |
| Output support | Bounded residual did not bound the final action | An extreme cached action plus a bounded residual can remain outside training support | Add frozen per-step/coordinate residual bounds, final support projection, and activation-rate gates |
| Gripper | Continuous loss alone did not match downstream binary semantics | Small continuous changes around the threshold can reverse open/close behavior | Freeze the exact normalized threshold and use continuous plus threshold classification loss |
| Architecture selection | Smallest passing model was automatically selected | It could deliberately choose an underfit model and lower closed-loop probability | Evaluate all three on an isolated architecture-selection split; use a composite score and one-standard-error simplicity rule |
| Split reuse | C2 model selection and C3 checkpoint selection reused the same validation trajectories | Repeated selection over the same values biases estimates | Divide 35 train demos/task into 24 fit, 5 architecture selection, and 6 checkpoint validation |
| Calibration status | V1 called 320 calibration demos unopened | PAIR P4/P4B already consumed all 320 | Rename them development-calibration; reserve the 280 still-sealed locked-test demos for final offline/mechanism evidence |
| Simulator arithmetic | V1 required 6 + 4 + up to 50 disjoint initial states/task | LIBERO provides only 50 official states/task | Use IDs 0-5 for R0, 6-9 for R1 holdout, and 10-49 for the 1,600-condition final population |
| Prior exposure | V1 treated new CAC seed manifests as outcome-blind | State IDs 0-9 have been exposed across earlier project stages | Treat every ID 0-9 as development-exposed, regardless of whether CAC saw it |
| Fake independence | Reserve seeds could be counted as new trials | Repeated seeds on the same official state are clustered and may be deterministic | V2 does not use reserve seeds and caps independent final pairs at 1,600 |
| Repair opportunity | V1 trained before learning whether D62 has terminal loss to repair | A 5-point CAC gain is impossible if cache is already near dense; catastrophic cache may be unrepairable | Add C1H dense-vs-D62 terminal headroom screen before C2 |
| R1 label | V1 described dense-minus-R0 for the same residual adapter | The adapter still corrects cached base action, so its target remains dense-minus-cache | Freeze `A_D - A_C` for R0 and R1; initialize R1 from R0 rather than stack a second head |
| R1 replay | Observation identities alone were treated as replayable | R0 induces off-demo images and cache states that identifiers cannot reconstruct | Persist lossless current observations or equivalent, proprio, instruction, cached representation, provenance, and hashes server-side |
| R1 selection | On-policy records lacked an internal fit/validation split | A single on-policy training population invites checkpoint overfit | IDs 0-4 provide R1 fit records; ID 5 provides action-level checkpoint validation; IDs 6-9 stay closed-loop held out |
| Offline metric | “Normalized error” was not operationally defined | Different coordinate scales can dominate the result | Freeze training-only coordinate scaling and exact mean/first/P95/gripper definitions |
| Contract correlation | Query contracts could overlap within trajectories | Treating overlapping queries as independent understates uncertainty | Split and resample by complete trajectory; report overlapping windows but never query-level IID intervals |
| Model multiplicity | Multiple classes/seeds had no complete selection accounting | Favorable seed/model shopping could inflate development results | Freeze one configuration/class, three seeds, one selection score, and one-standard-error rule before values open |
| Gate uncertainty | Several V1 gates used only point estimates | Small development populations previously produced misleading positive point estimates in PAIR | Add trajectory/hierarchical paired intervals and explicit lower-bound requirements |
| Final multiplicity | G3-G6 had multiple directional claims without a testing order | Nominal 95% tests do not jointly control false positive risk | Use a fixed sequence: dense NI, cache repair, timing, mechanism; use Holm within ablations |
| Final power | V1 allowed 2,000 final pairs that do not exist | The 2-point NI claim can be underpowered at realistic discordance | Cap at 1,600 and require a scenario power table; explicitly state the roughly 10% discordance boundary at true equality |
| Mechanism gate | G6 lacked exact effect sizes, uncertainty, and population | It could be declared passed from weak favorable directions | Use sealed P1 locked test, exact 10% comparator and 5% ablation margins, independently trained models, and Holm bounds |
| Timing boundary | “Complete cycle” was not sufficiently operational | Excluding adapter data movement or source work creates artificial speedup | Define the timed boundary through normalized action availability and include all required cache/CAC work |
| Memory/storage | C3 had no explicit feature-store cap | Four-layer source features can consume tens of GiB | Freeze tile-level compact schema, 50 GiB C3 cap, 60k VLA-call cap, and model-unloaded adapter training |
| Recovery | V1 was both brittle and ambiguous after partial technical failure | Repeating completed units or discarding unfavorable partial data risks bias | Use append-only atomic units, sealed outcomes, resume-only missing units, and versioned outcome-blind technical recovery |
| Literature collision | V1 underweighted Action-JND and nearest residual adapters | Recent methods narrow the novelty space | Explicitly position CAC between A2C2, Latent Bridge, Action-JND/LAC, and Gated VLA-Cache; prohibit broad novelty claims |

## 3. Architecture feasibility

The corrected architecture is compatible with the actual pinned path:

- current projected visual patches already exist because PAIR's acceleration is
  downstream K/V reuse, not vision-encoder skipping;
- the physical source tracker retains the source `PreparedQuery` records needed
  to build four layer-resolved tile deltas;
- `Z_C` is already computed inside the action head and can be exposed before its
  final 4096-to-7 layer;
- the eight action queries attend to roughly 160 compact context tokens, so the
  adapter need not perform quadratic attention over the complete 512-token
  visual sequence; and
- the projected adapter is comfortably below the 8M-parameter boundary.

The main integration risk is not adapter size. It is preserving exact cache
chronology and dense-shadow isolation while collecting labels. C1 therefore
tests cache clones, tracker/salience mutation, reset identity, action-head hook
identity, source alignment, and exception cleanup before data collection.

## 4. Data and storage feasibility

Tile means retain one 4096D token for 32 current tiles and 128 layer-resolved
deltas. With eight 4096D action tokens and one instruction token, a BF16 raw
feature row is approximately 1.35-1.45 MiB before container overhead. Therefore:

- 4,200 C2 rows are expected to occupy roughly 6 GiB;
- 26,000 C3 rows are expected to occupy roughly 36-40 GiB; and
- the V2 caps of 10 GiB and 50 GiB are realistic but must be verified from the
  actual schema in C1.

At audit time TITAN reported approximately 341,130,036 KiB free on the project
filesystem. Disk is sufficient for the bounded feature store, but the
filesystem was already 82% full; C0/C1 must recheck margin immediately before
collection and no unrelated cleanup is permitted.

The prior P3R peak of 18,261 MiB leaves about 5.2 GiB beneath the frozen
23,552 MiB ceiling. The action-head hook and compact adapter are plausibly
within that margin, but only C1/C4 measurements can establish it.

## 5. Statistical feasibility

The decisive limitation is the final paired population, not GPU memory.
Exactly 40 untouched official initial states per task remain after reserving
IDs 0-9 for development, so final `n = 1,600`.

For a one-sided alpha-0.05 paired non-inferiority test with a 2-point margin and
true CAC-dense difference zero, 80% power requires paired discordance around
10% or lower. If CAC and dense often succeed/fail on different conditions even
with equal aggregate success, the final study can be underpowered. V2 does not
hide this limitation or inflate `n` with repeated seeds. C0 publishes the full
power grid, and C5/C6 report discordance without changing the gate.

The 5-point cache-repair gate also requires real headroom. C1H moves that
question before expensive training. Its two-stage rule is development-only and
cannot be used to tune features, profiles, or thresholds.

## 6. Novelty assessment

The novelty is narrower than “learned VLA correction”:

- A2C2 already learns residual action correction from current observations and
  a base action, but addresses within-chunk delay and uses expert residuals.
- Action ControlNet also adapts actions under delay/handoff effects.
- Latent Bridge predicts feature or full-layer K/V deltas and uses DAgger, but
  reconstructs latent state rather than only its action consequence.
- LAC and Action-JND learn task/action-aware decisions about which tokens may be
  cached; Gated VLA-Cache invalidates unsafe cache states.
- CAC instead holds a physically validated mixed-age cache fixed and asks if
  exact layer/tile corruption can be repaired downstream in action space.

The primary-source search found no method combining the exact CAC mechanism at
audit time. This supports an organic, scoped novelty claim, not a claim of being
the first residual VLA adapter or the best VLA caching method.

### Primary sources actually used in this audit

- [OpenVLA-OFT](https://www.roboticsproceedings.org/rss21/p017.html): pinned
  continuous action-chunk architecture and LIBERO evaluation context.
- [VLA-Cache](https://openreview.net/pdf?id=QZYZ0Xm58q): selective visual K/V
  reuse mechanism and efficiency/reliability comparison point.
- [LAC](https://arxiv.org/abs/2602.00686): learned task-aware token selector and
  cache-ratio predictor, demonstrating that learned compute allocation already
  occupies the selector route.
- [Action-JND](https://arxiv.org/abs/2608.21247): action-tolerance supervision
  for cache/pruning decisions, the closest current action-aware compression
  comparator.
- [Gated VLA-Cache](https://arxiv.org/abs/2608.10824): training-free confidence
  invalidation and evidence that cache headroom can depend strongly on the
  backbone/profile.
- [A2C2](https://arxiv.org/abs/2509.23224): lightweight residual action
  correction using current observations, base actions, policy features, time,
  and language; motivates the mandatory generic comparator.
- [Latent Bridge](https://arxiv.org/abs/2605.02739): feature/K/V delta prediction
  with R0 plus DAgger-style R1; motivates but does not duplicate action-only
  repair.
- [Action ControlNet](https://arxiv.org/abs/2606.25985): another frozen-backbone
  residual adapter, but for delayed asynchronous chunk handoff rather than
  same-query mixed-age cache corruption.

## 7. Remaining irreducible risks

These are scientific uncertainties, not missing protocol details:

1. D62 may not have a stable terminal deficit on the final population.
2. Its deficit may be caused by lost information that current projected tokens
   and corrupted action features cannot identify.
3. Dense-action imitation may lower offline error without placing the corrected
   policy in states where dense remains competent.
4. Sparse gripper errors may dominate success despite good average motion error.
5. CAC-dense paired discordance may make the 2-point NI gate underpowered.
6. A generic current-observation residual may perform as well as cache-specific
   CAC, eliminating the intended mechanism contribution.
7. Recent competing preprints may narrow novelty further before submission.

No protocol can remove these without observing scientific outcomes. V2 exposes
them in the least expensive defensible order.

## 8. Final decision

**Protocol V1: no-go.** It contains incorrect action pooling, impossible split
arithmetic, a false calibration-access assumption, underspecified comparators,
and no early terminal repair-opportunity test.

**Protocol V2: ready to begin at C0 only.** The method is technically plausible,
the study is scientifically coherent, the evidence populations are now honest,
and every known implementation/data/statistical ambiguity has an explicit test,
gate, or stop rule.

This conclusion is not a prediction that CAC will be positive. It is a judgment
that V2 can now produce an interpretable positive or negative answer without
needing another conceptual correction before execution.
