# PAIR-VLA: Research Direction and Execution Plan

> **Operational supersession note (2026-08-26):**
> PAIR_VLA_EXECUTION_PROTOCOL_V1.md is the authoritative phase-by-phase
> execution plan. This document is retained as the research rationale and audit
> history.

**Project:** SAVR

**Date:** 2026-08-26

**Status:** Version 1.2 after third-pass logic, resource, and statistical audit; research recommendation only; no experiment is authorized by this document
**Recommended direction:** **Provenance-Aware Interventional Reuse (PAIR-VLA)**

## 1. Executive decision

There is a credible positive-results direction that remains organically related
to SAVR:

> **Learn cache decisions from controlled replacements with the exact stale K/V
> states that the policy would actually reuse, while recording the true source
> time, layer, token, and camera of every reused entry.**

The proposed method, PAIR-VLA, uses a frozen VLA and an offline intervention
study to train a very small reuse-risk router. It does **not** train the 7B
backbone, use terminal success as a training label, or perform randomized
experiments during deployment.

This is the best current direction because four pieces of evidence align:

1. **The local bottleneck is now specific.** BRACE already demonstrated 12.23%
   complete-cycle acceleration, but its heuristic token choices changed every
   tested action. The remaining problem is selecting which stale values are
   tolerable, not establishing that partial K/V reuse can save time.
2. **Recent VLA work validates action-aware learned selection.** Action-JND
   reports much better task retention than VLA-Cache at aggressive reuse, while
   LAC reports simultaneous speed and success improvements from learned caching.
3. **Recent causal-routing work validates intervention-derived supervision.**
   Direct masking interventions can reveal dependencies that attention maps and
   other observational proxies miss, and expensive intervention scores can be
   distilled into lightweight deployment-time gates.
4. **The repository already records the missing variable.** BRACE maintains
   exact per-layer, per-token source ownership, source age, camera identity, and
   source observations. This makes actual stale-value interventions feasible
   without rebuilding the full caching stack.

The recommendation is positive but not unconditional. PAIR-VLA becomes a paper
method only if it passes the outcome-blind label, generalization, and physical
latency gates below. The paper result must still be established by locked
closed-loop LIBERO experiments.

### 1.1 Second-pass corrections

A second adversarial review found two weaknesses in version 1.0 and corrected
them before implementation:

1. **Dense-action imitation was too indirect.** A cached action can differ from
   the dense prediction while moving closer to the demonstrated expert action,
   or match the dense prediction while preserving a dense-policy error. The
   primary offline target is now the **excess expert imitation loss caused by
   the actual stale intervention**. Dense-action distortion remains a secondary
   diagnostic and an on-policy consistency target where expert actions are
   unavailable.
2. **The measured speed reserve was too narrow.** BRACE's best 12.23%
   complete-cycle reduction leaves only 2.23 percentage points above the 10%
   paper floor. Version 1.1 introduced a nominal 15% raw screen; version 1.2
   supersedes that shortcut with an explicit routed-mixture lower-bound
   calculation. PAIR may not assume the existing P2-D50 profile has sufficient
   headroom.

The review also tightened treatment fidelity: primary labels must come from the
real cache-stack suffix computation with a complete source ledger. Single-layer
tensor swaps are diagnostics only because they do not reproduce compounded
upstream staleness.

### 1.2 Third-pass corrections

A third red-team review found that version 1.1 still treated model queries too
independently and used an insufficiently explicit speed gate. Version 1.2 makes
five additional corrections:

1. **Demonstrations must be sampled at deployment query times.** The official
   LIBERO evaluator executes an eight-action chunk before querying the model
   again. Sequential expert frames are therefore sampled at the exact
   deployment cadence $t_q,t_q+8,t_q+16,\ldots$, with end-of-trajectory and
   padding rules reproduced from the pinned pipeline. Adjacent-frame stale
   substitutions are diagnostics only.
2. **Reuse is evaluated as a recursive contract, not independent masks.** A
   deployed stale value at query $q+1$ depends on the decisions and source
   ledger produced at query $q$. Primary labels therefore replay bounded
   one-, two-, and four-query reuse contracts from clean anchors, carrying the
   real mixed cache and provenance forward.
3. **Expert demonstrations isolate cache corruption but not policy-induced
   state shift.** Expert-regret labels are used for the offline mechanism test;
   they are not presented as closed-loop evidence. One predeclared on-policy
   consistency/aggregation round remains necessary if the development screen
   reveals transfer failure.
4. **Raw saving alone is not a sufficient headroom gate.** The outcome-blind
   cost table must show an algebraically feasible route to the 10% net target
   after the frozen minimum service rate, router/provenance overhead, resets,
   and dense fallbacks. A nominal 15% raw profile no longer passes by itself.
5. **Confirmatory trial count must be powered for paired non-inferiority.** The
   official 500 trials per suite are a reference minimum, not an automatic
   guarantee of a narrow two-point paired interval. The final count is frozen
   from a blinded discordance estimate and a predeclared power calculation.

## 2. Precise research question

**Can a lightweight router trained on the measured action effect of actual,
provenance-resolved stale K/V substitutions preserve VLA task success at reuse
levels that deliver real end-to-end acceleration?**

The corresponding hypothesis is:

> At the same physical reuse budget, a router trained on actual stale-state
> interventions will produce lower action distortion and higher terminal task
> success than visual similarity, attention, age-only, and existing heuristic
> caching policies.

This is a narrower and more defensible claim than “state awareness makes
caching safe.” It directly targets the failure exposed by SAVR, ACR, and BRACE.

## 3. What the literature establishes

Only sources that determine the method or evaluation are included here.

### 3.1 Direct VLA caching neighbors

- [VLA-Cache](https://arxiv.org/abs/2502.02175) selects visually stable,
  low-relevance tokens and applies layer-dependent temporal K/V reuse. It is the
  essential heuristic comparator.
- [LAC](https://arxiv.org/html/2602.00686) learns a token selector and reuse-ratio
  predictor through the frozen VLA's task loss. It reports a 1.76x wall-clock
  speedup and a 1.9-point average LIBERO success improvement. It establishes
  that learned allocation can yield a positive frontier, but its selector uses
  optical flow and end-to-end gradients rather than actual-source intervention
  labels.
- [Action-JND](https://arxiv.org/html/2608.21247) trains a lightweight estimator
  to find generic feature perturbations that preserve the VLA action. It then
  ranks VLA-Cache candidates with one token ordering shared across layers. On
  OpenVLA-OFT it reports gains of 6.2 and 10.1 success points over VLA-Cache at
  60% and 80% reuse. It is the closest action-aware comparator.
- [Gated VLA-Cache](https://arxiv.org/abs/2608.10824) uses output confidence to
  invalidate a cache. It is a query-level recovery comparator, not a source-
  resolved token selector.
- [DySta](https://arxiv.org/abs/2602.03983) learns static/dynamic visual
  representations and a recache gate. It strengthens the case for learned
  temporal structure, but requires architectural training and does not label
  decisions with actual stale-value effects.

### 3.2 Why observational importance is insufficient

- [Learning What Matters](https://arxiv.org/html/2607.21692) shows in controlled
  routing tasks that attention may include obsolete evidence or miss causal
  chains, while intervention-derived evidence trains much better routers.
- [CausalGate](https://arxiv.org/html/2607.22720) measures the output effect of
  zeroing transformer modules, then distills that expensive intervention signal
  into lightweight gates with no intervention cost at inference.

These papers do not solve VLA temporal caching. They support the methodological
principle that the router should be supervised by the measured consequence of
the proposed computation change, not merely by attention or appearance.

### 3.3 What randomization can and cannot claim

[Error Certificates for KV-Cache Eviction](https://arxiv.org/html/2607.21475)
proves that known-probability sampling can restore identifiability after
permanent LLM cache eviction. Its real-workload study also finds that the
certificate supports attribution and scheduling, not general failure
prediction.

PAIR-VLA borrows the discipline of controlled, randomized interventions for
**offline label collection**. It does not copy the eviction theorem, claim an
online certificate, or infer terminal success from an attention-error bound.

### 3.4 Evaluation scale and compute boundary

[OpenVLA-OFT](https://arxiv.org/html/2502.19645) uses eight-action chunks and
reports LIBERO results over 500 trials per suite. Its backbone training used
eight 80-GB GPUs, which is outside this project's resource boundary. PAIR-VLA
therefore keeps the foundation model frozen and trains only a compact router
from stored features and intervention labels.

The official [OpenVLA-OFT RLDS dataset
pipeline](https://github.com/moojink/openvla-oft/blob/main/prismatic/vla/datasets/rlds/dataset.py)
exposes trajectory-level observations and actions and explicitly constructs a
future action window. This makes expert-regret labeling feasible in principle,
subject to P0 verifying the exact pinned constants, normalization, padding, and
dual-camera fields rather than assuming compatibility from the paper alone.

The official [LIBERO evaluation
loop](https://github.com/moojink/openvla-oft/blob/main/experiments/robot/libero/run_libero_eval.py)
only requeries the model when its action queue is empty and recommends executing
the complete action chunk. The pinned [LIBERO
constants](https://github.com/moojink/openvla-oft/blob/main/prismatic/vla/constants.py)
set that chunk to eight actions. Consequently, offline source age is measured
in **model-query intervals**, and the primary demonstration sequence must
reproduce the eight-environment-step query cadence rather than treating every
stored video frame as a new cache opportunity.

### 3.5 Verified local resource boundary

A read-only, project-scoped TITAN check on 2026-08-26 found:

- /home/ved/SAVR uses 32,316,980 KiB (about 30.82 GiB);
- the containing filesystem has 380,667,804 KiB free (about 363.03 GiB);
- checkpoints are the largest project component at about 14.84 GiB, followed by
  environments at about 8.73 GiB and cache at about 3.82 GiB; and
- prior BRACE evidence measured about 18.4 GiB peak aggregate model memory
  beneath the unchanged 23-GiB project GPU cap.

Disk capacity is therefore not the present blocker. The old 25-GiB storage cap
was explicitly scoped to the completed Phase-1 environment installation and is
not silently reused as authorization for a new dataset. P0 must still measure
the dataset's unpacked/cache footprint and request a new bounded storage and
transfer allowance. GPU feasibility remains conditional on serialized label
generation with one live contract, one selected idle GPU, and measured peak
reservation below 23 GiB.

## 4. Novelty boundary

### 4.1 Proposed contribution

The organic contribution is not “another learned cache router.” It is the
combination of these three linked mechanisms:

1. **Actual-stale intervention supervision:** substitute the precise stale K/V
   vectors that a temporal cache would use, not an arbitrary learned feature
   perturbation or a zero mask.
2. **Exact temporal provenance:** condition every decision on the true source
   query, age, layer, camera, and patch—not simply the previous frame or one
   shared token ranking.
3. **Structured interaction calibration:** train and validate on multi-token,
   multi-layer reuse masks so the router is not justified solely by isolated
   token effects.

The literature search through 2026-08-26 did not find a VLA caching paper with
this exact supervision target and provenance model. This is a provisional
novelty conclusion, not proof that unpublished work does not exist. The search
must be rerun immediately before submission.

### 4.2 Claims PAIR-VLA must not make

- It is not the first learned VLA cache selector; LAC already occupies that
  claim.
- It is not the first action-aware compression method; Action-JND occupies that
  claim.
- It does not provide formal robot safety or a terminal-success guarantee.
- It does not establish token-level causality independent of the intervention
  distribution and fixed policy state.
- It is not a positive result until the locked closed-loop evaluation passes.

## 5. Formal method

### 5.1 Cache state and provenance

Let \(q\) index **model queries**, not raw environment frames. For the pinned
LIBERO deployment, consecutive normal queries occur after eight executed
actions. At model query \(q\), decoder layer \(l\), and visual token \(i\), let

\[
K^q_{l,i},V^q_{l,i}
\]

denote the freshly computed key and value. The cache ledger records the actual
source query

\[
\tau_{q,l,i}\leq q,
\]

its age \(d_{q,l,i}=q-\tau_{q,l,i}\) in model-query intervals, camera \(c_i\),
patch identity, and the stored pair

\[
K^{\tau_{q,l,i}}_{l,i},V^{\tau_{q,l,i}}_{l,i}.
\]

The dense same-stack action chunk is

\[
A_q^0=f_\theta(O_q,S_q,L;C_q^{\mathrm{fresh}}),
\]

where \(O_q\), \(S_q\), and \(L\) are the current observations, robot state,
and instruction. The backbone parameters \(\theta\) remain frozen.

### 5.2 Actual-stale intervention

For a structured intervention mask \(Z\), define a mixed cache

\[
C_q(Z)_{l,i}=
\begin{cases}
(K^{\tau_{q,l,i}}_{l,i},V^{\tau_{q,l,i}}_{l,i}), & Z_{l,i}=1,\\
(K^q_{l,i},V^q_{l,i}), & Z_{l,i}=0.
\end{cases}
\]

The intervened action is

\[
A_q(Z)=f_\theta(O_q,S_q,L;C_q(Z)).
\]

For deployment-aligned LIBERO demonstrations, let \(A_q^*\) be the normalized
expert action chunk used by the OpenVLA-OFT training pipeline. Define the frozen
imitation loss

\[
\ell(A,A_q^*)=
\sum_{h=1}^{H}\gamma_h
\left[
\sum_{j\in\mathcal{C}}w_j
\frac{|A_{q,h,j}-A^*_{q,h,j}|}{\sigma_j+\epsilon}
+\lambda_g\mathbf{1}\{g_{q,h}\neq g^*_{q,h}\}
\right].
\]

The primary single-query intervention label is the signed excess expert regret

\[
R_q^*(Z)=\ell(A_q(Z),A_q^*)-\ell(A_q^0,A_q^*).
\]

Positive values mean that the stale intervention worsens imitation of the
expert relative to same-stack dense inference. Negative values are preserved
rather than clipped during analysis, because a stale intervention may
occasionally regularize a dense error. A one-sided risk model may use
\(\max(0,R_q^*(Z))\) only after the signed distribution is reported.

The secondary label, available on demonstrations and on-policy query states,
is dense-action distortion

\[
D_q(Z)=
\sum_{h=1}^{H}\gamma_h
\left[
\sum_{j\in\mathcal{C}}w_j
\frac{|A^0_{q,h,j}-A_{q,h,j}(Z)|}{\sigma_j+\epsilon}
+\lambda_g\mathbf{1}\{g^0_{q,h}\neq g_{q,h}(Z)\}
\right].
\]

Here \(\mathcal{C}\) contains continuous translation and rotation coordinates,
\(\sigma_j\) is a frozen training-set scale, \(g\) is the gripper decision,
and \(\gamma_h\) discounts later elements of the eight-action chunk. All
weights and normalizations must match the pinned OpenVLA-OFT dataset pipeline
and be frozen before intervention labels are inspected.

Because the action loss may hide a geometrically important accumulated error,
the study also reports a secondary chunk-end diagnostic: integrated translation
and rotation displacement error plus gripper-transition mismatch after applying
the eight predicted deltas. It cannot replace the pinned imitation loss or be
selected post hoc as the primary target.

Expert regret is closer to the VLA's supervised task objective than dense-action
parity, but it is still an offline surrogate. Terminal task success remains the
primary paper outcome.

### 5.3 Recursive reuse contracts

Independent masks do not reproduce deployed reuse. Let a contract

\[
\mathcal{Z}_{q:B}=(Z_q,Z_{q+1},\ldots,Z_{q+B-1})
\]

start from a clean anchor and run for \(B\in\{1,2,4\}\) model queries. After
each query, the actual mixed cache and complete source ledger are carried into
the next query. The contract label is

\[
R_{q:B}^{*}=\sum_{b=0}^{B-1}\beta_b R_{q+b}^{*}(Z_{q+b}),
\]

with all \(\beta_b\), reset behavior, and truncation rules frozen in P0. The
router is trained on both per-query regret and contract-level upper-tail regret.
This makes source age and compounded reuse consequences part of the treatment,
instead of reconstructing every intervention independently from a fresh cache.

### 5.4 Structured intervention design

Pure one-token ablations are inadequate because reuse effects can interact
across layers and tokens. Label collection therefore uses two complementary
designs:

1. **Local groups:** camera × spatial tile × layer band × source-age bin. These
   estimate interpretable local stale effects and are diagnostic only.
2. **Profile contracts:** randomized mask sequences drawn from an outcome-blind,
   physically measured profile frontier. These expose multi-token and
   cross-layer interactions and cumulative source aging at realistic compute
   budgets and supply the primary training examples.

Sampling probabilities are logged and stratified so both cameras, early and
late layers, gripper-transition states, age bins, and budget tiers receive
coverage. The estimand is explicitly conditional on this intervention design.

Every primary profile contract must run through the actual partial-reuse cache
suffix with its complete source ledger. Constructing a dense pass and replacing
an isolated stored K/V tensor does not reproduce the hidden-state, later-layer,
or later-query consequences of deployed reuse and cannot serve as a primary
label.

### 5.5 Router inputs

The deployment router may use only information available before the reuse
decision:

- actual source age, source query, layer, camera, and patch identity;
- current-versus-actual-source raw patch change;
- current-versus-source projected visual-feature change only if that projection
  is already available before the decision and its complete cost is charged;
- current and source proprioceptive states and their difference;
- previous action summary and gripper-transition indicator;
- instruction/task embedding already available to the policy;
- prior-step attention or salience when available without a new dense pass; and
- the candidate physical reuse profile.

It must not use current dense actions, current terminal outcomes, future frames,
a hidden dense pass, or any feature produced by computation that the selected
cache profile is supposed to skip.

### 5.6 Learned risk and cache decision

A compact model \(r_\phi(x_{q,g})\) predicts the intervention risk of candidate
group \(g\). A second small set-level calibrator receives aggregated group
scores, budget, camera composition, and layer composition to estimate
excess expert regret \(\widehat{R}_q^*(Z)\) on demonstration states and
dense-action distortion \(\widehat{D}_q(Z)\) as an auxiliary target. Training
occurs only on stored features and labels; no gradient passes through the VLA.

At deployment, PAIR chooses among a frozen set of physically measured cache
profiles and ranks eligible groups by predicted risk. It selects the greatest
measured compute saving whose calibrated risk stays below a threshold chosen on
the calibration split:

\[
Z_q^*=\arg\max_{Z\in\mathcal{Z}_{\mathrm{frozen}}} S(Z)
\quad\text{s.t.}\quad
U_\alpha\!\left(\widehat{R}_q^*(Z)\right)\leq\delta.
\]

\(S(Z)\) is measured complete-cycle saving, not FLOPs, and \(U_\alpha\) is a
held-out upper calibration bound. This bound is an empirical risk control, not
a closed-loop safety guarantee.

Hard fail-closed rules force dense recomputation when provenance is missing,
an age cap is exceeded, inputs are nonfinite, the router is out of calibration,
or a gripper-transition veto fires.

## 6. Why positive results are plausible

The method has a credible positive mechanism, not a guarantee:

- **Selection is the measured local failure.** P2-D50 already proves that the
  stack can convert partial K/V reuse into 12.23% complete-cycle saving. PAIR
  now requires a larger raw headroom profile and targets its reliability with
  expert-regret supervision.
- **The supervision matches the deployment corruption.** Action-JND learns
  tolerance to a generic feature perturbation. PAIR asks whether the exact
  direction and age of the stale value are tolerable at the exact layer where
  reuse occurs, and whether that substitution worsens imitation of the
  demonstrated expert action.
- **Intervention labels are stronger than proxies in adjacent work.** Direct
  intervention supervision has outperformed attention imitation in controlled
  routing and pruning studies.
- **Training is locally feasible.** Expensive work is frozen-model inference;
  the learned router is small and can train from stored rows on CPU or a single
  GPU. Full-backbone backpropagation is avoided.
- **The repository has most of the instrumentation.** SourceLedger,
  SourceTracker, source-aware patch comparison, fixed profiles, and the SDPA
  sidecar already cover the main provenance and interception requirements.

PAIR therefore has a better balance of novelty, feasibility, and positive
mechanism than the alternatives considered. It should be pursued as a staged
research program, not assumed successful in advance.

## 7. Main failure modes and controls

| Risk | Why it matters | Required control |
|---|---|---|
| Offline imitation regret does not guarantee terminal success | Expert-action loss is closer to the training objective but does not capture all long-horizon consequences | Use expert regret for training and screening; terminal success remains the locked primary outcome |
| Adjacent demonstration frames do not match deployment queries | OpenVLA-OFT normally executes all eight actions before the next observation query | Build training sequences at the pinned eight-step query cadence; verify padding, no-op filtering, normalization, and action-window truncation |
| Independent interventions miss recursive staleness | Source age and cache contents at a later query depend on earlier reuse decisions | Generate primary labels from one-, two-, and four-query contracts that carry the real ledger and cache forward from a clean anchor |
| Same-stack dense differs from optimized core-FR | BRACE observed this mismatch even with no reuse | Define regret as cached-minus-dense expert loss on the same stack; report core-FR separately; never use exact core-FR parity as a PAIR label |
| Demonstration states are exogenous to the tested cached actions | Expert trajectories isolate cache corruption but do not reproduce state drift caused by cached policy actions | Keep the offline claim narrow; add held-out dense-policy consistency data and permit one frozen on-policy aggregation round only after the first screen |
| Isolated K/V swaps are not deployed caching | Upstream stale hidden states affect later layers and masks | Use actual cache-suffix execution for primary labels; keep isolated swaps diagnostic |
| Isolated effects do not add | Multiple stale entries interact through attention and residual paths | Include realistic structured masks and a set-level calibrator; report interaction residuals |
| Dense-state labels miss routed-policy states | Closed-loop reuse changes the future state distribution | After the first successful screen, allow one frozen on-policy data-aggregation round, then refreeze |
| Action L1 misses accumulated end-effector error | Small per-step errors can sum into a materially different chunk endpoint | Keep pinned expert imitation loss primary and report integrated translation/rotation plus gripper-transition error as a frozen secondary diagnostic |
| Gripper changes are rare | A regressor can minimize average error while missing decisive transitions | Oversample transition states and use a hard gripper mismatch/veto term |
| Router overfits tasks or episode fragments | Token rows from one trajectory are highly correlated | Split by complete task and trajectory prefix; never randomly split individual token rows |
| Router feature requires the computation it is meant to skip | A current projected feature may erase the measured speed advantage or create circular decision timing | Admit only chronologically available features, charge their full synchronized cost, and exclude features produced downstream of the decision |
| Demonstration replay uses expert previous actions as router inputs | Deployment sees the policy's previous predicted/executed chunk, not an oracle expert-history feature | Carry the model-produced prior action summary through each contract; expert actions are labels only |
| Task-language features memorize the training tasks | The router could look good on the same LIBERO instructions without learning stale-value risk | Split by task where possible, add no-instruction/task-identity ablations, and state clearly whether final training uses demonstrations from evaluation task identities |
| Router overhead and fallbacks consume the speedup | A 15% raw profile used on 70% of queries loses the 10% target before modest overhead | Freeze a conservative service-rate/fallback assumption and require the measured cost algebra to admit at least 10% net saving; then verify the complete routed mixture physically |
| Intervention data become too expensive | Every contract needs several frozen-model queries and multiple masks | Use a small identifiability pilot first, label contracts sequentially, freeze query/hour/storage caps, and scale only after P3 pilot gates pass |
| Multiple live cache banks exceed GPU memory | Prior peak was about 18.4 GiB under a strict 23-GiB cap | Keep at most one active contract and bounded live source ring on GPU, serialize mask evaluations, and remeasure peak reservation at every GPU phase |
| Stored K/V labels consume impractical disk | A full source bank per query can be hundreds of MiB before overhead | Stream trajectory windows, retain compact provenance/features/labels, and discard reproducible K/V tensors after each bounded contract |
| Optional data silently exceeds an old phase-specific cap | The compressed dataset is about 9.5 GiB and its unpacked footprint is not yet frozen | Treat the old 25-GiB Phase-1 cap as historical, record current project/free-space evidence, estimate unpacked/cache growth, and obtain a new explicit storage cap before download |
| Recent comparator is unavailable | Action-JND and Gated VLA-Cache are very recent | Audit code availability first; if a faithful strongest comparator cannot be run, narrow the paper claim rather than imply superiority |
| Threshold shopping produces a false positive | Many budgets and risk thresholds can create a lucky point | Separate development, calibration, and locked test populations; one final confirmatory run |
| Matched-latency comparator is chosen after success is seen | Post-outcome matching can select a favorable baseline point | Freeze comparator identities and matching rule from the outcome-blind cost table before P6 |
| Fixed 500-trial evaluation is underpowered | A two-point paired margin can require more than 500 trials when method outcomes are discordant | Estimate discordance without method labels/outcomes being tuned against it, freeze a paired non-inferiority power calculation, and increase trials or widen the claim before confirmation |
| Literature collision appears | The field is moving rapidly | Repeat the primary-source search before implementation freeze and submission |

## 8. Candidate comparison under the revised project criteria

Scores are 0--5 ordinal assessments, not probabilities.

| Direction | Organic novelty | Positive mechanism | Relation to SAVR findings | Local feasibility | Decision |
|---|---:|---:|---:|---:|---|
| Reimplement LAC or Action-JND | 1 | 5 | 4 | 2 | Strong reproduction, weak new paper |
| Generic learned router | 1 | 3 | 4 | 4 | Reject: crowded and proxy-supervised |
| Feature/KV delta predictor | 1 | 4 | 4 | 1 | Reject: direct collisions and training cost |
| Online randomized error certificate | 4 | 2 | 4 | 2 | Preserve as analysis idea; too much online uncertainty/overhead |
| Action correction after cached inference | 2 | 3 | 3 | 2 | Reject: repairs outputs, not stale internal state |
| **PAIR-VLA** | **4** | **4** | **5** | **4** | **Recommended staged direction** |

PAIR is selected because it creates a distinct scientific question and can be
tested using the existing stack. It is not selected because a positive result
is guaranteed.

## 9. Staged execution plan

Each phase ends in a written checkpoint. A failure stops the direction; it does
not trigger an immediate redesign.

### Phase P0 — protocol, data, and collision freeze

**Work**

- rerun the primary-source novelty search;
- inspect availability and licenses for Action-JND, LAC, Gated VLA-Cache, and
  VLA-Cache code;
- verify that the official modified LIBERO trajectories expose sequential
  images, proprioception, actions, language, and the future eight-action window
  through the pinned OpenVLA-OFT preprocessing;
- reproduce the official eight-action queue chronology on demonstration
  trajectories and freeze query-aligned indexing, padding, terminal truncation,
  no-op filtering, normalization, and gripper conversion;
- freeze the exact coordinate space for expert/policy loss and the simulator-
  faithful composition used by the endpoint diagnostic; do not naively add
  rotations if the environment applies a different action convention;
- define prior-action router features from the model-produced contract history,
  not from oracle expert actions;
- record the current project-only disk footprint and filesystem free space,
  estimate compressed plus unpacked/cache growth, and obtain separate approval
  for a new project-local storage cap before the approximately 9.5-GiB optional
  training dataset is downloaded;
- map every proposed feature and real cache-suffix intervention to BRACE;
- freeze expert-loss weights, endpoint diagnostic, normalization, groups, age
  bins, one-/two-/four-query contract design, clean-anchor/reset rules,
  profile-search bounds, staged query/hour/storage caps, splits, seeds, and
  statistical tests;
- freeze the minimum routed service-rate assumption and maximum admissible
  router/provenance/reset overhead used by the P2 headroom algebra;
- classify every router input by the point at which it becomes available and
  exclude any input requiring computation the chosen profile would skip;
- define whether final router training uses demonstrations from the same task
  identities as evaluation and freeze task-held-out/no-instruction ablations;
- audit whether immutable earlier dense ledgers can be reused under exact
  checkpoint, code, environment, task, seed, and preprocessing identity;
- define same-stack dense and optimized core-FR roles; and
- produce a data schema and artifact manifest.

**Gate P0**

Advance only if the novelty distinction survives; expert-action chunks and
query-aligned observations match deployment semantics; recursive primary
contracts are implementable without backbone gradients; and projected
storage/runtime fit the one-GPU boundary.

### Phase P1 — CPU/synthetic correctness

**Work**

- implement structured interventions and provenance features behind the
  existing cache interface;
- test exact source resolution, age caps, camera separation, layer alignment,
  deterministic replay, and mask accounting;
- test eight-step query alignment and exact contract reset/replay;
- prove on synthetic tensors that the primary path executes the requested
  partial-reuse suffix rather than a post-hoc dense tensor overlay;
- verify that the all-fresh mask exactly matches same-stack dense and logged
  stale substitutions match the requested source tensors at every step of a
  recursive contract; and
- add malformed-provenance and nonfinite-input fail-closed tests.

**Gate P1**

All correctness and provenance tests pass. No model or simulator outcome is
inspected.

### Phase P2 — outcome-blind physical headroom frontier

**Work**

- benchmark a bounded grid extending beyond P2-D50 in reuse budget, layer
  suffix, camera allocation, and age while preserving the same stack;
- measure synchronized accelerated-query and complete-cycle time, peak memory,
  and provenance cost with no router and no terminal outcomes;
- fit an outcome-blind monotone cost table for the candidate profiles;
- measure bounded router-feature/provenance overhead with synthetic scores;
- freeze comparator points and the matched-latency/reuse rule without success
  outcomes; and
- calculate the conservative routed-mixture saving

  \[
  S_{\mathrm{net,LB}}=
  \rho_{\min}S_{\mathrm{raw,LB}}-
  O_{\mathrm{router,UB}}-
  O_{\mathrm{reset/fallback,UB}},
  \]

  where every term and the minimum service rate \(\rho_{\min}\) were frozen in
  P0.

**Gate P2**

At least one technically valid profile must stay inside the memory cap, have a
positive raw complete-cycle saving lower bound, and satisfy
\(S_{\mathrm{net,LB}}\geq10\%\). The raw threshold is therefore determined by
the frozen service-rate and overhead assumptions rather than a nominal 15%.
Otherwise PAIR stops before dataset labels are generated.

### Phase P3 — expert-regret intervention study

**Work**

- collect actual source banks from deployment-aligned demonstration query
  sequences;
- stream source banks per bounded trajectory window and retain compact
  provenance/features/labels rather than persisting full K/V tensors;
- run a bounded identifiability pilot before scaling label generation;
- run repeated all-fresh controls, diagnostic local interventions, and primary
  one-/two-/four-query full-cache-suffix profile contracts from clean anchors;
- compute per-query and contract-level signed excess expert regret, secondary
  dense-action distortion, and the frozen chunk-end diagnostic;
- quantify technical noise, individual effects, interaction residuals, and
  gripper-transition coverage;
- compare actual-source interventions with generic perturbation, previous-frame
  substitution, attention, image change, and age-only scores; and
- use complete trajectories—not token rows—as the split unit.

**Frozen minimum gates**

- repeated identical masks must be stable relative to between-mask variation;
- signed expert-regret labels must vary beyond numerical/control noise;
- multi-query labels must show either measurable cumulative structure or clear
  evidence that a one-query contract is sufficient;
- actual-source/provenance features must achieve held-out Spearman correlation
  of at least 0.30 with structured-mask excess expert regret; and
- the source-aware model must reduce held-out upper-tail positive expert regret
  by at least 15% versus the strongest zero-cost heuristic at the same measured
  reuse budget.

These thresholds are screening gates, not final paper claims.

### Phase P4 — offline router training and locked validation

**Work**

- train a compact group-risk model and set-level calibrator on expert regret,
  with dense-action distortion and contract upper-tail regret as auxiliary
  targets;
- use task-held-out and trajectory-prefix-held-out validation;
- calibrate dense fallback thresholds without terminal outcomes;
- test no-provenance, previous-step-only, no-expert-regret, no-interaction,
  shared-camera, no-age, and no-instruction/task-identity ablations; and
- freeze one router, one fallback rule, and at most three physical profiles.

**Gate P4**

The locked router must beat visual change, attention, age, random, VLA-Cache
heuristics, and the strongest faithfully runnable action-aware control on
held-out positive expert regret at matched measured compute. It must not depend
on task IDs unavailable at deployment or leak future information.

### Phase P5 — net physical overhead and headroom

**Work**

- measure synchronized complete-cycle latency for same-stack dense, every
  comparator, and the fully deployed PAIR router;
- include dense fallbacks at the locked calibration-set rate;
- replay the actual recursive cache/reset lifecycle and isolate feature, router,
  provenance, sorting, reset, and fallback overhead;
- verify peak memory below the project cap; and
- repeat measurements over the frozen balanced query set.

**Gate P5**

PAIR must retain at least 10% net complete-cycle acceleration over same-stack
dense, with a positive bootstrap lower confidence bound, and must not exceed the
memory cap. Failure ends PAIR before simulator success is opened.

### Phase P6 — paired closed-loop development screen

**Work**

- run paired initial states on a development suite;
- compare same-stack dense, the strongest matched-latency cache baseline, and
  PAIR;
- use terminal success as primary, with latency, reuse, fallback, expert-regret
  transfer, and action distortion as secondary outcomes; and
- inspect failure videos only after the frozen aggregate report exists.

**Gate P6**

Advance only if PAIR improves the paired success--latency frontier over the
strongest cache baseline and retains the 10% speed floor. A merely positive
offline result, microbenchmark, or tie created by near-zero reuse does not pass.

One predeclared on-policy consistency-data round is allowed only if P6 shows a
clear demonstration/deployment shift. It gathers query states under the locked
PAIR policy, uses same-state dense-action distortion because expert actions are
unavailable, preserves the original router as an immutable comparator, and
cannot access the final confirmatory population. It is a bounded correction,
not repeated online tuning.

### Phase P7 — locked confirmatory evaluation

Use all four LIBERO suites and the official tasks. Fifty paired initializations
per task (500 trials per suite per primary method; 2,000 paired trials across
all suites) is the starting minimum, not an automatically sufficient final
count. The primary non-inferiority estimand is a predeclared suite-stratified
overall paired success difference; per-suite intervals are mandatory but are
not individually claimed as two-point non-inferiority unless separately
powered. Before opening confirmatory method outcomes, freeze the final count
from an a priori paired non-inferiority power calculation using the P6
discordance estimate and two-point margin. If the required count is
unaffordable, widen the claim or stop; do not run an underpowered confirmatory
study. The minimum primary methods are:

1. same-stack dense refresh;
2. strongest faithful cache comparator at matched latency or reuse; and
3. PAIR-VLA.

Core ablations may use a smaller, predeclared paired population, but must not
replace the primary comparison.

Immutable earlier dense episodes may replace duplicate dense computation only
if P0 proves exact semantic identity and the reuse decision is frozen before
PAIR outcomes. Otherwise all primary methods are rerun on the paired population.

**Positive-paper gates**

1. PAIR is non-inferior to dense success with a predeclared margin of 2
   percentage points using a paired confidence interval.
2. PAIR is at least 10% faster in synchronized complete-cycle latency, with the
   lower confidence bound above zero improvement.
3. PAIR has significantly higher paired success than the strongest cache
   baseline at matched latency/reuse, or achieves significantly lower latency
   at matched non-inferior success.
4. Expert-regret supervision, actual stale values, and exact provenance each
   contribute in frozen ablations.
5. Results are reported by suite and task, including all failures and intervals.

If these gates pass, the paper is a positive result. If PAIR improves the cache
baseline but misses dense non-inferiority or the speed floor, it is useful
evidence but not the targeted positive paper.

### Phase P8 — paper and artifact release

- write the method and results from the locked analysis outputs;
- release configs, exact commands, seeds, environment identity, analysis code,
  and compact result ledgers;
- provide the repository URL and data-availability statement;
- distinguish expert imitation, action fidelity, task reliability, and physical
  safety; and
- rerun citation, artifact, figure, and venue-format audits.

## 10. Statistical and reproducibility design

- Pair methods on identical task, initial state, and seed where the simulator
  permits.
- Report Wilson intervals for individual success rates and paired intervals for
  success differences; use McNemar-type paired tests for primary binary
  comparisons.
- Freeze the confirmatory non-inferiority sample size from the P6 discordant-pair
  rate, the two-point margin, one-sided type-I error, target power, and expected
  attrition. The primary calculation is for the suite-stratified overall
  estimand; per-suite non-inferiority requires separate power.
- Bootstrap complete-cycle latency by query/episode blocks, not individual GPU
  events.
- Split router data by whole tasks and trajectory prefixes; token rows from the
  same trajectory never cross splits.
- Freeze all thresholds before opening the corresponding outcome population.
- Preserve raw records, analysis scripts, semantic hashes, git revision,
  checkpoint identity, CUDA/hardware identity, and exact launch commands.
- Treat exploratory analyses as exploratory and keep them out of confirmatory
  claims unless independently retested.
- Use a predeclared hierarchical testing order: dense non-inferiority, net
  latency, then matched-comparator frontier. Do not promote a later test after
  an earlier gate fails.

## 11. Project-control rules

1. One method identity: PAIR-VLA. A failed gate is not renamed into a new
   version with changed scientific claims.
2. No terminal outcome is read before the physical overhead gate passes.
3. No threshold is changed after its evaluation population is opened.
4. No positive claim is made from expert regret, action distortion, CUDA time,
   FLOPs, or a single suite alone.
5. No paper method is compared against an unfaithful or technically broken
   baseline.
6. No full-backbone training, multi-GPU method change, or new platform is
   introduced without a new feasibility decision.
7. Every phase publishes a status report stating pass, fail, or technical stop,
   with no silent retry.
8. If PAIR fails P2, P3, P4, or P5, stop before closed-loop evaluation and
   consult the advisors rather than beginning another immediate method cycle.

## 12. Expected workload

- P0--P1 are engineering and verification work and should be relatively short.
- P2 is a bounded physical timing frontier and should stop quickly if the local
  stack cannot create enough raw headroom.
- P3 is inference-heavy. Contracts multiply model calls, so it begins with a
  bounded identifiability pilot and scales only after label stability and
  predictability pass. Labels are generated serially on one GPU; masks are not
  retained as simultaneous GPU cache banks.
- The optional training trajectories add approximately 9.5 GiB compressed;
  unpacked/cache growth is measured before a new storage cap is authorized.
- P4 trains only small models and is inexpensive relative to VLA inference.
- P5 is a bounded net timing study.
- P6 is the first simulator-success phase.
- P7 is the longest phase because a proper four-suite, paired evaluation
  requires thousands of episodes across methods.

The plan deliberately spends most compute only after novelty, label quality,
generalization, and physical headroom have passed.

## 13. Third-pass plausibility assessment

These judgments are evidence levels, not numerical probabilities:

| Dimension | Assessment after third check | Decisive reason |
|---|---|---|
| Scientific logic | Moderate | Actual-stale expert regret directly targets the measured cache corruption, but remains a surrogate |
| Integration feasibility | High | BRACE already implements partial reuse and exact source provenance |
| GPU feasibility | Moderate-high | Prior peak was about 18.4 GiB under 23 GiB; recursive labels must be serialized |
| Disk feasibility | High | About 363 GiB is currently free; the optional dataset still needs a bounded new cap |
| Runtime feasibility | Moderate-high | Frozen inference and a small router fit one GPU, but contract labeling and confirmatory episodes require staged caps |
| Physical speed headroom | Moderate-low until P2 | The only demonstrated raw complete-cycle saving is 12.23%; no routed 10% lower bound exists yet |
| Offline label predictability | Moderate and unproven | Intervention supervision is well motivated, but cheap pre-decision features may not predict upper-tail regret |
| Closed-loop transfer | Moderate-low and decisive | Expert trajectories do not contain states caused by cached-policy errors |
| Organic novelty | Moderate | Exact stale-source regret plus recursive provenance contracts remains distinct from the located neighbors |
| Positive-paper route | Plausible but conditional | The route is coherent and feasible to falsify early, but it is not yet more likely than not on evidence alone |

PAIR should therefore be described as a **credible staged positive-paper
attempt**, not a high-confidence result. P2 (net headroom), the P3 pilot
(identifiable/predictable regret), and P6 (closed-loop transfer) are the three
decisive uncertainty reductions. Passing all three would materially raise the
plausibility; failure of any one ends this method identity.

## 14. Final recommendation

Proceed with **PAIR-VLA**, starting with Phase P0 only.

This direction is:

- **organic:** it grows directly from the observed failure of heuristic SAVR/
  BRACE selection;
- **novel enough to investigate:** it supervises on actual provenance-resolved
  stale values and recursive reuse contracts rather than generic perturbations
  or observational proxies;
- **positive-result plausible:** closely related learned/action-aware methods
  show that better token selection can preserve success at aggressive reuse;
  and
- **locally feasible:** it reuses the existing BRACE cache and provenance stack
  and avoids 7B-model backpropagation.

It is not guaranteed to work, and the third check does not justify a precise
success probability. The decisive advantage of this plan is that its main
uncertainties are now separated: physical headroom is tested before labels,
label identifiability before training, and closed-loop transfer before the
confirmatory campaign.
