# BRACE Formal Method Specification V1

**Method:** Branch-Rollout Adaptive Cache Execution (BRACE)

**Date:** 2026-08-25

**Status:** Formal design only; no BRACE implementation, model run, GPU
operation, branch outcome, or positive result exists

**Governing protocol:** `docs/BRACE_EXECUTION_PROTOCOL_V2_1.md`

**Companion audit:** `docs/BRACE_FORMAL_METHOD_AUDIT_V1.md`

**Authority:** This document does not authorize downloads, external-source
patching, model/checkpoint loading, GPU use, policy outcomes, protected data, or
work outside `/home/ved/SAVR`. BRACE-B1 remains the only eligible task.

## 1. Purpose and claim boundary

This specification converts Protocol V2.1 into an implementation-ready method.
It defines the cache mechanics, controller inputs, learning target, router,
calibration, intervention, algorithms, records, and verification contracts.

BRACE is not a new cache primitive. It uses a faithfully corrected VLA-Cache-
style visual-KV substrate and learns **when to enter a bounded cache contract**.
Its provisional scientific contribution is the combination of:

1. clean-anchor, mixed-source cache contracts;
2. replay-verified paired terminal-effect supervision;
3. training on cache-induced trajectory states; and
4. an empirically risk-controlled pre-forward reject option.

The method is paper-positive only if the sealed experiment passes every V2.1
success gate. A coherent architecture does not imply that the label is
predictable, that caching is fast on TITAN, or that BRACE beats existing work.

## 2. Exact base-policy interface

At policy query `t`, the frozen OpenVLA-OFT policy receives

\[
o_t=(I_t^s,I_t^w,z_t,\ell),
\]

where `s` is the scene camera, `w` is the wrist camera, `z_t` is current
proprioception, and `ell` is the instruction. The frozen dense policy is

\[
A_t^F=\pi_F(o_t;\phi)\in\mathbb{R}^{H_a\times d_a},
\]

with project constants expected to be `H_a=8` action steps, `d_a=7` action
dimensions, and `dim(z_t)=8`. Implementations must obtain these values from the
pinned runtime constants and fail if they differ; they may not silently hard-
code them.

The pinned visual path produces two ordered projected token blocks:

\[
X_t^v=E_v(I_t^v)\in\mathbb{R}^{N_v\times d},\qquad
X_t^{vis}=[X_t^s;X_t^w].
\]

For the accepted checkpoint, each camera is expected to yield 256 tokens and
the combined visual block 512 tokens. Current proprioception is projected only
after the visual blocks. The actual multimodal sequence map must be discovered
from the runtime inputs and verified against the pinned source. No method may
assume a fixed 34-token prompt, fixed absolute text span, or a sequence layout
inferred from another checkpoint.

Optimized full refresh (FR) is the oracle. With BRACE disabled, the integrated
path must reproduce its actions, projected inputs, sequence positions, and
terminal behavior under the V2.1 parity rules.

The pinned OpenVLA-OFT Transformers fork changes SDPA to non-causal,
bidirectional attention over all non-padding multimodal positions. Therefore a
visual token's deeper-layer K/V can depend on current proprioception, prompt,
and action-placeholder context even when its pixels are unchanged. BRACE never
equates pixel similarity with K/V equivalence. Pixel change is one eligibility
signal; robot/action drift and terminal-effect supervision address the remaining
context dependence empirically.

## 3. Transformer and cache notation

Let `Omega_vis` be the visual token positions in the complete multimodal
sequence and `Omega_nv` all nonvisual positions. For decoder layer `l`, dense
self-attention forms

\[
Q_t^l=H_t^lW_Q^l,\quad
K_t^l=H_t^lW_K^l,\quad
V_t^l=H_t^lW_V^l,
\]

followed by the unchanged attention, residual, normalization, and MLP
operations of the pinned model.

For every visual position `j` and layer `l`, BRACE maintains

\[
C_{t,l,j}=(K_{t,l,j},V_{t,l,j},\rho_{t,l,j},a_{t,l,j}),
\]

where `rho` is the exact source query and `a=t-rho` is the source age. Cache
identity also includes camera, patch position, model/checkpoint, sequence-map,
preprocessing, episode, anchor, profile, dtype, device, and tensor shape.

Every source query has an immutable source record containing both preprocessed
camera images, normalized proprioception, prompt/instruction digest, action-mask
and sequence-map digests, previously executed-action summary, RNG/configuration
identity, and query counters. A K/V entry points to this record through `rho`.
Image provenance alone is insufficient for a bidirectional multimodal decoder.

Nonvisual tokens are never temporally reused by BRACE. Text, proprioception,
action placeholders, and other nonvisual positions are recomputed according to
the unchanged OpenVLA-OFT forward.

## 4. Dense anchor

A dense anchor at query `tau` performs the complete FR path and creates:

- dense action chunk `A_tau^F`;
- complete visual K/V entries at every layer;
- exact per-token source `rho=tau`;
- preprocessed scene/wrist images and patch map;
- the runtime sequence-position map;
- an optional dense semantic map; and
- a configuration and provenance digest.

The semantic map uses instruction-to-vision and anchor-action-to-vision
attention from a predeclared set of dense layers and runtime-derived query
positions. For query family `q` in `{instruction, action}`:

\[
r^q_{\tau,v,j}=\frac{1}{|L_g||Q_q||H|}
\sum_{l\in L_g}\sum_{u\in Q_q}\sum_{m\in H}
\operatorname{Attn}_{\tau,l,m,u,j}.
\]

The final salience is the maximum of separately rank-normalized instruction and
action maps, conservatively protecting a patch salient to either family.
`Q_instruction` and `Q_action` are derived from the tokenizer, action mask, and
runtime sequence map; neither is a fixed numeric slice. This action-aware
salience follows established pruning practice and is not a BRACE novelty claim.
For head `m`, the sidecar uses the exact post-RoPE Q/K tensors, scale, and mask
from the accepted dense attention call:

\[
\operatorname{Attn}_{l,m}=\operatorname{softmax}
\left(Q_{l,m}K_{l,m}^{\top}/\sqrt{d_h}+M_l\right).
\]

The preferred implementation exposes those tensors through a no-mutation tap
inside the selected dense layer and computes a detached sidecar map after the
forward. Capturing pre-RoPE linear projections and pretending they are attention
is invalid. The sidecar must not feed back into the anchor action.

Calling `output_attentions=True` on the pinned SDPA implementation falls back to
manual attention and is therefore not the default gate path. Attention is usable
only if B2/B3 proves that sidecar capture does not change the attention backend,
dense action, cache, or claimed timing semantics. A second off-path dense pass,
an eager-only substitute, or an accelerated forward may not secretly define the
deployed gate. If parity fails, attention-derived profiles are removed before
outcomes; the method may retain only profiles whose gate is produced by the
accepted dense path.

## 5. Patch/source change operator

For current preprocessed camera patch `P_t(v,j)` and the exact source image of
cache entry `(l,j)`, define a bounded change score

\[
d_{t,l,v,j}=\lambda_1\frac{\|P_t(v,j)-P_{\rho_{t,l,j}}(v,j)\|_1}{D_P}
+\lambda_2\frac{1-\cos(P_t(v,j),P_{\rho_{t,l,j}}(v,j))}{2},
\]

with nonnegative fixed weights summing to one and normalization `D_P` fixed by
the image representation. B3 chooses at most a bounded predeclared set of
`(lambda_1,lambda_2)` and thresholds using parity/speed evidence only.

For a patch with accepted coordinate-wise preprocessed bounds `[a_r,b_r]`, set
`D_P=sum_r(b_r-a_r)`. Define cosine similarity with a frozen positive epsilon
and clamp both component scores to `[0,1]`; a zero-norm pair is equal only when
both patches are exactly zero. Nonfinite scores invalidate the contract. These
rules must be known-answer tested so constant-color or zero-valued patches
cannot silently become maximally reusable.

The operator must run on the exact center-cropped, resized representation whose
patch grid maps to visual tokens. It must be vectorized and device-resident when
possible. PIL/OpenCV/NumPy round trips, host-device synchronization, and source-
image transfers count in latency. A low-resolution approximation may be used by
the router, but not to assert exact cache-position eligibility unless parity is
separately established.

## 6. Clean-provenance cache profiles

A profile `p` contains:

\[
p=(b_{p,l,v},\tau_{p,v},m_{p,v},A_{p,v},L_p,
\eta_{p,z},\eta_{p,a}),
\]

where:

- `b` is the maximum reused-token budget by layer and camera;
- `tau` is the source-change threshold;
- `m` is the number/fraction of anchor-salient tokens protected from reuse;
- `A` is the maximum token age; and
- `L_p` is the set of decoder layers at which reuse begins or grows;
- `eta_z` bounds current proprioceptive drift from every live K/V source and
  the previous query; and
- `eta_a` bounds source-relative previous-action motion/change and always
  vetoes unexpected gripper transitions.

P1 sets all wrist-camera reuse budgets to zero. P2 permits stricter bounded
wrist reuse. State/action envelopes are selected from pre-outcome behavior-trace
quantiles and fixed before terminal labels. The profile grid contains at most
six base profiles and is chosen outcome-blind as required by V2.1.

### 6.1 Eligibility

A visual token is eligible to reuse at layer `l` only when

\[
e_{t,l,v,j}^{p}=
\mathbb{1}[a_{t,l,v,j}\le A_{p,v}]
\mathbb{1}[d_{t,l,v,j}\le\tau_{p,v}]
\mathbb{1}[j\notin G_{\tau,p,v}]
\mathbb{1}[\text{all provenance checks pass}],
\]

where `G` is the protected dense-anchor semantic set. A profile without a valid
semantic map may set `G` to all visual tokens (equivalent to no acceleration)
or be removed; it may not fabricate task relevance from an accelerated pass.

Let `R_t^p` be the distinct live source queries referenced by candidate profile
`p`. Normalize proprioception with the frozen checkpoint statistics, and let
`m(A)` concatenate per-action-dimension first, last, mean, standard deviation,
and maximum-absolute executed commands. Before constructing any token set,
require the exact-source context envelope

\[
D_z=\max_{r\in R_t^p}\|\widetilde z_t-\widetilde z_r\|_\infty
\le\eta_{p,z}^{source},\qquad
\|\widetilde z_t-\widetilde z_{t-1}\|_\infty
\le\eta_{p,z}^{step},
\]

and

\[
D_a=\max_{r\in R_t^p}
\|m(A_{t-1}^{exec})-m(A_{r-1}^{exec})\|_\infty
\le\eta_{p,a}^{source}.
\]

The current executed-motion magnitude has a separate frozen upper bound, and
any unexpected gripper transition or unavailable source context vetoes reuse.
Envelope failure aborts to FR for the current query. This does not prove
reliable reuse; it prevents obviously changed multimodal context from being
treated as static solely because pixels are similar.

### 6.2 Nested layerwise reuse

The corrected VLA-Cache substrate removes selected visual hidden states before
specified decoder layers and leaves their previous K/V entries at the original
sequence positions. Once a token is removed, its current hidden state cannot be
reintroduced at a deeper layer. Therefore reusable sets must be nested:

\[
U_{t,l-1}^{p}\subseteq U_{t,l}^{p}\subseteq\Omega_{vis}.
\]

Construct them recursively by adding only the most static eligible tokens up
to budget `b_{p,l,v}`. Budgets must be nondecreasing after the first pruning
layer. Any profile violating nestedness is mechanically invalid, regardless of
its apparent speed.

Because an entered token remains removed at every deeper pruning layer, define
suffix eligibility

\[
\bar e_{t,l,v,j}^{p}=\prod_{k\in L_p:k\ge l}e_{t,k,v,j}^{p}.
\]

A token may enter `U_{t,l}` only when suffix-eligible; rank candidates by their
maximum source-change score over the same layer suffix. This prevents an early
drop from trapping a token that a deeper layer required to recompute.

At layer `l`, active queries are

\[
\Omega_{t,l}^{act}=\Omega_{nv}\cup(\Omega_{vis}\setminus U_{t,l}^{p}).
\]

The model computes current Q/K/V and MLP work only for active hidden states.
The cache performs position-preserving `index_copy` updates for active entries;
entries in `U` retain their previous K/V and sources. Current text/action
queries still attend to the complete position-aligned K/V cache.

For every layer/token,

\[
\rho_{t,l,j}=\begin{cases}
t,&j\notin U_{t,l}^{p},\\
\rho_{t-1,l,j},&j\in U_{t,l}^{p}.
\end{cases}
\]

This layerwise definition is why a scalar cache age is invalid.

## 7. Bounded cache contract

A contract is `c=(p,h)` with `h` in `{1,2,4}`. It starts only at the first
policy query after a verified dense anchor.

State is

\[
q_t=(mode,p,h,r,\tau,C_t,G_\tau),
\]

where `r` is the remaining accelerated-query allowance.

Transitions are:

1. `ANCHOR -> CONTRACT(p,h)` if the router accepts `c`;
2. `CONTRACT -> CONTRACT` after a valid accelerated query with `r <- r-1`;
3. `CONTRACT -> ANCHOR` when `r=0`;
4. any invariant/rejection/timeout/OOD condition `-> ANCHOR` before action
   generation, executing FR for that query.

The contract cannot switch profile, reset age, increase budget, or extend its
horizon. Aborted assignments remain contract assignments for intent-to-treat
analysis. During a paired branch experiment, the first abort executes FR and
locks FR for the remainder of that assignment's `h`-query treatment window; it
may not start a different contract. During ordinary deployment, the aborting FR
creates a new anchor and the router may make a new decision at the following
query. This distinction preserves a well-defined local treatment while B6/B7
test repeated online decisions.

## 8. Router information set

For each candidate contract `c`, build features `x_t(c)` before the current VLA
forward. No current dense hidden state, logit, attention, action, reward,
success, privileged simulator state, or future observation is permitted.

### 8.1 Patch summaries

For each camera, aggregate current-to-actual-source change and age across
layers into patch-aligned maps:

- mean and maximum change;
- mean and maximum normalized age;
- dense-anchor salience;
- protected-token indicator; and
- candidate-profile eligible/reuse indicator.

The primary router does not ingest the full maps. Per camera it uses exactly 20
scalars: change mean, maximum, top-4 mean, three quartiles, two fixed-threshold
exceedance fractions, salience-weighted change, protected-region change
fraction, two-dimensional change centroid, spatial spread, age mean/maximum/
median/90th percentile/fraction-at-limit, eligible fraction, and planned reuse
fraction. Empty-set conventions are fixed before training. This reduces
overfitting and preprocessing cost.

### 8.2 Robot/action/history features

Include the eight normalized proprioceptive coordinates and their one- and
two-query deltas; per-action-dimension first, last, mean, standard deviation,
and maximum-absolute summaries of previously executed steps; gripper transition;
normalized query index, executed-step count, action-queue position, and anchor
age; and an eight-category prior-abort one-hot. Do not also flatten the full
chunk. With the 16 instruction features below, the pinned `8`-state/`7`-action
stack produces exactly 128 scalar features. Candidate profile and horizon enter
through the separate learned embedding. A runtime dimension mismatch fails
closed rather than changing the architecture.

Action features describe the portion actually executed by the environment and
its queue state, not merely the full chunk predicted at the prior query.

### 8.3 Instruction features

The default instruction representation is a 16-bin signed feature hash of the
pinned tokenizer IDs, with a published fixed hash seed and L2 normalization,
computed once per episode. It is deterministic,
requires no new language model, and prevents a large learned projection from
overfitting the small branch dataset. A frozen mean of base input embeddings is
an optional B5 challenger only if its one-time and amortized costs, leakage
controls, and held-out-task behavior pass. It is not part of the default method.

All continuous features are normalized using training groups only. Missing or
nonfinite required values force FR.

## 9. Router architecture

The primary architecture is intentionally sample-efficient:

1. deterministic feature compressor producing at most 128 scalar features;
2. `h_x = Linear(d_x,32) -> SiLU -> Dropout`;
3. concatenate `h_x` with an 8-dimensional learned profile/horizon embedding;
4. `Linear(40,16) -> SiLU`; and
5. `Linear(16,1)` harm-ranking score `s_theta(x,c)`.

With 128 inputs and at most three deployable contract identities, this is 4,825
trainable parameters including biases and the `3 x 8` embedding. The count is
recomputed and asserted from the instantiated model; it must remain below
10,000. The only model challenger is L2-regularized logistic regression on the
identical features.
The bounded validation grid contains:

- hidden model versus logistic model;
- L2 coefficient in a three-value log grid;
- dropout in `{0,0.1}` for the hidden model; and
- five fixed optimization seeds.

No CNN, transformer, current VLA feature probe, unbounded architecture search,
or per-task model is allowed in V1. If the small models cannot predict harm,
BRACE stops rather than purchasing accuracy with latency-erasing inference.

No pointwise monotonicity constraint is imposed on age, image change, or reuse
ratio. Those variables are plausible risk indicators, but the project has no
right to assume that more reuse is harmful in every state. Monotonic trends are
reported diagnostically.

## 10. Paired terminal-effect target

Let `S_i^F,S_i^c` be terminal success for FR and assigned contract arms from
the same reconstructed branch. Define signed effect

\[
\Delta_i(c)=S_i^F-S_i^c\in\{-1,0,1\},
\]

and labels

\[
Y_i^H=\mathbb{1}[\Delta_i=1],\qquad
Y_i^B=\mathbb{1}[\Delta_i=-1].
\]

`Y^H` is the router target. Beneficial, neutral-success, and neutral-failure
cases remain separate in every record and analysis. A beneficial case is not
deleted or relabeled as noise.

The estimand is policy-specific: incremental terminal harm of applying
contract `c` for its complete bounded horizon, followed by the frozen common
continuation policy, from states generated by the versioned behavior mixture.
It is not an intrinsic safety probability or a guarantee under arbitrary future
controllers.

## 11. Exact branch construction

A valid treatment state is the first policy query after a dense anchor. For
each arm independently:

1. reset to the same published task, initial state, and seed;
2. replay the identical low-level action prefix through normal `env.step()`;
3. at the recorded anchor observation, run the same dense anchor query;
4. require matching anchor action, cache, sequence map, and provenance digests;
5. execute the identical dense-anchor action chunk;
6. require matching pre-treatment simulator/observation/counter digests;
7. reject the branch if the anchor chunk terminated the episode; otherwise
   apply either FR or the assigned contract for `h` policy queries, filling the
   remainder with FR after a contract abort; and
8. use the same versioned continuation policy thereafter.

Re-running a dense pass at the treatment observation is invalid because it
changes the contract source and shifts the intervention. Restoring MuJoCo
coordinates without replaying controller, observation, wrapper, action-queue,
and cache history is also invalid.

Contract assignment `mu(c|x)`, behavior-mixture probability, branch-time
inclusion probability `q(x)`, and arm order are randomized and logged before
outcomes. Environment, model, and library RNG identities and deterministic-mode
settings are restored and digested for each arm. Duplicate FR/FR and contract/
contract arms must have zero unexplained terminal discordance.

## 12. Training objective

Only the assigned contract label is observed at a sampled state. Define the
design weight

\[
w_i=\frac{1}{q_i\,\mu(c_i\mid x_i)}.
\]

Use stabilized, predeclared clipping for optimization only:

\[
\widetilde w_i=\min(w_i,w_{max})/\operatorname{mean}[min(w,w_{max})].
\]

Train the ranking model with class-balanced weighted binary cross-entropy:

\[
\mathcal L(\theta)=\frac{1}{N}\sum_i\widetilde w_i
\left[-\alpha Y_i^H\log\sigma(s_i)
-(1-Y_i^H)\log(1-\sigma(s_i))\right]+\lambda\|\theta\|_2^2.
\]

Class balance `alpha` is computed from training groups only. Because class
balancing and clipping distort probability calibration, `sigma(s)` is never
reported as the true harm probability. The model supplies a ranking score;
risk is estimated separately on representative calibration groups.

Set `alpha` to the weighted negative/positive ratio on training groups; a split
with either class absent cannot train a router. Freeze `w_max` from the
outcome-blind design-weight distribution before labels. The MLP uses AdamW at
learning rate `3e-4`, batches of at most 64 branch rows, gradient-norm cap `1`,
at most 200 epochs, and patience 20 on grouped model-selection validation loss.
The L2 grid is `{1e-5,1e-4,1e-3}`. Logistic regression uses the same L2 grid and
an L-BFGS solver capped at 1,000 iterations. The nine architecture/regularization
configurations (six MLP, three logistic) and five seeds are the complete model
search; failed convergence rejects that configuration and does not expand it.

Training, model-selection validation, calibration, and held-out splits are
disjoint and grouped by complete task/initial-state/seed episodes. Five seeds
are trained; the frozen validation rule selects one seed or averages ranks
across all five. Calibration does not choose architecture, and no held-out
branch or confirmation outcome participates.

After the B4 screen, B5 may retain at most three deployable contracts. Other
profiles remain reported development conditions but cannot enter the router.
This cap limits assignment-weight variance, joint-selection multiplicity, and
the chance that a small router simply memorizes contract-specific noise.

## 13. Joint selective routing and calibration

For threshold vector `lambda`, define candidate acceptance

\[
g_{\lambda,c}(x)=\mathbb{1}[s_\theta(x,c)\le\lambda_c]
\mathbb{1}[c\text{ is mechanically eligible}]
\mathbb{1}[x\text{ is within calibration support}].
\]

Let `T_hat(c)` be the frozen synchronized B3 contract-cycle time. The deployed
router is the **joint** decision

\[
\pi_\lambda(x)=
\begin{cases}
\arg\min_{c:g_{\lambda,c}(x)=1}\widehat T(c),&\text{if any accepts},\\
FR,&\text{otherwise}.
\end{cases}
\]

Independent per-contract calibration is insufficient because selecting the
fastest accepted contract creates an additional multiple-selection effect.
Calibrate and report the complete `pi_lambda` rule on representative groups.

Define service coverage and selective harm:

\[
\kappa(\lambda)=P[\pi_\lambda(X)\ne FR],
\]

\[
R_{sel}(\lambda)=
\frac{E[Y^H(\pi_\lambda(X))\mathbb{1}[\pi_\lambda(X)\ne FR]]}
{\kappa(\lambda)}.
\]

Also report unconditional incremental harm

\[
R_{pop}(\lambda)=E[Y^H(\pi_\lambda(X))
\mathbb{1}[\pi_\lambda(X)\ne FR]].
\]

Use logged assignment/inclusion probabilities to estimate the joint rule and
report effective sample size, maximum weight, overlap, and unclipped/clipped
sensitivity. For sampled record `i`, let `c_i` be its randomized contract and
`pi_i=pi_lambda(x_i)`. Define context weight `v_i=1/q_i` and matched-outcome
weight `w_i=1/[q_i mu(c_i|x_i)]`. The primary Hájek estimates are

\[
\widehat R_{pop}(\lambda)=
\frac{\sum_i w_i\mathbb{1}[c_i=\pi_i]
\mathbb{1}[\pi_i\ne FR]Y_i^H}{\sum_i v_i},
\]

and

\[
\widehat\kappa(\lambda)=
\frac{\sum_i v_i\mathbb{1}[\pi_i\ne FR]}{\sum_i v_i}.
\]

Then `R_hat_sel=R_hat_pop/kappa_hat` when coverage is nonzero. Report the
Horvitz--Thompson form additionally only when the complete sampling-frame size
is known. A branch with assigned contract different from the deployed choice
contributes no observed outcome for that choice; imputing it from the score is
prohibited.

Thresholds form at most 20 predeclared joint candidate policies, not a
combinatorial post-hoc search. Evaluate these candidates on the disjoint
model-selection validation groups with the propensity-aware estimates above,
simultaneous Bonferroni-corrected diagnostics, and 10,000 fixed-seed
episode-group bootstrap resamples. Freeze exactly one joint policy before
calibration. Calibration may only accept or reject that policy; it may not move
to another threshold or contract after observing calibration outcomes.

The dedicated representative calibration then samples exactly one query
boundary per independent episode group i.i.d. from a frozen, declared
episode-balanced task/seed/query mixture. An ineligible boundary is retained as
an FR decision. At a served
boundary, execute the already-frozen joint policy's selected contract and its
paired FR arm directly—there is no randomized multi-contract off-policy match.
Define `Y_i^pi=0` for FR decisions and the observed paired harmful indicator for
served decisions. Thus the `Y_i^pi` are direct Bernoulli observations of the
frozen policy's local population harm under the declared mixture.

For `k=sum_i Y_i^pi` among `n` independent episode groups, compute the exact
one-sided 95% Clopper--Pearson upper limit

\[
U^H=\begin{cases}
\operatorname{Beta}^{-1}(0.95;k+1,n-k),&k<n,\\
1,&k=n.
\end{cases}
\]

After validation freezes the joint policy but before calibration outcomes, use
conservative B3 complete-cycle timings to derive `kappa_min`, the minimum service
coverage that still projects at least 12% net critical-path reduction when every
served decision uses the slowest contract the frozen policy can select and all
router/rejection overhead is charged, leaving a two-point development buffer
above the final 10% gate. If no `kappa<=1` satisfies it, stop. For observed
served count `s`, compute the exact one-sided 95% coverage lower limit
`L_kappa=Beta^{-1}(0.05;s,n-s+1)`, with `L_kappa=0` when `s=0`.

Accept only if `U^H<=0.01`, `L_kappa>=kappa_min`, and task/contract diagnostics
show no catastrophic stratum. Power `n` before labels;
even with zero harm, at least 299 independent groups are needed because
`1-0.05^(1/n)<=0.01`. Clustering, attrition, or a nonrepresentative query sampler
requires a larger prospective design or rejection. This direct final
calibration replaces a statistically inefficient multi-contract OPE guarantee;
the Hájek/HCOPE calculations remain development sensitivity analyses, not the
acceptance certificate.

This branch-level threshold does **not** reserve one point of the final 2-point
episode-success margin. An episode can contain multiple dependent routing
decisions, so no additive conversion is valid. B5 is a conservative local
development filter; B6 measures repeated-decision shift and B7/B8 alone can
establish episode-level non-inferiority.

This is an empirical development rule, not a distribution-free or physical-
safety guarantee. If overlap or effective sample size is inadequate, no router
operating point exists. B7/B8 closed-loop success remains the decisive test.

## 14. Deployment algorithm

```text
BRACE_QUERY(observation o_t, state q_t):
    validate episode, model, checkpoint, sequence, camera, and cache identity
    if q_t.mode is ANCHOR or any invariant fails:
        action, cache, gate, ledger = DENSE_FORWARD(o_t)
        return action, NEW_ANCHOR_STATE(cache, gate, ledger)

    if this is the first query after the anchor:
        candidates = mechanically eligible frozen contracts
        features = PRE_FORWARD_FEATURES(o_t, candidates, q_t.ledger)
        accepted = calibrated joint-router accepts
        if accepted is empty:
            action, cache, gate, ledger = DENSE_FORWARD(o_t)
            return action, NEW_ANCHOR_STATE(cache, gate, ledger)
        contract = lowest measured-cycle-time accepted contract
        q_t = START_CONTRACT(contract)

    reuse_sets = NESTED_REUSE_SETS(o_t, q_t.contract, q_t.ledger)
    validate positions, nestedness, sources, ages, budgets, and ring buffer
    if validation fails:
        action, cache, gate, ledger = DENSE_FORWARD(o_t)
        return action, NEW_ANCHOR_STATE(cache, gate, ledger)

    action, cache = ACCELERATED_FORWARD(o_t, reuse_sets, q_t.cache)
    ledger = POSITION_PRESERVING_LEDGER_UPDATE(reuse_sets)
    decrement remaining horizon
    mark next query ANCHOR when horizon expires
    return action, updated state
```

The router runs once per contract, not once per accelerated query. Hard-abort
checks and current-source eligibility still run every query and count in
latency.

## 15. Training-data algorithm

```text
COLLECT_BRANCH(behavior transcript, sampled anchor, assigned contract c):
    precommit run identity, contract, probabilities, order, and caps
    for arm in randomized [FR, c] order:
        reset environment
        replay exact low-level prefix to anchor observation
        run and verify dense anchor; execute its action chunk
        verify the pre-treatment branch state
        execute assigned h-query treatment with immutable ITT logging
        after the first contract abort, force FR for the remaining treatment queries
        continue with the versioned common policy to terminal outcome
    validate controls and reconcile both arms
    emit signed terminal effect and all four paired categories
```

Training code never receives protected-population outcomes. Branch artifacts
contain scalar/router features and digests; persistent raw K/V tensors are not
required after verification.

## 16. Timing and complexity accounting

Let dense query time be `T_F`, router-once cost `T_R`, eligibility/check cost at
accelerated query `k` be `T_E,k`, accelerated forward time `T_p,k`, `a_k`
indicate that the assigned contract is still active at query `k`, and `b_k`
indicate its first abort at that query. Thus `b_k <= a_k`, and after an abort
all later `a` and `b` values are zero. Fixed-window contract-cycle time is

\[
T_{cycle}(c)=T_F+T_R+\sum_{k=1}^{h}
[(a_k-b_k)(T_{E,k}+T_{p,k})+b_k(T_{E,k}+T_F)
+(1-a_k)T_F].
\]

The synchronized reduction is

\[
G(c)=1-\frac{T_{cycle}(c)}{(h+1)T_F}.
\]

Report both assignment-level intent-to-treat `G` and served-only diagnostics.
The positive method requires at least 10% net critical-path reduction in sealed
closed-loop evaluation, not merely kernel/FLOP savings.

Cache storage is

\[
M_{KV}=2B\sum_l N_{vis}H_{kv,l}d_{head,l},
\]

plus source images, ledgers, temporary active-token tensors, and runtime
workspace. Measure peak allocated and reserved memory; symbolic estimates do
not satisfy B3.

## 17. Implementation boundary

BRACE implementation, if B1 and later gates authorize it, belongs in a new
`src/savr/brace/` package. It must not alter historical SAVR/ACR behavior.

| Planned module | Responsibility |
|---|---|
| `types.py` | Frozen identities, profiles, contracts, states, branch labels |
| `sequence_map.py` | Runtime camera/text/proprio/action position discovery |
| `ledger.py` | Per-layer/token K/V source ownership and validation |
| `patch_change.py` | Vectorized preprocessed current-source scores |
| `profiles.py` | Outcome-blind profile grid and nested reuse sets |
| `cache_adapter.py` | Transactional corrected DynamicCache integration |
| `anchor.py` | Dense anchor, optional semantic map, parity checks |
| `router_features.py` | Deterministic <=128-dimensional pre-forward features |
| `router.py` | Logistic/MLP scoring and joint contract selection |
| `branch_replay.py` | Exact reset-prefix-anchor-treatment reconstruction |
| `records.py` | Immutable propensity, arm, query, and terminal records |
| `statistics.py` | Grouped IPW, risk--coverage, NI, multiplicity, power |
| `timing.py` | Synchronized component/cycle/ITT timing and memory |

All external integrations remain isolated and pinned. The official VLA-Cache
evaluator is not a valid scientific baseline without the documented minimal
correction.

## 18. Mandatory unit and integration properties

Before any outcome:

1. runtime sequence spans match the actual multimodal tensor;
2. no fixed prompt/text indices exist;
3. nested reuse sets and nondecreasing layer budgets are enforced;
4. a token can enter only when suffix-eligible at every deeper pruning layer;
5. cache `index_copy` preserves sequence positions;
6. every layer/token source transition matches the reuse set and resolves to a
   complete immutable multimodal source record;
7. source-image ring-buffer eviction cannot remove a live source;
8. scene/wrist token maps cannot alias or swap;
9. dense-anchor attention capture uses post-RoPE tensors and has
   action/backend/timing parity or is
   disabled;
10. P0/BRACE-disabled execution matches FR;
11. visual K/V changes under isolated proprio/action perturbations are measured
    and exact-source context envelopes abort as specified;
12. nonvisual positions are always current;
13. router features use no forbidden current-forward information;
14. joint selection, not independent thresholds alone, is calibrated;
15. propensity zero/near-zero and low-effective-sample-size cases reject;
16. modified prefix, anchor, cache, gate, and sequence maps reject;
17. duplicate-arm terminal discordance stops collection;
18. ITT records retain aborts and technical failures, and branch aborts fill
    the remaining assigned horizon with FR;
19. analysis code passes synthetic known-answer tests; and
20. all existing SAVR/ACR tests remain unchanged and pass.

## 19. Falsifiable hypotheses

| Hypothesis | Required evidence | Failure consequence |
|---|---|---|
| H1: exact simulator replay through the anchor-action boundary exists | B1 equality and negative controls | End BRACE |
| H2: corrected partial K/V and real anchor execution are faithful | B2 synthetic tests and B3 real parity/provenance | End or new protocol |
| H3: at least one clean contract has physical headroom | B3 >=8% development cycle gain | Stop before outcomes |
| H4: contract harm is neither absent nor ubiquitous | B4 representative prevalence | Static result or stop |
| H5: cheap pre-forward features rank harm | B5 held-out separation | End learned router |
| H6: terminal supervision beats cheaper labels | matched risk--coverage ablation | No BRACE contribution |
| H7: joint routing survives cache-induced shift | B6 drift and B7 closed loop | Stop or static method |
| H8: BRACE beats corrected cache/gates and is not dominated | B7/B8 complete frontier | No competitive positive paper |

These hypotheses prevent the implementation from treating the desired positive
result as an assumption.

## 20. Research grounding

- OpenVLA-OFT defines the pinned parallel continuous action-chunk policy and L1
  head: <https://www.roboticsproceedings.org/rss21/p017.html>
- VLA-Cache supplies the partial visual-KV substrate and layer-adaptive reuse:
  <https://openreview.net/forum?id=QZYZ0Xm58q>
- Gated VLA-Cache demonstrates the strength of a cheap confidence invalidator:
  <https://arxiv.org/abs/2608.10824>
- VLA-ADP uses text relevance and previous-action dynamics on OpenVLA-OFT:
  <https://openreview.net/forum?id=ea6j8k8Rnw>
- VLA-Pruner combines semantic and temporal action relevance:
  <https://arxiv.org/abs/2511.16449>
- SpecPrune-VLA uses previous global attention and action-aware pruning:
  <https://arxiv.org/abs/2509.05614>
- SP-VLA shows that action-history scheduling plus token pruning is already an
  established contribution: <https://openreview.net/forum?id=RwdGIIjPlC>
- LAC learns token selection/cache ratios end to end:
  <https://arxiv.org/abs/2602.00686>
- AC2-VLA jointly routes temporal, spatial, and depth computation:
  <https://arxiv.org/abs/2601.19634>
- Selective classification motivates measured risk--coverage rather than
  unconditional confidence claims:
  <https://papers.neurips.cc/paper_files/paper/2017/hash/4a8423d5e91fda00bb7e46540e2b0cf1-Abstract.html>
- Off-policy evaluation establishes the need for overlap and propensity-aware
  estimates: <https://proceedings.mlr.press/v70/wang17a.html>
- High-confidence off-policy evaluation supplies the conservative bounded-
  importance-weighted policy screen:
  <https://ojs.aaai.org/index.php/AAAI/article/view/9541>
- DAgger motivates the single explicitly versioned aggregation round:
  <https://proceedings.mlr.press/v15/ross11a.html>

Recent 2026 preprints, code availability, licenses, and claimed results must be
revalidated before B2, B7, and submission.

## 21. Next eligible work

Implement B1 only: transcript, exact simulator prefix replay through a
designated scripted anchor-action boundary, equality checks, and corrupted-
prefix/direct-state negative controls. B1 represents anchor metadata but does
not run the VLA or create a real K/V cache. Real anchor action/cache parity is a
B3 responsibility. Do not implement the cache, router, loss, or GPU path until
B1 passes and B2 is separately authorized.
