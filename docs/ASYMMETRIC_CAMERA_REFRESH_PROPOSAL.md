# Asymmetric Camera Refresh for Efficient Multi-View VLA Inference

Status: RESEARCH AND METHOD PROPOSAL — NOT IMPLEMENTED OR VALIDATED

Working method name: **Asymmetric Camera Refresh (ACR)**

Proposed primary variant: **State-Aware Asymmetric Camera Refresh (SA-ACR)**

Last updated: 2026-08-02

## 1. Executive summary

The original SAVR methods made one binary decision for the complete visual
prefix: either recompute both camera views or reuse both. That all-or-nothing
choice is poorly matched to a wrist-plus-scene camera system. The wrist camera
moves with the robot and observes contact-sensitive local geometry; the fixed
third-person camera provides comparatively stable global context.

ACR replaces the single refresh bit with a camera-specific refresh vector.
For the two-camera OpenVLA-OFT configuration:

- the wrist-camera representation is recomputed on every policy query;
- the third-person scene representation may be reused when a conservative,
  causal scene gate declares it safe;
- the fresh wrist tokens and fresh-or-cached scene tokens are concatenated in
  their original positions;
- current proprioception and the unchanged action head are used on every
  query.

The central hypothesis is that **the method can preserve the dynamic,
contact-relevant view while exploiting temporal redundancy only in the stable
global view**. This is materially different from SAVR1-3: no action is ever
predicted with an entirely stale visual prefix.

There are strong, testable reasons to pursue this route:

1. The project's own forensic evidence found 372 wrist-camera threshold
   exceedances versus only two third-person exceedances among concealed
   camera changes in the best SAVR1 setting.
2. SAVR3 logged 279 wrist-image triggers versus eight full-image triggers.
3. The pinned OpenVLA-OFT implementation processes each camera independently
   through the same DINOv2/SigLIP towers, concatenates per-camera patch tokens,
   and then applies a token-wise projector. The computation can therefore be
   factored by camera without changing weights or token order.
4. Prior VLA research supports the broader premises that adjacent visual
   computation is redundant, that action/task phase should regulate
   computation, and that multi-view inputs should not necessarily receive
   uniform compute. ACR combines these ideas in a deterministic,
   training-free, temporal camera cache.

These reasons make ACR plausible, not guaranteed. It must be rejected if
camera-factorized Full Refresh does not reproduce upstream outputs, if scene
reuse harms success beyond a predeclared margin, if savings are too small, or
if a final novelty review finds the method already established.

## 2. Relationship to the completed SAVR work

### 2.1 What remains valid

The following established infrastructure remains useful:

- the pinned OpenVLA-OFT model, action head, checkpoint, and LIBERO stack;
- the projected-visual-token boundary before current proprioception;
- context-safe cache identity and invalidation;
- immutable episode/query records;
- component-level CUDA timing and call accounting;
- exact wrapped-FR parity checks;
- the principle that task success is the primary constraint;
- final-holdout protections and reproducibility rules.

### 2.2 What changes fundamentally

SAVR1-3 treated the two-camera representation as one indivisible cache item:

\[
z_t = E_\phi(o_t^{S}, o_t^{W})
\]

and either recomputed or reused all of \(z_t\). ACR instead represents the
visual prefix as ordered camera blocks:

\[
z_t = [z_t^{S}; z_t^{W}],
\]

where \(S\) denotes the fixed scene camera and \(W\) denotes the wrist camera.
Each block has its own refresh state. For the proposed primary policy,
\(z_t^{W}\) is always fresh while \(z_t^{S}\) is selectively cached.

### 2.3 What must not happen

- SAVR3 may not be tuned or rerun.
- States `0-9` from LIBERO-Spatial may not be called fresh validation for
  another post-hoc whole-prefix SAVR variant.
- The ACR proposal may not be presented as a successful method before new
  correctness and outcome evidence exists.
- Existing negative results must remain visible in the paper's project
  history and limitations.

## 3. Empirical motivation from this project

### 3.1 The cameras exhibited strongly different dynamics

The SAVR1 forensic analysis examined 783 reuse queries from
`savr-s25-h2`. In 374 cases, at least one individual camera exceeded the shared
threshold even though their mean did not. The exceedances were:

| Camera | Individual exceedances hidden by the mean |
|---|---:|
| Wrist | 372 |
| Third-person scene | 2 |

This is direct evidence that averaging the views combined two different
temporal regimes. The moving wrist camera dominated hidden changes while the
scene camera was rarely the offending view.

### 3.2 Independent safety gates made whole-prefix reuse vanish

SAVR2 corrected camera averaging by allowing either camera to veto reuse.
SAVR3 strengthened the wrist gate further. It recovered success from 52/100
for the best SAVR1 setting to 69/70 on its validation population, but full
prefix reuse fell to 9/944 queries (0.95%). Its trigger totals included:

| Trigger family | Count |
|---|---:|
| Wrist-image change | 279 |
| Full-image change | 8 |
| Translation direction reversal | 412 |
| Insufficient stable-fresh history | 899 |
| Prefix skip-budget cap | 424 |

The safety logic was working as designed, but a wrist veto unnecessarily
forced recomputation of the scene-camera representation too. ACR removes that
coupling.

### 3.3 Whole-prefix staleness changed actions immediately

For all 87 SAVR1 episodes with comparable FR prefixes, the first action-hash
mismatch occurred exactly at the first full-prefix reuse. Under ACR, the
policy will still receive current wrist evidence at every query. This does not
prove that actions will remain safe, but it directly targets the most serious
weakness of the prior method: generating a complete action chunk without any
fresh visual representation.

### 3.4 The available compute is worth targeting, but bounded

The FR pilot measured visual backbone plus projector time at 15.874% of policy
query CUDA time. ACR cannot remove all of that work because one camera always
refreshes. Its likely end-to-end gain is therefore modest. The method should be
judged on measured per-camera visual FLOPs/time and total query latency, not
inflated theoretical savings.

## 4. Relevant research and positioning

### 4.1 Foundation system

OpenVLA fuses DINOv2 and SigLIP features before a Llama 2 language backbone.
OpenVLA-OFT adds parallel decoding, continuous actions, and action chunking and
supports multiple images. These properties define the proposed implementation
boundary.

### 4.2 Visual caching and pruning

VLA-Cache reuses computation for minimally changed visual tokens. VLA-ADP
changes token retention according to action dynamics and task phase. VLA-IAP
uses interaction-aligned visual-token pruning. Together, they support the
general premise that VLA visual processing contains exploitable redundancy and
that uniform computation is not always necessary.

### 4.3 Multi-view selective computation

The closest identified work is *Selective Perception for Robot: Task-Aware
Attention in Multimodal VLA* (2026), which trains a lightweight router using
the prompt and wrist observation to estimate the relevance of other views and
attenuate computation for low-utility views. Multi-camera distillation and
view-scaling work also show that camera views have different information roles.

This creates a real novelty risk. ACR should not be positioned merely as
"skip an unimportant camera." Its proposed distinction is:

- temporal reuse rather than dropping or attenuating the scene view;
- preservation of both view-token blocks in their original sequence;
- a training-free deterministic controller rather than a learned router;
- explicit cache-age, state/action, and fail-closed safety semantics;
- direct evaluation of camera-factorized caching in a chunked OpenVLA-OFT
  policy.

A complete full-text and code comparison with Selective Perception, VLA-Cache,
VLA-ADP, and any later camera-routing work is mandatory before claiming
novelty.

## 5. Problem formulation

Let a policy query at time \(t\) receive:

- scene image \(o_t^{S}\);
- wrist image \(o_t^{W}\);
- language instruction \(x\);
- current robot state \(s_t\);
- recent predicted action chunks \(\hat a_{t-1}, \hat a_{t-2}, \ldots\).

Let the frozen visual encoder/projector for one image be:

\[
z_t^c = P_\psi(E_\phi(o_t^c)), \qquad c \in \{S,W\}.
\]

The general multi-camera refresh decision is a vector:

\[
\mathbf r_t = (r_t^{S}, r_t^{W}), \qquad r_t^c \in \{0,1\}.
\]

Full Refresh uses \((1,1)\). Whole-prefix reuse used only \((1,1)\) or
\((0,0)\). The primary ACR constraint is:

\[
r_t^{W} = 1 \quad \forall t,
\]

while \(r_t^{S}\) is chosen causally.

Let \(\tau_t^S\) be the last query that refreshed the scene block. The visual
prefix supplied to the unchanged policy is:

\[
\tilde z_t =
\left[
  r_t^S z_t^S + (1-r_t^S)z_{\tau_t^S}^{S};
  z_t^W
\right].
\]

The action chunk remains:

\[
\hat a_t = \pi_\theta(x, \tilde z_t, s_t),
\]

with unchanged parameters \(\theta,\phi,\psi\), action head, prompt, and
preprocessing.

The optimization target is to reduce visual cost subject to a predeclared
success constraint:

\[
\min_g \; \mathbb E[C_{\mathrm{vis}}(g)]
\quad \text{s.t.} \quad
S(g) \ge S(\mathrm{FR}) - \epsilon.
\]

The future protocol must define \(\epsilon\), the population, and the minimum
meaningful compute reduction before observing ACR outcomes.

## 6. Proposed SA-ACR method

### 6.1 Ordered per-camera token blocks

The model's input order is scene first, wrist second. SA-ACR preserves this
order exactly:

1. obtain or load the scene projected-patch block;
2. compute the current wrist projected-patch block;
3. concatenate `[scene_tokens, wrist_tokens]`;
4. append current proprioception through the existing projector;
5. execute the unchanged language backbone and action head.

No camera token is moved, resized, pooled, or replaced by a placeholder.

### 6.2 Scene-cache state

The scene cache stores:

- projected scene patch embeddings;
- the corresponding downsampled reference scene image;
- context identity: episode, task, instruction, checkpoint, configuration;
- tensor shape, patch count, dtype, device, and language dimension;
- last scene-refresh query and environment step;
- scene-cache age;
- robot state and action context at the last scene refresh;
- controller counters required for exact resume/recovery.

Any missing or incompatible value forces a scene refresh.

### 6.3 Always-fresh wrist path

The wrist view is recomputed on every policy query. This is a hard method
property, not a tunable threshold. It is motivated by:

- ego-motion from the moving end effector;
- rapid local geometry changes near grasp, contact, and release;
- occlusion and object motion in the hand;
- the project's overwhelming wrist-versus-scene change asymmetry.

This guarantee means SA-ACR never generates an action chunk from an entirely
cached visual prefix.

### 6.4 Scene-image change signal

Compare only the current third-person scene image with its most recent scene
refresh reference. Reuse should not depend on the wrist image because the wrist
view is always fresh.

The proposed signal retains the existing deterministic local-patch design:

1. normalize the scene image to `[0,1]`;
2. create a deterministic `32 x 32` representation;
3. partition it into an `8 x 8` grid;
4. compute mean absolute change for each patch;
5. aggregate the largest local changes rather than one global mean;
6. refresh the scene block when the aggregate exceeds
   \(\gamma_{\mathrm{scene}}\).

The future design phase should test whether robot-arm motion in the scene view
causes false positives and whether a fixed robot-region mask is scientifically
justified. No mask is assumed in this proposal.

### 6.5 Scene-relative state signal

The state signal should measure movement since the last **scene refresh**, not
merely since the preceding query. A candidate translation score is:

\[
\Delta_t^{\mathrm{pos},S} =
\left\|
\operatorname{norm}(p_t) -
\operatorname{norm}(p_{\tau_t^S})
\right\|_2,
\]

where \(p_t\) is end-effector position. Large accumulated workspace motion can
make global scene context stale even when each adjacent step is small.

Orientation and gripper state should not automatically veto scene reuse unless
new evidence shows they affect the scene-view requirement. The fresh wrist view
already captures many local consequences of these changes. This avoids
recreating SAVR3's overconservative all-signal OR gate.

### 6.6 Task-transition and action-risk signal

A small set of interpretable events may force scene refresh:

- a gripper open/closed transition;
- a large change in the endpoint of predicted translational motion;
- a translation direction reversal indicating a new coarse motion phase;
- a task/instruction/context change.

These signals are proposed candidates, not all mandatory. The planning phase
must determine a minimal primary rule before outcome collection. Adding every
SAVR3 veto would likely eliminate the intended savings.

### 6.7 Temporal safety

Proposed fail-closed temporal rules are:

- initial scene refresh at each episode/context;
- finite maximum scene-cache age \(H_S\);
- immediate invalidation on context or metadata mismatch;
- no reuse on invalid/non-finite signal input;
- optional warm-up before first scene reuse;
- no requirement for two globally stable fresh queries, because the wrist
  view is already fresh at every query;
- consecutive scene reuse is allowed up to \(H_S\), which is the main source
  of compute savings.

The exact horizon and whether a warm-up is necessary must be frozen later.

### 6.8 Proposed primary decision rule

A concrete candidate rule is:

\[
r_t^S = \mathbb I\left[
\begin{aligned}
&\neg \mathrm{validCache}_t^S \,\vee\,
\Delta_t^{\mathrm{scene}} > \gamma_{\mathrm{scene}} \,\vee\\
&\Delta_t^{\mathrm{pos},S} > \gamma_{\mathrm{pos}} \,\vee\,
q_t^{\mathrm{transition}} = 1 \,\vee\\
&h_t^S \ge H_S
\end{aligned}
\right],
\qquad r_t^W=1.
\]

Here \(q_t^{\mathrm{transition}}\) is a predeclared, minimal transition veto.
The future design protocol must define it exactly and must not add conditions
after viewing ACR outcomes.

### 6.9 Controller pseudocode

```text
begin episode/context:
    invalidate scene cache
    reset scene reference and counters

for each policy query t:
    preprocess current scene image and current wrist image
    compute scene-change, scene-relative state, and transition signals

    if scene cache invalid OR any frozen scene-refresh condition fires:
        scene_tokens = encode_and_project(current_scene)
        replace scene cache and scene reference
        scene_event = SCENE_REFRESH
    else:
        scene_tokens = load_cached_scene_tokens()
        scene_event = SCENE_REUSE

    wrist_tokens = encode_and_project(current_wrist)  # always fresh
    visual_prefix = concatenate(scene_tokens, wrist_tokens)
    append current proprioception using unchanged upstream path
    actions = unchanged_vla_and_action_head(visual_prefix)

    log signals, decision, cache metadata, component calls, and actions
```

## 7. Exact compatibility with pinned OpenVLA-OFT

### 7.1 Verified upstream computation

At pinned revision `e4287e94541f459edc4feabc4e181f537cd569a8`:

1. LIBERO prepares `full_image` and `wrist_image` separately.
2. Each image is transformed for the fused SigLIP/DINOv2 backbone.
3. The two transformed images are concatenated by channels in scene-first,
   wrist-second order.
4. The vision backbone splits the tensor into per-image six-channel blocks.
5. Each image independently passes through SigLIP and DINOv2.
6. Their features are concatenated along the hidden dimension per image.
7. Per-image patch blocks are concatenated along the token dimension.
8. `PrismaticProjector`, an MLP applied independently to each token, maps them
   to the language dimension.
9. Current proprioception is appended after the projected visual block.

There is no cross-camera mixing in the vision towers or projector before the
two token blocks enter the language model.

### 7.2 Factorization identity

Because the projector acts token-wise:

\[
P_\psi([u^S;u^W]) = [P_\psi(u^S);P_\psi(u^W)].
\]

Therefore the original fresh computation should be reproducible by processing
each camera separately and concatenating in the original order. This is the
critical technical reason ACR is feasible without changing weights.

The identity must still be tested on real tensors. Kernel invocation order,
dtype behavior, or implementation details could prevent bitwise equivalence.
Failure of exact or tightly bounded parity is a correctness stop.

### 7.3 Required adapter change

The existing adapter intercepts one whole-prefix
`_process_vision_features` call. ACR requires a new camera-factorized adapter
that:

- splits the processed input into exact scene/wrist channel blocks;
- runs the original per-image SigLIP/DINOv2 path only for fresh blocks;
- applies the unchanged projector per block;
- stores only scene projected tokens;
- concatenates scene then wrist projected tokens;
- returns the same complete shape expected by upstream code;
- leaves upstream source files unmodified.

This should be a separate adapter and controller. SAVR1-3 code and evidence
must remain unchanged.

## 8. Expected compute behavior

Let scene and wrist visual costs be \(C_S\) and \(C_W\), and let
\(\rho_S\) be the scene refresh rate. Then:

\[
C_{\mathrm{vis}}^{\mathrm{FR}} = C_S + C_W,
\]

\[
C_{\mathrm{vis}}^{\mathrm{ACR}} = \rho_S C_S + C_W + C_{\mathrm{gate}}.
\]

Ignoring gate overhead, relative visual savings are:

\[
1 - \frac{\rho_S C_S+C_W}{C_S+C_W}.
\]

If the two views cost approximately the same, this becomes:

\[
\frac{1-\rho_S}{2}.
\]

Illustrative, not measured, examples:

| Scene refresh rate | Scene reuse rate | Approx. visual-work reduction if views cost equally | Approx. query-time reduction using the measured 15.874% visual fraction |
|---:|---:|---:|---:|
| 80% | 20% | 10% | 1.59% |
| 50% | 50% | 25% | 3.97% |
| 20% | 80% | 40% | 6.35% |

These estimates are ceilings before gate, split, cache-copy, and synchronization
overheads. The actual paper must report measured component time, FLOPs/calls,
and end-to-end latency.

## 9. Why this method has a credible chance of working

### 9.1 It targets the measured bottleneck in the old controller

The prior controller rarely reused because the wrist view vetoed the entire
prefix. ACR converts that veto into the intended behavior: refresh the wrist
without automatically refreshing the scene.

### 9.2 It retains current perception where manipulation is most dynamic

The wrist camera moves with the end effector and observes grasp/contact
geometry. Keeping it fresh addresses the most plausible safety failure of
whole-prefix reuse while allowing a fixed global view to serve as temporal
memory.

### 9.3 It preserves global context rather than dropping a camera

When the scene view is reused, its tokens remain present in the original token
positions. The policy receives global context plus current local perception.
This is less destructive than removing the scene camera entirely.

### 9.4 It matches the actual model factorization

The pinned model already loops over images independently before token
concatenation. ACR follows that structure instead of forcing an artificial
separation after cross-view fusion.

### 9.5 It is training-free and reversible

No weights, demonstrations, router training, or action-head changes are
required. A failed cache condition can always fall back to camera-factorized
FR. This keeps the causal interpretation and implementation scope narrow.

### 9.6 It creates a smoother efficiency control surface

Whole-prefix SAVR had a binary choice between all visual work and none. ACR
adds an intermediate operating mode: half-view refresh. This can produce
meaningful visual savings without the severe information loss associated with
an entirely stale prefix.

## 10. Research questions and hypotheses

These are hypotheses, not established results.

### Primary question

Can training-free, state-aware reuse of only the fixed scene-camera token block
reduce multi-view visual computation while preserving task success relative to
two-view Full Refresh?

### Primary hypothesis

SA-ACR will maintain success within a predeclared paired margin while reducing
scene-camera encoder/projector calls by a meaningful amount.

### Mechanism hypotheses

1. **Camera asymmetry:** the fixed scene view admits substantially more reuse
   than the wrist view.
2. **Fresh-local-perception:** always-fresh wrist tokens prevent most failures
   caused by complete visual staleness.
3. **State-aware safety:** accumulated end-effector displacement and a minimal
   transition veto improve safety beyond scene-image change alone.
4. **Ordered-token compatibility:** cached scene plus fresh wrist tokens remain
   compatible with the unchanged language/action stack.
5. **Graceful trade-off:** partial-camera refresh yields a smoother
   success-compute frontier than whole-prefix reuse.

## 11. Proposed comparison policies

The later experimental protocol should consider:

| Policy | Scene camera | Wrist camera | Purpose |
|---|---|---|---|
| Two-view FR | Fresh every query | Fresh every query | Correctness/performance oracle |
| Whole-prefix SAVR3 | Both reused or both fresh | Both reused or both fresh | Historical negative reference; no rerun on consumed split |
| Scene-periodic ACR | Fixed periodic refresh | Always fresh | Clock-based partial-camera baseline |
| Scene-visual ACR | Scene-image gate + horizon | Always fresh | Isolate visual-only scene gating |
| SA-ACR | Scene image + selected state/action safety + horizon | Always fresh | Proposed primary method |
| Wrist-only | Omitted or fixed null treatment | Always fresh | Test value of global context; likely ablation only |
| Scene-only | Always fresh | Omitted | Demonstrate wrist-view contribution; safety-bounded ablation only |

Matched baselines should be compared at similar **scene-camera encoder cost**,
not only at the old whole-prefix skip percentage.

## 12. Metrics required by the new method

### Task performance

- episode success and paired difference from FR;
- per-task and per-state success;
- horizon failures and technical failures separately;
- failure timing relative to scene reuse;
- action divergence from FR where comparable.

### Camera-specific efficiency

- scene refresh and reuse counts/rates;
- wrist refresh count, which must equal query count;
- per-camera DINOv2 and SigLIP call counts;
- per-camera projector calls;
- visual FLOPs or a reproducible proxy;
- per-camera and total visual CUDA time;
- total synchronized policy-query latency;
- complete episode wall time;
- cache memory and controller overhead.

### Correctness

- token-block shapes and ordering;
- tensor dtype/device identity;
- scene cache age and context identity;
- fresh wrist-image hash at every query;
- current proprioception at every query;
- exact downstream language/action-head counts;
- checkpoint and upstream cleanliness.

## 13. Correctness gates before any outcome experiment

1. **Factorized FR parity:** separately encode scene and wrist, concatenate,
   and match original two-view projected tokens and actions.
2. **Camera isolation:** changing only the wrist input must change only the
   wrist pre-LLM token block before concatenation; likewise for scene.
3. **Scene reuse proof:** a reuse query must invoke zero scene towers/projector
   work and exactly one wrist path.
4. **Fresh wrist proof:** the token block must correspond to the current wrist
   observation, not its cached predecessor.
5. **Current state proof:** current proprioception must be appended on both
   scene refresh and scene reuse.
6. **Fail-closed behavior:** cache mismatch, invalid signal, context change,
   unsupported FiLM, or unexpected model calls must force FR or stop.
7. **Immutable logging:** every camera decision and component event must be
   reconstructable from stored records.
8. **Unchanged source:** no edit to the pinned OpenVLA-OFT or LIBERO trees.

## 14. Evaluation route to be planned later

This proposal does not freeze a new experiment. The next planning document
should:

1. identify a genuinely independent development population, preferably a
   previously unused LIBERO suite supported by the existing four-suite
   checkpoint;
2. verify available tasks, initial states, and baseline success before
   assigning development/confirmation roles;
3. preserve Spatial states `10-49` and seeds `7/17/27` unless a new protocol
   explicitly defines a stronger untouched final evaluation;
4. freeze candidate count, thresholds, margins, resource limits, and negative
   stops before ACR outcomes;
5. use staged promotion so an unsafe configuration cannot consume the whole
   budget;
6. require matched camera-specific compute baselines and independent
   confirmation before a final holdout.

No exact split or threshold is asserted here because that is the next planning
task and must be verified against available assets first.

## 15. Proposed ablations

If the primary method becomes eligible, the paper should isolate:

1. always-fresh wrist versus whole-prefix reuse;
2. scene image only versus scene image + state;
3. transition veto on/off;
4. scene-cache horizon;
5. local top-k patch signal versus global mean;
6. cached projected tokens versus cached pre-projector camera features;
7. fixed periodic scene refresh at matched compute;
8. wrist-only and scene-only input, if technically and scientifically safe;
9. task phase or first-reuse timing;
10. per-camera timing asymmetry.

Ablations must follow primary selection. They cannot be used to retroactively
choose the main result.

## 16. Failure modes and falsification criteria

### Technical risks

- separate camera execution may not be bitwise equivalent because of kernel
  ordering or preprocessing details;
- FiLM-conditioned vision would complicate per-camera caching;
- cache copies/synchronization may erase theoretical savings;
- the scene and wrist views may have unequal costs;
- token-position or attention interactions may make mixed-age blocks unstable.

### Scientific risks

- global scene context may change in task-critical ways despite low pixel
  change;
- a stale scene can conflict with a fresh wrist view;
- the scene camera may be essential during long reaching motions;
- the achievable end-to-end speedup may be too small for a strong paper;
- an adjacent method may already cover the claimed contribution;
- success may remain sensitive to even one stale camera block.

### Predeclared reasons to abandon or redesign

The later protocol should stop if:

- factorized FR fails parity;
- bounded scene reuse violates component/cache invariants;
- low reuse already exceeds the frozen success margin;
- observed camera-specific savings are below a predeclared meaningful floor;
- the controller requires outcome-driven threshold changes;
- novelty cannot be distinguished from prior multi-view routing/caching work.

## 17. Anticipated manuscript changes

The current manuscript describes one scalar full-prefix refresh decision. ACR
requires a substantial rewrite rather than a small patch.

### Title

Candidate direction:

> Asymmetric Camera Refresh for Efficient Multi-View Vision-Language-Action
> Inference

Retaining "State-Aware" in the title should depend on whether state signals
remain in the frozen primary controller and are shown to contribute.

### Abstract

Rewrite around:

- uniform visual processing across heterogeneous cameras;
- always-fresh wrist plus selectively cached scene tokens;
- training-free camera-factorized inference;
- success, camera-specific compute, and end-to-end latency;
- only measured findings after evaluation.

### Introduction

Replace the generic consecutive-frame redundancy argument with the stronger
multi-view mismatch:

- cameras have different motion and information roles;
- all-or-nothing refresh couples their compute schedules;
- a dynamic wrist view should not force recomputation of a stable global view;
- a stable global view should not justify a stale wrist view.

The completed SAVR negative evidence can motivate the method transparently.

### Related work

Add dedicated subsections for:

- multi-view robot perception;
- visual token caching/pruning in VLAs;
- camera/view routing and selective perception;
- action- or phase-aware adaptive compute.

The comparison with Selective Perception must be explicit.

### Notation

Replace scalar camera observation/cache notation with:

- \(o_t^S,o_t^W\): scene and wrist images;
- \(z_t^S,z_t^W\): per-camera projected tokens;
- \(r_t^S,r_t^W\): per-camera decisions;
- \(\tau_t^S\): last scene refresh;
- \(h_t^S\): scene-cache age;
- \(C_S,C_W\): per-camera compute;
- \(\gamma_{\mathrm{scene}},\gamma_{\mathrm{pos}},H_S\): scene-gate values.

### Problem formulation

Replace the binary full-prefix refresh objective with vector-valued camera
refresh and camera-specific compute. State explicitly that the method uses
mixed-age visual blocks while preserving order and current state.

### Method

Completely replace the current method section with:

1. multi-view feature factorization;
2. always-fresh wrist pathway;
3. scene-cache contents and identity;
4. scene change/state/transition signals;
5. scene refresh rule;
6. token concatenation and unchanged downstream policy;
7. camera-specific cost model;
8. failure-safe behavior.

### Method figure

The new figure should show two parallel paths:

- wrist image -> encoder/projector every query;
- scene image -> gate -> refresh or scene cache;
- ordered concatenation -> fresh proprioception -> unchanged VLA/action head.

Use distinct colors for fresh wrist, fresh/cached scene, and unchanged
downstream computation.

### Baseline table

Replace FR/PR/VOR/SAVR-only comparison with camera-specific policies and state
whether each view is fresh, periodic, gated, cached, or absent.

### Experimental section

Add:

- per-camera call/timing instrumentation;
- factorized-FR parity;
- independent development and final splits;
- paired success margin;
- camera-specific compute matching;
- component and end-to-end efficiency;
- failure and mixed-age token analysis.

### Results and discussion

If positive, center the result on the success-versus-camera-compute frontier,
not raw scene reuse alone. If negative, connect back to the preserved SAVR
negative archive and report whether even camera factorization fails.

## 18. Proposed contributions if validated

Only after positive evidence, the paper could claim:

1. a training-free camera-factorized cache for multi-view VLA inference;
2. an asymmetric policy that keeps dynamic wrist perception fresh while
   selectively reusing stable scene context;
3. exact integration with an unchanged chunked OpenVLA-OFT policy;
4. an empirical study of camera-specific redundancy, success, and latency;
5. evidence that sensor-granular refresh offers a safer trade-off than
   whole-prefix reuse.

Until validation, these are proposed contributions.

## 19. Immediate next planning deliverables

After this proposal is accepted, the project should create—before coding or
new outcomes:

1. an ACR literature and novelty audit with full-text method comparisons;
2. a pinned source-boundary implementation design;
3. an independent split and evidence-reuse audit;
4. a frozen correctness protocol;
5. a staged calibration and positive/negative gate protocol;
6. explicit GPU, time, storage, and artifact limits;
7. a manuscript migration checklist that preserves the negative SAVR history.

## 20. Primary sources

- Kim et al., *OpenVLA: An Open-Source Vision-Language-Action Model*:
  https://arxiv.org/abs/2406.09246
- Kim, Finn, and Liang, *Fine-Tuning Vision-Language-Action Models:
  Optimizing Speed and Success* (OpenVLA-OFT):
  https://arxiv.org/abs/2502.19645
- Liu et al., *LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot
  Learning*: https://arxiv.org/abs/2306.03310
- Xu et al., *VLA-Cache: Towards Efficient Vision-Language-Action Model via
  Adaptive Token Caching in Robotic Manipulation*:
  https://arxiv.org/abs/2502.02175
- Pei et al., *Action-aware Dynamic Pruning for Efficient
  Vision-Language-Action Manipulation*: https://arxiv.org/abs/2509.22093
- Cheng et al., *VLA-IAP: Training-Free Visual Token Pruning via Interaction
  Alignment for Vision-Language-Action Models*:
  https://arxiv.org/abs/2603.22991
- Son et al., *Selective Perception for Robot: Task-Aware Attention in
  Multimodal VLA*: https://arxiv.org/abs/2602.15543
- Acar et al., *Visual-Policy Learning through Multi-Camera View to
  Single-Camera View Knowledge Distillation for Robot Manipulation Tasks*:
  https://arxiv.org/abs/2303.07026
- Xie et al., *Multi-Camera View Scaling for Data-Efficient Robot Imitation
  Learning*: https://arxiv.org/abs/2604.00557

## 21. Proposal decision

ACR is recommended for the next planning phase because it is directly
motivated by the project's camera-specific evidence, matches the pinned model's
computational structure, and removes the all-or-nothing failure mode without
making the wrist view stale. The recommendation is conditional on a full
novelty audit, factorized-FR parity, independent evaluation design, and
predeclared success/efficiency gates.
