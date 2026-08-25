# BRACE Research and Feasibility Audit

**Method:** Branch-Rollout Adaptive Cache Execution (BRACE)  
**Working title:** *Branch Before You Cache: Interventional Rollout Supervision for Reliable VLA Acceleration*  
**Date:** 2026-08-24  
**Status:** Research specification only; no implementation, model run, or BRACE result exists

## 1. Verdict

BRACE is a technically feasible and provisionally novel replacement for the
superseded OPCCR proposal, but it is still a research hypothesis rather than a
reliable path to a guaranteed positive paper.

The key correction is the supervision target. OPCCR proposed labeling cache
risk using immediate disagreement between full-refresh and cached actions. A
newer controlled study reports that cache-collapse failures can have negligible
single-step action deviation and that its tested early action-level detectors
were near chance. BRACE instead derives labels from paired closed-loop outcomes
after an explicitly randomized cache intervention from the same reconstructed
state.

Current calibrated judgment:

| Event | Pre-pilot plausibility |
|---|---:|
| Functional VLA-Cache path on one TITAN RTX | 70--80% |
| At least 10% synchronized query-wall acceleration | 60--70% |
| Exact replay-prefix branch reconstruction | 70--85% |
| Sufficient harmful/safe intervention labels for learning | 50--65% |
| Router predicts harmful interventions at useful coverage | 45--60% |
| Competitive positive result versus current cache baselines | **30--45%** |

These are engineering/research judgments, not statistical probabilities. If the
physical cache, branch-determinism, label-prevalence, and held-out-predictability
gates all pass, the conditional chance of a competitive result rises to roughly
55--65%. Before those gates, describing success as “highly likely” would be
misleading.

## 2. Exact problem

The project has already established two facts:

1. reuse decisions alter the action trajectory and therefore the future state
   distribution; and
2. encoder-only camera reuse has insufficient end-to-end latency headroom on
   the pinned model, whose residual downstream computation accounted for 83.7%
   of measured query CUDA time.

VLA-Cache attacks the correct downstream computation by reusing selected visual
KV entries. Its published OpenVLA-OFT result also shows that meaningful speed is
physically possible. The unsolved question is not whether caching can work on
average, but whether a deployment-time controller can identify when a specific
cache intervention will damage closed-loop behavior without first running the
full model.

The published bar is already high. VLA-Cache reports OpenVLA-OFT average success
of 97.4% versus 96.8% for vanilla and latency of 62.59 ms versus 79.05 ms.
Gated VLA-Cache reports essentially tied OpenVLA-OFT averages (95.8% for both
full inference and VLA-Cache, 95.7% for its gate). Action-JND reports a learned
adaptive operating point of 96.20% versus 95.35% for VLA-Cache, while adding
some latency relative to that cache baseline. The Gate, Not the Cache reports
18--22% faster critical-path serving when dense refresh is moved into actuation
slack. AC2-VLA jointly routes temporal reuse, token pruning, and layer execution
from visual/language/action context and reports up to 1.79x speedup with
comparable task success. These results come from different backbones,
benchmarks, implementations, and hardware and cannot be combined numerically,
but they show why beating FR alone is not publishable.

## 3. Method in one sentence

During training, BRACE reconstructs a visited policy state under identical
history, branches it into full-refresh and cache-treatment rollouts, observes the
paired effect on task outcome, and trains a lightweight fail-closed controller
to choose the fastest cache profile whose predicted harm remains below a frozen
risk threshold.

## 4. Interventional design

### 4.1 State distribution

Training states are sampled from actual rollouts of a behavior mixture over full
refresh and cache profiles. A second aggregation round, if earned, samples from
the first BRACE controller. Thus the router is trained on states caused by cache
decisions rather than only on clean full-refresh traces.

Sampling of branch points must be outcome-blind and stratified by:

- task and initial state;
- episode progress;
- cache profile and cache age;
- recent gripper transition and action magnitude;
- scene and wrist motion;
- previous refresh reason.

Stratification may enrich rare transition regimes for training, but prevalence,
risk--coverage, and closed-loop claims must be estimated on a separately sampled
deployment-representative population. Inclusion probabilities are recorded so
an enriched training set is never misreported as the natural cache-harm rate.

### 4.2 Treatment

At a sampled query boundary, compare:

- `T=0`: full refresh for the current action chunk;
- `T=p`: one frozen cache profile for the current action chunk.

Both arms then use the same frozen continuation policy. Round 0 uses full refresh
as the continuation; a later aggregation round may use the current BRACE policy.
This isolates the current compute intervention instead of confounding it with an
entirely different future policy.

To capture cache-age and compounding effects, a separately randomized burst
ablation may apply the same cache profile for 1, 2, or 4 consecutive query
chunks before the common continuation. Burst length is an input to the effect
model and is never chosen after observing outcomes.

### 4.3 Outcome and label

Each paired branch produces a utility vector:

1. terminal task success;
2. time or environment steps to success;
3. termination type and remaining episode budget;
4. optional task-predicate diagnostics used only if they are predeclared and
   never exposed to the deployment router.

The primary categorical effect is:

- **harmful:** FR succeeds and cache fails;
- **beneficial:** cache succeeds and FR fails;
- **neutral-success:** both succeed;
- **neutral-failure/unresolved:** both fail.

The method must retain all four categories. It may not discard full-refresh
failures and relabel the remaining data as if FR were infallible. For router
training, the primary risk target is the harmful category; beneficial and both-
failure cases are separate auxiliary targets.

Task success remains the primary label because LIBERO rewards are sparse.
Action distance, image distance, and predicate progress may be features or
diagnostics, but cannot replace the paired terminal-outcome label.

A frozen subset of branch points must also receive duplicate-arm controls
(`FR` versus `FR` and cache-profile versus the same cache-profile). Any terminal
discordance between identical treatments estimates intrinsic replay/model
variation rather than a cache effect. The pilot must stop if this discordance is
not negligible relative to the harmful-pair rate; it may not be subtracted away
after outcomes are known.

### 4.4 Deployment decision

Let `p` index predeclared cache profiles and let `r_theta(x_t,p)` estimate the
probability that profile `p` causes harm relative to current-step FR under the
frozen continuation distribution. BRACE selects:

`p* = fastest profile p whose calibrated upper risk estimate is at most epsilon`.

It selects FR when:

- no cache profile meets the risk bound;
- the input is rejected as out of distribution;
- cache provenance or age is invalid;
- the maximum reuse horizon is reached;
- profile-switching invariants fail; or
- an episode reset or task transition occurs.

The controller does not claim formal safety. It estimates task-reliability risk
within the evaluated simulator distribution.

## 5. Cache profiles

BRACE does not learn arbitrary token masks in its first version. That would
create a large action space and collide more directly with LAC and Action-JND.
It chooses among a small frozen menu:

| Profile | Description | Purpose |
|---|---|---|
| P0 | Full refresh | Oracle, fallback, and control |
| P1 | Conservative scene-only visual-KV reuse; wrist fresh | Evidence-driven low-risk option |
| P2 | Moderate VLA-Cache-compatible reuse across eligible views/layers | Main speed option |
| P3 | Aggressive reuse stress profile | Generates informative interventions; deployment only if independently earned |

Exact token ratios, layers, task-relevance filtering, clean-refresh horizon, and
profile-switch semantics must be frozen after the microbenchmark and before any
branch outcome is opened.

The method contribution is the outcome-supervised profile controller, not the
underlying VLA-Cache implementation.

## 6. Simulator-fork feasibility

### 6.1 Why direct state restoration is invalid

Pinned LIBERO exposes `get_sim_state()` and `set_state()`, but the flattened
MuJoCo state does not contain all causal state needed for an exact episode fork.
Robosuite also maintains, outside MuJoCo:

- controller goals and interpolator state;
- observable sampling clocks, delayed values, filters, and observation caches;
- environment `timestep`, `cur_time`, `done`, and horizon bookkeeping;
- wrapper state and potentially random-number-generator state;
- the VLA action queue, previous observations, and cache provenance.

LIBERO's `regenerate_obs_from_state()` forces observables to update but does not
restore these omitted variables. Therefore a mid-episode `set_state()` fork is
not accepted as causal evidence.

### 6.2 Accepted reconstruction

Each branch must:

1. create or fully reset a clean environment with the same task, seed, and
   published initial state;
2. replay the exact recorded low-level action prefix through normal
   `env.step()` calls;
3. reconstruct the same query schedule, previous observations, action queue,
   and cache history;
4. compare the reconstructed state and observation against the factual branch;
5. begin the intervention only after every equivalence gate passes.

This is slower than `set_state()` but much more defensible. Prefix replay is
mostly simulator work; cache-history reconstruction adds VLA calls but can be
bounded by sampling only a few branch points per episode.

### 6.3 Equivalence requirements

Before accepting a branch, reconcile:

- MuJoCo time, qpos, qvel, actuator state, and task-object state;
- task success/reward and episode counters;
- robot proprioception and gripper state;
- both camera observations, using exact hashes if deterministic or a frozen
  pixel tolerance after diagnosing rendering nondeterminism;
- next-step response under an identical probe action;
- VLA query index, action-queue position, cache profile, age, reusable-token
  indices, and clean-base provenance.

Any tolerance must be justified before treatment outcomes are inspected.

## 7. Cache-state feasibility

The official OpenVLA-OFT VLA-Cache implementation exists and uses a modified
Transformers 4.47 branch. Its `DynamicCache` mutates key/value arrays in place
with `index_copy_`, while reusable-token indices and layer schedules are stored
in mutable model configuration. Consequently:

- branch arms cannot share a live cache object;
- profile switching must reset every global configuration field;
- a cache snapshot/clone helper needs exact tests for keys, values, positions,
  seen-token count, attentions, mask indices, and profile metadata;
- failures must invalidate both branch-local and deployment cache state.

For a 7B LLaMA-class decoder with 512 visual tokens, 32 layers, hidden size
4096, K and V, and BF16 storage, the visual KV tensors alone are approximately
256 MiB. Prior-attention tensors and runtime overhead may add substantially more,
but VLA-Cache was evaluated on a 24 GiB RTX 4090 and TITAN provides 24 GiB per
GPU. Actual peak memory remains a mandatory measured gate.

The pinned project uses Torch 2.2.0/CUDA 11.8 and a different OpenVLA-OFT
Transformers fork. VLA-Cache therefore requires an isolated project-local
environment or a carefully reviewed patch; it must never overwrite the proven
SAVR environment.

## 8. Router feasibility

The router may use only pre-inference information:

- current and previous low-resolution scene/wrist images;
- temporal differences or compact motion features;
- current proprioception and its delta;
- previous executed action and action-queue position;
- task embedding computed once per episode;
- candidate profile, cache age, last-clean age, and reusable-token/mask summary;
- prior refresh reasons and inexpensive provenance metadata.

Privileged simulator state, success predicates, future observations, current
full-model hidden states, or the current FR action are forbidden at deployment.

Because the initial branch dataset will be small, training a large visual model
from scratch is not credible. The default should be a frozen compact pretrained
image encoder or a very small temporal CNN plus metadata MLP. The exact encoder,
weights, license, download size, and latency are frozen before branch outcomes.

The router must be compared against:

- profile-only and task-only priors;
- pixel/motion thresholds;
- the same architecture trained on immediate action-disagreement labels;
- the same architecture trained only on FR-distribution states;
- confidence-gated VLA-Cache when reproducible.

If raw/pre-inference features cannot predict the paired harmful label, BRACE
stops. Adding current full-VLA features after that failure would defeat the
latency premise and constitutes a different method.

## 9. Data and compute estimate

Historical measured costs:

- A4: 100 full episodes in 1.61 model-run hours;
- Phase 6: 1,000 FR/SAVR episodes in 24.83 reconciled GPU-run hours;
- FR query CUDA mean: approximately 1.267 seconds on TITAN.

A 30-episode branch pilot with three branch points per episode and two paired
arms is expected to require roughly 3--6 GPU-hours, depending on branch position
and cache reconstruction. A 100-episode, three-profile development collection
could require approximately 20--40 GPU-hours. These estimates must be replaced
by measured pilot throughput before scale-up.

Storage can remain modest by saving compressed router inputs, actions, state
digests, and outcomes rather than persistent KV tensors. Raw immutable evidence
must retain enough information to reproduce every branch identity and replay.

## 10. Novelty audit

| Nearest work | Existing contribution | BRACE distinction |
|---|---|---|
| VLA-Cache, NeurIPS 2025 | Training-free task-aware temporal KV reuse | Uses it as the physical substrate; does not claim the cache. |
| LAC, 2026 preprint | End-to-end learnable token selector and cache-ratio predictor trained with differentiable VLA task loss | BRACE learns a discrete execution profile from paired closed-loop intervention outcomes on cache-induced states. |
| AC2-VLA, 2026 preprint | Action-context router jointly controls cognition reuse, token pruning, and layer skipping using action/feature self-distillation | BRACE cannot claim generic action-aware adaptive computation; its distinction is paired terminal-effect supervision, replay verification, and fail-closed profile risk. |
| Gated VLA-Cache, 2026 preprint | Previous-action logit-margin invalidation | BRACE does not use previous model confidence as its supervision or sole gate. |
| Action-JND, 2026 preprint | Action-tolerance supervision from clean-feature perturbations | BRACE labels actual cache treatments by paired terminal rollout effects. |
| The Gate, Not the Cache, 2026 preprint | Isolates self-harvested gate collapse and uses unconditional off-path dense refresh | BRACE accepts the detector warning, uses raw pre-inference inputs, and performs dense paired branches only during training. |
| VLA-ATTC, 2026 preprint | Relative action critic selects among candidate actions with extra test-time compute | BRACE selects compute/cache profiles, keeps the base action policy frozen, and adds no candidate-action deliberation at deployment. |
| SAFE / adaptive failure probes | Predict episode failure from policy hidden features | BRACE estimates the incremental harm of a cache intervention before the current VLA forward, not general policy failure after hidden features exist. |
| DAgger / ThriftyDAgger | On-policy aggregation and selective expert intervention | Established methodological ancestry; BRACE applies paired compute interventions and does not claim DAgger itself. |
| General counterfactual policy evaluation | Simulator interventions for policy analysis | BRACE specializes paired interventions to learned VLA cache execution and an end-to-end measured reliability--latency frontier. |

Searches through 2026-08-24 found no established paper combining replay-verified
same-state cache interventions, paired terminal effect labels, cache-induced
data aggregation, and pre-inference profile routing for VLA acceleration. This
is provisional novelty only. The phrase “causal” must be limited to the paired
simulator intervention; it does not establish real-world causal validity.

## 11. Failure register

| Risk | How it could create a false positive or wasted project | Required control |
|---|---|---|
| Incomplete simulator snapshot | Branches differ before treatment | Replay from initial state; reject direct mid-episode state-only forks. |
| Replay nondeterminism | Outcome difference is not caused by caching | Exact/tolerance equivalence plus identical-action negative controls. |
| Controller/observable hidden state | Same qpos but different next dynamics | Replay normal `env.step()` history and next-action probe. |
| Wrong cache reconstruction | Treatment starts from different model history | Digest all cache/provenance fields and prove reconstruction parity. |
| In-place cache mutation | One branch contaminates the other | Independent exact cache clone or complete reconstruction per arm. |
| Global model-config leakage | Profile order changes results | Transactional profile context with restoration tests and randomized arm order. |
| Sparse terminal reward | Too few harmful labels | Aggressive stress profiles, stratified branch points, minimum class-count gate. |
| FR itself fails | Cache is blamed for base-policy failure | Preserve four paired outcome categories; never assume FR success. |
| One-step label misses compounding | Router appears safe but repeated reuse fails | Cache-age inputs, burst interventions, and later on-policy aggregation. |
| Long-horizon credit ambiguity | Current cache treatment is blamed for unrelated later failure | One intervention plus common continuation; report horizon sensitivity. |
| Continuation-policy target drift | A label under FR continuation does not predict harm under BRACE continuation | One explicitly versioned aggregation round; compare effect stability across continuation policies before deployment. |
| Selection bias | Only suspicious states receive labels | Randomized branch sampling within frozen strata; log inclusion probability. |
| Prefix leakage across train/test | Near-duplicate states inflate accuracy | Group splits by complete episode/initial state/task, never by branch row. |
| Severe class imbalance | High accuracy with no useful harm detection | AUPRC, risk--coverage, class counts, and matched-service false-negative rates. |
| Unobservable harm | Router inputs cannot see decisive stale-token shift | Predictability gate; stop instead of adding expensive current VLA features. |
| OOD overconfidence | Router caches novel states | Explicit rejection/ensemble and fail-closed thresholds. |
| Wrist-always-fresh too slow | Reliable profile has no paper-level speed | Physical profile timing gate before branch collection. |
| Aggressive profile distribution mismatch | Stress labels do not transfer to deployable profiles | Profile-conditioned model and per-profile held-out results. |
| Router overhead | Reported cache gain disappears end to end | Include preprocessing, transfer, and routing in synchronized timing. |
| Newer-GPU transfer | RTX 4090/A100 results do not transfer to TITAN sm_75 | Local microbenchmark is authoritative. |
| Strong baseline already positive | BRACE beats FR but not the state of the art | Require Pareto improvement versus official VLA-Cache and strongest reproducible gate. |
| Unavailable baseline code | Weak comparison is mistaken for novelty | Document availability; faithfully reimplement only if verifiable; no invented head-to-head. |
| Simulator privileged leakage | Router quietly uses task oracle | Schema-level deployment feature allowlist and ablation. |
| Repeated threshold tuning | Final positive result is selected post hoc | Separate development and sealed confirmation populations. |
| Multiple operating points | Cherry-picked frontier point | Freeze selection rule and report the full risk--latency curve. |
| Compute explosion | Branch collection consumes days before feasibility is known | Fixed query/hour/storage caps at every phase. |
| Simulator-only result | Paper overclaims robot deployment | Limit claims to checkpoint, suites, and simulator; physical validation is separate. |
| Rapid novelty collision | A 2026 preprint duplicates the contribution | Repeat primary-source/code search before implementation freeze and submission. |

## 12. Positive-result definition

BRACE is paper-positive only if a sealed evaluation shows both:

1. task-success non-inferiority to optimized FR under a predeclared paired
   margin; and
2. a statistically supported improvement in the success--latency Pareto
   frontier over official VLA-Cache and the strongest reproducible cache gate.

Beating FR latency alone is insufficient because VLA-Cache already does that.
Recovering success only by refreshing nearly every step is also insufficient.

## 13. Sources

Primary or official sources used in the audit:

1. VLA-Cache, NeurIPS 2025: <https://openreview.net/forum?id=QZYZ0Xm58q>
2. OpenVLA-OFT, RSS 2025: <https://www.roboticsproceedings.org/rss21/p017.html>
3. LIBERO source and environment wrapper: <https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/libero/libero/envs/env_wrapper.py>
4. LAC: <https://arxiv.org/abs/2602.00686>
5. Gated VLA-Cache: <https://arxiv.org/abs/2608.10824>
6. The Gate, Not the Cache: <https://arxiv.org/abs/2608.00391>
7. Action-JND: <https://arxiv.org/abs/2608.21247>
8. VLA-ATTC: <https://arxiv.org/abs/2605.01194>
9. DAgger: <https://www.ri.cmu.edu/pub_files/2011/4/Ross-AISTATS11-NoRegret.pdf>
10. ThriftyDAgger: <https://users.cs.utah.edu/~dsbrown/readings/thrifty_dagger.pdf>
11. SIMPLER / simulated policy evaluation: <https://arxiv.org/abs/2405.05941>
12. CREST / causal interventions in manipulation simulation: <https://arxiv.org/abs/2103.16772>
13. AC2-VLA: <https://arxiv.org/abs/2601.19634>

Recent preprints must be revalidated before any manuscript submission.

### Source snapshots inspected for feasibility

- project OpenVLA-OFT pin: `e4287e94541f459edc4feabc4e181f537cd569a8`;
- VLA-Cache repository: `a4909880573868dee2769343d52e793c0341678b`;
- VLA-Cache Transformers fork:
  `9a90a37acacf453433168db8d7769b7ea3c40c06` (Transformers 4.47.0);
- pinned LIBERO: `8f1084e3132a39270c3a13ebe37270a43ece2a01`;
- pinned robosuite package: 1.4.1;
- LAC repository snapshot: `fd191e0566944a3368be44909d36b535079e474b`
  (README only at audit time; no usable implementation present).
- AC2-VLA repository snapshot: `835b0fd2a36c070bf44b7ad58f5a8b75805157d2`
  (usable CogACT/SIMPLER research code, but not a drop-in OpenVLA-OFT/LIBERO
  baseline).

## 14. Recommendation

Proceed only through the stop-fast protocol in
`docs/BRACE_EXECUTION_PROTOCOL_V1.md`. The first two phases test replay
validity and physical acceleration before collecting any branch outcome. If
either fails, stop BRACE without router training.
