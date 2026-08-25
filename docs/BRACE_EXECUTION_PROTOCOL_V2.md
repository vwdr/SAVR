# BRACE Execution Protocol V2

**Method:** Branch-Rollout Adaptive Cache Execution (BRACE)

**Working title:** *Branch Before You Cache: Outcome-Supervised Cache Contracts for Reliable VLA Acceleration*

**Date frozen:** 2026-08-24

**Status:** Research protocol prepared; no BRACE implementation or result exists

**Supersedes:** `docs/BRACE_EXECUTION_PROTOCOL_V1.md` before implementation

**Authority:** This document does not authorize a model load, GPU run, outcome
collection, download, or protected-population access

## 1. Objective and positive-result standard

Determine whether a lightweight pre-inference controller can select a bounded,
clean-provenance visual-KV reuse contract that improves the closed-loop
task-reliability--efficiency frontier of the pinned OpenVLA-OFT checkpoint.

A positive result requires both:

1. paired task-success non-inferiority to optimized full refresh (FR) under a
   predeclared margin; and
2. a statistically supported Pareto improvement over official VLA-Cache and
   the strongest reproducible reliability-preserving baseline, including
   actuation-slack refresh when technically representable.

Latency alone is insufficient. Report critical-path wall time, total GPU work,
memory, cache-service coverage, and task success. BRACE makes no formal robot-
safety claim.

## 2. Corrections from V1

V2 fixes four material problems found during the independent re-audit:

1. **Cumulative treatment:** labels cover bounded reuse bursts, not only one
   cached query followed by FR.
2. **Clean gate provenance:** deployable profiles never derive task-relevance
   or layer schedules from an accelerated forward.
3. **Observable staleness:** the router receives token-level age, provenance,
   and current-versus-source difference maps rather than only scalar cache age.
4. **Competitive baseline:** unconditional actuation-slack dense refresh is a
   required comparison or a measured technical exclusion.

V2 also replaces the unbounded per-query profile policy with bounded **cache
contracts**. This aligns the training intervention with deployment and prevents
indefinite self-reinforcing reuse.

## 3. Frozen system boundary

The following remain unchanged:

- OpenVLA-OFT checkpoint, prompt, preprocessing, normalization, action head,
  proprio projector, and action-chunk semantics;
- LIBERO task definitions, initial states, horizon, reward, success predicate,
  camera configuration, and control frequency;
- optimized FR as the correctness and timing oracle;
- one policy-query boundary per action chunk;
- writes on TITAN restricted to `/home/ved/SAVR`;
- at most one explicitly coordinated GPU unless a later protocol separately
  authorizes more.

BRACE may select computation contracts. It may not alter, correct, blend, rank,
or post-process the base policy's predicted actions.

## 4. Cache-contract state machine

### 4.1 Clean anchor

A **clean anchor** is an FR query whose forward skips no visual computation.
It creates:

- current scene and wrist images;
- complete visual K/V state;
- clean text-to-vision attention and layer statistics;
- a query identity and per-layer/per-token provenance ledger;
- the base action chunk.

Only a clean anchor may define task-relevance masks or layer-reuse schedules for
a later contract. Information produced by an accelerated forward may update K/V
entries and their source identities, but may not become the semantic gate.

### 4.2 Contract

A contract is `c = (p, h)`, where:

- `p` is one predeclared clean-anchored cache profile; and
- `h` is a maximum of 1, 2, or 4 accelerated policy queries.

At the first query after a clean anchor, BRACE selects either FR or one eligible
contract. A selected contract remains bounded by `h`; expiration forces FR.
Every accelerated query may terminate the contract early through a hard abort,
but it may not silently extend or become more aggressive.

### 4.3 Hard aborts

Any of the following causes immediate FR and a new clean anchor:

- task, episode, seed, prompt, camera, or model identity change;
- missing or inconsistent K/V, semantic-gate, token-source, or layer metadata;
- contract age or maximum horizon violation;
- current image/source-patch difference outside the profile envelope;
- missing current proprioception;
- router OOD rejection or invalid numerical output;
- profile switch not explicitly allowed by the state machine;
- exception, timeout, memory violation, or failed invariant.

Hard abort conditions are frozen before outcome collection. Their execution is
included in intent-to-treat results and latency accounting.

## 5. Deployable and diagnostic profiles

| ID | Role | Definition |
|---|---|---|
| P0 | Oracle/fallback | Optimized full refresh; creates a clean anchor. |
| P1 | Deployable conservative | Scene-camera eligible K/V may be reused; wrist tokens always refresh; semantic mask and layer schedule come only from the last clean anchor. |
| P2 | Deployable moderate | Eligible scene and wrist K/V may be reused; clean-anchor semantic mask and layer schedule only; stricter per-token source-difference and age caps on wrist tokens. |
| P3 | Diagnostic baseline | Official self-harvested VLA-Cache behavior using prior accelerated attention. Never deploy through BRACE unless later evidence and a new decision explicitly earn it. |
| P4 | System baseline | Unconditional dense refresh during action execution, supplying a clean gate/KV base for the next serve query. Not a BRACE-selectable contract. |

Exact token thresholds, clean semantic mask, per-layer reuse schedule, maximum
token ages, camera rules, and contract-switch rules are frozen after physical
microbenchmarking and before any terminal branch outcome is opened.

P1/P2 may update recomputed K/V entries. Every entry retains its source query
identity. They may not replace the clean semantic gate with accelerated
attention.

## 6. Provenance ledger and router inputs

### 6.1 Exact cache ledger

For every layer and visual token, record:

- camera and patch identity;
- source query identity and source image digest;
- token age and clean-anchor age;
- whether the current query recomputed or reused it;
- K/V tensor position, dtype, shape, and content digest;
- semantic-gate identity and whether it came from P0;
- profile, contract identity, remaining horizon, and abort history.

Mixed-age caches are expected. A scalar `cache_age` is not accepted as complete
provenance.

### 6.2 Deployment feature allowlist

The router may use only information available before the current VLA forward:

- current low-resolution scene and wrist images;
- last-clean anchor thumbnails;
- current-to-source patch-difference maps for the candidate profile;
- per-camera token-age maps and layerwise age/reuse summaries;
- clean semantic mask and semantic-gate age;
- current proprioception and its delta;
- previous executed action, query index, and action-queue position;
- task embedding computed once per episode;
- candidate profile/horizon and prior hard-abort reasons.

The source-difference map must reflect the actual source query for each cached
token. Comparing only with the immediately previous image is insufficient.

Forbidden deployment inputs include:

- success/reward predicates or privileged simulator state;
- future observations or actions;
- the current FR action;
- current dense-VLA hidden states, logits, attention, or uncertainty;
- labels or statistics derived from the protected evaluation population.

Feature extraction, device transfer, routing, and abort checks are included in
end-to-end timing.

## 7. Interventional supervision

### 7.1 Branch state

Sample clean-anchor-adjacent query states from actual behavior-mixture rollouts.
Each branch is reconstructed by a fresh environment reset to the same task,
seed, and published initial state followed by exact low-level action-prefix
replay through normal `env.step()` calls.

Direct mid-episode MuJoCo `set_state()` alone is prohibited because it omits
controller, observable, wrapper, timing, action-queue, and cache state.

### 7.2 Treatment

At an accepted branch point, randomize between:

- `T=0`: FR for the next `h` policy queries; and
- `T=(p,h)`: the exact deployable cache contract for up to `h` queries,
  including its frozen hard-abort behavior.

After the assigned horizon, both arms use the same versioned continuation
policy. Round 0 uses FR continuation. A later on-policy aggregation round uses
the frozen BRACE-v0 continuation.

Contract horizons 1, 2, and 4 are mandatory experimental conditions. They may
be screened sequentially for cost, but single-query evidence alone cannot
advance BRACE.

### 7.3 Outcomes

Primary paired categories are:

- harmful: FR succeeds, assigned contract fails;
- beneficial: assigned contract succeeds, FR fails;
- neutral-success: both succeed;
- neutral-failure/unresolved: both fail.

Also record time/steps to success, termination type, remaining horizon, actual
cached queries served, hard aborts, and total/critical-path compute.

The primary causal analysis is **intention to treat** using assigned contracts,
including aborts. Served-cache analyses are secondary because conditioning on
successful service can introduce selection bias.

### 7.4 Negative controls

Before outcomes, reserve branch points for:

- FR versus FR;
- P1 contract versus the identical P1 contract;
- P2 contract versus the identical P2 contract when P2 is eligible;
- randomized arm execution order;
- a deliberately corrupted prefix that the validator must reject.

Identical-treatment terminal discordance must be negligible relative to the
harmful-contract rate under a frozen rule. It may not be subtracted post hoc.

## 8. Router objective and deployment rule

For contract `c`, let `r_theta(x_t,c)` estimate the probability that assigning
`c` causes paired terminal harm relative to FR under the versioned continuation
policy. Let `U_theta(x_t,c)` be a calibrated upper risk bound and `L(c)` the
measured net critical-path latency including router overhead.

BRACE chooses the lowest-latency eligible contract satisfying
`U_theta(x_t,c) <= epsilon`; otherwise it selects FR.

The initial model is deliberately small:

- frozen compact image encoder or small temporal CNN;
- token-age/source-difference map encoder;
- metadata/proprio/action MLP;
- profile/horizon-conditioned risk head;
- explicit calibration and OOD rejection.

The exact encoder, weights, license, architecture, parameter count, and latency
budget are frozen before comparative training results are opened.

Mandatory router baselines:

- prevalence and contract prior;
- cache metadata only;
- current/previous-image motion only;
- current/source-patch difference without learning;
- action-disagreement-supervised router with the same capacity;
- BRACE without token provenance;
- BRACE trained only on FR-continuation states.

## 9. Protected populations

Historical SAVR/ACR outcomes remain development evidence. Preserve untouched
LIBERO initial states 10--49 and seeds 7/17/27 across supported suites until a
final population map is frozen.

- development and branch training: already-opened states 0--9 plus separately
  declared development seeds;
- train/validation/test split: group by complete task/initial-state/seed
  episode, never by branch row;
- prevalence and risk--coverage: deployment-representative sampling stratum;
- optional enrichment: outcome-blind transition/cache-age strata with recorded
  inclusion probabilities;
- final confirmation: untouched state/seed groups and, if feasible, at least
  one task group not used for router training.

No protected outcome may influence features, contracts, thresholds, sample
sizes, or operating points.

## 10. Phase B1 — Replay-verified branch harness

**Resource class:** project-local CPU/OSMesa simulator only; no VLA or CUDA.

Implement a complete episode transcript with every low-level action, query and
chunk boundary, observation digest, state/counter digest, reward/success,
termination, and action-queue index.

For at least three tasks spanning free motion, contact, and gripper transition:

1. record a deterministic scripted trajectory;
2. replay early, middle, and late prefixes twice in fresh environments;
3. compare MuJoCo state, counters, reward/success, proprioception, cameras, and
   the next transition under an identical probe action;
4. prove that one modified prefix action is rejected;
5. prove that direct-state-only restoration is rejected.

Acceptance requires complete transcripts, exact discrete equality, justified
pre-outcome numeric/image tolerances where exact equality is unavailable, and
100% next-transition equivalence at accepted branches.

## 11. Phase B2 — Cache/provenance correctness

**Resource class:** CPU and synthetic tensors only.

Use an isolated project-local dependency plan. Do not overwrite the proven SAVR
environment. Reconcile the project's Transformers 4.40.1 OpenVLA-OFT fork with
the VLA-Cache Transformers 4.47.0 fork by reviewed patch or isolated runtime.

Implement and adversarially test:

- exact DynamicCache clone/restore;
- per-layer/per-token source ledger;
- clean semantic-gate identity and age;
- transactional restoration of all mutable model configuration;
- P0--P4 state transitions and reset semantics;
- P1/P2 camera/token/layer rules;
- 1/2/4-query contract expiry and every hard abort;
- randomized profile/arm order without state leakage;
- source-difference and age-map construction;
- immutable treatment assignment and intent-to-treat records.

Acceptance requires synthetic parity, order independence, zero cross-arm cache
mutation, exact P0-disabled equivalence, and a frozen B3 resource command.

## 12. Phase B3 — Physical cache and slack microbenchmark

**Resource class:** one separately authorized aggregate-idle TITAN GPU; no
simulator outcome. Maximum 300 balanced real-model queries.

Measure P0, P1, P2, official P3, and P4 components where technically possible:

- synchronized critical-path and total query wall time;
- amortized clean-anchor/contract-cycle wall time for horizons 1, 2, and 4;
- GPU busy time and component CUDA time;
- router/provenance preprocessing time;
- peak allocation/reservation;
- per-profile reusable tokens and actual K/V ages;
- action/tensor repeatability and P0 parity;
- clean-gate versus self-harvested-gate identity;
- contract reset, switching, and failure restoration;
- actual action-chunk execution slack and whether an asynchronous dense pass
  completes before the next query.

Do not claim P4 latency from artificial simulator sleep. Report the measured FR
duration and the frozen real-time control budget. If dense refresh exceeds the
available action slack, report its overflow and total GPU work.

Advancement requires:

- P0 matches the proven FR path within frozen tolerances;
- at least one clean-provenance P1/P2 contract yields at least 10% synchronized
  accelerated-query reduction and a projected 8% net contract-cycle reduction
  after mandatory clean anchors, expiry refreshes, routing, and provenance work;
- no cache/provenance/reset invariant failure;
- peak reservation below the unchanged 23 GiB cap;
- exact source/checkpoint restoration and GPU release;
- a measured disposition for P4 rather than silently omitting it.

If only official self-harvested P3 is fast, stop BRACE.

## 13. Phase B4 — Horizon and label-prevalence pilot

Requires B1--B3 acceptance and a separately frozen outcome authorization.

### 13.1 Stage B4a: technical branch pilot

- 12 development episodes;
- at most two preassigned branch points per episode;
- one B3-eligible clean profile at horizon 1 and 2;
- mandatory duplicate-arm controls;
- maximum 48 primary paired contracts plus controls;
- frozen 2-GPU-hour cap replaced by measured B3 throughput.

Open outcomes only after branch equivalence and immutable records reconcile.
Stop on unexplained duplicate-arm discordance, cache-state mismatch, arm-order
effect, or cost overrun.

### 13.2 Stage B4b: scientific prevalence pilot

If B4a passes:

- 30 development episodes;
- at most three outcome-blind branch points per episode;
- horizons 1, 2, and 4 for one speed-eligible clean profile;
- maximum 270 primary paired contracts plus frozen controls;
- provisional 5--9 GPU-hours, replaced by B4a measurements;
- no router training.

Before outcomes, freeze prevalence bands, duplicate-control tolerance, task
balance, terminal-completion requirement, and scaled cost rule.

Recommended advancement defaults:

- at least 20 harmful and 40 neutral-success pairs for the deployable profile,
  with horizon-specific reporting;
- no task supplies more than half of harmful pairs;
- at least 95% of planned branches are terminal and analyzable;
- identical-treatment discordance is both low in absolute terms and materially
  below harmful prevalence;
- development scaling is projected below 40 GPU-hours.

Stop if single-query harm is absent but longer-horizon harm exists and cannot be
predicted before contract entry; this would invalidate the chosen controller.
Stop if harm is too rare to learn within budget or so common that useful cache
coverage is implausible.

## 14. Phase B5 — Conditional data expansion and predictability

After B4, freeze the required branch count from measured class prevalence,
learning curves, task balance, and cost. Recommended minimum harmful counts are:

- 100 training;
- 30 validation;
- 60 held-out test.

Sixty held-out harmful examples are the minimum for an approximate 95% upper
miss-rate bound near 5% when zero are missed; the final requirement may be
larger. Preserve a representative evaluation stratum separate from enrichment.

Stop rather than scale if the projected branch collection exceeds 40 GPU-hours.

Evaluate AUPRC, AUROC, calibration, false-negative harm rate, and risk--coverage
with episode-grouped intervals. Report every metric by task, contract horizon,
cache age, camera motion, and profile. Advancement requires:

- clear improvement over prevalence, metadata, and motion baselines;
- a predeclared lower confidence bound above random ranking;
- material harm reduction at service coverage that retains the B3 net speed;
- calibrated risk bounds on held-out episode groups;
- router overhead within budget;
- no prohibited features or group leakage.

No profile/horizon may be deployed merely by borrowing evidence from a more
aggressive contract. Each claimed contract needs either enough held-out harmful
examples to bound detector misses or enough representative assigned trials to
bound its natural harm rate under a predeclared exact/binomial calculation.

If prediction requires current dense-VLA features, BRACE fails its deployment
premise and stops.

## 15. Phase B6 — One on-policy aggregation round

Only after B5 passes:

1. freeze BRACE-v0 contracts, features, calibration, and threshold;
2. collect separate development rollouts under BRACE-v0;
3. sample new contract branches outcome-blind from BRACE-induced states;
4. use BRACE-v0 as the common continuation after assigned horizons;
5. train BRACE-v1 once;
6. compare v1 against v0 on a held-out branch population;
7. quantify label drift between FR and BRACE continuations.

Proceed only if v1 preserves or improves risk--coverage and label drift remains
within a frozen bound. A second aggregation round requires a new protocol and
cost justification.

## 16. Phase B7 — Paired closed-loop development

Required executable methods:

1. optimized FR;
2. official self-harvested VLA-Cache;
3. the best static clean-provenance contract;
4. periodic FR at matched cache-service coverage;
5. strongest reproducible confidence gate;
6. P4 actuation-slack refresh, or a measured hardware/control-budget exclusion;
7. action-disagreement router with matched capacity;
8. BRACE-v1;
9. BRACE without provenance maps;
10. BRACE without on-policy aggregation.

Action-JND, LAC, and AC2-VLA receive direct comparisons only when compatible
code/checkpoints exist. Otherwise use their published results as context and do
not fabricate a head-to-head.

Advancement targets, finalized by paired power analysis before outcomes:

- success non-inferior to FR within a two-percentage-point margin;
- at least 10% net synchronized critical-path reduction including routing;
- statistically supported Pareto improvement over official VLA-Cache or the
  strongest reliability-preserving executable baseline;
- lower total GPU work than P4 if P4 wins critical-path latency;
- no task-level catastrophic regression;
- complete intent-to-treat, service, abort, horizon, and provenance accounting.

## 17. Phase B8 — Sealed confirmation

Only after B7 passes:

- freeze code, dependencies, checkpoint, contracts, router, thresholds, seeds,
  population, sample size, and analysis;
- select the minimum confirmatory comparison set needed for the claim rather
  than automatically repeating all development ablations;
- determine episode count from paired discordance and the non-inferiority
  margin; do not treat 500 episodes/method/suite as automatically sufficient;
- evaluate all claimed LIBERO suites on untouched groups;
- report paired success differences and intervals, binomial success intervals,
  grouped bootstrap latency/compute intervals, and the complete frontier;
- retain technical stops, aborts, exclusions, and all four outcome categories;
- repeat novelty and code-availability searches before submission.

No threshold, contract, operating point, or claim may be chosen from B8.

## 18. Resource and evidence policy

- Default to one GPU selected only after user coordination and aggregate-idle
  telemetry; never inspect or interfere with other users' processes.
- Every phase has a frozen run identity, query/episode/hour/storage cap, terminal
  summary, recovery rule, and outcome-blind monitoring rule.
- Never modify the proven SAVR environment; use isolated project-local paths.
- Never use `sudo`, change server services/permissions, or write outside
  `/home/ved/SAVR`.
- Record source commits, licenses, download sizes, commands, GPU ID/UUID,
  dependency lock, checkpoint hashes, timing definitions, and artifact hashes.
- Report simulation-only evidence as simulation-only. Physical deployment and
  energy claims require corresponding measurements.

## 19. Stop conditions

BRACE stops without reinterpretation if any of the following occurs:

- replay-equivalent branches cannot be established;
- clean-provenance caching cannot be integrated with P0 parity;
- no clean profile produces meaningful net speed;
- only self-harvested-gate caching is fast;
- horizon-sensitive harm cannot be learned from pre-inference features;
- useful labels exceed the development budget;
- the router loses its speed after preprocessing/transfer;
- a simpler executable baseline dominates its reliability--efficiency frontier;
- sealed non-inferiority or task-level reliability gates fail;
- novelty is independently preempted.

## 20. Next eligible task

Implement and verify **B1 only**. B1 is unchanged in spirit by the cache-method
redesign and produces generally useful branch-replay infrastructure. Do not
begin B2, install VLA-Cache, load the model, use a GPU, or collect policy
outcomes without a separate checkpoint decision.

## 21. Primary related work governing this protocol

1. VLA-Cache: <https://openreview.net/forum?id=QZYZ0Xm58q>
2. The Gate, Not the Cache: <https://arxiv.org/abs/2608.00391>
3. Gated VLA-Cache: <https://arxiv.org/abs/2608.10824>
4. Action-JND: <https://arxiv.org/abs/2608.21247>
5. LAC: <https://arxiv.org/abs/2602.00686>
6. AC2-VLA: <https://arxiv.org/abs/2601.19634>
7. ActionCache: <https://arxiv.org/abs/2607.06370>
8. LoopVLA: <https://arxiv.org/abs/2605.09948>
9. DySL-VLA: <https://arxiv.org/abs/2602.22896>
10. DAgger: <https://www.ri.cmu.edu/pub_files/2011/4/Ross-AISTATS11-NoRegret.pdf>

Recent preprints and code availability must be revalidated before B2 and again
before manuscript submission.
