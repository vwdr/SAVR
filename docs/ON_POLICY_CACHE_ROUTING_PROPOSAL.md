# On-Policy Counterfactual Cache Routing for VLA Inference

**Working acronym:** OPCCR  
**Working paper title:** *Learning When to Reuse: On-Policy Cache Routing for Reliable VLA Inference*  
**Status:** Method proposal and gated research plan; no implementation or OPCCR outcome exists yet  
**Base stack:** Pinned OpenVLA-OFT + LIBERO + VLA-Cache-compatible decoder KV reuse

## 1. Research question

Can a cache router trained on states caused by its own reuse decisions preserve full-refresh task performance while obtaining meaningful end-to-end speedup from decoder visual-KV reuse?

The question follows directly from the completed SAVR/ACR experiments. The core failure was not merely a poor image threshold: reuse changed the predicted action, the action changed the future state, and an offline replay rule then operated outside the distribution on which it had been selected. Meanwhile, encoder-only scene reuse could preserve behavior but had at most 1.6202% idealized wall-clock benefit at the measured reuse rate because 83.7% of query time was downstream.

## 2. Hypothesis

A lightweight router trained with **on-policy counterfactual supervision** will reject cache reuse at intervention-sensitive states more accurately than:

- visual-change thresholds;
- an otherwise identical router trained only on full-refresh trajectories;
- previous-step model-confidence gates; and
- ungated VLA-Cache.

When coupled to token-level decoder KV reuse, this should create at least one operating point that is non-inferior in task success to full refresh within a predeclared margin while reducing synchronized end-to-end query latency.

## 3. Method

### 3.1 Two inference paths

At control step \(t\), define:

- \(F(o_t, s_t, \ell)\): the full-refresh VLA action chunk;
- \(C(o_t, s_t, \ell, K_{t-1})\): the cached candidate, where eligible scene-camera decoder KV entries may be reused;
- \(K_{t-1}\): cache state from the previous accepted refresh/reuse sequence;
- \(g_\theta(x_t)\): router-estimated risk that serving \(C\) instead of \(F\) will materially change the action.

The inference decision is

\[
u_t =
\begin{cases}
C(o_t,s_t,\ell,K_{t-1}), & g_\theta(x_t) \le \tau,\ q_\theta(x_t) \le \tau_q,\ h_t < H,\\
F(o_t,s_t,\ell), & \text{otherwise},
\end{cases}
\]

where \(q_\theta\) is an out-of-distribution or novelty score, \(h_t\) is the consecutive-reuse horizon, and \(H\) is a frozen maximum horizon. The policy fails closed to full refresh.

### 3.2 Camera asymmetry

The first implementation always refreshes wrist-camera information and permits reuse only for eligible scene-camera visual KV entries. This is a reliability prior grounded in the original SAVR evidence: 372 of 374 concealed per-camera violations came from the wrist view.

Camera asymmetry is not itself the claimed contribution. It is an evidence-based constraint that reduces the router's search space. A symmetric variant is an ablation only after the asymmetric system passes.

### 3.3 Router inputs

The router may use only information available before the expensive VLA path is selected:

- low-resolution current and previous scene frames;
- low-resolution current and previous wrist frames;
- frame differences or compact motion/flow features;
- current proprioceptive state and state delta;
- previous executed action chunk and action delta;
- cache age and consecutive-reuse count;
- task-language embedding computed once per episode;
- optional inexpensive cache metadata that does not require the current VLA forward pass.

The default architecture is a small dual-view visual encoder plus multilayer perceptron for state/action/cache metadata, followed by:

1. a risk head predicting unsafe cache intervention;
2. a novelty/reject head; and
3. an optional calibrated risk score.

No feature that requires first running the full model may be used at deployment. That would create a false speedup.

### 3.4 Counterfactual label

At each data-collection state, compute both \(a_t^F=F(\cdot)\) and \(a_t^C=C(\cdot)\), but execute only the action chosen by the behavior-mixture policy. Define a continuous discrepancy

\[
d_t = w_p d_{\mathrm{pos}}(a_t^F,a_t^C)
    + w_r d_{\mathrm{rot}}(a_t^F,a_t^C)
    + w_g d_{\mathrm{grip}}(a_t^F,a_t^C),
\]

with token-wise or chunk-wise aggregation frozen before outcome evaluation. The binary unsafe-intervention label is

\[
y_t = \mathbb{1}[d_t>\delta \;\lor\; m_t=1],
\]

where \(m_t\) captures a predeclared categorical mismatch such as opposite motion direction or gripper-state disagreement.

The thresholds and weights must be selected using development data and task-scale normalization, then frozen. Terminal success is not used as a per-step training label because it is sparse and confounded; it remains the primary closed-loop evaluation outcome.

### 3.5 On-policy data aggregation

The training distribution is collected iteratively:

1. **Round 0:** collect paired outputs on full-refresh trajectories.
2. Train router \(g_{\theta_0}\).
3. **Round \(k\):** roll out a frozen mixture of full refresh and router-directed cache decisions. At every visited state, query both paths for supervision, but execute exactly one recorded action.
4. Add all visited states, including apparently safe states, to the aggregate dataset.
5. Retrain from the aggregate dataset with task-, episode-, and round-balanced sampling.
6. Stop after the predeclared round count or when held-out policy-induced risk metrics cease improving.

This procedure is DAgger-inspired, but the teacher corrects a **compute intervention** rather than learning the robot policy itself. The base VLA weights remain frozen.

### 3.6 Reject and clean-refresh safeguards

The router refreshes when any of the following holds:

- predicted intervention risk exceeds \(\tau\);
- novelty score exceeds \(\tau_q\);
- maximum cache age \(H\) is reached;
- episode initialization, task transition, or reset occurs;
- an invariant or cache-shape check fails.

These safeguards are defense-in-depth, not evidence of formal safety.

## 4. Claimed novelty

The proposed scientific contribution is the combination of:

1. **cache-induced on-policy data aggregation** rather than selecting a gate only from clean/full-refresh trajectories;
2. **paired current-state counterfactual supervision** from full-refresh and cached VLA outputs;
3. a **reject-capable, fail-closed cache router** that uses only pre-inference signals; and
4. **evidence-driven multi-view asymmetry** with fresh wrist information and selectively reused scene KV.

The work does not claim novelty for KV caching, DAgger, confidence calibration, multi-camera perception, or maximum-horizon refresh separately.

## 5. Nearest baselines and required ablations

Minimum baseline set:

1. optimized full refresh;
2. official or faithfully reproduced VLA-Cache;
3. the best implementable confidence-gated VLA-Cache baseline;
4. offline-only router trained on Round 0 data;
5. OPCCR with on-policy aggregation;
6. OPCCR without novelty rejection;
7. OPCCR without wrist-always-fresh asymmetry;
8. fixed periodic clean refresh at matched reuse.

Action-JND should be included if official code and compatible weights are available before the baseline freeze. If unavailable, report this explicitly and compare at least against its published operating points without implying an experimental head-to-head.

The decisive scientific ablation is **offline-only versus on-policy-aggregated training** at matched cache-service rate. Without this, the paper cannot attribute improvement to the proposed solution to distribution shift.

## 6. Stop-fast execution protocol

No phase may inspect a frozen holdout before all choices for that phase are committed. All outputs, failures, hashes, hardware identifiers, and commands must be preserved.

### R0 — Research and specification

**State:** Complete with this proposal and `POSITIVE_RESULTS_DIRECTION_AUDIT.md`.

Acceptance:

- exact contribution and collision boundary documented;
- internal evidence reconciled;
- architecture path identified;
- no result claim made.

### R1 — CPU/local integration audit

Tasks:

- pin the VLA-Cache-compatible dependencies and source revisions;
- map scene/wrist token indices and layer cache layout;
- implement unit tests with synthetic tensors;
- verify fail-closed routing, cache invalidation, and no deployment-only feature leakage;
- produce a measured GPU resource estimate before requesting a run.

Stop if the intended cache entries cannot be separated correctly or the deployment router requires a full current VLA pass.

### R2 — One bounded GPU feasibility microbenchmark

Requires explicit authorization for the frozen attempt. Use one available GPU and record its ID; do not interfere with other jobs.

Maximum scope: 200 inference queries, no simulator campaign.

Required checks:

- actual peak memory below the server's conservative cap;
- deterministic cache bookkeeping and action-shape parity;
- full-refresh equivalence when reuse is disabled;
- synchronized end-to-end wall timing, CUDA timing, and router overhead;
- at least 10% end-to-end query-wall reduction for the cache path versus optimized FR under a representative reuse pattern;
- router overhead budget no greater than 5% of FR query wall time.

Stop the direction if the physical accelerator cannot meet the timing gate. A classifier cannot rescue a cache path with insufficient end-to-end headroom.

### R3 — Development data aggregation

Use only predeclared LIBERO development tasks and initial states. Keep final tasks/states sealed.

Before collection, freeze:

- task and initial-state partitions;
- number of aggregation rounds;
- behavior-mixture schedule;
- action-discrepancy definition;
- dataset balancing and deduplication;
- maximum number of visited states;
- seed policy.

Query both FR and cache paths at every visited state; log which action was executed. Collect complete states, not only router-selected reuse states, to avoid selective-label bias.

### R4 — Router training and offline held-out test

Primary test: on a held-out set of policy-induced states, compare the on-policy router against the same architecture trained only on Round 0 full-refresh states.

Report:

- AUROC and AUPRC for unsafe intervention;
- false-negative unsafe-reuse rate at matched service rates;
- calibration error and risk-coverage curves;
- results by task, collection round, cache age, and camera-motion stratum;
- router wall latency and memory.

Proceed only if on-policy aggregation materially improves risk-coverage behavior at the intended service range. Exact numerical gates must be frozen after the class prevalence pilot and before model comparison; they may not be set after seeing the comparative result.

### R5 — Paired closed-loop development evaluation

Compare FR, ungated cache, confidence gate, offline-only router, and OPCCR on paired initial states. The same episode states/seeds must be used for each method where the benchmark permits.

Provisional paper-level targets, to be finalized by a power analysis before outcomes are opened:

- task success non-inferior to FR within a 2 percentage-point margin using a paired 95% interval;
- at least 15% reduction in synchronized end-to-end query wall time versus optimized FR;
- router overhead at most 5% of FR query wall time;
- meaningful cache service rate (provisional target at least 30% of eligible scene KV decisions);
- no task-level catastrophic regression, with the exact bound frozen in advance.

Failing R5 ends method development. Do not tune on the failed episodes and retest the same population.

### R6 — Independent confirmation

Only after R5 passes:

- freeze code, checkpoint, thresholds, cache policy, task populations, seeds, and analysis scripts;
- evaluate across the standard LIBERO suites rather than only LIBERO-Object;
- use an episode count justified by an a priori non-inferiority power calculation (provisional minimum: 500 episodes per method over a suite, subject to the paired design and observed FR rate);
- use multiple seeds when stochasticity exists, or explicitly state when benchmark initial states are deterministic;
- preserve a never-tuned final holdout;
- report all methods and failures, not only the selected operating point.

Stop before manuscript claim expansion. A second checkpoint or embodiment is desirable for external validity but must not be implied if unavailable.

## 7. Metrics and statistical analysis

Primary outcomes:

- terminal task success;
- synchronized end-to-end query wall latency;
- success non-inferiority difference versus FR;
- cache service/reuse rate.

Secondary outcomes:

- CUDA time by component;
- control frequency and episode wall time;
- router latency, memory, and rejection rate;
- action-discrepancy false-negative rate;
- calibration and risk-coverage curves;
- cache age and refresh-reason distribution;
- task- and motion-stratified success.

Use paired confidence intervals when the same initial states are evaluated. Report binomial intervals for absolute success rates and bootstrap paired intervals for latency and rate differences. Correct or clearly label multiple exploratory comparisons. Never call terminal success “safety.”

## 8. Bias controls

- Freeze the final holdout before method training.
- Separate feasibility, development, and confirmation populations.
- Keep an outcome-blind monitor during long runs.
- Log every terminal episode, interrupted run, exclusion, and retry.
- Make the optimized FR path the timing oracle.
- Include router computation and cache maintenance in latency.
- Disallow post-hoc threshold selection on final success.
- Preserve negative results and technical stops.
- Re-run the novelty collision audit before submission.

## 9. Resource plan

The design targets one 24 GiB GPU. OpenVLA-OFT's reported inference footprint and the approximate visual-KV size suggest feasibility, but R2 is authoritative. The on-policy collection phase is expensive because it computes two candidate actions per visited state even though only one is executed; therefore development populations must be bounded and checkpointed.

The plan deliberately excludes:

- CUDA graph capture;
- multi-GPU model sharding;
- new timing methodologies that make FR comparison ambiguous;
- system-wide dependency changes;
- large downloads without a storage estimate and explicit approval.

## 10. Positive-paper decision

A positive-results manuscript is justified only if both conditions hold on the sealed confirmation evaluation:

1. OPCCR meets the predeclared task-success non-inferiority criterion; and
2. OPCCR produces meaningful synchronized end-to-end speedup over optimized FR and a better reliability–efficiency frontier than the strongest implemented cache baseline.

If it passes, the paper's central result is that **training a cache router on the distribution created by cache interventions closes part of the replay-to-rollout reliability gap while retaining downstream KV-cache efficiency**.

If it fails, the result is preserved as an extension of the negative evidence. Metrics, margins, populations, or titles must not be changed merely to manufacture a positive paper.

## 11. Immediate next action

Begin R1 only: create a pinned, CPU-testable integration design and exact GPU resource estimate. Do not start a GPU run, simulator evaluation, or model download until R1 passes and the single R2 attempt is explicitly authorized.
