# ACR V5-B Output-Blind Development Screening Preflight

Status: **FROZEN BEFORE V5-B CANDIDATE OUTPUTS**

Authorization date: 2026-08-10

Machine freeze: `configs/acr/v5_b_output_blind_preflight.json`

## 1. Question

Does the CPU-verified IR-SA-ACR state machine admit a useful, auditable scene-
reuse region on outcome-free Full-Refresh development traces while satisfying
its isolation, cache, gripper, and hard-cap invariants?

This is a telemetry screening phase. It cannot establish task success, online
reuse, GPU efficiency, simulator behavior, or a positive paper result.

## 2. Authorization and boundary

The user approved the logical V5 next steps on 2026-08-10 and confirmed the
V5-B checkpoint. This phase may perform deterministic CPU replay only after
this protocol, its machine configuration, analysis code, verifier, and tests
are committed and synchronized.

V5-B may not:

- read episode success, failure, timing, or protected outcome fields;
- use Goal, final, reserve states, or reserve seeds;
- use a GPU, model, simulator, or network download;
- implement or measure an optimized executor;
- modify the controller, adapter, prior evidence, or manuscript; or
- change a candidate, gate, bootstrap, or selection rule after output.

## 3. Frozen input

The only data input is the outcome-free companion trace from the completed A4
upstream Full-Refresh Object development run:

- run: `acr-a4-upstream-fr-object-dev00-09-v01`;
- tasks: Object `0-9`;
- initial states: `0-9`;
- seed: `0`;
- episodes: exactly `100`;
- query traces: exactly `1,773`;
- encoded trace bytes: exactly `13,415,489`; and
- ordered relative-path/content SHA-256:
  `3ce22a1d1de7d33ed0a6bcdb52b32f42800d732ec93aed0bfed593f1e536b34b`.

The trace schema contains only query identity, scene representation,
normalized end-effector position, action chunk, transition telemetry,
component counts, and integrity hashes. The V5-B loader requires the exact
allowlist and rejects any record containing `success`, `failure`, `reward`,
`timing`, or other non-schema fields.

The replay uses Full-Refresh observations/actions. It estimates where the
controller would request reuse on those traces; it does not simulate the
closed-loop trajectory that reuse would have produced.

## 4. Disclosed use of prior development evidence

The threshold anchors were frozen originally from the A4/A5 development route
and were later used by V4-A:

- scene low/high: `0.2476380718954248` / `0.30046895424836606`;
- translation low/high: `0.5479944908411765` / `0.685919037527938`.

V4-A showed that the old horizon-2 controller did not satisfy the intended
maximum-streak-one contract and that the direction-reversal variants did not
meet the complete frozen gates. V5-B therefore does not pretend to be
independent of V4 development. It prospectively evaluates the separately
implemented latch and excludes the direction-reversal veto from the primary
family.

Because the latch now independently limits every reuse interval, the family
starts at the prior high anchor and extrapolates by fixed half-range increments
rather than selecting a previously observed best candidate. Independent Goal
confirmation remains required after all development selection.

## 5. Frozen candidate family

For threshold level $a$, define

\[
\tau_S(a)=S_{low}+a(S_{high}-S_{low}), \qquad
\tau_Q(a)=Q_{low}+a(Q_{high}-Q_{low}).
\]

Use exactly $a\in\{1.0,1.5,2.0\}$ and hard prefix caps
$\rho\in\{0.35,0.40\}$, producing exactly six candidates:

| Candidate | $a$ | Scene threshold | Translation threshold | Hard cap |
|---|---:|---:|---:|---:|
| `v5-a100-b35` | 1.0 | 0.30046895424836606 | 0.685919037527938 | 0.35 |
| `v5-a100-b40` | 1.0 | 0.30046895424836606 | 0.685919037527938 | 0.40 |
| `v5-a150-b35` | 1.5 | 0.3268843954248367 | 0.7548813108713188 | 0.35 |
| `v5-a150-b40` | 1.5 | 0.3268843954248367 | 0.7548813108713188 | 0.40 |
| `v5-a200-b35` | 2.0 | 0.3532998366013073 | 0.8238435842146996 | 0.35 |
| `v5-a200-b40` | 2.0 | 0.3532998366013073 | 0.8238435842146996 | 0.40 |

Every candidate uses:

- `IsolatedACRController`;
- identity `acr-isolated-controller-v1`;
- SA-ACR policy;
- horizon `1`;
- warm-up queries `0` and `1`;
- scene and normalized-translation thresholds;
- action-history-derived gripper-transition veto;
- post-reuse refresh latch;
- cache-age/latch agreement; and
- no task-, state-, episode-, or success-dependent rule.

No candidate may be added, removed, or relabeled after execution.

## 6. Frozen accounting

For each candidate, report:

- completed queries, refreshes, and reuses;
- reuse point estimate and episode-cluster bootstrap 95% interval;
- maximum reuse streak;
- maximum prefix reuse fraction;
- post-reuse refresh count;
- gripper-transition reuses;
- isolation-state mismatches;
- cache/horizon/hard-cap reason counts; and
- theoretical logical visual-component reduction.

The existing two-camera accounting assigns equal logical scene and wrist
visual blocks. Reuse skips the scene block and retains the wrist block, so

\[
\text{logical visual reduction}=0.5\times\text{reuse fraction}.
\]

This is theoretical component accounting, not measured CUDA or wall time.

Bootstrap unit: episode. Seed: `5102026`. Resamples: exactly `10,000`.
Confidence interval: percentile 95%. No outlier deletion.

## 7. Eligibility gates

A candidate is eligible only if all gates pass:

1. exactly 100 episodes and 1,773 contiguous queries are replayed;
2. maximum completed reuse streak is one;
3. every completed prefix respects its candidate hard cap;
4. gripper-transition reuses equal zero;
5. isolation-state mismatches and invariant failures equal zero;
6. at least one post-reuse forced refresh occurs;
7. reuse point estimate is at least `0.35`;
8. bootstrap reuse lower bound is at least `0.30`;
9. logical visual-reduction point estimate is at least `0.175`; and
10. logical visual-reduction lower bound is at least `0.15`.

Gates 7-10 encode the same minimum reuse requirement in two views but are
retained explicitly for report reconciliation. They do not claim measured
speed.

## 8. Selection and stop rule

Among eligible candidates, select exactly one lexicographically:

1. smallest threshold level $a$;
2. smallest hard prefix cap;
3. smallest isolation-risk exposure (which must already be zero); and
4. smallest reuse point estimate at or above the gate.

This chooses the least permissive eligible method. If no candidate passes,
V5-B stops negative before V5-C. Do not relax the threshold grid or gates.

If one candidate passes, V5-B authorizes only preparation of the V5-C CPU
executor-correctness protocol. It does not authorize executor code, GPU use, or
online evaluation by itself.

## 9. Determinism and publication procedure

The analyzer must:

1. verify the input count, byte count, ordered path/content hash, exact schema,
   record semantic hashes, episode count, and contiguous query indices;
2. replay every candidate twice from clean controller state;
3. require byte-identical candidate summaries;
4. compute every gate and selection mechanically;
5. emit one semantic-hashed runtime JSON only after all candidates complete;
6. run an independent verifier against the frozen config; and
7. publish the runtime record, human report, status, milestone, and decision in
   a new reviewable checkpoint.

No candidate output may be inspected until the analyzer has produced all six
candidate summaries and the completed run disposition.

## 10. Resources and protected data

Resource cap:

- zero GPUs;
- zero model queries;
- zero simulator episodes/resets;
- zero downloads;
- zero new task outcomes;
- at most `256 MiB` of new artifacts; and
- CPU wall time at most `1,800` seconds.

Goal states `0-9`, all states `10-49`, reserve seeds `7/17/27`, and all final
populations remain unopened. All TITAN reads/writes remain strictly inside
`/home/ved/SAVR`.
