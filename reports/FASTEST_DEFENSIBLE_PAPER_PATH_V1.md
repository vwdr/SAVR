# Fastest defensible paper path — 2026-09-26

## User direction and interpretation

User wants the fastest path to a paper and does not want an advisor email now.
No email is prepared or sent. Speed does not authorize hiding comparator results,
changing frozen gates or claiming novelty for an established method.

The shortest candidate is a **comparative empirical systems paper**, not another
new-algorithm proposal. That distinction is an explicit scope assumption. It may
contain positive acceleration results while acknowledging that the algorithms
are established. Acceptance and a sufficient original empirical contribution
are not established. A new-method requirement cannot be met by this reframing.

## Additional local check of existing data

Source: immutable `results/adaptive-screen-v03/records.json`, already hash-verified.
No new data, server access, GPU use, code/config changes or threshold selection.
These are exploratory summaries of completed development data.

| Arm | Success /40 | Actual rollout mean query time, ms | Mean episode wall time, s | Median queries per episode |
|---|---:|---:|---:|---:|
| Dense512 |39|1207.30|29.89|16|
| Fixed384 |39|1022.78|26.66|16|
| Adaptive384 |38|1085.94|27.84|15.5|
| Adaptive256 |39|930.65|24.32|15|

Query means weight every recorded query equally, so episodes with more queries
contribute more. Each arm encounters its own observations; these are NOT matched
input speed comparisons or independent confirmations. Episode wall time includes
simulator cost and trajectory length. Native control episodes are excluded because
their shadow calls deliberately inflate timing.

Every primary episode in every arm required at least ten policy queries. Thus
the saved seven-query timing microbenchmark is shorter than every actual primary
rollout. Do not claim a useful short-episode niche for fixed384 based on that
microbenchmark; the robot data do not contain such episodes. Adaptive256 is
numerically faster than fixed384 in the controlled test and in these rollout
summaries, with the same observed success pattern. Do not omit this comparator.

## Candidate paper question

How do simple spatial token deletion and a backend-aligned adaptive selector
trade off task success and total inference cost in a fixed, resource-constrained
OpenVLA-OFT deployment, when startup, selection and output costs are all counted?

This is a candidate empirical question, NOT a first-in-literature claim or a
new pruning method. Existing measurements support positive compression outcomes
in development. They do not establish broad novelty, unseen-state performance,
equivalence or superiority. The contribution must be an informative, reproducible
comparison and explanation of tradeoffs, not the bare existence of compression.
The failed backend-method novelty gate remains failed.

## Shortest bounded completion sequence

1. Assemble an evidence-led manuscript outline/claim map using existing verified
   data. Explicitly call it a comparative study. All four arms and unfavorable
   historical results remain available; do not pool different development studies
   or pretend the failed trained corrector supplies the positive contribution.
2. Before another run, specify what independent evidence would materially improve
   this study and its contribution beyond existing reports. A confirmation must
   use exposure-audited states, explicit uncertainty/precision and paired analysis,
   with startup and natural-trajectory timing and all relevant comparators.
   No arbitrary sample count or outcome-dependent stopping. No launch until a
   complete prospective protocol exists. Do not use a [0,0] development bootstrap
   interval as proof of equal performance.
3. Run at most the justified, fixed confirmation—not repeated attempts to obtain
   favorable results. Independently reconcile its full evidence.
4. Write the complete paper around the actual findings and clearly label prior
   methods, our adaptations and remaining limitations. Audit every abstract claim
   against evidence. A completed draft is not a guarantee of peer-reviewed acceptance.

## Boundaries and current status

No new GPU worker or heartbeat is active. No confirmation protocol has been
frozen. No journal-ready or positive-method paper is claimed. No new architecture
or renamed method is proposed. Existing frozen artifacts remain unchanged.
The next paper route is empirical and conditional, not an assurance that one
more experiment solves the contribution problem. If this scope is unacceptable,
genuine method development requires a different plan and additional time.
