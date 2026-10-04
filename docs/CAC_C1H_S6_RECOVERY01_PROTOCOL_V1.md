# CAC C1H S6 Recovery 01 Protocol V1

**Date:** 2026-09-01  
**Authorization:** user approved one versioned recovery  
**Boundary:** mandatory outcome-free qualification, then one simulator attempt;
stop before C2

## Recovery scope

The source attempt `cac-c1h-headroom-s6-v01` stopped after two pre-episode
calls, with zero episode attempts and no outcomes, because its wrapper omitted
the already computed `action_hidden` and `base_action` tensors from its return
mapping. Recovery 01 changes only that return plumbing and the tests that
authenticate it.

It does not change the checkpoint, official evaluator, corrected substrate
`D62_BAL_PT1_S4C_V1`, historical D62 profile values, maximum reuse age, dense
reset, preprocessing, action chunking, LIBERO tasks, state IDs, pair order,
seeds, episode horizons, terminal-success outcome, Gate H, resource caps, or
stop-before-C2 rule.

## Required code contract

When `capture_cac=True`, `policy_query` must return:

- unnormalized actions with shape `8 x 7`;
- `action_hidden` with shape `1 x 56 x 4096`;
- normalized `base_action` with shape `1 x 8 x 7`;
- cache and prepared-query objects.

Missing or malformed fields fail closed. A direct synthetic unit test must
prove successful propagation and rejection of both missing tensors. This test
exists specifically because source-string checks did not detect the source
attempt's omission.

## Mandatory outcome-free GPU qualification

Before any simulator attempt, run the exact worker in qualification-only mode
on one idle GPU. It may use one frozen Stage-1 state to construct the official
observation, but it may not launch a terminal episode or inspect reward,
success, or done fields. It must make exactly four model calls:

1. released official dense evaluator with reversible boundary capture;
2. corrected custom dense path with CAC capture;
3. corrected D62 anchor;
4. corrected D62 recursive reuse.

The qualification passes only if:

- official/custom hidden, normalized-action, and unnormalized-action maximum
  errors are all at most `1e-6`;
- the official path does not mutate the custom policy observation;
- action processing is finite and shape-correct;
- D62 anchor and reuse both return finite `8 x 7` actions and valid caches;
- no terminal episode or outcome is opened;
- checkpoint restoration and aggregate-memory limits pass.

If qualification fails, preserve evidence and stop. Do not launch C1H.

## Single simulator attempt

Only after a complete, semantically hashed passing qualification exists may the
actual worker start. The worker independently verifies that qualification
before loading the model and repeats the same four controls before episodes.

Stage 1 remains 120 paired conditions and 240 episodes. Terminal records remain
sealed in memory until all 240 complete. The existing 240-episode extension
opens only if the original Stage-1 Gate-H ambiguity rule requires it. Monitoring
is limited to process health, progress count, artifact bytes, elapsed time, and
aggregate selected-GPU telemetry until an immutable stage summary exists.

Any technical failure is preserved and stops without automatic retry. A
completed run is independently analyzed, synced to the local review mirror,
and reported. Passing Gate H makes C2 eligible but does not authorize it or
constitute a positive paper result by itself.

