# ACR Version 3 Phase V3-D Preflight

**Authorization:** user approved Phase V3-D on 2026-08-04.

**Outcome access at freeze:** V3-C correctness/latency evidence and historical
A4/A5 evidence only. No V3 simulator outcome exists.

## Frozen paired scope

Run exactly one Batched Full Refresh (BFR) episode and one frozen SA-BDP-ACR
episode for each LIBERO-Object task `0-9`, initial state `3-9`, seed `0`.
There are 70 paired identities, 70 episodes per policy, and 140 total attempts.
For pair index `task_id * 7 + (state_id - 3)`, BFR runs first when the index is
even and V3 runs first when it is odd. This produces 35 first positions per
policy while keeping each comparison adjacent in one model process.

The V3 controller remains `acr-t25-h2-b30` exactly. BFR uses the same ordered
two-camera batched visual path as every V3 refresh, but has no controller or
cache. V3 reuse executes no logical scene work and always recomputes the wrist,
proprioception, and downstream policy. Sequential FR is not rerun; only the
immutable A4 records for matching states `3-9` may supply that latency point.

## Outcome-blind execution

The runner may record terminal success booleans but must not aggregate, print,
inspect, or gate them until all 140 terminal episode records exist. During the
run, monitoring is limited to process health, resource caps, record counts,
and technical/invariant failures. Scientific failures are terminal records
and are never retried. Any technical, invariant, resource, or operator failure
stops the matrix and preserves every existing record.

## Mechanical gate

V3-D passes only if every condition holds:

1. exactly 70 completed terminal episodes per policy and zero technical failures;
2. V3 loses at most two aggregate successes versus BFR;
3. V3 loses at most one success on every task versus BFR;
4. V3 scene reuse is at least 20%;
5. V3 visual CUDA work is at least 10% below BFR;
6. V3 synchronized query-wall ratio is at most 0.98 versus immutable sequential FR;
7. V3 synchronized query-wall ratio is at most 1.00 versus contemporaneous BFR;
8. every physical/logical work, cache, record, source, checkpoint, and restoration invariant passes.

Timing uses all query measurements except the first three global queries of
each policy. Point values are sums divided by retained query counts. No timed
outlier is removed. Failure stops before Goal; passage still stops and asks
for separate V3-E authorization.

## Hard resources and safety

- one responsibly selected GPU and one model process;
- 43,200 cumulative seconds;
- exactly 140 maximum attempted episodes;
- 2,147,483,648 cumulative V3-D result bytes;
- no download and no write outside `/home/ved/SAVR`;
- no inspection or interference with unrelated university files, processes,
  services, environments, or allocations.

The configuration, runner, analyzer, tests, and this preflight must pass local
and TITAN CPU checks and be merged/synchronized before the first episode.
V3-E, protected populations, and manuscript editing remain unauthorized.
