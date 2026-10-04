# PAIR-VLA Phase P3 Technical Stop 03

Date: 2026-08-29  
Attempt: `pair-p3-physical-v03`  
Decision: **TECHNICAL STOP AFTER COMPLETE TIMING POPULATION; NOT YET ANALYZED**

## What completed

- All 688 frozen model queries completed.
- All 96 randomized paired timing blocks were written and individually
  semantic-hashed.
- The artifact contains the exact 8-profile by 3-horizon by 4-repetition
  schedule.
- Peak aggregate GPU memory was 18,265 MiB, below the strict 23,552-MiB limit.
- No expert actions, simulator state, terminal outcomes, or success fields were
  accessed.
- Checkpoint metadata was restored and no backup remains.

## Why terminalization failed

After the final block, the worker constructed the CPU-only router-overhead
mock using all eight timing-profile aliases. The frozen router supports at
most six profile categories because its scientific categories are the six
base physical profiles; the two additional timing variants differ only in
protected-tile allocation. `PairRouter.initialize` therefore rejected the
eight aliases before the worker summary was written.

This is a terminal bookkeeping/integration error, not a cache execution error
and not a scientific result. The timing blocks must not be analyzed until a
separately authorized recovery produces a valid terminal summary.

## Immutable evidence

| Artifact | SHA-256 |
|---|---|
| V03 worker source | `3f58f896a3a9c4308b61e50c4e145daa626c68614c2193d1aced5b7d0342870e` |
| V03 OpenVLA helper source | `75fd1b9131e9eb9c050ad988c7d593a2cb9a630519fe4f0c24ab83e635a8bdbb` |
| V03 configuration | `c1c81cb5a4f7ae99325d7ebd4ffcc619cf12f0af9ef37ec5b8697b40ed75701b` |
| V03 preflight | `95a3e0c0b4d0040c5d20109d63a5ed2438781649d6018d70ab81d262025f91ba` |
| 96-block timing population | `e9d56ebe73d989ecb3cde7027629792fa5edc658a77ca77f2a4eb39fd2e2f652` |
| Technical stop | `5f5f0103dc1d84a0a428911f5cd2072343af2b93892086cb76346c917ca06c4b` |

Copies of the exact runtime worker and helper source are preserved under
`reports/pair_p3/v03_runtime_sources/`.

## Proposed zero-query recovery

A recovery can remain within the original cumulative 700-query boundary: V02
attempted one warm-up and V03 completed 688, totaling 689; the recovery uses
zero model queries.

The recovery would:

1. require CUDA to be hidden and refuse model, checkpoint, data, simulator, or
   network access;
2. authenticate the exact source, configuration, preflight, block, and stop
   hashes above;
3. validate all 96 block identities, semantic hashes, action-structure-only
   records, query accounting, protected fields, and resource limits;
4. use the pinned V03 control flow to establish that sidecar, all-fresh,
   sequence-map, profile warm-up, and exact reused-K/V checks passed before any
   timing block could be appended;
5. initialize the router mock with the six `base_profile_id` categories and
   measure its CPU p99 plus reset p99;
6. conservatively derive sidecar overhead from matched dense versus cache-anchor
   records and use the recorded per-query mask/provenance costs;
7. use peak aggregate GPU memory as the conservative memory gate because V03
   did not preserve PyTorch's peak-reserved counter; and
8. write a separate immutable recovery summary and only then run the frozen
   outcome-blind timing analysis.

The original V03 directory remains unchanged. The recovery must stop before P4
regardless of whether the P3 timing gate passes.
