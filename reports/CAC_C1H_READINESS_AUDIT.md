# CAC Phase C1H Readiness Audit

**Date:** 2026-08-31  
**Decision:** READY FOR ONE FROZEN C1H ATTEMPT; STOP BEFORE C2

## Purpose

C1H asks only whether the frozen D62 cache substrate leaves a plausible and
non-collapsed amount of terminal task-success loss for a learned correction to
repair. It does not evaluate CAC and cannot be used to choose CAC features,
architecture, or training settings.

## Frozen experiment

- Compare dense OpenVLA-OFT with uncorrected `D62_BAL_PT1` on identical paired
  LIBERO initial-state conditions.
- Stage 1 contains state IDs 0--2: 120 conditions and 240 episodes.
- Open state IDs 3--5 only if the prespecified Stage-1 result is ambiguous;
  the cumulative maximum is 240 conditions and 480 episodes.
- Use all four 10-task suites, seed 7, the official suite horizons, eight
  actions per query, and exactly balanced paired arm ordering.
- D62 uses the validated recursive mixed-age source tracker and performs a
  complete dense reset after at most four cached query intervals.
- Hold partial success outcomes in worker memory. Write terminal records only
  when an exact 240-episode stage is complete.

## Mechanical Gate H

Stage 1 proceeds when dense success is at least 75%, D62 success is at least
50%, the dense-minus-D62 gap is 8--35 percentage points, and at least two suites
favor dense. It stops when dense is below 75%, D62 is below 50%, the gap is at
most 2 points, or the gap exceeds 35 points. Every other result opens the
prespecified extension. On all 240 cumulative conditions, proceed requires
dense at least 75%, D62 at least 50%, a 5--35 point gap, and at least two suites
favoring dense; otherwise stop.

## Anticipated failures and controls

| Risk | Frozen control |
|---|---|
| Partial-result steering or optional stopping | No partial outcomes are written; extension is opened only by the coded Gate-H ambiguity rule |
| Cache implementation drift | Authenticate the previously validated P3 physical helper and D62 profile; regression-test source ages and mask logic |
| Paired-condition mismatch | Derive both arms from each authenticated C0 condition and require exact paired coverage |
| Suite/task normalization mismatch | Verify C0 task language, suite-local task index, normalization key, and official horizon before each rollout |
| Recursive age violation | Mandatory complete reset after four cached intervals; source tracker independently rejects age above four |
| Invalid actions or simulator failure | Fail closed, write a technical stop, preserve progress evidence, and do not retry automatically |
| GPU interference or OOM | Use one idle GPU only; stop below 23,552 MiB aggregate/reserved memory; never affect other processes |
| Scope expansion | No training, CAC inference, locked states 10--49, new downloads, or C2 work |

## Verified preflight

- 32 C1H/C1/P3 unit and regression tests passed on TITAN.
- Config, protocol, C0 population, C1 pass, source, runner, and analyzer hashes
  reconcile on TITAN.
- All four GPUs were idle at the preflight snapshot; GPU 0 was selected with
  6 MiB used and 0% utilization.
- Project filesystem had 348,212,461,568 bytes available, well above the
  required margin.
- The immutable output root `results/cac-c1h-headroom-v01` did not exist.

## Resource and stopping boundary

One model process and one selected GPU; at most 480 episodes, 20,000 model
queries, 10 hours, 1 GiB of artifacts, and strictly less than 23,552 MiB GPU
memory. No automatic retry is authorized. Completion applies Gate H once,
publishes the evidence, and stops before C2.
