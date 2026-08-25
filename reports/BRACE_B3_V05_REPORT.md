# BRACE-B3 v05 Physical Microbenchmark Report

Date: 2026-08-25  
Disposition: **COMPLETE; STOPPED NEGATIVE**

## Result

V05 completed the full permitted physical workflow without a technical stop.
The frozen analyzer rejected advancement because no profile jointly preserved
action parity and passed both speed gates. This is a valid negative B3 result,
not another integration failure.

The run completed 356 model queries: 22 optimized core-FR, 302 cache-suite,
and 32 VLA-ADP queries. The remaining 32-query VLA-Pruner allocation was not
reassigned because the pinned upstream release lacks an imported VLA-Cache
utility. This reviewed exclusion preserved the query cap and produced no
automatic retry.

## Profile results

| Profile | Accelerated-query reduction | Complete-cycle reduction | Speed gate | Timed action parity |
|---|---:|---:|---|---:|
| P1-S25 | 7.60% | 4.82% | Fail | 0/42 |
| P1-S50 | 11.82% | 7.68% | Fail | 0/42 |
| P2-D25 | 11.02% | 7.30% | Fail | 0/42 |
| P2-D50 | 18.71% | 12.23% | Pass | 0/42 |

P2-D50 therefore demonstrated the targeted physical speedup, but not the
required reliability. All 168 timed cached actions differed numerically from
their dense cache-stack references, although their timed gripper decisions
matched. Warm-cycle maximum action differences were approximately 1.00--1.03;
timed maximum differences were approximately 0.85--0.95.

## Other gates

- Optimized core-FR versus dense cache-stack P0 parity failed on all three
  deterministic inputs. Maximum absolute differences were 0.222--0.368, while
  gripper decisions matched.
- Corrected VLA-Cache reuse parity failed in all 10 paired comparisons.
- The dense SDPA sidecar passed exactly: zero action difference, exact gripper
  decisions, and all 32 expected attention calls.
- Cache provenance/reset invariants passed.
- Peak memory passed: 18,419 MiB aggregate and 19,094,568,960 worker-reserved
  bytes, both below the strict 23 GiB boundary.
- Query/identity accounting and the measured P4 disposition passed.
- Dense cache-stack completion was approximately 1.220 s p50, exceeding the
  400 ms eight-action control window without artificial sleep.

Five conjunctive gates passed and five failed. The failed gates were P0/FR
parity, all-profile parity, corrected VLA-Cache parity, at least one joint
profile speed gate, and mandatory comparator disposition (which is conjunctive
with corrected VLA-Cache parity).

## Interpretation

The central tradeoff is now measured directly: more aggressive cache reuse can
provide meaningful physical acceleration, but under the frozen BRACE contracts
it changes predicted actions beyond the declared tolerance. Because task
outcomes were intentionally unavailable in B3, these deviations cannot be
claimed safe or harmless. The frozen B3 gate therefore correctly stops BRACE
before B4.

This result does not support a positive-results paper for BRACE as currently
defined. Any continuation would require a newly researched method or a
scientifically justified redesign—not a threshold search, post-hoc tolerance
change, or v05 retry.

## Integrity and evidence

- Run source revision:
  `77f321da074fb719c594160b96a9cc3fc1f1050a`.
- Analysis semantic SHA-256:
  `26219851a78d35142ef956950e59b37114a8a893b562007308daf791f54f503b`.
- Launch file SHA-256:
  `7ab09aed44013215d93d449ec5047fcbb2637fa3269146f99e511e8b31261666`.
- Run-summary file SHA-256:
  `97610434137524275861e28e0ab21b9a49655c38d893ca73d2fe33ff255c9c64`.
- Analysis file SHA-256:
  `fd2f8077bff6214de6b24e860dd5a28f8bf95b38b5a1bf6f3e077f27af8ddd9d`.
- All run-summary, analysis, and worker semantic hashes independently verified.
- No simulator outcomes or protected task-success fields were accessed.

The complete machine-readable evidence is preserved under
`results/brace-b3-physical-v05/`. B4 is ineligible and unauthorized.
