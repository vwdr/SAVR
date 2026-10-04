# PAIR P4B Independent Confirmation Report

Date: 2026-08-30  
Run: `pair-p4b-confirmatory-v01`  
Status: **SCIENTIFIC STOP**

## Execution integrity

The single frozen attempt completed exactly 1,784 model calls on GPU 0, with
776 intervention records, 560 feature records, and 304 contract records. Peak
aggregate GPU memory was 17,799 MiB, below the 23,552 MiB cap. Exact repeats,
all-fresh controls, suite-by-horizon balance, tail concentration, resources,
and every protection gate passed. No retry, simulator, terminal outcome,
locked-test value, or persisted raw action array was used.

## Confirmatory results

| Frozen endpoint | Point | One-sided 95% lower | Required | Result |
|---|---:|---:|---:|---|
| Structured Spearman | 0.2001 | 0.0999 | point >= 0.30; lower > 0.15 | Fail |
| Matched-service CVaR90 improvement | 3.14% | -1.07% | point >= 15%; lower > 0% | Fail |
| Horizon-2/4 CVaR90 improvement | 1.62% | -- | point >= 10% | Fail |

Five of eight gate groups passed. The frozen analyzer therefore returned
`scientific_stop`. P4B did not confirm P4's favorable 40-contract estimates:
the larger independent population showed materially weaker ranking and tail
improvement, not merely wider uncertainty.

Task-cluster sensitivity lower bounds were 0.0897 for Spearman and -0.37% for
tail improvement, consistent with the primary stop. Non-gating diagnostics
showed the strongest horizon-4 signal (Spearman 0.3741; 10.17% tail
improvement), but horizon 2 had negative tail improvement and no subgroup may
rescue the failed prespecified pooled gates. The fixed-effect P4/P4B Spearman
summary was 0.2204 and is secondary only.

## Decision

PAIR stops before P5. The locked-test population remains sealed. The protocol
forbids P4C, post-hoc gate changes, router redesign on P4B outcomes, or using
secondary subgroups to reverse the decision. These data remain valid evidence
for understanding why the development signal did not generalize.
