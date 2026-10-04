# PAIR-VLA Phase P3 Report

Date: 2026-08-29  
Phase: P3 — outcome-blind physical headroom  
Decision: **SCIENTIFIC STOP; DO NOT ADVANCE TO P4**

## Result

The physical cache path produced substantial complete-cycle savings at deeper
reuse profiles and longer horizons, but no point satisfied every frozen P3
gate. The decisive failure was the 2% total feature/router/provenance/reset
overhead ceiling: observed conservative overhead bounds were 3.41% to 5.35%.
All memory, source-provenance, shape, service, control, artifact, and protected-
population gates passed.

The strongest point was `D62_BAL_PT1` at horizon 4:

- raw complete-cycle saving: 20.89%;
- one-sided 95% raw-saving lower bound: 20.54%;
- conservative net-saving lower bound: 14.37%;
- total overhead upper bound: 4.91%;
- service rate: 100%; and
- peak aggregate GPU memory: 18,265 MiB versus the 23,552-MiB limit.

It clears the positive raw-saving and 10% net-saving requirements but fails the
predeclared 2% overhead ceiling. The frozen rule is conjunctive, so this point
cannot pass P3.

## Frontier summary

| Profile | Horizon | Raw saving | Raw 95% lower | Net lower | Total overhead upper | Decision |
|---|---:|---:|---:|---:|---:|---|
| D50_BAL_PT0 | 4 | 15.00% | 14.66% | 10.25% | 5.35% | FAIL overhead |
| D50_BAL_PT1 | 4 | 15.29% | 14.84% | 10.38% | 5.24% | FAIL overhead |
| D50_BAL_PT2 | 4 | 15.68% | 15.24% | 10.66% | 4.82% | FAIL overhead |
| D50_SCENE_PT1 | 4 | 15.69% | 15.51% | 10.85% | 4.91% | FAIL overhead |
| D59_BAL_PT1 | 2 | 16.41% | 16.31% | 11.41% | 4.09% | FAIL overhead |
| D59_BAL_PT1 | 4 | 20.08% | 19.70% | 13.79% | 5.22% | FAIL overhead |
| D62_BAL_PT1 | 2 | 17.41% | 17.10% | 11.96% | 4.45% | FAIL overhead |
| D62_BAL_PT1 | 4 | 20.89% | 20.54% | 14.37% | 4.91% | FAIL overhead |

Other points also failed the 10% net-saving threshold and/or the overhead
ceiling. The full 24-point result is preserved in the analysis JSON.

## Gate reconciliation

| Gate | Result |
|---|---|
| At least one fully valid headroom point | **FAIL: 0/24** |
| Positive raw-saving lower bounds | PASS for all measured points |
| Net-saving lower bound at least 10% | PASS for 8/24 points |
| Total overhead at most 2% | **FAIL for all 24 points** |
| Service rate at least 70% | PASS; 100% throughout |
| Exact source/provenance invariants | PASS |
| Sequence/shape invariants | PASS |
| Sidecar, all-fresh, and warm-up controls | PASS |
| Memory strictly below 23,552 MiB | PASS; peak aggregate 18,265 MiB |
| Query/block/resource caps | PASS; 688 queries and 96 blocks |
| Protected outcome boundary | PASS; no expert actions or outcomes opened |
| Artifact cap | PASS; 541,141 bytes reconciled |

## Technical history and recovery

V01 stopped with zero queries at LIBERO's external configuration prompt. V02
stopped on its first warm-up because of a CausalLM output-field mismatch. V03
then completed all 688 queries and 96 timing blocks but failed while creating
the post-run CPU router mock because eight timing aliases were passed to a
six-category router.

The user-approved recovery added zero model queries. It authenticated the exact
V03 source and evidence hashes, inferred pre-block control success from the
hash-pinned fail-closed control flow, measured the router/reset mock with the
six base profiles, and wrote a separate terminal summary. Original V01-V03
evidence was never deleted, overwritten, resumed, or treated as a completed run.

## Scientific interpretation

The downstream cache substrate has real timing reserve: aggressive profiles
showed roughly 20% raw saving and retained more than 14% conservative net lower
bound at horizon 4. However, the current PAIR feature/mask/provenance path is
too expensive for the frozen 2% overhead envelope. Therefore P3 does not
authorize collecting expert-regret labels, training the router, or opening
simulator outcomes.

This result localizes the obstacle to implementation overhead rather than GPU
memory, cache correctness, source diversity, or lack of raw decoder savings.
Any continuation would require a separately researched and approved protocol
revision focused on reducing or reformulating the decision-path overhead; it
cannot be presented as a passing PAIR P3 result.

## Evidence

- `results/pair-p3-physical-v03/blocks.jsonl`
- `results/pair-p3-physical-v03/technical_stop.json`
- `results/pair-p3-physical-v03-recovery01/worker_summary.json`
- `results/pair-p3-physical-v03-recovery01/analysis.json`
- `reports/PAIR_P3_TECHNICAL_STOP_01.md`
- `reports/PAIR_P3_TECHNICAL_STOP_02.md`
- `reports/PAIR_P3_TECHNICAL_STOP_03.md`

P4 remains unauthorized. No commit, push, manuscript edit, or unrelated server
change was performed.
