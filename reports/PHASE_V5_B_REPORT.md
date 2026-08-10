# Phase V5-B Output-Blind Screening Report

Status: **COMPLETE — ELIGIBLE CANDIDATE SELECTED**

Run ID: `acr-v5b-output-blind-screening-v01`

Frozen preflight revision: `e901faede5e4d4ad151d48a37e6909daa02578ab`

## 1. Disposition

V5-B passes. Three of six predeclared candidates satisfied every frozen gate.
The lexicographic safety-first rule selected the least permissive eligible
candidate:

`v5-a100-b40`

This permits preparation of the V5-C CPU executor-correctness protocol. It does
not establish task success, measured CUDA reduction, wall-time speed, online
reuse, or a positive paper result.

## 2. Integrity and population

- Input: outcome-free A4 upstream Full-Refresh Object companion trace.
- Population: tasks `0-9`, initial states `0-9`, seed `0`.
- Episodes: `100/100`.
- Query records: `1,773/1,773`.
- Trace bytes: `13,415,489`.
- Ordered path/content SHA-256:
  `3ce22a1d1de7d33ed0a6bcdb52b32f42800d732ec93aed0bfed593f1e536b34b`.
- Candidate replays: two complete repetitions, byte-identical.
- Independent result verifier: passed with zero errors.
- Success/failure/reward/timing fields: sealed and rejected by schema.

## 3. Candidate results

| Candidate | Cap | Reuses | Reuse rate | Reuse 95% CI | Logical visual reduction | Logical 95% CI | Max streak | Eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `v5-a100-b35` | 0.35 | 533 | 30.06% | [28.94%, 30.98%] | 15.03% | [14.47%, 15.49%] | 1 | No |
| `v5-a100-b40` | 0.40 | 629 | 35.48% | [34.18%, 36.54%] | 17.74% | [17.09%, 18.27%] | 1 | **Yes — selected** |
| `v5-a150-b35` | 0.35 | 533 | 30.06% | [28.94%, 30.98%] | 15.03% | [14.47%, 15.49%] | 1 | No |
| `v5-a150-b40` | 0.40 | 630 | 35.53% | [34.23%, 36.60%] | 17.77% | [17.12%, 18.30%] | 1 | Yes |
| `v5-a200-b35` | 0.35 | 533 | 30.06% | [28.94%, 30.98%] | 15.03% | [14.47%, 15.49%] | 1 | No |
| `v5-a200-b40` | 0.40 | 631 | 35.59% | [34.29%, 36.65%] | 17.79% | [17.15%, 18.33%] | 1 | Yes |

The cap-0.35 candidates failed the frozen 35% reuse/17.5% logical-work point
gates and corresponding lower-bound gates. All cap-0.40 candidates passed.
The selection rule then chose threshold level `1.0`, the smallest eligible
level, yielding `v5-a100-b40`.

## 4. Selected-candidate gate reconciliation

| Gate | Frozen requirement | Observed | Result |
|---|---:|---:|---|
| Population | 100 episodes / 1,773 queries | 100 / 1,773 | Pass |
| Maximum reuse streak | 1 | 1 | Pass |
| Prefix-cap violations | 0 | 0 | Pass |
| Gripper-transition reuses | 0 | 0 | Pass |
| Isolation-state mismatches | 0 | 0 | Pass |
| Invariant failures | 0 | 0 | Pass |
| Post-reuse refreshes | at least 1 | 605 | Pass |
| Reuse point | at least 35% | 35.48% | Pass |
| Reuse lower 95% bound | at least 30% | 34.18% | Pass |
| Logical visual reduction point | at least 17.5% | 17.74% | Pass |
| Logical reduction lower 95% bound | at least 15% | 17.09% | Pass |

Maximum observed prefix reuse was exactly the frozen `0.40` cap.

## 5. Interpretation

This is the first positive V5 mechanism result: the corrected latch can retain
a predeclared useful offline reuse region while mechanically preventing
consecutive reuse. The fact that all cap-0.40 candidates were similar also
shows that the prefix budget, rather than the more aggressive thresholds,
largely controls this trace population. Selecting the lowest threshold is
therefore consistent with the frozen safety preference.

The result is deliberately limited. Full-Refresh traces do not reproduce the
closed-loop observations/actions that IR-SA-ACR would cause. The 17.74% value
counts logical scene-versus-wrist visual components; it is not measured GPU
time. Task success and realized speed remain unknown.

## 6. Resources and protection

- GPUs: `0`.
- Model queries: `0`.
- Simulator episodes/resets: `0`.
- Downloads: `0`.
- New task outcomes: `0`.
- Goal states `0-9`: unopened.
- States `10-49` and reserve seeds `7/17/27`: unopened.
- Manuscript: unchanged.
- TITAN work: only `/home/ved/SAVR`.

## 7. Evidence

- freeze: `configs/acr/v5_b_output_blind_preflight.json`
- analyzer: `scripts/analyze_acr_v5_b.py`
- independent verifier: `scripts/verify_acr_v5_b_result.py`
- machine result: `reports/runtime/acr_v5_b.json`
- result semantic SHA-256:
  `8a9f15b818b58ed2868d4b1123a222a4c062507161ab7de911d8d233f3b1efec`
- machine-file SHA-256:
  `46d2033d1ea409062bbe6cc57afc46359c1047fa68c565ed6216ca8121c88080`

## 8. Next gate

Prepare and freeze V5-C before implementing any executor change. V5-C must
separate reference and optimized paths, preserve controller decisions, verify
refresh/reuse parity and failure-state integrity, and remain CPU-only. GPU use
still requires a later V5-D freeze and explicit user coordination.
