# PAIR P4B Design and Power Audit

Date: 2026-08-30  
Status: **PROPOSED — NO UNUSED-CALIBRATION, LOCKED LABEL, OR GPU ACCESS**

## Evidence used

- Completed P4 structured calibration contracts and frozen router/proxy
  predictions.
- P1 trajectory-index metadata and the completed P4 schedule identities.
- No unused-calibration or locked expert action, observation, regret, model
  output, reward, success, or simulator field was opened.

## Independent capacity

The P1 index contains 40 tasks with the identical split pattern: 35 train,
8 calibration, and 7 locked trajectories. After excluding the exact P4
schedule, every task retains 27 unused train, 6 unused calibration, and 7
untouched locked trajectories. The independent development capacity is exactly
240 unused calibration trajectories, balanced as 60 per suite. The proposed
allocation uses all 240 as structured confirmations, with 80 per horizon and
20 per suite-by-horizon cell. Forty all-fresh controls use one separate unused
training trajectory per task. All 280 locked trajectories remain sealed for
P5.

## Planning calculations

Suite-by-horizon-stratified empirical resampling used 20,000 replicates and
only the 40 P4 structured calibration contracts. It reproduces P4B's balanced
within-cell selection rule. Values are plug-in projected one-sided 95% lower
bounds, not guarantees:

| n | Spearman lower | CVaR90-improvement lower | Nonpositive CVaR fraction |
|---:|---:|---:|---:|
| 48 | 0.1224 | 0.36% | 4.92% |
| 84 | 0.1803 | 3.79% | 0.49% |
| 120 | 0.2092 | 8.52% | 0.00% |
| 156 | 0.2238 | 8.07% | 0.03% |
| 204 | 0.2425 | 12.31% | 0.00% |
| 240 | 0.2503 | 13.41% | 0.00% |

The design-aligned P4 plug-in tail improvement is 20.43%, compared with
21.48% under P4's original horizon-only matching. This small difference is
evidence that P4B's stricter suite balancing does not manufacture the planning
signal.

Fisher-z planning for a one-sided 95% Spearman lower bound greater than 0.15:

| Assumed true correlation | Approx. n for 80% power | Approx. n for 90% power |
|---:|---:|---:|
| 0.3452 (P4 point) | 145 | 200 |
| 0.3000 | 250 | 345 |
| 0.2750 | 363 | 502 |

The 240-contract design therefore has a credible chance if the P4 effect
transfers, but it is not powered against a substantially attenuated effect.
Using fewer than 120 structured contracts would be difficult to justify for
the CVaR endpoint. Using all 240 is the most defensible available choice.

## Principal risks and controls

- **Pilot optimism:** P4 selected the router seed and proxy before calibration,
  but its effect size can still be optimistic. P4B uses no P4B tuning.
- **Tail discreteness:** P4 CVaR used very few tail contracts. P4B serves 168
  contracts, producing a 17-contract CVaR90 tail.
- **Shared task identity:** The primary claim is conditional on the fixed 40
  tasks. A task-cluster bootstrap is reported as sensitivity rather than
  silently treating task identity as unseen.
- **Confirmation-population exhaustion:** Every unused calibration trajectory
  is used. A failed or label-bearing interrupted run ends development
  confirmation; there is no P4C.
- **Distribution shift:** The untouched calibration trajectories may differ
  from the P4 sample. This is precisely the independent test and cannot be
  corrected post hoc.
- **Phase-order integrity:** Locked test remains fully sealed so P5 can perform
  its one permitted offline-test evaluation after router scaling/freezing.
- **Closed-loop gap:** Passing offline regret does not prove task success. P5
  and later closed-loop work require separate authorization and evidence.

## Recommendation

P4B is justified as one final independent development reliability
confirmation. The full
240-contract design should be used; a smaller convenience sample would weaken
both primary endpoints. Execution should begin only after the protocol,
configuration, scheduler, worker, analyzer, and outcome-sealing preflight are
implemented and independently reconciled.
