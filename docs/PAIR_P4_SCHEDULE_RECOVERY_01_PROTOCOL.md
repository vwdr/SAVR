# PAIR P4 Schedule Recovery 01 Protocol

## Authorization required

This versioned amendment corrects only the pre-GPU gripper-transition stratum
used by the P4 scheduler. It requires explicit user approval before execution.
It does not authorize any other population, method, gate, resource, retry, P5,
locked-test label, simulator outcome, or manuscript change.

## Authenticated technical stop

- Invalid schedule: `configs/pair/p4_schedule_v1.jsonl`
- Invalid schedule SHA-256:
  `d7d2c8b9b151c4cbabf0e90769365e249d3bdcaf0752a0bbcc5fc8efdb3ae5c6`
- Technical report: `reports/PAIR_P4_SCHEDULE_TECHNICAL_STOP_01.md`
- Failure identity: 200 requested positive-transition strata, 200 fallbacks,
  and zero observed transitions.
- Scientific/model use: zero regret values, model calls, GPU use, locked-test
  labels, or simulator outcomes.

## Sole correction

The checkpoint marks the LIBERO gripper coordinate `mask=false`; P1 therefore
retains it in `[0,1]`. For schedule stratification only, transform each retained
coordinate as `execution_coordinate = 2 * coordinate - 1`, then classify its
sign. A contract is a transition when this execution sign changes within any
of the `h` evaluated future chunks or relative to the immediately preceding
action.

This exactly implements P0's already-frozen gripper execution diagnostic. It
does not change expert normalization for regret, any router feature, action
label, threshold, schedule count, task/split allocation, horizon, mask,
profile, repeat, query budget, or gate.

## Recovery identities and stop rules

- Recovery run: `pair-p4-pilot-v02-schedule-recovery01`
- Schedule: `configs/pair/p4_schedule_v02_recovery01.jsonl`
- Summary: `reports/pair_p4/schedule_summary_v02_recovery01.json`
- The v1 schedule remains immutable and ineligible.
- The recovered schedule must contain 400 anchors with the unchanged frozen
  allocations and must report transition support by split/category.
- Any remaining inability to represent both transition strata stops P4 before
  GPU execution.
- No automatic retry is permitted.
