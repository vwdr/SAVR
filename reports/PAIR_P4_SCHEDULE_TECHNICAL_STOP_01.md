# PAIR P4 Schedule Technical Stop 01

Date: 2026-08-29 EDT  
Schedule: `p4_schedule_v1.jsonl`  
Decision: **invalid schedule; no GPU launch**

## Finding

The frozen CPU scheduler produced 400 balanced anchors but reported 200/200
fallbacks for requested positive gripper-transition strata and zero observed
transitions. The checkpoint metadata explains the result: all four LIBERO
normalization keys mark the gripper coordinate as unnormalized (`mask=false`),
so it remains in `[0,1]`. Testing `coordinate >= 0` therefore always returns
the same sign.

P0 already defines the correct execution diagnostic: map `[0,1]` to `[-1,1]`,
then sign-binarize. The scheduler omitted that mapping. This is a schedule
integration error, not a PAIR method or expert-regret result.

## Accounting and protection

- 400 schedule rows generated and preserved
- 0 model loads or model queries
- 0 regret values computed or accessed
- 0 dense/cache action values accessed
- 0 expert action arrays persisted
- 0 locked-test labels
- 0 simulator outcomes
- CUDA hidden; no GPU selected or used

## Evidence

- Invalid schedule SHA-256:
  `d7d2c8b9b151c4cbabf0e90769365e249d3bdcaf0752a0bbcc5fc8efdb3ae5c6`
- Summary SHA-256:
  `b22808b82549389a3b692fc620658eb46d0409d4487329ee001a11bb1faad88d`
- Summary semantic SHA-256:
  `a61a822e746ed3b891c98120bf2687f64eb29026fa625e1424b820344458a61c`

## Boundary

The invalid schedule must never be used for GPU execution. A versioned
schedule amendment must apply the P0-defined `[0,1] -> [-1,1] -> sign`
diagnostic, use new immutable output identities, preserve every other P4
population/method/gate/resource rule, and receive explicit approval before
regenerating the schedule.
