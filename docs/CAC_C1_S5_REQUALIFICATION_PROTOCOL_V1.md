# CAC C1 S5 Requalification Protocol V1

Date: 2026-09-01  
Parent substrate: `D62_BAL_PT1_S4C_V1`  
Purpose: requalify CAC tensors and systems feasibility after official OpenVLA semantic correction

## Governing boundary

S4 passed corrected D62 substrate requalification. Historical C1 remains useful
engineering evidence, but its action states, `Z_C`, base actions, features, and
timing were produced before the official action-readout correction. S5 must
therefore rerun C1 from a new immutable root.

This is outcome-free systems and tensor qualification. It does not train the
adapter, evaluate repair quality, run a simulator, or establish positive task
success.

## Frozen method

Preserve the original C1 design:

- one frozen D62 profile, whose tile selections were shown by S4 to be
  invariant on all eight audited observations;
- exact action-head penultimate `Z_C` and base-action reproduction;
- current visual tile features, physical-source deltas, proprioception,
  instruction-only embedding, and 12-column provenance;
- the 5,070,599-parameter zero-initialized correction adapter;
- recursive ages 1--4, fresh reset, branch isolation, streaming record, and
  complete-cycle timing at horizons 2 and 4; and
- the original memory, storage, headroom, chronology, and exactness gates.

Use corrected identity `D62_BAL_PT1_S4C_V1`. The historical profile values may
be loaded from `D62_BAL_PT1` only because S4 proved 0/8 selection materiality;
the result must never relabel the historical action implementation as corrected.

## Mandatory independent official parity

Before warmup or internal C1 controls:

1. load through the S3-qualified official OpenVLA loader guard and official
   OpenVLA-OFT source tree;
2. execute one released `evaluation.get_action` call on the frozen C1 anchor;
3. independently capture the exact official action-head input and normalized
   action output using the reversible S3/S4 oracle;
4. execute one corrected dense custom call on the identical prepared anchor;
5. compare official/custom 56x4096 hidden states, normalized 8x7 actions, and
   unnormalized 8x7 actions at maximum absolute error `1e-6`;
6. prove image-copy isolation and the documented normalized-state mutation;
7. persist only shapes, maxima, and booleans—not raw actions or hidden tensors.

Any mismatch stops C1 before internal feasibility evidence is interpreted.

## Frozen call schedule

- independent official/custom parity: 2 calls;
- model warmup: 4 calls;
- sidecar and all-fresh controls: 4 calls;
- paired action-head-hook controls: 8 calls;
- two recursive age-1--4 cycles plus resets: 12 calls;
- branch-isolation controls: 3 calls;
- complete-cycle timing: 64 calls;
- total: exactly 97 calls; hard cap: 160.

No existing C1 control is removed to offset the new oracle calls.

## Exit gates

All must pass:

- independent official hidden/normalized/unnormalized parity at `1e-6`;
- dense/all-fresh and `fc2(Z_C)` error at `1e-6`;
- sidecar, cache, action, recursive, and reset exactness;
- maximum physical source age exactly 4 and never above 4;
- clean-clone cache and tracker isolation plus RNG/salience immutability;
- finite corrected tensor shapes/dtypes/devices and exact source chronology;
- zero-initialized adapter bitwise bypass;
- exactly 5,070,599 adapter parameters and 1,380,352-byte numeric record;
- median D62 horizon-4 complete-cycle gross saving at least 8%;
- peak aggregate selected-GPU memory below 23,552 MiB;
- artifacts below 4 GiB and exact checkpoint restoration; and
- exactly 97 model calls.

## Resources and protection

One explicitly selected idle GPU, one model process, 3,600-second wall limit,
offline project-confined caches, no downloads, no training, no simulator, no
expert actions, no terminal outcomes, no raw action persistence, and no
automatic retry. All server writes remain inside `/home/ved/SAVR`.

## Decision boundary

A pass makes corrected CAC technically feasible and re-establishes C1 evidence.
It does not show that the adapter can repair cached actions or preserve task
success. Stop before C1H regardless of result; C1H requires separate approval.
