# ACR Version 3 Phase V3-D Technical Recovery

## Preserved stop

The first V3-D run started one BFR episode and stopped before recording a
completed model query or executing an action. The pinned evaluator returned a
valid action list, while the BFR adapter's default finite checker called
`torch.isfinite` as if that list were a tensor. The immutable summary records
one `TypeError`, zero query records, zero aggregated outcomes, clean source and
checkpoint restoration, and GPU release. No episode success field was opened.

This is a runner integration defect, not a method, controller, task, timing,
cache, batching, or scientific failure. The original run and all of its files
remain immutable at `results/acr-v3d-paired-object-dev03-09-v01`.

## Frozen correction

Pass both V3 adapters an explicit action-finite checker that converts the
pinned evaluator's list/NumPy output to a NumPy array and checks every element.
The check remains outside the timed inference boundary. A regression test must
accept a finite list and reject a list containing infinity or NaN. No other
production or analysis behavior changes.

## One-time recovery

Use new run ID `acr-v3d-paired-object-dev03-09-recovery-v01` and rerun the
complete unchanged 70-pair scientific matrix. The failed start remains counted,
so cumulative allowance is 141 attempted episode starts: one preserved
technical start plus 140 recovery attempts. The 43,200-second and 2-GiB caps
remain cumulative across both runs. Outcome blindness, counterbalancing,
population, policies, controller, success/reuse/visual/wall gates, no-retry
rule for scientific failures, and V3-E stop are unchanged.

Recovery may launch only after this correction, its tests, and
`configs/acr/v3_d_recovery.json` pass locally and on TITAN, merge to GitHub,
and synchronize to the server.
