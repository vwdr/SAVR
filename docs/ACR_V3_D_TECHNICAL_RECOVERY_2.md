# ACR Version 3 Phase V3-D Technical Recovery 2

The first recovery corrected list-action validation and completed the BFR
episode for the first pair. Before the paired V3 query, controller reset
rejected the context because the runner used the method label
`sa-bdp-acr-t25-h2-b30-v01` where the unchanged controller requires its frozen
configuration identity `acr-t25-h2-b30`. The recovery stopped with 35 BFR
queries, zero V3 queries, clean restoration, and GPU release. Success remains
unopened while the official matrix is incomplete.

This is an identity-wiring defect outside the timed method. Freeze the context
mapping as `batched-fr -> batched-full-refresh` and
`sa-bdp-acr-t25-h2-b30-v01 -> acr-t25-h2-b30`. Add a regression assertion for
both identities. No controller value, cache behavior, batching path, model,
task, timing boundary, gate, or analysis rule changes.

Preserve both prior run roots and exclude the partial BFR episode from official
analysis. Run the full unchanged 140-episode matrix in one new model process
under `acr-v3d-paired-object-dev03-09-recovery-02-v01`. Cumulative allowance
is 143 starts: one initial technical start, two first-recovery starts, and 140
final-recovery starts. Wall time and artifact limits remain cumulative at
43,200 seconds and 2 GiB. Outcome blindness and the scientific no-retry rule
remain unchanged. Merge, synchronize, and CPU-verify this checkpoint before
launch; stop before V3-E regardless of the result.
