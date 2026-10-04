# Current-frame pilot feature collection v1

2026-09-15. Implements the authorized learning pilot after the integrated
qualification passes. No change to the primary 384-token budget or sample selection.

Prerequisite: independently verified completed learning qualification (80 model
calls, 16 records, 128 diagnostic updates, zero episodes). Freeze its completed
summary and analysis hashes in the collection config before launch. Failed or
incomplete qualification forbids collection. Diagnostic adapter weights are not
loaded for collection or research evaluation.

Use exactly the frozen 4,000 observations in
`configs/openvla/current_frame_training_inputs_v1.json`, SHA-256
`2fd13ced3e5a9a2b810eb299a5c1667f8c1de3ee2c5487a4a0683e61847d9ae8`.
Keep the 3,200 fit / 800 validation roles separate. Both roles are from the
historical training split; validation trajectories never enter optimizer fitting.
No expert actions, calibration observations or locked observations are accessed.

For every observation, run the qualified 512-token teacher and then the qualified
384-token feature extractor sequentially, on the identical current images, state,
instruction and normalization key. One frozen backbone is loaded. Preserve the
exact released preprocessing, readout, action-head and original rotary positions.
No stale images, action execution, history or cache. Previous-step metadata is
only a label and is never a model input in the fixed current-frame query.

Store 4,000 numeric records using the qualified BF16-bit-pattern/FP32 schema, with
immutable sample membership, raw observation hash, input ancestry and source hashes.
The legacy `backbone_sha256` field identifies the authenticated backbone/runtime
ancestry configuration, not a hash of a single weight shard. Label this explicitly
in the run summary. Teacher labels are normalized model actions, not human labels.
Role membership lives in the frozen collection manifest and must be enforced by
the later trainer; a record's `split=train` alone is insufficient authorization.

Exactly 8,000 backbone calls, 4,000 feature records, zero optimizer steps and zero
simulator episodes. Preflight verifies all source files, sample selection, raw
observation hashes/schema and source/test-code identities before GPU use. Require
at least 30 GiB free in the project filesystem before launch. Predicted features
are about 17.1 GB (15.9 GiB); hard total output cap 20 GiB, retain a 10 GiB free-space
floor throughout. Memory cap <23,552 MiB on coordinated GPU 0, wall-time cap eight
hours. No downloads, installs, other GPU or automatic retry. Runtime caches stay
inside the new result directory. Existing files and failed evidence are immutable.

Verify each record's round-trip immediately within the worker; do not expose
its labels, action differences or losses to the assistant while active. Monitor
only owned health, count, bytes, elapsed time and aggregate selected-GPU telemetry.
Unblind only after completed immutable summary. CPU analysis independently checks
all identities, roles, counts, hashes and numeric record contracts. No efficacy
claim follows from a complete dataset. Sync status/manifests and verified evidence
locally; do not push generated features or model weights to GitHub.

After collection, freeze the matched fitting configuration (including optimizer,
steps, shuffle order, shuffled-label diagnostic and final checkpoint rule) before
starting either research adapter. No best-checkpoint selection using rollouts.
The subsequent robot evaluation must test incremental benefit over compression
alone, not relabel the 384-token screen's existing gain as adapter benefit.
