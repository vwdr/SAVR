# Results Logging Contract

Every experimental run must emit:

1. one run manifest conforming to `schemas/run_manifest.schema.json`
2. one episode record per episode conforming to `schemas/episode_result.schema.json`
3. raw per-step timing and refresh-decision records in a documented tabular format
4. stdout/stderr logs with secrets removed

## Required provenance

- UTC timestamps
- SAVR git revision and clean/dirty state
- upstream repository revisions
- model/checkpoint identifier and immutable revision when available
- benchmark suite, task, initial-state ID, and seed
- policy and every threshold/horizon parameter
- hardware identity and software versions
- exact launch command or structured equivalent

## Timing

Record separately:

- environment step time
- preprocessing time
- refresh decision time
- visual encoding time
- downstream policy/action decoding time
- total policy time
- total control-step time

GPU timings must be synchronized. Model loading and warm-up must not be mixed into steady-state latency.

## Refresh records

Each step should include:

- whether a refresh occurred
- cache age before the decision
- image/state/action change scores
- threshold values
- trigger flags
- maximum-horizon trigger

## Aggregation

Aggregate only from preserved raw records. Report policy-level and task-level distributions, paired comparisons with FR, and uncertainty. Failed or interrupted episodes must remain visible and carry an explicit status/reason.
