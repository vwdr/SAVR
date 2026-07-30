# Phase 5 Smoke and External-Baseline Protocol

Status: **APPROVED FOR EXECUTION**

Approved by the user on 2026-07-29 for uninterrupted execution through the
end-of-Phase-5 checkpoint. Safety, integrity, and scientific stop rules remain
in force.

## 1. Purpose and claim boundary

Phase 5 answers two narrow feasibility questions:

1. Can FR, PR, VOR, and SAVR complete real LIBERO trajectories through the
   common project-owned interface while producing internally consistent
   refresh and timing records?
2. Can the official VLA-Cache implementation be validly pinned and exercised
   against the same OpenVLA-OFT checkpoint and task stack?

This phase is a structural smoke test. Its three initial states are not enough
for method selection, non-inferiority testing, latency claims, threshold
calibration, or paper-level success comparisons.

## 2. Frozen core-policy matrix

- Suite: `libero_spatial`
- Task ID: `0`
- Initial-state IDs: `0`, `1`, `2`
- Seed: `0`
- Policies: `FR`, `PR`, `VOR`, `SAVR`
- Episodes: exactly `12`
- Checkpoint revision:
  `638918f3d1c2e43a39a8a20772bdb8b91835e4b7`
- OpenVLA-OFT revision:
  `e4287e94541f459edc4feabc4e181f537cd569a8`
- Action-chunk length/open-loop steps: `8`
- Inputs: primary image, wrist image, and current 8-D proprioception
- Model loading: once for the complete core matrix

The fixed run order counterbalances policy position across the three states:

| Order | State | Policy |
|---:|---:|---|
| 1 | 0 | FR |
| 2 | 1 | PR |
| 3 | 2 | VOR |
| 4 | 0 | SAVR |
| 5 | 1 | FR |
| 6 | 2 | PR |
| 7 | 0 | VOR |
| 8 | 1 | SAVR |
| 9 | 2 | FR |
| 10 | 0 | PR |
| 11 | 1 | VOR |
| 12 | 2 | SAVR |

Each episode receives a new simulator reset, cache context, controller, action
queue, and record namespace. No cached tensor or controller history may cross
an episode boundary.

## 3. Diagnostic policy configurations

These settings are selected to exercise refresh and reuse paths predictably;
they are not learned, calibrated, or candidates for the final evaluation.

- FR: refresh every query.
- PR: period `2`.
- VOR: image threshold `1,000,000`; maximum reuse horizon `2`.
- SAVR: image, state, and action thresholds `1,000,000`; maximum reuse
  horizon `2`.

The deliberately unreachable finite signal thresholds make VOR reuse governed
by the horizon and SAVR reuse governed by conservative history warm-up plus
the horizon. Invalid inputs still fail closed to refresh. Phase 6 must replace
these diagnostic settings using calibration data disjoint from final
evaluation.

## 4. Required records and invariants

For every policy query, preserve:

- episode, policy, state, query, and environment-step identifiers
- action shape and SHA-256 digest
- refresh/reuse event, trigger set, signal scores, thresholds, and cache ages
- synchronized total and component CUDA timing
- wall timing and component invocation counts
- fresh state input and finite-action validation

For every episode, preserve:

- terminal status and success
- control-step and policy-query counts
- refresh and reuse counts/rate
- trigger totals
- trajectory SHA-256 over executed actions and resulting robot state
- latency summaries and peak GPU memory

Required invariants:

- each terminal episode has at least one query
- `refresh_count + reuse_count == query_count`
- FR refreshes every query
- PR follows its period after the forced empty-cache refresh
- VOR and SAVR never exceed cache age `2`
- a reuse query executes zero vision-backbone and zero visual-projector calls
- a refresh query executes each visual component exactly once
- every query executes the language model and action head exactly once
- controller/query indices and immutable record counts are contiguous
- all actions and recorded numeric values are finite
- all 12 planned policy/state pairs reach a terminal record
- checkpoint protected files and inventory are restored exactly

The three-state outcomes are reported descriptively. A success difference
does not pass or fail a policy in this phase unless it accompanies a runtime,
instrumentation, or deterministic-protocol discrepancy.

## 5. Resource and safety bounds

- Server writes: only `/home/ved/SAVR`
- GPU: one qualifying GPU selected from three aggregate-only idle samples
- No process identities, termination, reprioritization, or allocation changes
- Core run wall-clock cap: `2 hours`
- Core artifact cap: `1 GiB`
- External-baseline compatibility work wall-clock cap: `2 hours`
- External-baseline added storage cap: `8 GiB`
- No new model checkpoint or dataset download
- No system-wide installation, `sudo`, or modification of the validated
  `envs/openvla-oft` environment
- No manuscript edit and no Phase 6 calibration

Stop the affected run on a non-finite action, record/schema failure,
unexplained component-count mismatch, checkpoint-restoration failure, artifact
cap, time cap, loss of exclusive visibility of the selected GPU, or any action
that would leave the project boundary.

## 6. Official VLA-Cache compatibility audit

Pin:

- official repository: `https://github.com/siyuhsu/vla-cache`
- repository commit:
  `a4909880573868dee2769343d52e793c0341678b`
- required Transformers fork commit:
  `9a90a37acacf453433168db8d7769b7ea3c40c06`

The official code uses token-level language-model KV reuse, not SAVR's
projected visual-feature reuse. It therefore requires a separate compatibility
path and must not be installed over the validated SAVR environment.

Audit sequence:

1. Pin the official source inside `third_party/vla-cache`.
2. Compare its OpenVLA-OFT and Transformers interfaces with the pinned core
   stack.
3. Verify imports in an isolated project-local environment or environment
   clone, within the storage cap.
4. Reuse the existing combined checkpoint; do not download a second model.
5. Run at most one task-0/state-0/seed-0 episode only if the official path has
   valid previous-frame semantics, checkpoint compatibility, and complete
   terminal/error reporting.
6. Otherwise produce a documented technical exclusion with exact source,
   interface, and runtime evidence.

The official evaluation source currently assigns `prev_images` from the
current frame after appending that frame to its replay list. This appears to
make the current and previous images identical. The audit must verify this
behavior. Do not silently patch it or claim an official comparison from a
semantically invalid run. Any proposed correction must be isolated, minimal,
clearly labeled, and deferred for review if it would change the published
baseline.

## 7. Phase exit gate

Phase 5 may close only when:

- the 12 core episodes have terminal records and all invariants reconcile;
- any policy-specific instrumentation difference is explained;
- VLA-Cache is either compatibly exercised or excluded with reproducible
  technical evidence;
- tests, lint, schema validation, upstream cleanliness, checkpoint integrity,
  and artifact/resource audits pass;
- a Phase 5 report and checkpoint pull request are published; and
- GitHub, TITAN, and the local review folder are synchronized after acceptance.

Execution stops at the Phase 5 checkpoint for user review. Phase 6 remains
unauthorized.
