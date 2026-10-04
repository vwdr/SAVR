# Current-frame adapter fitting: completed offline pilot

Completed 2026-09-18 04:03:14.862717 UTC; verified 2026-09-18.

## Result

The run completed without a technical stop. The visual adapter met the two
predeclared offline triage conditions: lower mean action L1 than uncorrected
384-token compression and the shuffled-label visual control. Its improvement over
the action-only adapter was small. No robot episode was run and no deployment
latency was measured in this stage. This is not a positive closed-loop result.

## Validation results

800 observations from fitting-disjoint trajectories, 20 per task over 40 tasks.
These observations come from the historical training pool and form an offline
development set, not an independent final benchmark. Targets are normalized
dense-policy actions. Mean L1 averages all eight actions and seven components.
Lower L1 means closer agreement with the dense policy, not necessarily better
robot behavior. Gripper disagreement compares predicted and teacher signs.

| Arm | Mean normalized action L1 | Gripper-sign disagreement |
| --- | ---: | ---: |
| Uncorrected 384-token compression | 0.03489551 | 6.71875% |
| Action-only correction | 0.03332298 | 6.546875% |
| Current-visual correction | 0.03305615 | 6.81250% |
| Visual adapter, shuffled training labels | 0.16091179 | 13.890625% |

The visual adapter reduced mean L1 by 5.271% relative to uncorrected compression.
The action-only adapter already reduced it by 4.506%. Adding visual input improved
L1 by only 0.801% relative to action-only. Visual gripper disagreement increased
by 0.09375 percentage points relative to uncorrected compression. These mixed
diagnostics must remain visible when interpreting the result.

| Suite | Uncorrected | Action-only | Visual | Shuffled visual |
| --- | ---: | ---: | ---: | ---: |
| LIBERO-Spatial | 0.04321236 | 0.04086577 | 0.04050865 | 0.17348399 |
| LIBERO-Object | 0.02995610 | 0.02919429 | 0.02880850 | 0.17874090 |
| LIBERO-Goal | 0.03375689 | 0.03138999 | 0.03112372 | 0.14846598 |
| LIBERO-Long (libero_10) | 0.03265668 | 0.03184188 | 0.03178375 | 0.14295630 |

All suites have 200 validation observations. These are descriptive offline means;
no significance, equivalence or robustness claim is made. The within-task
shuffled-label control tests alignment, not every possible source of leakage.
The action-only ablation has fewer parameters and is not capacity-matched.

## Frozen execution and checks

- Three arms, identical fitting observations and order: 3,200 fitting records,
  batch 16, ten epochs, exactly 2,000 AdamW updates each (6,000 total).
- Seed 7, learning rate 1e-4, zero weight decay, clip norm 1, FP32, final-only
  checkpoints. No early stopping, validation tuning or hyperparameter search.
- 29 CPU tests and authenticated-data/real-batch preflight passed before dispatch.
- All three checkpoint hashes sealed before any validation tensor decoding.
  Exactly zero validation records entered fitting. No update followed validation.
- Shared parameter initialization was reconstructed on CPU and checked against
  the saved runtime identities; visual and shuffled visual had identical starts.
- Exact save/reload predictions passed for each arm. All 12 summary-listed
  artifact hashes passed on the server and after local copying. Local independent
  checks reconciled sample IDs, task balance, access roles, schedules, 6,000 progress
  records, resource caps and metrics recalculated from stored predictions.
- Completed analysis and summary hashes were independently authenticated locally.
  No partial losses or validation outcomes were inspected during fitting.
- Elapsed 2,356.204 seconds (39.27 minutes); peak aggregate GPU 907 MiB; peak
  own allocated memory 486.009 MiB; 62,488,536 artifact bytes before summary.
  All limits passed. No backbone calls, simulator episodes or automatic retry.

## Evidence identities and locations

Run directory: `results/current-frame-adapter-fit-v01`, present under both
`/home/ved/SAVR` on TITAN and `/Users/veddwivedi/Documents/VLA/SAVR` locally.
Technical log: `reports/current-frame-adapter-fit-v01-terminal.log`.

| Artifact | SHA-256 |
| --- | --- |
| Frozen fitting configuration | `cafa427d5e5e4a5e5a88f2f93ac85b39b49871b5a689a009f89bc2f6bfc8c031` |
| Worker summary | `c863b31bff5ea15f4c93bed21fda430734711642df56aa3e791e9a402c466627` |
| CPU analysis | `d9ed6108eef40d4c4466343e04baea42d58a88ba5e06a44368f89bfcbb64a2de` |
| action_only-final.pt | `8a01608b9690bf2e62017968fbc2ab4afd86325503ccfd3322994ab1838b8640` |
| visual-final.pt | `15489080d2752d3cf90a7b9cb016cb4c95c9a395b335b400d63e74c33f6bb951` |
| visual_shuffled-final.pt | `1524bbd2527a4ad362c1d4dff7ab33a17c2bd2cbfded8bb523c507cd75627cc9` |

## Decision and limitations

Advance only to planning a prospectively frozen closed-loop development
evaluation of dense inference, compression alone, action-only correction and
visual correction. Include all extraction, transfers and adapter overhead in
complete-query timing. Test whether any incremental success benefit survives
the extra latency and the action-only comparison. Preserve the selected 384-token
budget and final weights; do not use these results to select extra epochs or a
different checkpoint. No learner-state refinement or new training is launched.

The present result is single-seed, offline teacher imitation on demonstration
observations. Closed-loop distribution shift, teacher failures, the small visual
increment and the worse gripper diagnostic are unresolved. The prior 39/40 screen
is near ceiling and cannot alone demonstrate adapter recovery. A positive paper
would require actual closed-loop evidence and broader independent validation.

Status and evidence were synchronized. All remote operations remained inside
`/home/ved/SAVR`, through ssh titan. No unrelated workload was inspected or changed,
no GPU was allocated for analysis, and nothing was pushed to GitHub. The completed
fitting heartbeat is to be deleted after final status synchronization.
