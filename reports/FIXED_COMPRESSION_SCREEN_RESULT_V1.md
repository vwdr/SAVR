# Fixed current-frame compression: completed development screen

Run completed 2026-09-13 at 22:34:20 UTC. Reviewed 2026-09-14 UTC.
Evidence: `results/fixed-compression-screen-v01`.

## Result and frozen decision

The screen completed without a technical stop. The 384-token setting matched
the dense baseline's success on all 40 paired conditions (39 successes and one
shared failure), with 15.32% lower mean complete-query time in the controlled
timing module. The 256-token setting succeeded on 37/40 with 32.50% lower mean
query time. Both settings passed the predeclared substrate criteria. The frozen
preference therefore selects **384 tokens**, without any post-result gate change.

This is a promising measured compression tradeoff on consumed development tasks.
It is not confirmatory evidence of preserved success probability, a novelty claim
for spatial deletion, or evidence that a learned corrector improves the policy.
No adapter was trained or tested. Stop before data generation and training.

## Design and primary results

Both camera views were freshly encoded at every query. The fixed stratified
selector retained 512, 384 or 256 visual tokens before decoder layer zero, with
all nonvisual positions, original rotary IDs, bidirectional attention and released
action readout retained. There was no stale cache, routing or replanning.

The same 40 consumed task/state-0/seed-7 conditions received all three arms, in
the frozen rotating order. Eight separate native controls bracketed the four
suites. These controls are not pooled into the primary success denominator.

| Suite | 512 tokens | 384 tokens | 256 tokens |
|---|---:|---:|---:|
| Spatial | 10/10 | 10/10 | 10/10 |
| Object | 10/10 | 10/10 | 9/10 |
| Goal | 9/10 | 9/10 | 8/10 |
| Long (LIBERO-10) | 10/10 | 10/10 | 10/10 |
| Total | 39/40 | 39/40 | 37/40 |

Every task failure is retained:

| Task | 512 | 384 | 256 |
|---|---|---|---|
| Object: pick up the BBQ sauce and place it in the basket | Success | Success | Failure |
| Goal: open the top drawer and put the bowl inside | Failure | Failure | Failure |
| Goal: push the plate to the front of the stove | Success | Success | Failure |

All other 37 conditions succeeded in every arm. Full canonical task identities,
all 40 paired rows, query counts and action/episode costs are preserved in the
frozen configuration and `analysis.json`/`records.json`.

For 384 there were zero discordant paired successes. For 256 there were two
dense-only successes and no compressed-only successes. Success differences were
0 and -5 percentage points respectively. The prespecified descriptive bootstrap
intervals were [0, 0] and [-12.5, 0] percentage points. The zero-width 384 interval
is a limitation of resampling this sample with no observed discordances; it is
NOT proof of zero uncertainty or equivalence on new episodes. There were 10,000
task-level draws stratified by suite, seed 7. No significance/noninferiority claim.

## Timing and complete episode costs

The controlled module used eight consumed training trajectories, two frames and
four measured rounds. Twelve warm-up calls were excluded by their frozen labels;
192 calls were measured (64 per arm). All query preprocessing, transfers,
selection/mapping, decoder/head work, validation and float32 conversion remained
inside the synchronized timing boundary. No corrector overhead was included.

| Controlled timing (ms), mean / median / p95 | 512 | 384 | 256 |
|---|---:|---:|---:|
| First query, 32 per arm | 1198.62 / 1198.18 / 1201.98 | 1015.21 / 1014.30 / 1019.18 | 809.61 / 809.38 / 813.04 |
| Second query, 32 per arm | 1198.81 / 1198.26 / 1202.48 | 1014.85 / 1014.35 / 1018.48 | 808.73 / 808.17 / 811.89 |
| Combined, 64 per arm | 1198.72 / 1198.21 / 1202.28 | 1015.03 / 1014.35 / 1018.75 | 809.17 / 808.91 / 812.63 |

Mean time reductions: 384 = 15.3237%; 256 = 32.4968%. The descriptive paired
trajectory-bootstrap intervals were [15.3030%, 15.3472%] and [32.4769%, 32.5169%].
All frames/rounds in each trajectory were resampled together. These tight
intervals concern eight fixed short traces, not uncertainty over unseen tasks,
hardware conditions or general deployment behavior.

| Primary episode measurement, all failures included | 512 | 384 | 256 |
|---|---:|---:|---:|
| Mean episode wall time (s) | 29.90 | 26.57 | 23.98 |
| Median episode wall time (s) | 25.14 | 22.28 | 19.60 |
| p95 episode wall time (s) | 57.96 | 46.03 | 42.44 |
| Mean policy queries per episode | 19.05 | 19.225 | 20.575 |
| Mean executed actions | 149.20 | 150.025 | 160.40 |
| Total primary policy queries | 762 | 769 | 823 |
| All-query mean / median / p95 (ms) | 1206.56 / 1203.97 / 1235.80 | 1022.19 / 1019.80 / 1047.19 | 814.89 / 812.63 / 815.88 |

No primary arm replanned or discarded queued actions. Lower episode wall times
are descriptive because the resulting trajectories and, for 256, successes
differ. Do not equate short-trace query savings with matched-quality deployment
acceleration without broader evaluation.

## Native controls and technical reconciliation

All eight native controls and all four corresponding primary dense sentinels
succeeded. No success-repeatability flag or dense-baseline concern triggered.
Historical and contemporaneous dense totals are both 39/40.

| Suite | Native before / after steps | Before / after queries | Observation-command traces equal |
|---|---:|---:|---|
| Spatial | 79 / 78 | 10 / 10 | No |
| Object | 128 / 128 | 16 / 16 | Yes |
| Goal | 118 / 118 | 15 / 15 | Yes |
| Long | 247 / 247 | 31 / 31 | Yes |

All 144 same-input native-versus-512 shadows had zero head/normalized/raw-action
discrepancy and byte-identical processed commands. Both paths executed 32 layers.
This establishes conditional agreement on those observations, not exact independent
rollout reproducibility. Two native repeats per suite are limited diagnostics.

Every frozen technical condition reconciled:

- 128 episode records: 120 primary and eight controls, in the exact frozen order.
- 204 timing records: 12 warm-up and 192 measured, with exact frame/arm identities.
- 2,846 model calls = 204 timing + 762 dense + 769 at 384 + 823 at 256 + 2 x 144
  native/shadow calls. This is below the 7,000-call cap.
- Exact horizons, query/action queue accounting, no replanning/discards, finite
  outputs/costs, retained budgets and complete native parity records verified.
- 4,222.59 seconds (70.38 minutes) below six hours. Peak aggregate GPU-0 memory
  16,504 MiB below 23,552 MiB. Pre-analysis artifacts 2,184,255 bytes below 512 MiB.
- Owned PID 1333597 exited; GPU 0 was subsequently idle at 6 MiB / 0% utilization.
- Completed summary and all artifact/source hashes verified. Worker reports
  unchanged checkpoint and authenticated files. Independent local reconciliation
  passed; episode/timing/parity streams equal their bundled records.
- No task exclusion, technical retry, source/config/gate changes, training or
  GitHub push. Existing historical evidence remains unchanged.

## Selection and implications for the next phase

Both budgets meet the >=10% controlled mean time-saving and <=6/40 success-loss
rules. No baseline/control concern blocks selection. The predeclared gentler-
budget preference selects 384. The alternate 256 result is retained, not hidden.

At 384, this development population has no observed compression-induced terminal
failures to recover. That creates a ceiling for demonstrating the corrector's
incremental benefit. The shared dense failure cannot automatically be labeled a
compression error. Do not treat eligibility as a reason to claim recovery, train
without a frozen evaluation question, or silently switch to 256 after the result.
Any later change of primary budget/evaluation must be a prospective, justified
protocol revision with the current selection and consumed exposure preserved.

The measured 384 mean saving is approximately 183.69 ms/query before correction.
Corrector work must fit within that saving for any net speed benefit on these
traces. If a later protocol requires retaining 10% total query saving, only about
63.82 ms/query would remain for added work at these means. These are arithmetic
planning margins, not measurements of adapter runtime or guaranteed headroom.

Next boundary: separately plan/freeze the actual-data feature extraction and
learning pilot, including how incremental benefit will be evaluated despite this
ceiling. Stop before data generation or training in the present phase.

## Provenance

Protocol: `docs/CURRENT_FRAME_COMPRESSION_SCREEN_V1.md`.
Configuration: `configs/openvla/fixed_compression_screen_v1.json`.
Runtime/checkpoint/input ancestry are authenticated by the completed 112-call
qualification and original four-suite OpenVLA-OFT reference configurations.

- Config SHA-256: `cc0100823806caac8770f4bb533dda181985b6224a3275997af0793b19590ac3`.
- Worker SHA-256: `6d7a77e9a097f5d7ade95b568f100ae6a8eb130c4cb57710663ef737a416cf72`.
- Summary SHA-256: `56bfa17c0d7779275f35ad651a2bebd4e3b72ca9b40be3c255c616f92af977a3`.
- Analysis SHA-256: `7baad5f415951301ae73353f6efdf4515a4431596f5e7d40d47fddb233d176b4`.

CPU analysis already executed once, with exclusive output creation:

```sh
CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 envs/openvla-oft/bin/python scripts/analyze_fixed_compression_screen.py --run results/fixed-compression-screen-v01
```

Evidence and status synchronized to the local SAVR project. All server file work
used ssh titan within /home/ved/SAVR. No unrelated university files, processes,
allocations or configuration were inspected or changed. Monitoring is retired
after this report; no new GPU workload is launched.
