# Current-frame correction: completed robot-development pilot

Completed 2026-09-20 04:38:29 UTC. Verified and analyzed 2026-09-20.

## Decision

**The frozen positive-development criteria were not met.** This was a complete,
technically valid evaluation, not an integration failure. The visual adapter did
not improve aggregate success over compression alone and did not beat the
action-only adapter. Its offline error improvement did not establish a net
closed-loop success improvement in this pilot. No publication-ready result is claimed.

## Primary results

All four policies were evaluated on the same 120 development starting conditions:
40 tasks, three states per task (1/2/3), seed 7. Eight separate native controls
bring the total to 488 episodes. The prior state-0 screen is not pooled here.

| Policy | Successes | Success rate | Controlled mean query time | Reduction versus dense |
|---|---:|---:|---:|---:|
| Dense, 512 tokens | 120/120 | 100.00% | 1,201.22 ms | — |
| Compression alone, 384 tokens | 117/120 | 97.50% | 1,017.60 ms | 15.29% |
| Compression + action-only adapter | 118/120 | 98.33% | 1,018.95 ms | 15.17% |
| Compression + visual adapter | 117/120 | 97.50% | 1,019.80 ms | 15.10% |

Each controlled timing estimate uses 64 measured queries over eight two-frame
trajectories and four rounds per policy. All adapter feature extraction, checks,
copies and output conversion are included. Simulator observation resizing is not
included in this controlled block; these are not whole-task completion times.
Visual correction costs approximately 2.20 ms (0.216%) more than plain compression.
The roughly 15% dense-relative acceleration comes from compression, not the adapter.

## Per-suite results

| Suite (30 conditions each) | Dense | Compression | Action-only | Visual |
|---|---:|---:|---:|---:|
| Spatial | 30 | 30 | 30 | 30 |
| Object | 30 | 30 | 30 | 29 |
| Goal | 30 | 30 | 30 | 29 |
| Long-horizon (LIBERO-10) | 30 | 27 | 28 | 29 |

Visual correction succeeded on two conditions where compression failed, but
failed on two conditions where compression succeeded. Net improvement: zero.
Its long-horizon count was better, but losses elsewhere canceled that advantage.
Do not select the favorable suite after observing results or call it confirmation.
Action-only had one paired gain and no paired loss versus compression; one event
does not establish a reliable learned-method advantage, particularly with rollout
variation and only one training/evaluation seed.

## Every frozen scientific criterion

| Criterion | Observed | Decision |
|---|---|---|
| Dense >=108/120; native controls 8/8 | 120/120; 8/8 | Pass |
| Visual >=3 net successes above compression | 0 | Fail |
| Visual >=1 net success above action-only | -1 | Fail |
| Visual loses <=2 successes versus dense | Loses 3 | Fail |
| No suite loses >2 versus compression | Maximum loss 1 | Pass |
| Visual controlled mean query reduction >=10% versus dense | 15.103% | Pass |

All six were required. Three passed and three failed. No gate or checkpoint was
changed. Compression itself is near ceiling at 117/120, so the prespecified
three-success gain required recovering all three net failures. This limits the
pilot's ability to demonstrate modest improvements, but does not justify relaxing
the criterion after seeing the result. Visual correction also introduced failures,
so lack of headroom alone does not explain its zero net improvement.

## Uncertainty and reproducibility limitations

Frozen descriptive paired 95% bootstrap intervals use 10,000 draws, seed 7.
Success resampling uses task clusters stratified by suite, keeping three states
together. Controlled timing resamples paired trajectory clusters.

| Comparison | Paired gains / losses | Net success difference | Descriptive 95% interval |
|---|---:|---:|---:|
| Compression minus dense | 0 / 3 | -2.50 pp | [-5.00, 0.00] pp |
| Action-only minus dense | 0 / 2 | -1.67 pp | [-4.17, 0.00] pp |
| Visual minus dense | 0 / 3 | -2.50 pp | [-5.83, 0.00] pp |
| Visual minus compression | 2 / 2 | 0.00 pp | [-3.33, 2.50] pp |
| Visual minus action-only | 2 / 3 | -0.83 pp | [-4.17, 2.50] pp |
| Action-only minus compression | 1 / 0 | 0.83 pp | [0.00, 2.50] pp |

The visual/dense controlled time-reduction interval is [15.089%, 15.119%].
These narrow timing intervals characterize this small fixed-input timing block,
not variation across GPUs, sessions or all robot states. The success intervals
are not confirmatory tests or proof of noninferiority. Near-ceiling empirical
bootstrap intervals can omit unobserved failures. The states were historically
exposed development conditions, not an independent test set.

All eight native controls succeeded, and all 144 same-observation native/shadow
comparisons passed with the frozen action/readout checks. However, independent
before/after observation-and-command traces differed in Spatial, Goal and
LIBERO-10; only Object matched exactly. Matched initial conditions do not imply
deterministic future trajectories. This is particularly relevant to one-success
differences and does not invalidate same-observation deployment qualification.

## Additional measured costs (all primary episodes, including failures)

| Policy | Mean online query time | Policy queries | Mean executed steps | Mean episode wall time |
|---|---:|---:|---:|---:|
| Dense | 1,205.75 ms | 2,283 | 148.81 | 29.78 s |
| Compression | 1,021.95 ms | 2,380 | 155.29 | 27.27 s |
| Action-only | 1,023.19 ms | 2,407 | 157.08 | 27.56 s |
| Visual | 1,024.09 ms | 2,394 | 155.77 | 27.46 s |

Online query timing includes evaluator observation preparation. Episode wall
time also includes simulator and bookkeeping work. Different policies produced
different trajectories and failures; these means are descriptive and must not be
presented as a paired causal gain in successful task completion. Complete median,
p95, simulator, controller and nonquery-preparation statistics are in analysis.json.

## Technical completion and audit

- Exactly 16 integrated qualification records (96 backbone calls), 272 timing
  records (16 warmup, 256 measured), 488 episodes and 144 native shadow records.
- 10,120 total backbone calls: 96 + 272 + 9,464 primary queries + 288 native/shadow
  calls. No training, automatic retry, extra state selection or checkpoint changes.
- Elapsed 15,569.69 seconds (4 h 19 min 30 s), within the 16-hour cap.
- Peak aggregate GPU-0 memory 16,598 MiB, below 23,552 MiB. Worker exited normally.
- Recorded artifact size before summary 4,791,004 bytes, below 512 MiB. Ephemeral
  runtime-cache files may disappear on process exit; immutable evidence is
  authenticated by its nine-file hash manifest, not by cache byte equality.
- Full frozen CPU analyzer passed, including ancestor/source/data verification,
  bundle/stream consistency, all resource bounds and execution accounting.
- Complete result directory and technical terminal log copied locally. Independent
  standard-library verification (without importing worker/analyzer decision code)
  reproduced all four success counts, per-suite counts, paired gains/losses, query
  means and every frozen scientific gate, and verified all nine evidence hashes.
- No new run, training, GPU retry or GitHub push. Stop here pending direction.

## Evidence identifiers

Run: `results/current-frame-robot-pilot-v01` on TITAN and local review repository.

Config SHA-256: `9e3f6728be56821d5da73185f310f1119adda32e2a4fc6e50672cc2e71612c8e`

Summary SHA-256: `699de8203703ea508ab5668584ed2d7d52452f53af6dc20a539809a50221588e`

Analysis SHA-256: `4ef6ae0873f5c0d24b5caffcd223a9673d067f951080af4d2492da093e4fba70`

Independent local verification: `PYTHONDONTWRITEBYTECODE=1 python3 scripts/verify_current_frame_robot_copy.py`.
The verifier was added after completion for auditing only; it does not replace or
modify the frozen analyzer, gates, evidence, policies or run configuration.
