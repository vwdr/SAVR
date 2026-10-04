# Four-arm development comparison v03 — completed 2026-09-26

## Summary

The implementation qualification and robot comparison completed without a
technical stop. Fixed compression and adaptive 50%-retention compression each
matched dense inference at 39/40 successes. Both reduced controlled mean query
time by about 15.5%. Fixed compression did not demonstrate superiority over the
stronger adaptive arm. This is a positive development tradeoff, not a confirmed
novel method or a paper-ready result.

## Verified completion and provenance

- Run: `results/adaptive-screen-v03`; owned PID 3074656 has exited.
- Completed UTC: 2026-09-26 19:34:20.767302. Worker elapsed 5688.629 seconds
  (94.81 minutes), within the six-hour cap.
- 168 episodes: four primary arms with 40 matched conditions each, plus eight
  native controls. Exactly 240 timing records, including 16 excluded hardware
  warmup calls, and 288 shadow-parity records.
- 3728 model calls, below 7000. Peak aggregate GPU0 memory 16,506 MiB, below
  23,552 MiB. Remote result-directory size after analysis 3,487,948 bytes by
  `du -sb`, below 512 MiB. GPU UUID is unchanged from the frozen configuration.
- All eight immutable artifact hashes and nine frozen source hashes verified.
  Stream records exactly match the bundled records and the frozen schedules.
- Frozen CPU analyzer passed with CUDA hidden and one BLAS thread. Separate
  standard-library verifier `scripts/verify_adaptive_screen_v03_independent.py`
  imports neither worker nor analyzer. It independently checks record order,
  counts, action queues, actual token budgets, call accounting, timing regimes,
  parity, caps, source hashes, per-arm/per-suite outcomes, controls and triage;
  its computed values agree with the frozen analyzer.
- Configuration SHA256:
  `ae1cbd7d0dbaccb3ff14a335cae6b451baa2b6ea62f25968e52c5ea5e3c8c768`.
- Immutable summary SHA256:
  `e60854bb0b443e9a093c8b58668fc95788ec9a201398d60e1e35ff3c492ee72a`.
- Analysis SHA256:
  `0687bb3ef3efd868226c0b61b40be81f7694e37b87e6bf772e85710cc8ee9b58`.

## All four arms

| Arm | Success | Controlled mean query time, ms | Reduction vs dense | Startup, ms | Steady, ms | Mean episode wall time, s |
|---|---:|---:|---:|---:|---:|---:|
| Dense, 512 tokens | 39/40 | 1203.65 | — | Not separate | Not separate | 29.89 |
| Fixed, 384 tokens | 39/40 | 1018.71 | 15.36% | No dense startup | 1018.71 overall | 26.66 |
| Adaptive, 384 tokens after startup | 38/40 | 1122.29 | 6.76% | 1204.67 | 1060.51 | 27.84 |
| Adaptive, 256 tokens after startup | 39/40 | 1016.12 | 15.58% | 1205.25 | 874.28 | 24.32 |

Adaptive startup is three dense queries per episode and is included in the
primary all-query timing. The controlled test uses seven-query traces from two
saved frames per trajectory: 24 startup and 32 steady queries per adaptive arm.
It is not a natural-trajectory latency estimate. Mean episode wall time is
reported separately and includes trajectory length and simulator costs, so it
must not be interpreted as a controlled inference-speed comparison.

Per-suite success counts (spatial / object / goal / long-horizon, ten each):
dense 10/10/9/10; fixed 10/10/9/10; adaptive384 9/10/9/10;
adaptive256 10/10/9/10. Fixed and adaptive256 share the same success/failure
pattern as dense. Adaptive384 has one additional failure.

## Frozen gates and controls

- Dense minimum 39/40: passed exactly. All eight native controls succeeded.
- Native-before/after outcome repeatability: passed for all four suites.
  Observation/command traces were identical in three suites but differed in
  LIBERO-Goal. Report this rollout variation; do not assert full determinism.
- All 288 shadow rows had exact executed-command equality and zero recorded
  hidden, normalized-action and raw-action error, including 144 adaptive-zero
  comparisons. The original 1e-6 bound was not loosened.
- Maximum six-success loss: all three compressed arms pass the development
  threshold. Minimum 10% all-query time reduction: fixed384 and adaptive256
  pass; adaptive384 fails. No automatic arm selection was made.
- No missing or converted error episodes, no training, no retry, no changed
  scientific gates, no modified frozen evidence or configuration.

## Uncertainty and claim limits

The analyzer's paired descriptive bootstrap intervals for success differences
are [0,0] for fixed384 and adaptive256 and [-7.5,0] percentage points for
adaptive384. The degenerate [0,0] intervals arise from zero observed discordant
pairs in only 40 development conditions. They do NOT prove population equality
or noninferiority. Do not present them as confirmatory confidence statements.
Timing intervals describe only eight controlled traces, not hardware or task
generalization. The conditions were previously used in development; one seed
and one GPU/checkpoint do not establish an independent result.

The adaptive implementation is the documented native-SDPA local selection
adaptation, not an exact upstream VLA-Pruner numerical or timing reproduction.
Static subsampling itself is not established as novel by these experiments.

## Confirmation decision

The screen supports considering ONE independent confirmation of the speed–success
tradeoff. It does not support launching a study premised on fixed compression
beating adaptive compression: adaptive256 matched its success, was slightly
faster overall, and was clearly faster in steady-state queries. At matched
384-token retention fixed compression was faster and had one more success,
but the stronger 256-token comparator must not be omitted from the conclusion.

Before any confirmation launch, prospectively freeze the claim, exposure-audited
evaluation population, sample size justified by desired precision, seeds, resource
cap, stopping rules and paired analysis. Dense, fixed384 and adaptive256 are
necessary comparators for a broad tradeoff claim; retain adaptive384 if making
a matched-retention claim. Include natural-trajectory timing rather than selecting
only a favorable startup-heavy microbenchmark. Do not tune the method on this
confirmation population or repeatedly expand evaluation until it is positive.

A confirmed compression benefit alone would not establish a novel method
contribution. The next planning decision must state what new scientific claim
this confirmation would test, rather than promise publication from replication.
No confirmation was launched in this monitoring turn. No GitHub push, manuscript,
advisor email or training was performed. Evidence and status are synced locally;
the completed-run heartbeat has been deleted. Remote writes remain within
`/home/ved/SAVR`; no unrelated university resource was modified.
