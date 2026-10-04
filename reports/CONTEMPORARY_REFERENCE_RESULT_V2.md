# Contemporaneous dense and SpecPrune reference: completed development evaluation

Date: 2026-09-13. Run: `results/contemporary-reference-v02`.

## Finding and decision

The frozen evaluation completed without a technical stop. Dense inference
succeeded on 39/40 conditions, matching the historical aggregate. The pinned
same-checkpoint SpecPrune adaptation succeeded on 32/40. Its controlled mean
query time was 53.80% lower, but its success count was lower by seven tasks.
This establishes a working current-frame compression comparator with a measured
success–cost tradeoff. It does not establish a reliability-preserving improvement,
the efficacy of the proposed learned corrector, or an exact reproduction of the
SpecPrune publication.

Stop before compression screening and training. The planned 384/256-current-token
screen remains the next separately frozen step. The comparator's 17.5-percentage-
point success loss exceeds the plan's approximately 15-point triage target; do
not automatically adopt this setting as a suitable corrector training substrate.
The fixed-token candidates have not been tested here. Neither their success nor
the ability of a learned corrector to recover task success is established.

## Population and primary outcomes

Forty previously consumed development conditions, one initial state per task,
seed 7, one released OpenVLA-OFT four-suite checkpoint. Each condition received
one dense and one SpecPrune episode in the frozen balanced order. Eight native
episodes were separate controls, not extra primary tasks. All failures retained.

| Suite | Dense success | SpecPrune success | Success-repeatability flag |
|---|---:|---:|---|
| Spatial | 10/10 | 7/10 | None |
| Object | 10/10 | 10/10 | None |
| Goal | 9/10 | 7/10 | None |
| Long (LIBERO-10) | 10/10 | 8/10 | None |
| Total | 39/40 | 32/40 | None |

There were eight dense-only successes, one SpecPrune-only success, and 31 paired
successes. The success difference was -17.5 percentage points. The prespecified
descriptive task bootstrap interval was [-30, -5] percentage points (10,000 draws,
seed 7, stratified by suite). This development interval is not a significance,
noninferiority, or unseen-task generalization claim. The identical historical
and current dense totals do not imply identical trajectories.

## Cost measurements

The matched-input timing module used eight consumed training trajectories, two
frames per trace, four measured rounds and eight separately labeled warm-up
calls. Each arm has 64 measured calls. The profile was fixed to coarse selection;
these short traces do not represent the full deployment controller mix.

| Controlled query time (ms) | Dense mean / median / p95 | SpecPrune mean / median / p95 |
|---|---:|---:|
| First query (32 per arm) | 1203.79 / 1203.13 / 1207.93 | 554.33 / 553.64 / 558.76 |
| Second query (32 per arm) | 1204.60 / 1204.23 / 1208.58 | 558.31 / 553.19 / 579.11 |
| Combined (64 per arm) | 1204.20 / 1204.12 / 1208.33 | 556.32 / 553.45 / 577.66 |

Mean controlled-query reduction: 53.80%, with a descriptive paired trajectory-
bootstrap interval of [53.48%, 54.09%]. All rounds and both frames were resampled
together within each of the eight trajectory clusters. This narrow interval
does not quantify uncertainty over other tasks, hardware, or deployment states.

Primary episode measurements include failures, validation, controller costs and
all replanning. Native shadow controls are excluded from primary timing by design.

| Episode measurement | Dense | SpecPrune |
|---|---:|---:|
| Mean episode wall time (s) | 30.21 | 23.63 |
| Median episode wall time (s) | 25.37 | 18.75 |
| p95 episode wall time (s) | 58.31 | 48.95 |
| Mean queries per episode | 19.15 | 26.625 |
| Mean executed actions per episode | 150.05 | 191.90 |
| All-query mean / median / p95 (ms) | 1213.90 / 1211.02 / 1241.80 | 583.46 / 578.80 / 643.21 |
| Total primary policy queries | 766 | 1065 |
| Total replans / discarded actions | 0 / 0 | 142 / 694 |

These episode costs occur on different resulting trajectories and success
outcomes. Lower episode wall time must not be presented as acceleration at
preserved task quality. The complete paired task records and additional cost
statistics are in `analysis.json`, including unsuccessful episodes.

## Native controls and conditional inference equivalence

All eight native controls succeeded. Each corresponding primary dense sentinel
also succeeded, so no suite triggered the frozen success-repeatability flag.
Nevertheless, independent rollout traces were not identical in two suites.

| Suite | Before / after steps | Before / after queries | Observation–command traces equal |
|---|---:|---:|---|
| Spatial | 78 / 80 | 10 / 10 | No |
| Object | 128 / 128 | 16 / 16 | Yes |
| Goal | 117 / 118 | 15 / 15 | No |
| Long | 247 / 247 | 31 / 31 | Yes |

On all 144 native queries, the shadow dense bridge received the same observation.
All head, normalized-action and raw-action maximum discrepancies were zero;
processed float32 command chunks were byte-identical, with 32 layers observed
on both paths. This supports conditional equivalence on the tested inputs.
It does not establish exact independent-rollout reproducibility or identify
the cause of the remaining rollout variation. Two controls per suite cannot
precisely estimate task-success variance.

## Frozen technical reconciliation

- Completed immutable summary at 2026-09-13T15:44:16.514248+00:00; no technical stop.
- Exactly 88 ordered episodes: 40 dense, 40 SpecPrune and eight native controls.
- Exactly 136 ordered timing records: eight warm-ups and 128 measured calls.
- Exactly 2,255 model calls = 136 timing + 766 dense + 1,065 SpecPrune + 2 x 144
  native/shadow calls, below the 6,000-call cap.
- All episode identities, paired order, horizons, controller assignments,
  generated/executed/discarded action accounting and finite cost checks passed.
- All native parity records, fixed timing identities and counts passed.
- Worker elapsed time 2,973.42 seconds (49.56 minutes), below six hours.
- Peak aggregate GPU-0 memory 16,506 MiB, below 23,552 MiB. Owned PID 1301012
  exited; subsequent selected-GPU telemetry was 6 MiB and 0% utilization.
- Before analysis, all run artifacts occupied 1,834,824 bytes, below 512 MiB.
  Frozen worker also checked the output cap before writing its summary.
- Frozen configuration, source and artifact identities verified by the CPU
  analyzer. Worker reports checkpoint and authenticated inputs unchanged.
- Original runtime: Torch 2.2.0+cu118, Transformers 4.40.1, 32 SDPA layers.
- CPU-only analysis passed with CUDA hidden. Independent local reconciliation
  also passed; JSONL episode/timing/parity records equal their bundled records.
- No training, automatic retry, gate changes, excluded failures or GitHub push.

The evaluation summary's `hook_neutrality_passed: false` is an existing mode
marker: the worker sets it to `mode == hook_qualification`. It is not a failed
comparison in this evaluation. The launch instead authenticates the separate
completed recovery qualification with 24/24 checks and 80/80 calls. The frozen
sources and summaries are preserved without rewriting this potentially confusing
field. The current evaluation independently passed all 144 native shadow checks.

## Provenance and reproduction

Protocol: `docs/CONTEMPORANEOUS_REFERENCE_EVALUATION_PROTOCOL_V1.md`.
Launch: `docs/CONTEMPORARY_REFERENCE_LAUNCH_V2.md`.
Result bundle and dedicated terminal log are copied to the local SAVR project.

Full SHA-256 identifiers:

- Executable config: `afa1bc948880b072b74c9f150e7dbab77fe90468040746c73b82af603f91ec67`.
- Worker: `99e784f05d71820c285c4bb0481baa88b83aca61b30c57841dc871ca70f95d94`.
- Analyzer: `a7c85a58c52bcc0b508746541e501a5a99c9868c284cf7b2c99379b0c986aa2a`.
- Completed summary: `6c8fc825f55ba2154a48df3ae3171f674e47ebd363e620efa5f929082148326b`.
- Analysis: `1d70a8ba05e53a5ed45641873e6f4c88efa8ee1230e63815d2a2f6224bd53385`.

Analysis command (already executed once; output is exclusive-create):

```sh
CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 envs/openvla-oft/bin/python scripts/analyze_contemporary_reference_v2.py --run results/contemporary-reference-v02
```

All server operations used `ssh titan` within `/home/ved/SAVR`, apart from the
explicitly authorized owned-process health and aggregate selected-GPU telemetry.
No unrelated server files, processes, allocations or configuration were changed.
