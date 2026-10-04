# Original-runtime dense baseline: forty-task development result

Date: 2026-09-10. Status: completed and reconciled; no technical stop or retry.
Classification: baseline task-coverage check, not a new-method result.

## Main result

The original OpenVLA-OFT policy succeeded on 39 of 40 fixed development
conditions (97.5% of this particular population), with no visual caching.

| Suite | Successes | Episodes |
|---|---:|---:|
| Spatial | 10 | 10 |
| Object | 10 | 10 |
| Goal | 9 | 10 |
| Long | 10 | 10 |
| Total | 39 | 40 |

This provides broader evidence that the original runtime is a useful dense
reference. It does not establish a successful acceleration method, reproduce
the authors' full published benchmark, or explain all historical cache failures.
Only one previously consumed initial state per task and one seed were tested.

## Fixed design and relation to earlier work

All forty tasks were selected before execution, ten per suite, from previously
consumed development conditions. Initial-state ID was 0 and seed was 7. The
same original checkpoint, runtime, released preprocessing, gripper conversion,
ten settling steps, eight-action chunks and suite-specific horizons were used.
Every closed-loop query used native bidirectional Llama SDPA attention.

The 32-call offline reference/restoration check preceded simulation; its causal
calls were diagnostic only. No closed-loop causal arm, cache, compression,
adapter training, protected test data, sample extension or threshold tuning
was introduced. CPU tests and real-checkpoint parity checks were required
before simulator outcomes were evaluated.

The preceding pilot and this run share eight conditions. All eight had the
same success outcome and the same executed step count:

| Repeated task (short description) | Earlier original / current outcome | Steps in each |
|---|---|---:|
| Bowl between ramekin and plate onto plate | Success / success | 78 |
| Bowl from table center onto plate | Success / success | 94 |
| Alphabet soup into basket | Success / success | 128 |
| BBQ sauce into basket | Success / success | 121 |
| Open middle drawer | Success / success | 118 |
| Open top drawer and put bowl inside | Failure / failure | 300 |
| Turn stove on and put moka pot on it | Success / success | 247 |
| Bowl into bottom drawer and close it | Success / success | 222 |

There were zero outcome or step-count disagreements. This is repeatability of
these recorded endpoints, not proof that entire trajectories or all numerical
intermediates were bitwise identical. Matching condition identifiers, task names,
state IDs, seeds and semantic identifiers were checked. Full identifiers and
all forty outcomes appear in the accompanying JSON. Do not pool the repeated
episodes with the pilot as independent evidence.

## Complete failure inventory

There was one failure:
`libero_goal / open_the_top_drawer_and_put_the_bowl_inside`,
initial-state ID 0, seed 7, condition
`7fb637b98e47c76f8e20704f1012be67e0eeed0e4be9dcb29e2560748d2643e8`.
It reached the 300-step task horizon without terminal success.

This same condition failed in both attention arms of the preceding pilot.
The current record therefore does not attribute that failure specifically to
causal attention. Terminal counts and episode length do not identify whether
the cause was grasping, placement, task completion detection, or another issue.
No physical mechanism is asserted, and the failed condition was not excluded,
tuned or rerun. Any detailed failure investigation needs a separately scoped
diagnostic with suitable trajectory evidence.

## Integrity and resources

- Completed forty episode records and eight offline observation records,
  comprising 32 offline calls and 762 closed-loop calls: 794 total.
- Maximum original-evaluator/helper discrepancy: 2.8203848589924974e-08,
  below 1e-6. Maximum restoration discrepancy: zero.
- All forty authenticated source/checkpoint/input files were unchanged, with
  eight consumed HDF5 sources authenticated again after completion.
- The CPU-only analyzer passed on TITAN and locally with matching outcomes.
  It reconciled completion, hashes, configuration, full schedule, source/runtime
  identity, finite reference checks, attention layers, queue accounting,
  initial-state hash format and resource ceilings.
- Twelve analyzer and six selection tests passed again locally. All 29 CPU
  tests had passed on TITAN before launch.
- Elapsed worker time: 1,413.89 seconds (23.56 minutes), below two hours.
  No latency or acceleration claim is made from total worker duration.
- Peak aggregate GPU-0 memory: 16,624 MiB, below the strict 23,552-MiB ceiling.
  Post-completion GPU-0 telemetry: 6 MiB, 0% utilization; owned PID 982803 exited.
- Run directory size at completion monitoring: 1,257,359 bytes including
  disposable runtime-cache files, below the 512-MiB cap.
- Six non-cache evidence files were synced locally, totaling 177,808 bytes.
- No technical stop, automatic retry, training, dependency installation,
  model/dataset download or unrelated university-process inspection occurred.
  No work outside /home/ved/SAVR on TITAN was authorized or performed.

## Decision and next boundary

Technical execution is complete, and the original policy shows strong task
coverage on this fixed development population. This supports moving to
**planning a published acceleration comparator on the authenticated original
runtime**, with explicit correctness and complete-query timing checks before
choosing or training a new method. Preserve the one failing condition.

This is not a post-hoc certification threshold. No broad reliability estimate
or independent generalization claim follows from one state per task. No direct
comparison is made between 39/40 and the historical 68/120 as if they were matched
populations. The attention-only pilot provides separate, limited evidence about
attention effects; this run does not estimate that effect across forty tasks.

Keep all historical evidence and qualify compatibility-runtime interpretations.
Do not infer that all previous negative findings were technical artifacts or
that corrected caching will succeed. Earlier original-runtime prefix/camera
experiments remain separate evidence.

Stop after this report. No comparator, cache, compression or learned-method
experiment has been launched. No GitHub push or manuscript/poster edits.

## Reproduction and evidence

Protocol: `docs/OPENVLA_ORIGINAL_BASELINE_40TASK_PROTOCOL_V1.md`.
Configuration: `configs/openvla/original_baseline_40task_v1.json`.
Configuration SHA-256:
`90a65540dfade976ad58924b8318576b38c310ac1d8831db31ceb3d4682daa27`.

Worker summary completed at 2026-09-10T23:21:19.911532+00:00.
Worker-summary SHA-256:
`302380da00e874dbe15941ad87f5bed5dfb511b0a23e4b7e826c947979291adf`.

Run evidence: `results/openvla-original-baseline-40task-v01`, on TITAN and
under /Users/veddwivedi/Documents/VLA/SAVR. The JSON report includes all six
evidence-file hashes, all forty task outcomes, the eight-repeat comparison and
the original worker summary.

Reconcile without GPU access:

```bash
CUDA_VISIBLE_DEVICES= python3 -B scripts/analyze_openvla_original_baseline.py \
  --root results/openvla-original-baseline-40task-v01 \
  --config configs/openvla/original_baseline_40task_v1.json
```

The completed monitor is to be removed after reporting; no recurring workload
or next experiment is required by this phase.
