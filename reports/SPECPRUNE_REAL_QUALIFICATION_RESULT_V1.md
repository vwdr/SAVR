# SpecPrune same-checkpoint real-model qualification: PASS

Date: 2026-09-11. Completed at 11:12:55 UTC on TITAN GPU 0.

## Result

All 40 scheduled model calls completed on the eight frozen, previously consumed
training observations. All numerical, token-accounting, source-integrity and
resource gates passed. No technical stop, automatic retry, training or simulator
episode occurred.

| Comparison with native evaluator | Maximum action-head input error | Maximum normalized-action error | Maximum unnormalized-action error |
|---|---:|---:|---:|
| Pruning disabled | 0 | 0 | 0 |
| All tokens retained, attention scores computed | 0 | 0 | 0 |
| Native evaluator restored after compression | 0 | 0 | 0 |

The frozen tolerance was 1e-6. The observed errors were exactly zero across all
eight observations. This verifies these adaptation paths on the actual 7B
checkpoint, beyond the earlier tiny-model CPU tests.

Every compressed query completed all 32 original bidirectional SDPA layers,
retained the required nonvisual and action-readout positions, and produced
finite 8x7 policy actions. The final layer retained **59 of 512 visual tokens**
on each observation. This is a final-layer token count, not a whole-query compute
reduction, temporal refresh-skipping percentage, or measured speedup. Earlier
layers execute with more tokens, and both camera images are freshly encoded.

The source's minimum-retention expression is `60 + prompt_tokens + 58`. Our
authenticated sequence has `prompt_tokens + 59` nonvisual tokens, leaving 59
visual tokens at that floor. This matches the pinned implementation's arithmetic;
we did not silently change it to fit the constant's “60” name. No claim of a
60-token observed floor is made.

## Integrity and resources

- Eight observations, five ordered modes each, exactly 40 model calls.
- Native → disabled → keep-all → compressed → restored for every observation.
- Exactly 32 layer records per adapted call; contiguous, nonincreasing sequence
  counts; correct final absolute action positions; independent state resets.
- All 46 authenticated source/checkpoint/input identities and eight input
  sources passed preflight and post-run verification. Checkpoint inventory unchanged.
- Peak aggregate selected-GPU memory: **15,679 MiB**, below the 23,552 MiB cap.
- Worker elapsed time: **107.50 seconds**, including setup and final checks.
  This is not a benchmark latency measurement.
- Owned PID `1038935` exited; GPU 0 returned to 6 MiB and 0% utilization.
- Completed summary and artifact hashes were checked independently on TITAN
  and locally. Five non-cache evidence files were copied to the local repository.
- No unrelated server files, environments, processes, services or GPU allocations
  were changed. No checkpoint/runtime edit, installation, or GitHub push occurred.

Config: `configs/openvla/specprune_real_qualification_v1.json`.
SHA-256: `8adc9dde86d08f8d717c96e4017250229c71d7f5062ebe697cbb4c53eb75e168`.
Worker: `scripts/run_specprune_qualification.py`.
SHA-256: `1dad144a4a5e88159ba0d7cf726fa464cee3da44dfbf3ddab2f9d1200816b2f3`.
Evidence: `results/specprune-real-qualification-v01/`.
Machine-readable report: `reports/SPECPRUNE_REAL_QUALIFICATION_RESULT_V1.json`.

## Interpretation and next step

The integration is qualified for the tested single-query paths. This does not
establish closed-loop success, acceleration, a successful learned correction,
or a positive-results paper. The original 39/40 dense development baseline is
unchanged; it was not rerun here.

Next: connect the comparator to the tested simulator loop, validate controller
and rolling two-camera frame history, and freeze a matched development evaluation
with a fair timing control. Do not copy the upstream evaluator's different
horizons or execution behavior. The timed dense control must share the decoder-
only wrapper optimizations so their savings are not attributed to pruning.
No simulator benchmark or adapter training was launched after this checkpoint.
