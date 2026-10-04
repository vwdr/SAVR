# Original-runtime attention diagnostic: completed

Date: 9 September 2026. Classification: baseline diagnostic, not a new-method result.

## Result in plain language

The released model with its original attention succeeded on **7 of 8** fixed
development conditions. The same weights and runtime with a causal-attention
control succeeded on **5 of 8**. The original succeeded on both selected
long-horizon tasks where the causal control failed. One Goal condition failed
in both modes.

This supports an attention-specific contribution to the runtime problem. It
does not show that attention explains every historical failure, that a corrected
cache works, or that a trained adapter will produce a positive paper.

## What was compared

Both arms used the same local four-suite OpenVLA-OFT checkpoint, original
Transformers 4.40.1 environment, images, preprocessing, normalization, action
head, simulator conventions, seed, and initial conditions. Only the Llama
attention mask differed. The original arm permits bidirectional attention
within a current query. The control blocks later token positions, reproducing
the causal information-flow restriction found in the compatibility runtime.

This was **not** an execution of the entire compatibility environment, and
neither arm reused a visual cache. The intervention was confined to Llama
attention, preserved padding, left vision attention unchanged, and was restored
after every query. No runtime or checkpoint file was modified.

The eight conditions were fixed before model execution: two lexicographically
selected task names per suite, initial-state ID zero, all previously consumed
in the historical headroom population. Arm order was balanced. No held-out data,
training, adaptive sample extension, partial-outcome inspection, or automatic
retry was used.

## Closed-loop outcomes

| Suite | Original attention | Causal control |
|---|---:|---:|
| Spatial | 2/2 | 2/2 |
| Object | 2/2 | 2/2 |
| Goal | 1/2 | 1/2 |
| Long | 2/2 | 0/2 |
| Total | **7/8** | **5/8** |

Five pairs succeeded in both modes, one failed in both, and two succeeded only
with original attention. None succeeded only with the causal control. The
observed difference is 25 percentage points **on this eight-condition subset**.
With only two discordant pairs, this is a small diagnostic observation rather
than a precise population-effect estimate or a broad benchmark result.

The two differing tasks were turning on the stove and placing the moka pot on
it, and placing a bowl in the bottom drawer and closing it. Both modes failed
the selected Goal task that requires opening the top drawer and placing a bowl
inside. The machine-readable report contains all task identifiers and pairs.

Do not compare 7/8 directly against the earlier 68/120 as if the populations
were matched. This diagnostic does not independently reproduce the authors'
published benchmark percentage.

## Independent reference and intervention checks

Before simulation, all eight previously consumed offline observations received
four calls: original evaluator, corrected dense helper, causal control, and
restored original evaluator. All 32 calls completed.

- The live checkpoint used `LlamaSdpaAttention` in all 32 Llama layers.
- Every original/reference and restored call satisfied the expected attention
  contract; no eager-attention fallback was used.
- Maximum original-evaluator/custom-helper discrepancy over the compared
  boundaries was `2.8203848589924974e-08`, below the frozen `1e-6` tolerance.
- The maximum discrepancy after restoring original attention was exactly zero.
- Changing attention changed normalized action outputs on all eight offline
  observations. Per-observation maximum absolute differences ranged from
  `0.294921875` to `1.0054779052734375`. These are normalized-action differences,
  not task-failure probabilities, physical distances, or safety measurements.

The helper was checked inside the **original** dependency stack, addressing the
previous common-dependency validation gap. The exact same model was used for
the attention-only intervention, avoiding a cross-version causal attribution.

## Technical completion and integrity

- Selected GPU: physical ID 0, UUID
  `GPU-bb2451d6-2989-a112-5c18-8892943710e4`.
- Completed 16/16 episodes, eight paired conditions, and 455 total policy calls
  (32 offline plus 423 closed-loop queries), below the 696-call ceiling.
- Elapsed worker time: 788.83 seconds, approximately 13.15 minutes.
- Peak aggregate GPU memory: 16,306 MiB, below the 23,552-MiB ceiling.
- Post-exit selected-GPU telemetry: 6 MiB, 0% utilization.
- Checkpoint inventory and authenticated checkpoint bytes were unchanged.
- Preflight authenticated 35 files, including checkpoint weights, source,
  metadata, and input manifests; the eight HDF5 sources were also authenticated.
- All 11 CPU intervention/episode tests passed on TITAN. Ten analyzer tests
  passed locally and on TITAN. The earlier 12 audit-regression tests also passed
  locally. The local intervention test invocation initially lacked the required
  import path; it was corrected before remote testing or GPU execution.
- Full analysis independently reconciled hashes, schedule order, conditions,
  32-layer observations, parity/restoration limits, action-queue counts, success
  totals, and resource conditions. Remote and local analyses agree.
- No model or dataset download, training, technical stop, automatic retry,
  unrelated process inspection, or change outside `/home/ved/SAVR` on TITAN.

TensorFlow GPU allocation was disabled in both arms, since TensorFlow is used
for image preprocessing. This run makes **no latency or acceleration claim**.
Do not compare its elapsed time with previous inference timing experiments.

## Research decision

The attention discrepancy is no longer merely a static-code suspicion: it
changes real-checkpoint actions and coincides with worse control on two paired
long-horizon conditions when other settings are held fixed. However, the
original baseline is not fully qualified by this small pilot, and it still
failed one selected Goal condition.

The next useful step is a **fixed, broader original-stack dense evaluation on
previously consumed development conditions**, spanning all 40 tasks, with an
explicit review of the common Goal failure. This should establish a defensible
baseline before testing a published comparator and choosing the next method.
Do not tune the method or thresholds using protected final-test data.

Keep the historical compatibility-stack measurements and stopping decisions,
but maintain the qualification on their interpretation as released-policy
experiments. Do not automatically invalidate the earlier original-runtime
whole-prefix/camera experiments, rerun all historical methods, or resume the
stopped CAC training protocol. Current-frame compression with learned action
correction remains a candidate hypothesis, not a selected validated solution.

**Stop here for this diagnostic. No subsequent experiment was launched.**

## Files and reproduction

Protocol: `docs/OPENVLA_ATTENTION_BASELINE_DIAGNOSTIC_V1.md`.
Frozen configuration: `configs/openvla/attention_baseline_diagnostic_v1.json`.
Configuration SHA-256:
`4dc04df06f60268cc12a9fb54a8e3d6cafd5cb630a88b8610806a4be65bbca35`.

Run: `results/openvla-attention-baseline-diagnostic-20260909-v01`.
Owned worker PID was 864123 and has exited. Worker summary SHA-256:
`ff833d835b6a42043d0bddaa78a09848c8114f187a5ffdda13fc4acb6f0180d1`.

The six non-cache evidence files have been copied to the same relative path in
`/Users/veddwivedi/Documents/VLA/SAVR`. Disposable runtime caches remain on TITAN.
Local evidence and the report allow review without a server connection.
Reconcile with:

```bash
python3 -B scripts/analyze_openvla_attention_diagnostic.py \
  --root results/openvla-attention-baseline-diagnostic-20260909-v01 \
  --config configs/openvla/attention_baseline_diagnostic_v1.json
```

Analyzer SHA-256:
`562388b505dfc66d2cd092a7a3f305eaa80512bb581e5a72a35aaf7f14db8501`.
Machine-readable report: `reports/OPENVLA_ATTENTION_BASELINE_DIAGNOSTIC_V1.json`.
No manuscript/poster edits or GitHub push were made as part of this diagnostic.
