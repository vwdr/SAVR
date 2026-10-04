# SpecPrune-OFT algorithm port: CPU qualification

Date: 2026-09-10 (US Eastern; final verification after midnight UTC).
Status: core algorithm and native-decoder integration implemented and CPU-tested.
Real-checkpoint qualification and simulator integration are not yet complete.

## Completed work

The isolated module `src/savr/openvla/specprune.py` now implements current-image
patch similarity, early text-guided selection, layerwise importance/pruning,
previous-query selection indices, episode confidence reset, action-controller
thresholds and frame-lookback arithmetic. Its decoder runner calls the existing
original SDPA layers with current tokens and their original rotary positions.
It deletes actual token rows rather than merely masking their outputs.

The auxiliary attention-score calculation does not replace the native SDPA
output. Attention matrices are needed only at selected layers and are not kept
for all 32 layers. No hidden states or K/V tensors survive into the next query.
Only confidence and selection indices are retained across queries. The real
action head continues to require the previously authenticated absolute readout
positions, not the upstream shifted tail slice.

No source, installed runtime, checkpoint, old experiment, manuscript, or poster
was edited. This module is an adaptation, not a claim of an exact paper reproduction.

## Independent tests and evidence

All **44 CPU tests passed on TITAN with CUDA hidden**, with no skips:

| Test group | Tests | What it establishes |
|---|---:|---|
| New algorithm port | 13 | Synthetic source equivalence, state handling, attention neutrality and tiny-model execution |
| Existing source identity audit | 6 | Pinned source/readout checks and drift rejection |
| Existing visual-compaction foundation | 14 | Protected positions, identity, actual deletion and bidirectional flow |
| Existing original-attention/episode regressions | 11 | Reference attention and episode-order behavior remain intact |

The full 32-layer selection test executes the pinned upstream `LlamaModel.forward`
on fake decoder layers providing deterministic attention matrices. The adaptation
receives the same matrices. Every pre-layer position map, final position map,
importance vector and completed prior-index set matches exactly in six cases:
prompt lengths 12/34/70, both controller profiles, uniform-attention ties,
pruning disabled and dynamic pruning disabled. These are six synthetic cases,
not an exhaustive proof or six robot experiments.

The source's patch-cosine selection matches the port for both camera offsets,
ordinary/precise budgets, altered regions, identical images and all-black images.
The source's initial zero-importance pruning point is exercised without changing
its layer schedule. Controller tests cover strict threshold boundaries, signed
vertical motion, queue-dependent mode exit, first-step behavior and frame lookback.
Repeated episode IDs reset correctly; confidence persists only within an episode.

A random 32-layer, width-32 LLaMA in the original project runtime passes both
float32 and bfloat16 tests. Pruning-disabled and explicit keep-all-with-scores
outputs equal native dense output exactly in the tested paths. Real deletion
produces finite outputs and the expected 56-state action readout. Returning to
dense execution restores the original output. This is a tiny CPU model, not the
7B checkpoint and not evidence about task success or GPU latency.

Log: `reports/specprune_algorithm_port_v1/cpu_tests.log`.
Tests: `tests/openvla/test_specprune_port.py` and the existing groups above.

## Source provenance and test harness

Official source: <https://github.com/alexwhz-sjtu/SpecPrune-VLA>, revision
`8091adc4b574ce9008d49a1dc9a210f4eec314c1`.
The previous eight-file archive is unchanged. One additional required helper,
`openvla-oft/experiments/robot/spec_prune_vla.py`, is archived separately at
`reports/specprune_algorithm_port_v1/upstream/spec_prune_vla.py`.
Its Git blob ID is `44f1ef83e18a8817135df1836305fe094797a6e1`, verified against
the pinned upstream tree and local bytes. MIT and Apache license text/notices
are retained beside the adapted module.

Tests extract only named upstream functions through ASTs, remove decorators,
and supply synthetic model objects/constants. No upstream loader is executed.
The first test discovery stopped because the original environment lacks
`skimage`. No package was installed. The test harness instead supplies an
independent, explicit patch-slicing implementation for that one block-extraction
dependency; the source cosine/ranking functions run unchanged. The production
port uses NumPy reshape/transpose. The final 44-test run completed normally.
This was a CPU test-harness issue, not a failed GPU experiment or task outcome.

## Explicit adaptations and remaining qualification constraints

1. **Policy semantics:** Keep the authenticated original attention and action-head
   readout. Preserve upstream selection windows separately: dynamic importance
   uses placeholders; prior global selection uses fixed absolute IDs [522,547).
   These are source implementation choices, not interchangeable with the action
   head's readout or guaranteed instruction-only tokens for every prompt length.
2. **Episode execution:** The source evaluator has different horizons and its
   action-controller branch places simulator execution inside `acting_steps >= 10`.
   Do not import this evaluation loop. The future bridge must preserve our dense
   evaluator's actual steps, initialization, horizons and action transforms;
   only the explicitly tested mode/replan rules may be added. Full controller
   and rolling paired-frame history still need that integration test.
3. **Pruning semantics:** Do not impose the earlier shared helper's one-token-per-
   camera floor on SpecPrune. This port follows the source's global retention
   rule while preserving all nonvisual states. Tokens may reappear in a new
   fresh query, but not after deletion within the same query.
4. **Masks and precision:** This runner supports unpadded, batch-one current
   queries. The real-model worker must assert that input contract. Native CUDA
   kernels and 4096-wide layers remain untested by these CPU checks. Tie ordering
   can differ across devices; do not claim CPU selection identity proves CUDA identity.
5. **Timing fairness:** The decoder-only runner omits unused vocabulary logits
   and the full hidden-state history. A timed dense comparator must share these
   same wrapper optimizations, or separately account for their cost. Do not
   attribute wrapper savings to pruning. Include scoring, frame preparation,
   CPU/GPU transfers, controller and validation costs in measured query time.
6. **State safety:** A query that stops early must not be resumed using partial
   confidence state. Abort and preserve evidence; do not automatically retry.

## Next action

Prepare and freeze a bounded real-checkpoint qualification worker using the
eight already consumed training observations. Compare native reference,
pruning-disabled adaptation, explicit keep-all-with-scores, bounded compression
and restored reference before any simulator benchmark. Numerical tolerances,
exact counts and resource caps must be frozen in its configuration before launch.
No adapter training or held-out-data access is authorized by this report.

The original 39/40 baseline result remains unchanged. No positive acceleration
result exists yet. Work was performed locally and through `ssh titan` exclusively
inside `/home/ved/SAVR` on the server. No GPU workload or GitHub push occurred.
