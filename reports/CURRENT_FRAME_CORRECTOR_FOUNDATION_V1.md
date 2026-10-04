# Current-frame corrector: implemented foundation

Date: 2026-09-12. Implementation evidence only; no learned robot-policy result.

## What is implemented

The proposed downstream correction module is now separate from historical CAC.
Its inputs are the current compressed action-head penultimate features
(batch x 8 x 4096), normalized base chunk (batch x 8 x 7), current projected
visual patches (batch x 512 x 4096), normalized robot state (batch x 8), and a
current instruction representation (batch x 4096). Actual feature extraction
and serialization still need integration with hard-compressed inference.

The visual version projects all patches to width 256, adds camera and 16-by-16
patch-position embeddings, and applies two eight-head cross-attention/MLP blocks
to eight action queries. The final seven-dimensional residual head starts at
zero. Its output is added to the normalized base actions without new clipping,
residual bounds, smoothing or routing. FP32 is the initial adapter precision.
No past representations, K/V reconstruction or cache-age signals are used.

The action-only ablation removes visual projection and cross-attention. It must
receive the same training labels, split and optimization budget when trained;
its parameter count is smaller, which must be disclosed rather than describing
it as parameter-matched. A dense-plus-corrector control remains in the larger
evaluation plan to separate generic adaptation from compression recovery.

Measured by summing the instantiated default modules' parameters on CPU:

- Visual corrector: 5,071,879 parameters.
- Action-only corrector: 3,485,959 parameters.

The detached-input boundary prevents a backward graph through the backbone.
Inputs created inside inference mode are cloned as ordinary training tensors
before trainable operations, avoiding PyTorch's inference-tensor save restriction.
Input checks and copying are real costs to include in future measured inference.

## CPU verification

Seven unit tests passed on TITAN with CUDA hidden:

- Exact zero-initialization identity for visual and action-only versions.
- No gradients entering frozen input features or teacher labels.
- Backpropagation from features created under inference mode.
- Tiny synthetic L1 fitting reduces loss and reaches visual-projection gradients.
- Changing visual information changes predictions after the output is nonzero.
- The action-only control does not consume visual inputs.
- Rejection of incorrect feature/label shapes and nonfinite inputs.

A separate production-dimension CPU check used batch one, all 512 x 4096 visual
features, and the default 256-wide module. Exact zero-init identity and one Adam
optimizer step passed with finite weights. This is a synthetic implementation
check, not a memory or latency qualification on the GPU, and not LIBERO training.

## Fixed compression foundation

The new deterministic selector supports 256 or 384 of 512 current visual tokens
and 512-token identity. Each camera's 16-by-16 grid is partitioned into 2-by-2
cells; retain two or three corners per cell, respectively, with a fixed spatial
pattern. This yields exactly balanced camera counts and original absolute token
IDs. Three tests verify counts, ordering, each stratum, dense identity and invalid
budget rejection. All nonvisual positions must still be protected using the
qualified compaction map when this selector is connected to the model.

This is a transparent diagnostic baseline, not a novelty claim or evidence that
uniform selection is competitive. The published comparator remains necessary.

## Remaining work

Connect actual hard-compressed inference to the authenticated penultimate
action features, current patch features and dense targets. Verify the feature
record contract on a small fixed sample, including float32 execution rounding,
instruction pooling and training/evaluation split provenance. Measure actual
generation throughput and correction overhead before freezing the bounded
training/pilot configuration. Do not materialize all 41,447 available queries.

No large-model training, robot-data training, corrective rollout or positive
method result has occurred. No old module, raw result or locked split was changed.
