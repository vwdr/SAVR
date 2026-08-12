# ACR V5-D V08 Inference-Semantics Recovery Protocol

Status: **AUTHORIZED FOR ONE AGGREGATE-SELECTED, FAIL-CLOSED GPU ATTEMPT**

Date: 2026-08-11

Authorization checkpoint: On 2026-08-11, the user explicitly approved V08
GPU selection and execution. The authorization is limited to one aggregate-only
selection and one frozen attempt with at most 111 model queries. It does not
authorize an automatic retry, simulator use, protected-outcome access,
manuscript changes, or V5-E.

## 1. Decision

V07 remains an immutable pre-correctness memory technical stop. V08 preserves
V07 and changes one raw-process property: after the sustained aggregate GPU
transition gate and before model initialization, the complete raw attempt runs
inside `torch.inference_mode()`.

V08 retains V07's exact `expandable_segments:True` allocator, model,
checkpoint, tensors, pre-capture lifecycle, graph bodies, one stream, shared
pool, capture/replay order, 111-query schedule, correctness/timing gates,
23 GiB cap, and claim boundary. The compiler process remains unchanged.

## 2. Root-cause evidence

The pinned official OpenVLA-OFT evaluator wraps action inference in
`torch.inference_mode()` and its outer robot utility also uses
`torch.no_grad()`. The custom V5-D tensor path called the same trainable modules
under default grad mode; `model.eval()` was present, but no grad or inference
guard covered input preparation, static-buffer copies, warm-ups, or capture.

PyTorch 2.2 explicitly states that evaluation mode is orthogonal to gradient
tracking. It states that inference mode excludes operations from the backward
graph and removes additional autograd tracking overhead:

- [PyTorch 2.2 autograd grad modes](https://docs.pytorch.org/docs/2.2/notes/autograd.html#locally-disabling-gradient-computation)
- [PyTorch 2.2 CUDA graph semantics](https://docs.pytorch.org/docs/2.2/notes/cuda.html#cuda-graphs)

A CUDA-hidden reproduction in TITAN's pinned PyTorch 2.2 environment used a
trainable evaluation-mode module and repeated `copy_` into one static output.
Under default grad mode, the destination had `CopyBackwards` and its reachable
backward graph grew from 10 to 16 to 22 nodes across three repetitions. Under
inference mode, the destination never required gradients and had no `grad_fn`.
CUDA remained uninitialized.

This mechanism maps directly to V07: its owned wrist and action outputs are
repeatedly populated during warm-up, and the model's parameters retain their
normal trainable flags even though the official evaluator performs inference.
V08 therefore corrects execution semantics rather than changing the model.

## 3. Frozen lifecycle

1. Use the unchanged aggregate-only selector and one physical GPU.
2. Run the unchanged compiler process with no allocator override or inference
   wrapper.
3. Require its exact zero-output technical failure, restoration, and fresh-raw
   permit.
4. Start one fresh raw process with V07's exact allocator environment.
5. Require V05's unchanged sustained transition gate.
6. Enter one thread-local `torch.inference_mode()` context immediately after
   the transition passes and before model initialization.
7. Attest `torch.is_grad_enabled() == False` and
   `torch.is_inference_mode_enabled() == True` before any model preparation.
8. Keep the context active through model setup, input preparation, owned-buffer
   population, eager warm-ups, both captures, all correctness queries, and all
   timed queries.
9. Record the active state in the raw attempt/final record, exit in `finally`,
   and require exact restoration of the prior thread-local state.
10. Enforce every unchanged memory, correctness, timing, ordering, restoration,
    resource, and 111-query gate.

## 4. Alternatives considered

- **More allocator tuning or `empty_cache`: rejected.** V07 reduced the
  fragmentation indicator but active allocation itself reached 23.0152 GiB.
- **Fewer warm-ups: rejected.** This weakens CUDA-graph preparation validity
  and changes the frozen timing method.
- **Two-GPU sharding/offload: deferred.** It requires a new model partition and
  inter-device timing method and is unnecessary before correcting inference
  semantics.
- **Quantization or precision changes: rejected for V08.** They alter model
  numerics and require new correctness tolerances and baselines.
- **Bare-decoder/no-cache/output pruning: promising but deferred.** Transformers
  4.40.1 documents that `use_cache=True` returns key/value state and
  `output_hidden_states=True` returns every layer, while the regression head
  needs the final hidden state. Applying that optimization fairly would require
  a separately frozen common-path baseline; it is not mixed into V08.

## 5. Stop rules

- Any inherited inference context, failed state attestation, unsupported
  inference tensor, graph-capture error, or failed state restoration: preserve
  and stop.
- Any OOM or memory-cap breach: preserve and stop without retry.
- Any correctness failure: stop before timing.
- Only a complete 111-query record may be analyzed.
- No simulator, success/reward field, final population, or manuscript may be
  accessed. V5-E remains unauthorized.

## 6. Expected result and uncertainty

V08 has a strong mechanistic basis because it removes backward-graph retention
that should never exist during evaluation and matches the official evaluator's
semantics. It is expected to reduce active preparation memory by substantially
more than V07's 230 MiB cap excess, but this is a hypothesis until the single
frozen real-GPU attempt completes. CUDA-graph compatibility and exact numerical
parity remain mandatory empirical gates.
