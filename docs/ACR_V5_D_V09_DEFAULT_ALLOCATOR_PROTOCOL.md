# ACR V5-D V09 Default-Allocator Recovery Protocol

Status: **FROZEN FOR PRE-GPU IMPLEMENTATION; GPU SELECTION NOT AUTHORIZED**

Date: 2026-08-11

## 1. Decision

V08 remains an immutable technical stop that confirmed its inference-semantics
memory hypothesis. V09 preserves V08 and changes one raw-process property: it
removes `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, returning the raw
process to PyTorch's default native allocator before PyTorch is imported.

V09 preserves V08's model, checkpoint, tensors, inference-mode lifecycle,
pre-capture warm-ups, graph bodies, capture stream, shared pool, capture and
replay order, 111-query schedule, correctness and timing gates, 23 GiB cap,
compiler-first waterfall, and scientific claim boundary.

## 2. Evidence and causal hypothesis

V07 introduced expandable segments only to recover a 230 MiB memory excess.
V08 then identified and corrected the actual active-memory cause. With the
experimental allocator still enabled, V08 reduced peak reservation from
23.2246 GiB to 15.2773 GiB, leaving 7.7227 GiB below the unchanged cap. The
allocator override is therefore no longer required for feasibility.

The remaining V08 failure occurred during the second graph capture, after both
eager warm-ups and the first capture succeeded. PyTorch 2.2 documents:

- the native allocator as the default and `expandable_segments` as an
  experimental option intended for changing allocation sizes;
- separate graph-private pools and the supported ordered shared-pool pattern;
- that capture errors may arise from graph-unsafe work and that asynchronous
  errors can surface at a later API boundary.

Primary sources:

- [PyTorch 2.2 allocator and CUDA-graph semantics](https://docs.pytorch.org/docs/2.2/notes/cuda.html)
- [PyTorch issue: expandable segments with CUDA graphs](https://github.com/pytorch/pytorch/issues/140419)
- [PyTorch issue showing the same generic capture-end error can mask an earlier unsupported operation](https://github.com/pytorch/pytorch/issues/128396)

The issue tracker is supporting evidence, not proof of V08's exact cause. The
falsifiable V09 hypothesis is narrower: if the experimental allocator is the
active compatibility factor, the unchanged second capture will complete under
the default allocator. If it fails again, V09 stops and the model graph body
must be diagnosed separately.

## 3. Frozen lifecycle

1. Keep the compiler process unchanged and require its exact zero-output
   technical failure before permitting a fresh raw process.
2. Require `PYTORCH_CUDA_ALLOC_CONF` to be absent in both processes and attest
   its absence in raw/final evidence.
3. Apply V08's sustained aggregate transition gate before PyTorch import.
4. Enter and attest V08's whole-attempt `torch.inference_mode()` lifecycle.
5. Perform the unchanged wrist/downstream pre-capture warm-ups.
6. Capture wrist then downstream on the same stream and shared private pool.
7. If both captures complete, enforce all seven correctness queries before
   any scheduled warm-up or timing query.
8. Permit at most the unchanged 111 full queries and finalize only a complete
   record that passes every resource, restoration, correctness, and timing
   integrity gate.

## 4. Rejected or deferred alternatives

- **Capture-mode relaxation:** rejected because it weakens safety and could
  conceal graph-unsafe work.
- **Dropping the shared pool or changing stream/order:** rejected because it
  changes the tested backend and memory/lifetime contract.
- **Disabling KV cache, bypassing the LM head, or returning only final hidden
  state:** promising but deferred because these alter the graph body and need a
  separately frozen eager oracle and parity proof.
- **Single combined graph:** deferred because it changes the executor boundary
  and component timing methodology.
- **Two-GPU sharding, quantization, offload, higher cap, or fewer warm-ups:**
  rejected for V09 as broader and scientifically confounded.

## 5. Stop rules and authorization boundary

- Any allocator override, failed inference attestation/restoration, OOM,
  memory-cap breach, capture failure, correctness failure, resource failure,
  or checkpoint-restoration failure stops the identity without retry.
- Only a complete 111-query record may reach analysis.
- No simulator, reward, success field, protected population, or manuscript may
  be accessed. V5-E remains unauthorized.
- Current authorization covers research, documentation, implementation, local
  tests, and CUDA-hidden TITAN verification only. Stop before GPU inspection or
  selection.

## 6. Expected result and uncertainty

V09 has substantial memory headroom and removes the only experimental runtime
setting left in the raw path. This makes it a logical, minimally invasive test,
not a guaranteed fix. The generic V08 error could instead originate inside the
downstream graph body. V09 is successful only if both captures, exact eager
parity, the full schedule, and every timing/resource gate complete.
