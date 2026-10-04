# OpenVLA Semantic Requalification Foundation Report

Date: 2026-08-31  
Status: S0--S2 complete; S3 launch package ready; stop before GPU execution

## Completed work

1. Preserved every historical BRACE, PAIR, CAC, and C1H artifact unchanged.
2. Audited official and custom preprocessing, prompt, proprio, multimodal,
   cache, hidden-state, action-head, and sidecar-position boundaries.
3. Confirmed the one-token regression-readout shift and its propagation through
   BRACE, PAIR, P4, D62 salience, and CAC.
4. Confirmed that ACR V5-D independently uses the official prompt-derived
   readout start and does not contain this tail-slice error.
5. Created a shared structural contract that derives official action-readout
   and instruction-only positions from the runtime action mask, projected-token
   count, and tokenizer offsets; it contains no negative action-tail indices.
6. Added a reversible official-boundary capture that records, for one released
   evaluator call, the exact pixel, visual, proprio, masked-language,
   multimodal, attention-mask, and action-head-input tensors. Every patched
   method is restored through normal and exceptional exits.
7. Added a corrected dense-forward primitive that consumes the structural
   layout and exposes both official-equivalent `use_cache=None` and
   cache-producing `use_cache=True` paths for the future parity diagnostic.

## Verification

- 15 dedicated semantic tests pass on TITAN.
- Tests span prompt lengths 5, 12, 22, and 37; malformed action masks;
  instruction overlap/duplication; sentinel off-by-one detection; hidden-length
  drift; plain and `torch.nn.Module` action heads; full boundary capture;
  exception restoration; and corrected dense selection/cache semantics.
- 33 combined semantic/C1H/C1/P3 checks pass.
- The complete suite records 526 passes and 9 subtest passes.
- The sole failure remains the unrelated historical ACR-V10 pre-attempt check,
  which expects now-completed V10 result directories to be absent.
- No GPU was selected, no model was loaded, and no simulator or outcome was
  accessed during this work.

## S3 package completion

The versioned S3 package now authenticates its configuration, code, checkpoint
metadata, project-local runtime configuration, and eight frozen cross-suite
observations. It fixes exactly 32 calls, alternates official/custom order,
records only hashes/shapes/maxima, stops on the first mismatch, seals outputs,
restores the checkpoint, samples aggregate selected-GPU memory, enforces the
30-minute/23,552-MiB bounds, and prohibits simulator/outcome/training access.

The dedicated semantic/package suite has 22 passes. The complete repository
suite has 536 passes and 9 subtest passes; the sole failure remains the same
unrelated historical ACR-V10 stale pre-attempt assertion. The outcome-free
remote preflight passes against the real local data and checkpoint inventory.

Historical shifted helpers remain unchanged so their authenticated evidence can
still be reproduced. They are legacy-only and may not be referenced by a new
experiment configuration. New work must use the shared official contract.

## Hashes

- shared contract:
  `9dc8bda16642cd3c76e7bc468e84b1de5899826d86cc9ee528e3c1fdc723459c`
- semantic tests:
  `1082212909b1dd9145a06d453018fa4ae6f4a2343bee306ecf62142e62fe29d3`
- package tests:
  `a51808e79cc221a3e4c60945f50c32e79b859125df7c4266b6f4ef673dccba1d`
- S3 configuration:
  `799b72151217fec8d5f362ebbbd1d39eb6b9dde697f5a31ff439517e70916617`
- protocol:
  `1979a894eca3b46c79b9978da278589602335f394715d33bc4ce3c3f3bc61cc7`
- audit:
  `c136e3be4c4f680c1bd6f714c99fcecd93f39434b4f7b7c174b701669e087d53`

## Boundary

Stop before S3 GPU/model execution. C1H and C2 remain unauthorized. The user
approved continuation into S3; GPU selection must still obey the one-device,
aggregate-only coordination boundary.
