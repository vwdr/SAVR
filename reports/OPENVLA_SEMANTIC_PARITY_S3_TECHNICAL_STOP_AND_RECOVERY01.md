# OpenVLA Semantic Parity S3 Technical Stop and Recovery 01

Date: 2026-08-31  
Classification: `TECHNICAL_STOP_NO_SEMANTIC_OR_METHOD_RESULT`  
Recovery status: `READY_NOT_AUTHORIZED`

## What happened

The authorized S3 v01 attempt loaded the model and then stopped before its
first policy query. The loaded official Llama configuration did not contain
the VLA-Cache-only fields `proportion_attn_var` and `reusable_patches`; the v01
worker tried to snapshot those fields as though they always existed.

The immutable stop records zero model calls, zero completed observations, no
actions, no semantic comparisons, no simulator or outcome access, and no
automatic retry. Peak aggregate GPU-0 memory was 15,243 MiB. Checkpoint bytes
and inventory were restored exactly, verified loader backups were removed, and
GPU 0 returned to 6 MiB/0%.

## Deeper integration audit

The stop exposed a second issue before it could affect evidence. The v01 worker
entered the VLA-Cache fork's source tree before model initialization. That
fork's loader synchronizes its modified `modeling_prismatic.py` into a local
checkpoint. Therefore it was not a sufficiently independent oracle for the
pinned released OpenVLA-OFT action semantics.

The repository already contains the official OpenVLA-OFT tree at revision
`e4287e94541f459edc4feabc4e181f537cd569a8`. Its evaluator returns actions
directly and its configuration has no `use_vla_cache` field, while the forked
API expects a four-item return and the extra flag. These are interface
differences, not policy differences.

## Recovery 01

Recovery 01:

1. imports and initializes through the pinned official OpenVLA-OFT source tree;
2. disables model-logic synchronization so the checkpoint's authenticated
   prompt-derived action readout remains the executed oracle;
3. inspects the loaded regression method and rejects the cache fork's magic
   tail readout;
4. adapts only the official evaluator's return into a four-item container for
   the already-audited worker, without changing action values;
5. discards only the fork-specific `use_vla_cache=False` constructor argument;
6. creates the two absent cache-control fields as dense `None` values after
   proving they were absent; and
7. leaves the eight observations, 32 calls, tensor boundaries, `1e-6` gate,
   memory/time limits, hashes, output sealing, and stop-before-S4 rule unchanged.

## Verification and boundary

The CUDA-hidden recovery preflight passes against the real source revisions,
checkpoint files, data hashes, official/fork API surfaces, project-local LIBERO
configuration, output-root exclusivity, and constructor compatibility. The
recovery config remains explicitly unauthorized. No recovery model load or GPU
query occurred.

One explicit approval is required for S3 Recovery 01. It remains one attempt
with no automatic retry. S4, C1 requalification, C1H, C2, training, and
simulator work remain blocked.
