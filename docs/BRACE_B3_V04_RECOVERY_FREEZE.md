# BRACE-B3 v04 Inference-Mode Recovery Freeze

Date: 2026-08-25

Status: Authorized; frozen before GPU selection

## Sole correction

Every custom cache-stack operation—preprocessing, vision projection, semantic
sidecar, dense forward, cached forward, action head, and provenance update—now
executes under `torch.inference_mode()`. A runtime assertion rejects any query
whose gradient mode is enabled.

The resolved configuration semantic SHA-256 is
`21b44336aac5db0f62131449a0f1eb2eaf2ce1c19bde118b4af577c390d9fc86`.

## Preflight conclusion

The pinned 4.47.0 cache stack instantiates `LlamaSdpaAttention` and calls the
same `torch.nn.functional.scaled_dot_product_attention` primitive wrapped by
the frozen post-RoPE sidecar. With inference mode, the prior autograd-only
activation retention is structurally impossible. Official core and comparator
paths already use no-gradient inference.

## Unchanged boundary

All scientific and resource settings remain identical to the v01 base:
method, profiles, thresholds, timing, parity, comparator dispositions, query
allocations and caps, strict 23 GiB memory gate, zero outcomes, and acceptance
rules. There is no automatic retry after v04. B4 remains unauthorized.
