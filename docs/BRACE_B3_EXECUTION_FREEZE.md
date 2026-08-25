# BRACE-B3 Physical Timing, Parity, and Slack Freeze

Date: 2026-08-25

Status: Frozen before implementation, GPU selection, or model query

Governing protocol: `docs/BRACE_EXECUTION_PROTOCOL_V2_1.md`

Configuration semantic SHA-256:
`30825fd30eb2c0566b30564740a212c51e09dc60f7e9c4bc7315b24b369dea11`

## Purpose

B3 tests hypotheses H2 and H3 only: whether the accepted corrected cache can
execute faithfully on the real OpenVLA-OFT checkpoint and whether at least one
clean-provenance contract has physical timing headroom on TITAN. It does not
run task episodes or measure success and cannot establish a positive paper.

## Frozen clean-profile grid

Four outcome-blind profiles use VLA-Cache's position-preserving partial-K/V
substrate at decoder layers 2, 6, 9, and 11:

| Profile | Family | Final scene reuse | Final wrist reuse | Scene/wrist max age |
|---|---|---:|---:|---:|
| P1-S25 | scene only | 64/256 | 0/256 | 4/0 |
| P1-S50 | scene only | 128/256 | 0/256 | 4/0 |
| P2-D25 | dual view | 64/256 | 32/256 | 4/2 |
| P2-D50 | dual view | 128/256 | 64/256 | 4/2 |

Budgets grow monotonically across the four pruning layers. Selection uses the
exact source image, a fixed 0.5 L1/0.5 cosine change score, immutable per-token
source ownership, context envelopes, and a dense-anchor semantic mask. P1
always refreshes the wrist camera; P2's wrist limits are stricter than its
scene limits. Horizons 1, 2, and 4 remain separate conditions.

The dense semantic sidecar may be retained only if a post-RoPE Q/K tap calls
the unchanged SDPA kernel and preserves backend, action, cache, and timing
semantics. `output_attentions=True` is not an accepted substitute. A failed
sidecar removes the affected profiles rather than weakening parity.

## Timing and query design

- one fixed-seed randomized schedule;
- one warm-up horizon-4 cycle per profile;
- six synchronized measured cycles per profile and horizon;
- p50/p95/p99 critical wall, CUDA, component, and cycle times;
- preprocessing, patch comparison, semantic sidecar, provenance, transfers,
  action head, and abort checks included;
- no outlier deletion; and
- 388 planned real-model queries under a hard cap of 420. Unused or failed
  comparator allocations cannot be reassigned.

The one-GPU process sequence is: optimized core FR; isolated VLA-Cache/P0/P1/
P2/P3; official VLA-ADP overlay; official VLA-Pruner isolated stack. Only one
model process may exist at once.

## Comparator adjudication

- Faithfully corrected VLA-Cache must receive real timing.
- VLA-ADP and VLA-Pruner must receive real official-code timing or an
  individually evidenced technical exclusion.
- SpecPrune-VLA is excluded from execution because its pinned repository lacks
  the advertised top-level method-specific license. Its source is not treated
  as licensed merely because the vendored OpenVLA-OFT subtree is MIT.
- Gated VLA-Cache is excluded from direct execution because no official code
  is discoverable and its published top-two discrete action-token logit margin
  is not defined for the pinned continuous L1 action head. Inventing a
  different confidence score would not be a matched reproduction.

These exclusions are limitations, never evidence that BRACE wins.

## P4 disposition

P4 uses the synchronized dense completion time against the real LIBERO control
window: eight queued actions at 20 Hz, or 0.4 seconds. No artificial sleep is
used. Completion, spillover, blocking, and total GPU work are reported.

## Advance and failure rule

B3 advances only if P0 matches optimized FR; at least one clean profile has
both 10% accelerated-query and 8% amortized contract-cycle reduction after all
overhead; peak reserved memory is strictly below 23 GiB; every cache,
provenance, and reset invariant passes; every mandatory comparator has valid
timing or a reviewed exclusion; and P4 has a measured disposition.

Any technical failure, cap breach, parity failure, missing disposition, or
absence of a passing clean profile stops before B4. There is no automatic
retry. B4 requires separate authorization.
