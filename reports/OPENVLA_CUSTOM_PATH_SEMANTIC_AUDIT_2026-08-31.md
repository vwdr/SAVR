# OpenVLA Custom-Path Semantic Audit

Date: 2026-08-31  
Scope: BRACE, PAIR, and CAC custom OpenVLA-OFT inference helpers  
Verdict: semantic requalification required before further GPU/simulator work

## Confirmed root cause

For a prepared query with:

- `P` prompt tokens after optional empty-token insertion;
- 513 projected visual/proprio tokens;
- 56 placeholder action tokens; and
- one stop token,

the complete multimodal length is `N = 513 + P + 57`.

The pinned released evaluator sets `NUM_PROMPT_TOKENS = P - 1` and reads:

```text
[513 + (P - 1), 513 + (P - 1) + 56) = [N - 58, N - 2)
```

Therefore, the official regression-head input is equivalent to
`last_hidden[:, -58:-2, :]`.

The BRACE/PAIR/CAC helper instead reads `last_hidden[:, -57:-1, :]`. It shifts
all 56 readout positions one token later, dropping the official first state and
adding a state the official evaluator does not use. Recovery 02 measured a
maximum absolute action difference of `0.18178841471672058` and stopped before
any episode.

## Boundary-by-boundary audit

| Boundary | Official and custom relationship | Status |
|---|---|---|
| Prompt construction | Same literal prompt | aligned |
| Empty-token insertion | Same token ID and condition | aligned |
| Image preparation | Same two-camera preparation and processor calls | aligned |
| Observation state | Upstream helper mutates state; Recovery 02 now isolates a copy | corrected |
| Proprio normalization/projection | Same statistics and BF16 projected token | aligned |
| Placeholder labels/action mask | Same model helpers; 56 contiguous action placeholders | aligned |
| Multimodal construction | Same model helper and 513 projected tokens | aligned |
| Dense cache input | Both begin without prior K/V; exact `use_cache` semantics still require parity attestation | qualification required |
| Regression-head states | Official `[N-58,N-2)` versus custom `[N-57,N-1)` | confirmed mismatch |
| Action unnormalization | Same model helper | aligned after hidden-state correction |
| Sidecar action positions | Custom positions follow the shifted placeholders, not official readout states | confirmed mismatch |
| Instruction positions | P3 computes exact instruction-token offsets but runtime salience currently uses all nonaction prompt positions | semantic correction required |

## Affected implementation and evidence

Active helpers containing the shifted readout include:

- `src/savr/brace/b3_openvla.py`;
- `src/savr/pair/p3_openvla.py`; and
- `src/savr/pair/p4_openvla.py`.

The P3 runtime position map also shifts action-attention queries and ignores its
already-computed instruction-only offsets. Consequently:

- BRACE and PAIR custom-path action comparisons are not authenticated as the
  pinned OpenVLA-OFT policy;
- the D62 dynamic selection substrate may choose different visual groups after
  official action/instruction salience is restored; and
- CAC C1's `Z_C`, base actions, internal equality controls, and adapter input
  examples were generated from the shifted readout.

Historical artifacts remain immutable. Their internal measurements are not
deleted, but claims requiring official-policy equivalence are provisional.
CAC's measured systems headroom remains useful as an engineering estimate and
must be remeasured after correction.

The ACR V5-D backend uses a prompt-count-derived start
`513 + number_of_prompt_tokens`, which matches the pinned released evaluator;
it does not contain the BRACE/PAIR/CAC tail-slice error.

## Why earlier checks did not catch it

C1 compared the custom path against repeated, all-fresh, hooked, and
action-head-reconstructed executions of that same custom path. These tests were
strong internal invariants but not an independent official oracle. Because all
branches shared the shifted hidden tensor, they agreed exactly. C1H Recovery 02
was the first test to invoke the pinned released helper and the custom helper on
the same observation; it correctly stopped.

## Containment decision

No further C1H retry or C2 work is permitted. Before another simulator run, the
project must:

1. replace every active magic tail slice with one shared, structurally derived
   official readout contract;
2. align action and instruction sidecar positions with explicit semantics;
3. pass CPU tests that make the former off-by-one implementation fail;
4. capture the exact hidden tensor received by the official action head as an
   independent real-model oracle; and
5. requalify D62 and CAC C1 under the corrected policy semantics.

## Authenticated upstream sources

- checkpoint `modeling_prismatic.py`:
  `f40ee7883e16aab1a2d89b6e8f31cc81f6b8055120b1fefe169e05c7031098fa`
- pinned `openvla_utils.py`:
  `0ee61ce0eb79f9a661ec4a99454878d8842cc0d6d62fb00c4b2ba5de4573631c`
- pinned `robot_utils.py`:
  `61109d1de8f7ed4cfc4d97a65b8ca5326fefe0d4da4b15c73732f77568d3e34d`
- pinned source `modeling_prismatic.py`:
  `58011d78a27a1b44a8729052b5fd8e06c4dec79fe531e3c91ef91538ccdc1e3a`

