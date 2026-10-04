# OpenVLA D62 S4 Recovery 01 Pass

Date: 2026-09-01  
Run: `openvla-d62-requalification-s4-v02-recovery01`  
Classification: corrected D62 substrate requalification pass

## Result

S4 Recovery 01 passed every frozen gate. The result establishes that the
corrected D62 cache path preserves official-policy semantics when reuse is
disabled, executes deterministic recursive ages 1--4 with exact cache/source
provenance, and resets exactly. It does not measure closed-loop task success or
establish a positive-paper result.

## Mandatory compact-position qualification

Before loading OpenVLA, the exact pinned VLA-Cache fork was exercised with a
tiny randomly initialized model:

- 2 tiny-model calls and 0 OpenVLA checkpoint loads;
- active sequence reduced from 592 to 464 tokens;
- exactly 128 designated visual positions removed;
- all 56 official action-readout positions preserved;
- exact sentinel position mapping passed; and
- missing-action, duplicate-map, and length-mismatch contracts were rejected.

The qualification summary is semantically valid and has SHA-256
`7e7d5417b4b1f07a3f605c06d2b9bcbf4fcdb6b3d432824b9d03c24d1e8967e6`.
GPU 0 returned to 6 MiB/0% before the OpenVLA attempt.

## OpenVLA requalification gates

- Complete/pass: true/true.
- Model calls: exactly 37.
- Official/all-fresh controls: 8 of 8.
- Maximum all-fresh hidden error: 0.0.
- Maximum all-fresh action error: 0.0.
- Recursive ages: 1, 2, 3, and 4 completed twice.
- Recursive repetitions: exact.
- Maximum physical source age: 4.
- Reused cache provenance: exact.
- Anchor/parent cache: unchanged.
- Cache clones: disjoint.
- Reset hidden/action errors: 0.0/0.0.
- Reset repetition: exact.
- Peak aggregate selected-GPU memory: 16,165 MiB, below 23,552 MiB.
- Artifact bytes: 89,017,449, below 268,435,456.
- Checkpoint restoration: protected bytes, inventory, and backup cleanup all
  exact; GPU 0 returned to 6 MiB/0%.

No simulator outcomes, terminal success fields, expert actions, or raw actions
were accessed or persisted. The evidence contains none of the forbidden raw or
outcome keys.

## Selection-materiality finding

Canonical instruction-only salience and official action-readout positions did
not change the protected tiles, ordered reuse positions, or onset assignments
on any of the eight frozen suite-balanced observations:

- `selection_materially_changed = false`;
- changed observations: 0 of 8.

This means the historical D62 tile-selection decisions happen to be invariant
on the audited population. Future work must still use the corrected substrate
identity `D62_BAL_PT1_S4C_V1`, because the action-readout implementation and
semantic qualification changed even though the selected tiles did not.

## Evidence integrity

- Worker summary SHA-256:
  `452488620591e74cf70ca96a800b240c991239634339a72ef953b1b55267dc3c`.
- Evidence manifest SHA-256:
  `d01413b2021a9131a2e7da49c4f23c3cece46023087b69d4fbe2d5a994176cfa`.
- Both embedded semantic hashes reconcile.
- Immutable parent stop remains preserved.
- No automatic retry occurred.

## Boundary

S4 is complete. Under the governing semantic protocol, S5/CAC C1
requalification is now technically eligible but remains unauthorized. S5 must
start from a new immutable root and remeasure corrected action states, base
actions, `Z_C`, features, and timing. C1H, C2, simulator work, and training
remain blocked.
