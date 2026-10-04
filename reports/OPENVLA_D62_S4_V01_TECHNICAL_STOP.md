# OpenVLA D62 S4 V01 Technical Stop

Date: 2026-09-01  
Run: `openvla-d62-requalification-s4-v01`  
Classification: technical integration stop; no S4 scientific decision

## Outcome

The authorized S4 attempt stopped fail-closed after 26 of 37 planned model
calls. It completed no simulator work, accessed no terminal outcomes or expert
actions, persisted no raw actions, and wrote no partial scientific manifest.
The checkpoint was restored exactly and GPU 0 returned to 6 MiB/0%.

The immutable stop records:

- error: `OpenVLASemanticError: language-model hidden-state layout changed`;
- elapsed time: 90.3419 seconds;
- peak aggregate selected-GPU memory: 16,165 MiB;
- stop semantic SHA-256:
  `b1f62e785565aaf49e57134eb1d8c9ad4cd84b3205b212e7c95d3d16a48c294f`;
- stop-file SHA-256:
  `c378d341617c5ef022a63369e3e3ad95fd24595aa558d5eef0558b7c402672db`;
- traceback SHA-256:
  `76b49e1ad35384882dac1c323898f1f861d5430b7f50bef05769874265c93e2`.

The deterministic 26-call execution path shows that the run traversed all
eight suite-balanced official/all-fresh controls (24 calls), then the first
recursive anchor and first recursive transition. Because the fail-closed
worker intentionally seals no partial scientific evidence, those controls are
not promoted to an S4 result.

## Root cause

S3 qualified the canonical action selector on dense, full-length output. S4's
first recursive reuse call activated the pinned VLA-Cache fork's token
compaction. At each pruning layer, that fork removes selected reused visual
positions from the active hidden sequence, sorts the surviving absolute
`cache_position` values, and returns the final position vector as the sole
entry in `output.attentions` when attention tensors are disabled.

The new canonical selector correctly rejected the compacted length because it
still required `hidden_length == full_sequence_length`. The action states were
not shown to be absent; the harness failed before mapping their surviving
absolute positions to compact tensor offsets. This is a dense-versus-compact
indexing contract defect, not an observed D62 action, cache-provenance, or
method failure.

Audited pinned source:

- VLA-Cache revision: `a4909880573868dee2769343d52e793c0341678b`;
- installed `modeling_llama.py` SHA-256:
  `34b00dd58c9887780a7947329cb96468a7fe1427e8fa49dc773ce2c1afc627d4`;
- pruning implementation: removes selected absolute positions from
  `hidden_states`, updates/sorts `cache_position`, and returns that vector with
  the model output.

## Correct structural repair

The repair does not restore the historical `[-57:-1]` tail slice and does not
infer offsets from compact length. Instead it:

1. extracts the explicit final absolute-position map from the pinned fork;
2. requires a rank-one map exactly aligned with the active hidden length;
3. requires positions to be sorted, unique, nonnegative, and within the
   authenticated full sequence;
4. requires every one of the 56 official action-readout positions to survive;
5. maps those absolute positions to active tensor offsets; and
6. selects exactly those 56 states in official order.

The shared repair is applied to BRACE, PAIR P3/P4, and the S4 path. Dense
selection remains supported and retains its strict full-length check.

## Verification completed without a model retry

- New adversarial tests cover compact visual pruning, exact mapped action
  values, a missing official action state, duplicate/unsorted maps, length
  mismatch, and malformed output contracts.
- The active code contains no `-57:-1` slice.
- All 63 focused CUDA-hidden OpenVLA/P3 tests pass on TITAN.
- No GPU model run, S4 retry, simulator call, or later-stage work was performed
  during diagnosis and repair.

## Recovery requirements

Any S4 Recovery 01 must be a new immutable attempt. Before loading OpenVLA, it
must exercise the pinned compact-position output contract with a tiny synthetic
model on the selected GPU and prove exact position-mapped selection after
visual pruning. Only if that qualification passes may the unchanged 37-call
S4 population run from a new output root.

The recovery must preserve the v01 population, D62 profile, scientific gates,
call schedule, resource caps, checkpoint protection, and outcome-free boundary.
No automatic retry is permitted. S5/C1, C1H, C2, training, and simulator work
remain blocked.
