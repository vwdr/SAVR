# Same-checkpoint SpecPrune-OFT comparator qualification

Date: 2026-09-10. Implementation protocol, not a frozen benchmark launch file.
User authorization: continue the published-comparator step after completing the
original-runtime forty-task dense baseline. Do not begin adapter training here.

## Purpose and fixed reference

Establish a credible existing acceleration comparator before evaluating our
current-frame compression plus learned action-correction hypothesis. A method
must be judged against the same released checkpoint, original attention,
preprocessing, action chunk, normalization and simulator execution conventions.
The 39/40 dense result is a development reference, not evidence for compression.

Use SpecPrune-VLA source revision
`8091adc4b574ce9008d49a1dc9a210f4eec314c1`. Preserve the unmodified source archive
and licenses. Label our result **same-checkpoint SpecPrune-OFT adaptation**, not
an exact reproduction of the paper's model, hardware or latency protocol.
Every divergence from source must have a rationale and a regression test.

## 1. Source contract and compaction foundation — completed

Evidence: `reports/SPECPRUNE_SOURCE_AND_COMPACTION_AUDIT_V1.md` and
`reports/specprune_source_audit_v1/source_contract_audit.json`.
Twenty CPU checks passed in the authenticated original TITAN environment.
The readout offset mismatch is established. Attention-path equivalence and full
controller/pruning behavior remain unqualified. No full-model execution occurred.

## 2. Isolated algorithm port — CPU core completed; real-model bridge next

Progress: `src/savr/openvla/specprune.py` implements selection, controller state
and direct original-decoder execution. Thirteen new tests pass, including six
full 32-layer synthetic comparisons against the actual upstream forward. All
44 selected new/existing CPU regressions pass on TITAN. Evidence and explicit
adaptations: `reports/SPECPRUNE_ALGORITHM_PORT_CPU_V1.md`.
This does not qualify the 7B checkpoint or the future simulator/controller bridge.

Implement under a new comparator module, never in checkpoint files, installed
Transformers, or the upstream reference archive. Reuse qualified image/state
preparation and action extraction. Do not copy historical BRACE/PAIR tail slicing.

Separate the implementation into current-frame selection, layerwise selection,
and episode/controller state. Match pinned source operations first; do not
replace them with a convenient fixed token budget while retaining its name.

Required CPU tests before model integration:

1. Both camera orderings and frame-difference indices, including first-query
   behavior and identical versus changed image regions.
2. Early-layer union of local, text-attention and previous-step indices; stable
   mapping from absolute IDs to compressed offsets and protection of nonvisual
   states. No dropped token may be silently reintroduced later in a query.
3. Layer-dependent attention query and key maps before and after each deletion.
   In particular, check rectangular masks at the early pruning transitions.
4. Importance update/prune order and ties. The pinned source prunes at layer 10
   before its first importance update at layer 14. Record the zero-score behavior
   rather than silently changing the schedule. Freeze any justified variant as
   an explicitly separate adaptation before reading task outcomes.
5. Previous-step indices and per-episode confidence reset independently of
   success. Back-to-back one-query episodes must also reset. Batch size one;
   no parallel-environment state sharing.
6. Controller threshold boundaries, coarse/precise mode switches and actual
   retained counts. Use source-defined preset labels, not claims inferred from
   the word “precise.” Keep optional layer skipping disabled initially.
7. Identity behavior with pruning disabled and with an explicit retain-all
   selector. No nonvisual token deletion or changed action readout is allowed.
8. Native SDPA output must remain bidirectional. If importance is computed by
   an auxiliary attention calculation, compare its scores against the equivalent
   manual formula, preserve all masking/scaling, and include its actual cost.
   Do not obtain scores by an unqualified switch of the policy attention backend.

Compare source and adaptation on deterministic synthetic tensors for the
unchanged mathematical operations. Any difference must be explained before
claiming source equivalence. A passing source check is not real-model parity.
The current compaction helper requires at least one retained token from each
camera. This is a qualification restriction, not an asserted SpecPrune rule.
Before using it in the full comparator, establish whether the frozen source
preset satisfies it or generalize the mapping checks to the source's actual
valid selections. Do not silently impose a new per-camera pruning floor.

## 3. Real-checkpoint qualification — only after stage 2

Completed 2026-09-11: all 40 calls passed the frozen single-query check, with zero
identity error. Evidence: `reports/SPECPRUNE_REAL_QUALIFICATION_RESULT_V1.md`.
This does not qualify the pending simulator/controller/frame-history bridge.

Prepare a new immutable configuration and preflight report, reusing the eight
previously consumed training observations from the authenticated input manifest.
Use one selected GPU only after a fresh aggregate availability check. Keep the
existing runtime and checkpoint untouched, all caches/logs project-local, and
network model loading disabled. No installation or new data download is needed.

The configuration must enumerate exact model calls, observation hashes,
pruning presets, numerical tolerances inherited from the verified reference,
wall-time/memory/artifact caps, source hashes, and the no-retry stop rule.
Do not launch from this prose document or choose tolerances after outputs exist.

Measure reference → pruning-disabled adaptation → explicit retain-all path →
reference restoration on the fixed observations. Verify action-head inputs,
normalized and executed action outputs, actual attention implementation, and
unchanged source hashes. Then validate bounded compressed calls for finite
outputs, action shape, final absolute position maps, current-camera use and
controller reset. Compressed actions are not required to equal dense actions:
their difference is the intervention to be evaluated, not a parity bug by itself.

Stop without a benchmark if identity parity fails or any bookkeeping invariant
fails. Preserve the exact technical evidence. Do not automatically retry with
different settings or substitute the historical weak dense reference.

## 4. Matched development evaluation — after qualification

Freeze a separate evaluation configuration before collecting task outcomes.
Use the same consumed task/state/seed conditions for dense and comparator arms,
including the known dense failure, with prespecified arm ordering. Retain every
episode. The forty-task reference can guide debugging but is not a held-out test.
Do not repeatedly tune against its successes while calling it confirmation.

Report paired success counts and uncertainty appropriate to the paired task
population. Do not pool repeated states as independent episodes. Different
observations visited by different policies are expected in closed-loop evaluation.

Collect a separate frozen same-observation timing sample to isolate computational
cost. Time complete queries including both fresh images, selection, attention
scoring, controller, backbone, head and required transfer/synchronization work.
Warm-up policy, call order and resource checks must be fixed in advance. Include
all scheduled calls, not only successful episodes. Also report observed
closed-loop query costs without pretending their observation distributions match.
Do not run dense and comparator models concurrently or hide comparator overhead.

No universal speed/success gate is invented here. The scientific pilot criteria
must be fixed before collecting those outcomes and scoped to its precision.
Small development results can justify further evaluation, not a paper claim.

## 5. Return to the proposed learned method

Only after a credible comparator exists: implement current-frame compression
with the zero-initialized small corrector, validate identity and runtime cost,
then train on a frozen development split and run a matched pilot. Do not claim
novelty merely from fixing a comparator's integration into our checkpoint.
The positive-result question remains whether correction improves the useful
success–latency tradeoff against both uncorrected compression and the qualified
existing method. Offline action-error reduction alone is insufficient.

## Operational boundaries and records

TITAN access is only through `ssh titan`, within `/home/ved/SAVR`. No unrelated
process inspection, allocation changes, installs, cleanup, or checkpoint edits.
Never resume old launchers because they happen to exist. Preserve immutable
result directories, data splits and frozen gates. All new evidence/status must
be synced to `/Users/veddwivedi/Documents/VLA/SAVR`; do not blanket-stage the
dirty repository or push unreviewed files. URTC artifacts are out of scope.
