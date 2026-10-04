# OpenVLA Semantic Requalification Protocol V1

Date: 2026-08-31  
Applies to: BRACE, PAIR, D62, and CAC  
Purpose: prevent another full run from discovering a basic policy-integration mismatch

## 1. Governing rule

An internal comparison is never sufficient to establish correctness of a
custom inference path. Every custom action path must first reproduce the exact
pinned released evaluator on the same observation, using an independently
captured official action-head input and action output.

No GPU-scale, simulator, terminal-outcome, training, or C2 work may begin until
the preceding stage is complete. Every technical failure is fail-closed and
cannot trigger an automatic retry.

## 2. Frozen reference

The reference policy is the current pinned OpenVLA-OFT checkpoint and evaluator
already authenticated in the repository. The official regression-head input is
captured by a temporary, reversible hook on `action_head.predict_action` during
the released `evaluation.get_action` call. This avoids recreating the reference
with the same custom indexing logic.

The hook must:

- capture a detached clone of the exact input tensor;
- call the original action head unchanged;
- execute exactly once per reference query;
- restore the original method in `finally`;
- prove that model parameters, RNG state, cache controls, and observation input
  are unchanged except for the documented upstream state-copy mutation; and
- persist only hashes, shapes, maxima, and pass/fail fields—not raw actions.

## 3. Stage S0 — containment and impact ledger

1. Preserve all historical result roots unchanged.
2. Mark BRACE/PAIR/CAC claims requiring official-policy equivalence as
   provisional; do not reinterpret outcomes before corrected requalification.
3. Record every active occurrence of custom action-state indexing and every
   consumer of its action/instruction position map.
4. Confirm that C1H and C2 are blocked.

Exit gate: the semantic audit and dependency inventory are complete.

## 4. Stage S1 — one canonical structural contract

Create one small shared module that derives, without negative tail indices:

- projected-token count from the prepared tensor;
- prompt count from the pre-action input IDs;
- the contiguous 56-placeholder mask;
- official readout start `projected_tokens + prompt_tokens`;
- official readout stop `start + 56`;
- exact instruction-only multimodal positions from tokenizer offsets; and
- visual, wrist, proprio, prompt, action-readout, placeholder, and stop spans.

Required invariants:

- action readout has exactly 56 unique in-range positions;
- its first position is exactly one before the first placeholder position;
- its final position is exactly one before the final placeholder position;
- instruction-only positions are nonempty and do not overlap action readout;
- scene, wrist, and proprio spans retain their authenticated positions;
- no active BRACE/PAIR/CAC code contains `-57:-1`; and
- ACR's already-correct prompt-derived behavior remains unchanged.

The shared contract becomes the sole source for action-head extraction and
sidecar action/instruction positions.

## 5. Stage S2 — CPU/static adversarial verification

Before GPU access, tests must cover:

1. multiple synthetic prompt lengths rather than only the historical 21-token
   prompt;
2. sentinel hidden states proving the official span is selected exactly and the
   former shifted span fails;
3. contiguous and malformed action masks;
4. missing, duplicate, overlapping, and out-of-range semantic positions;
5. exact instruction-offset mapping through BOS and projected-token insertion;
6. observation cloning against in-place upstream normalization;
7. four-item official return validation and invalid shapes/nonfinite actions;
8. hook installation, single invocation, exception cleanup, and restoration;
9. dense/reset/cache configuration restoration after every control;
10. source-hash drift, config-hash drift, output-root reuse, timeout, memory,
    checkpoint restoration, and traceback persistence; and
11. a repository scan that rejects active magic action-tail slices.

Run the focused suite and complete repository suite. Historical tests that
encode the shifted convention must be corrected explicitly, not weakened.

Exit gate: all relevant tests pass; unrelated historical failures are named and
demonstrably independent.

## 6. Stage S3 — bounded outcome-free real-model parity

This is a new versioned diagnostic, not a C1 or simulator retry.

Population:

- eight frozen offline observations: two from each LIBERO suite;
- varied instructions and states from already-authorized local artifacts;
- no simulator, reward, success, expert outcome, or locked state IDs.

For each observation, alternate reference/custom order and verify:

- exact observation-copy isolation;
- identical prepared pixel tensors, token IDs, attention mask, normalized
  proprio, projected tokens, and multimodal tensors;
- exact structural span agreement;
- custom hidden tensor versus independently captured official action-head input;
- normalized and unnormalized action maximum absolute error at most `1e-6`;
- deterministic repeated dense results;
- sidecar-off/on exact equality;
- `use_cache`/K/V production does not change dense hidden states or actions;
- complete cleanup, checkpoint restoration, and aggregate memory compliance.

Budget: one selected idle GPU, one model process, at most 32 model calls, 30
minutes, no downloads, no raw action persistence, and no automatic retry.

Any single mismatch stops the stage. The first mismatching boundary—not merely
the final action—is recorded.

## 7. Stage S4 — D62 substrate requalification

Only after S3 passes:

1. regenerate action/instruction salience with canonical positions;
2. verify zero-reuse/all-fresh cache execution against the official policy;
3. verify one anchor and each recursive age 1--4 transition;
4. re-establish physical source provenance, cache isolation, reset semantics,
   deterministic actions, and memory limits; and
5. determine whether corrected salience changes D62 selections materially.

If D62 changes, prior PAIR/D62 evidence remains historical and a new versioned
substrate is frozen. Do not silently reuse the old identity.

## 8. Stage S5 — CAC C1 requalification

Only after S4 passes, rerun C1 from a new immutable root because corrected
action states change `Z_C`, base actions, and potentially source features.

Mandatory additions to C1:

- official hidden/action parity before all internal controls;
- corrected action and instruction positions;
- the independent hook oracle included in the evidence summary; and
- the existing chronology, isolation, memory, timing, and adapter-shape gates.

The prior 22.41% saving is provisional until remeasured. C1 must pass again
before C1H becomes eligible.

## 9. Stage S6 — C1H eligibility

C1H may be reconsidered only if S3, S4, and the new C1 all pass. Its population,
paired schedule, Gate H, outcome sealing, and stop-before-C2 boundary remain
unchanged unless a separately reviewed protocol amendment is scientifically
necessary.

## 10. Status and approval gates

- S0--S2 are CPU/static engineering work and may proceed under the current
  instruction.
- Stop before selecting a GPU for S3 and report the complete readiness audit.
- S3, S4, C1 requalification, and C1H each require an explicit phase approval.
- C2 remains unauthorized regardless of intermediate results.

## 11. Definition of success

This protocol succeeds when the project no longer relies on circular internal
agreement: the custom cached path is demonstrably the same pinned policy when
reuse is disabled, and every later scientific result is downstream of that
independent fact.
