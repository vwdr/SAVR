# SpecPrune-OFT episode integration

Date: 2026-09-11. Implementation specification and CPU evidence, not a GPU
experiment authorization/configuration or a task-performance result.

## Purpose

Connect the qualified same-checkpoint comparator to the original OpenVLA-OFT
episode semantics. Preserve the original forty-task dense reference, archived
upstream code, completed qualification worker/configuration and raw results.
New implementation: `src/savr/openvla/specprune_episode.py`.

## Episode contract

The loop resets the simulator, copies the initial state, performs ten original
settling actions, and executes up to the original suite's action horizon.
It uses the original observation preparation and gripper processing. Each policy
query returns eight finite seven-dimensional actions. The loop executes each
popped action once. Terminal success ends the loop immediately; it never prompts
an additional query. Exceptions propagate as technical errors, not task failures.

With the controller disabled, the complete reset/prepare/query/action trace
matches `attention_diagnostic.execute_episode`. That reference itself matches
the released original evaluator in the existing regression test. Controller-
enabled execution adds only the pinned mode/replan rules after each executed
action. The first nine acting steps are executed normally, not omitted by the
upstream comparator's differently structured conditional. Episode horizons are
not extended to the comparator source's larger values.

Record executed steps, query count, replans and discarded queued actions. A
replan on a terminal step can discard actions but cannot generate another query.
Do not interpret discarded actions as executed or artificially free inference.

## Paired history and preprocessing

Store the scene and wrist images together on every acting step, before a
possible query. Both are the original evaluator's resized 224-by-224 uint8 RGB
images, before policy center cropping. A six-pair queue suffices for the pinned
lookback cap. Copies are read-only and cannot alias mutable simulator buffers.
All history and controller state reset before every episode, including when the
same task/state identifier is repeated.

The selected previous pair is determined by `EpisodeState.previous_frame_index`.
The first query uses the current pair. Later queries use the source's capped
control-step lookback, not simply the previous policy-query image. The source
action accumulator is consumed at query time. The selector increments the query
counter exactly once; the episode loop checks this rather than incrementing it
again. Partial state after an exception is never resumed automatically.

Policy-preprocess the current pair once and selected prior pair once per
compressed query. The same current PIL images supply both patch comparison and
the qualified semantic input preparer through an explicit one-use callback.
The preparer must not crop those images again. Prior pixels supply selection
signals only. Current pixels, not historical pixels, enter the vision encoder.
No cached vision output, transformer hidden state or K/V tensor is substituted.

This is an explicit camera-path adaptation: the upstream comparator stores an
unresized scene replay image but a resized wrist image. We store both at the
original policy-input boundary. This avoids changing the qualified dense image
pipeline and does not assert pixel equivalence to the source's replay route.
The original `EpisodeState` docstring describes prepared history generically;
this bridge delays the final policy crop until the pair is selected.

Low-change budgets remain the pinned values: coarse scene/wrist 236/230,
precise 240/236; cosine thresholds 0.986/0.98. Do not infer the aggressiveness of
these settings from the word “precise.” Other selection operations remain in
the unchanged, authenticated core module.

## Query and timing contract

`SpecPruneEpisodeQuery` uses the qualified current-image/state preparation,
original bidirectional SDPA decoder and authenticated 56-state head readout.
The dense control uses that same direct decoder with selection disabled. Neither
arm computes unused language-vocabulary logits or stores all layer outputs.
Dense inference does not incur unnecessary historical-image preprocessing.

The eventual timing worker must time the entire query call with accelerator
synchronization: current image processing, historical image processing where
needed, selection, auxiliary attention scores, backbone, head, action transfer
and required validation. Do not subtract these costs after seeing results.
Report history/controller bookkeeping and complete episode time separately,
including extra policy calls caused by replanning. Comparing per-query time
alone cannot establish faster episode execution. Native-wrapper timing can be
reported separately, not used to credit unrelated wrapper savings to pruning.

## Checks completed and limits

Thirteen new CPU tests cover exact controller-off episode traces, paired
per-control-step history, action/replan order, terminal behavior, reset of reused
state objects, malformed outputs, selector completion, immutable frame copies,
preprocessing count and current-versus-historical image routing. Neural-boundary
wiring tests use mocks; they are not real-model numerical parity measurements.

Together with the original algorithm/source/compaction/attention/qualification
tests, 65 tests pass with CUDA hidden and no skips. Authenticated sources and
checkpoint identities from the completed real-checkpoint configuration are
verified unchanged. Machine-readable evidence and test log are in
`reports/specprune_episode_integration_v1/`.

## Remaining work, in order

1. Freeze a bounded live integration check using already consumed development
   observations/conditions. Check the new dense query bridge against native
   actions, consecutive compressed queries with changing frames, persistent
   within-episode indices/confidence, and reset across episodes. Run the real
   simulator bridge with the original horizons; no task tuning during the check.
   Freeze calls/episode limits, identities, tolerances, resource caps and stop
   rules in the actual worker configuration before launch. A technical failure
   stops without retry and remains immutable evidence.
2. Freeze the matched development comparison only after that integration check.
   Include all forty consumed tasks and the known dense failure, balanced arm
   order, both task success and full cost, plus a separate fixed same-observation
   timing sample. Declare pilot decision criteria before collecting outcomes.
   These consumed conditions are development data, not held-out confirmation.
3. Return to the new learned action-correction hypothesis after establishing a
   credible comparator. Comparator qualification is not a novel contribution or
   a positive-result paper. No adapter has been trained by this integration.

No GPU experiment, new task-success measurement, training, installation or
GitHub publication occurred in this implementation step. TITAN writes remain
inside `/home/ved/SAVR`; local evidence is synced to the project repository.
