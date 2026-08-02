# Project Status

Last updated: 2026-08-02

## Current phase

ACR planning — **MASTER EXECUTION PROTOCOL PREPARED FOR REVIEW; NO ACR
IMPLEMENTATION OR OUTCOME PHASE IS ACTIVE**

The original whole-prefix SAVR program remains stopped negative. Its evidence
is preserved in `docs/NEGATIVE_RESULTS_PAPER_ARCHIVE.md` and the machine-readable
companion `docs/evidence/negative_results_summary.csv`. A materially different
Asymmetric Camera Refresh route is proposed in
`docs/ASYMMETRIC_CAMERA_REFRESH_PROPOSAL.md`.

`docs/ACR_EXECUTION_PROTOCOL_V1.md` now defines the staged novelty audit,
implementation, correctness, development, independent confirmation, matched
baselines, transfer, power, final evaluation, resource, and stop rules. Its
creation authorizes no ACR code, GPU run, simulator outcome, final-holdout
access, or manuscript modification. After user acceptance, the next eligible
step is ACR Phase A0 only.

### Legacy SAVR terminal state

Phase 6S-D — SAVR3 fresh development validation: **STOPPED NEGATIVE
(POSITIVE GATE NOT MET; PHASE 6S-E INELIGIBLE)**

Phase 0 completed after preparation PR #1 and ledger PR #2 were explicitly approved, merged, and synchronized. Phase 1 proposal PR #3 was subsequently approved and merged as `50eabfac111f65995ce515926aaa291d345c3cf2`.

The original Phase 6 remains stopped negative and unchanged. The user chose a
scientifically controlled redesign path and approved Phase 6R-A. A reproducible
offline diagnosis of the existing Phase 6 records found that FR-replay targets
underpredicted online reuse, two-camera averaging concealed frequent
wrist-camera changes, earlier reuse was associated with lower success, and
two-query reuse streaks remained unsafe. No new rollout or final-holdout
outcome was executed during Phase 6R-A.
The Phase 6R-A checkpoint was accepted under the phase authorization and
merged in PR #17 at `5d2f69038b76bf94d94bbabefb92b0aa91df72dc`.
On 2026-07-31 the user authorized all remaining Phase 6 work. Phase 6R-B
reviewed primary VLA-efficiency sources and froze the training-free SAVR 2.0
controller, staged calibration, comparison, power, resource, and stop rules in
`docs/PHASE6R_PROTOCOL_V1.md` before implementation or new outcomes.
The separate SAVR 2.0 signal/controller path and bounded Phase 6R-C runner are
now implemented. All `88` CPU tests pass, including local camera vetoes,
grouped signals, transition/temporal/budget rules, immutable records, adapter
reuse, schema compatibility, and the unchanged SAVR 1.0 suite.
The first bounded real-model run preserved exact FR parity and component
invariants but correctly vetoed the intended reuse because its initial-state
action chunk contained a gripper transition. The failed run remains immutable.
A no-simulator recovery using a hashed pre-existing FR trace is frozen in
`reports/PHASE6R_C_CORRECTNESS_RECOVERY_PLAN.md`; it keeps cumulative Phase
6R-C usage within `18/20` model queries.
The recovery subsequently passed all eight queries: the real reuse invoked
zero vision-backbone/projector calls, used current proprioception, and exactly
matched unmodified actions for the same input. Full evidence and hashes are in
`reports/PHASE6R_C_CORRECTNESS_REPORT.md`.
Phase 6R-D then deterministically derived `b05`, `b10`, and `b15` from all
1,309 existing FR queries. Repeated derivations were byte-identical. The
tracked Stage 1 matrix contains 90 maximum episodes and is frozen in
`configs/calibration/phase6r_d_stage1.json`.
Stage 1 completed all 90 episodes with zero technical failures. `b05` preserved
30/30 success but skipped 0%; `b10` achieved 6.72% skip with 29/30 success;
`b15` achieved 10.57% skip with 27/30 success. No candidate passed both frozen
gates, so Stage 2 and Phase 6R-E were not run.
The user subsequently authorized uninterrupted Phase 6S work through the first
predeclared positive method result. Phase 6S-A used only immutable Stage 1
records to localize the four failed trajectories. The disclosed post-hoc design
selects one final SAVR3 candidate: `b15` with a translation-direction-reversal
veto and wrist threshold `0.375`. Its implementation and one-shot 70-episode
states-`3-9` validation were frozen in `docs/PHASE6S_PROTOCOL_V1.md`.
SAVR3 implementation and correctness are complete: all 102 CPU tests and the
changed-file static checks pass. The shared, previously validated model/cache
adapter is unchanged. The frozen 70-episode configuration has semantic SHA-256
`10b93d3247f6bec35c7419e362627dffef597ddbcd5dd71f9509a6b66bb52289`.
The frozen SAVR3 run subsequently completed all 70 episodes with 69 successes,
944 queries, nine valid visual reuses (0.9534% skip), zero technical or
invariant failures, and exact checkpoint restoration. Task 9/state 4 was the
only unsuccessful episode. SAVR3 missed both the 70/70 success gate and the 5%
skip gate, so the negative stop applies and Phase 6S-E is ineligible.

The bounded Phase 1 installation and CPU-only simulator smoke test passed. The
Phase 2A checkpoint and Full Refresh smoke were accepted and merged in PR #6.
The approved Phase 2B all-task pilot is complete and passed its predeclared
feasibility threshold. Its evidence was accepted and merged in PR #8. All
technical Phase 2 exit criteria are satisfied. The user approved the Phase 3
transition. The bounded implementation and CPU tests were accepted and merged
in PR #10. All technical Phase 3 exit criteria are satisfied. The Phase 4
proposal was approved. Its expanded CPU suite and bounded six-query
real-model correctness matrix passed. The evidence was accepted and merged in
PR #13. All Phase 4 exit criteria are satisfied. The user subsequently
approved uninterrupted Phase 5 execution through its final checkpoint. The
bounded core-policy smoke and official VLA-Cache compatibility audit passed
their technical exit gates and were accepted in PR #15. The user authorized
uninterrupted Phase 6 execution through its final checkpoint. The Phase 6
calibration and power rules are frozen before outcome collection in
`docs/PHASE6_CALIBRATION_PROTOCOL.md`. FR subsequently completed all 100
calibration episodes successfully. None of the nine frozen SAVR candidates met
the 2-percentage-point success constraint; the best achieved `52/100`.
Accordingly, the frozen protocol stopped before matched baselines, power
confirmation, Phase 7, or any final-holdout execution.

## Research objective

Evaluate whether State-Aware Visual Refresh (SAVR), a training-free wrapper around an existing VLA policy, can reduce repeated visual encoding while preserving task success.

## Proposed decision rule

A visual refresh is triggered when at least one of the following exceeds its configured threshold or limit:

1. image change
2. robot-state change
3. recent action change
4. maximum visual-feature reuse horizon

Otherwise, cached visual features are reused.

## Required comparison policies

- **FR — Full Refresh:** recompute visual features every control step.
- **PR — Periodic Refresh:** recompute on a fixed interval.
- **VOR — Visual-Only Refresh:** use image change plus a maximum horizon.
- **SAVR:** use image, robot-state, recent-action, and maximum-horizon signals.

## Current completion state

Completed:

- SAVR concept and mathematical decision rule
- comparison-policy definitions
- intended metrics
- repository safety and reproducibility rules
- initial experiment plan and machine-readable result contracts
- bounded TITAN hardware/bootstrap diagnostics
- original LaTeX manuscript added unchanged with a recorded SHA-256 checksum
- preparation PR #1 merged with explicit approval
- authoritative `main` synchronized across GitHub, TITAN, and the local review copy
- milestone and decision ledgers prepared for review
- Phase 0 ledger PR #2 merged and synchronized
- official Phase 1 requirements, revisions, and asset sizes verified
- bounded Phase 1 resource proposal prepared
- Phase 1 resource proposal approved by the user
- project-local Micromamba/Python/OpenVLA-OFT/LIBERO environment installed
- exact Conda and pip inventories recorded
- OpenVLA-OFT imports verified without loading a model
- one CPU-only OSMesa LIBERO-Spatial render/action smoke test passed
- actual Phase 1 project storage measured at about `14.70 GiB`, below the `25 GiB` cap
- the empty `/home/ved/.libero` artifact from the interrupted upstream prompt was narrowly inspected, removed with `rmdir`, and verified absent
- Phase 1 accepted and completed
- pinned combined four-suite checkpoint revision and all 25 declared file sizes verified
- Phase 2A checkpoint storage remained within the approved additional-storage cap
- GPU 0 selected using only user-authorized aggregate utilization/memory evidence
- one unmodified Full Refresh LIBERO-Spatial task 0 / initial-state 0 / seed 0 episode completed successfully
- model fit one TITAN RTX with about `14.98 GiB` peak allocated memory
- checkpoint metadata restored byte-for-byte after the official local loading path
- approved Phase 2B pilot completed all `50/50` planned calibration episodes
- Full Refresh achieved `49/50` successes, with at least `4/5` on every task and no runtime errors
- steady-state visual backbone plus projector time measured `15.874%` of total query CUDA time and `15.873%` of policy-query wall time
- the Phase 2B run, component timing, resource use, and safety audit were reconciled in `reports/PHASE2B_PILOT_REPORT.md`
- Phase 2 accepted and completed after the explicit Phase 3 go decision
- pinned OpenVLA-OFT source boundary reconciled with the protocol: projected visual embeddings precede the fresh proprioception token
- bounded Phase 3 implementation design recorded in `docs/PHASE3_IMPLEMENTATION_DESIGN.md`
- common FR, PR, VOR, and SAVR controller interface implemented
- exact two-camera image, normalized state, and action-history signals implemented
- context-safe projected visual-feature cache and exception-safe OpenVLA-OFT adapter implemented without upstream edits
- immutable query and episode record storage implemented
- `29/29` CPU tests passed locally and in TITAN's pinned OpenVLA-OFT environment
- Ruff, mypy, byte compilation, package build, and upstream-cleanliness checks passed
- Phase 3 implementation evidence accepted and merged in PR #10
- bounded Phase 4 correctness, parity, timing, recovery, GPU, and stop-rule proposal prepared
- Phase 4 proposal explicitly approved and merged in PR #12
- expanded Phase 4 controller/cache/timing/recovery suite passed `44/44` tests in TITAN
- wrapped Full Refresh actions matched unmodified upstream exactly on both controlled queries
- real VOR reuse skipped the visual backbone/projector, preserved fresh state-B proprioception, and matched the unmodified state-B action chunk exactly
- all six immutable query records and the run manifest validated
- checkpoint metadata restored byte-for-byte and the selected GPU returned to its exact pre-run aggregate idle state
- Phase 4 correctness evidence accepted and merged in PR #13
- Phase 4 administratively closed after the explicit Phase 5 go decision
- bounded Phase 5 protocol frozen in `docs/PHASE5_SMOKE_PROTOCOL.md`
- all 12 fixed Phase 5 core-policy episodes completed with 283 reconciled query records
- FR refreshed on all 31 queries and succeeded on all three diagnostic states
- PR completed with 42 refreshes/42 reuses; VOR and SAVR each completed with 30 refreshes/54 reuses
- all reuse queries skipped both visual components while preserving complete downstream execution
- diagnostic PR/VOR/SAVR settings reached the horizon without task success, establishing that aggressive uncalibrated reuse is unsafe
- official VLA-Cache source and its required Transformers fork pinned in an isolated project-local environment
- official VLA-Cache evaluator technically excluded because its previous-frame path aliases the current frame and its episode path suppresses explicit errors
- Phase 5 independent evidence reconciliation passed
- Phase 5 evidence accepted and merged in PR #15 at merge commit `5a4046b2b689d71e2ef0a54a6b67629180d5cdd3`
- Phase 6 authorized through its final checkpoint
- Phase 6 calibration population, nine-setting SAVR grid, 2-percentage-point margin, paired power target, matching rules, resource limits, and holdout protections frozen before new outcome collection
- Phase 6 FR signal collection completed `100/100` episodes with `100/100` successes, `1,309` immutable query traces, and no infrastructure errors
- nine SAVR threshold/horizon settings were derived from the frozen FR trace hash before online evaluation
- the full SAVR grid completed `900/900` terminal episodes with no infrastructure errors and exact checkpoint restoration
- no SAVR candidate was eligible: the best setting, `savr-s25-h2`, achieved `52/100` successes and a paired difference of `-48` percentage points versus FR
- the predeclared negative-result stop rule was applied without relaxing thresholds, the margin, or the split
- matched VOR/PR runs, final power confirmation, Phase 7, and final-holdout execution were not performed
- the frozen SAVR3 validation completed 70/70 terminal episodes with 69 successes and 9/944 visual reuses
- all nine SAVR3 reuses exactly skipped both visual components, with zero technical or invariant errors
- SAVR3 failed the frozen 70/70 success and 5% skip gates, so no tuning or rerun was performed

Not completed:

- no dataset has been downloaded
- no eligible primary SAVR configuration exists
- no matched-budget VOR/PR calibration was run because the SAVR eligibility stop rule fired
- no paired final sample size was confirmed
- no positive SAVR performance claim is supported
- no Phase 5 success or latency comparison is approved as a paper-level claim
- no final holdout outcome has been inspected
- no Phase 6R-A GPU or simulator run was performed
- no SAVR3 matched-baseline or confirmatory evaluation is eligible

## Next authorized action

No further experiment is authorized by the frozen Phase 6S protocol. Preserve
the negative result, do not tune or rerun SAVR3, and never inspect the final
holdout. A new attempt would require a materially different method and new
independent development evidence.

## Candidate initial stack

OpenVLA-OFT with LIBERO remains the leading stack. Model loading, one-GPU
compatibility, baseline feasibility, cache interception, exact wrapped-FR
parity, real-tensor reuse, and complete trajectories through all four core
controllers are verified. The Phase 6 online grid found severe closed-loop
success degradation for every tested reuse setting. The current SAVR operating
region is therefore not eligible for final evaluation.

## Bootstrap checkpoint

Bootstrap was accepted on 2026-07-29 after:

- the private GitHub repository exists
- the repository is present at `/home/ved/SAVR`
- required context, safety, plan, schema, CI, diagnostic, and next-handoff files validate
- the initial commit is pushed
- no SAVR implementation or large experiment has begun
- missing inputs and risks are explicitly reported
- the final report confirms no modification outside `/home/ved/SAVR` on TITAN
