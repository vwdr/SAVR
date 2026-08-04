# Project Status

Last updated: 2026-08-04

## Current phase

ACR Version 3 Phase V3-D — **IN PROGRESS: TECHNICAL RECOVERY PREFLIGHT**

The original whole-prefix SAVR program remains stopped negative. Its evidence
is preserved in `docs/NEGATIVE_RESULTS_PAPER_ARCHIVE.md` and the machine-readable
companion `docs/evidence/negative_results_summary.csv`. A materially different
Asymmetric Camera Refresh route is proposed in
`docs/ASYMMETRIC_CAMERA_REFRESH_PROPOSAL.md`.

`docs/ACR_EXECUTION_PROTOCOL_V1.md` now defines the staged novelty audit,
implementation, correctness, development, independent confirmation, matched
baselines, transfer, power, final evaluation, resource, and stop rules. Its
creation authorizes no ACR code, GPU run, simulator outcome, final-holdout
access, or manuscript modification. The user authorized Phase A0 on
2026-08-02. Phase A0 completed its primary-source/code research, pinned-source
inspection, split/resource audit, and exact implementation design without an
ACR implementation or outcome. The novelty gate passes only for the narrowed
camera-block contribution recorded in `docs/ACR_NOVELTY_AUDIT.md`. The pinned
source boundary and bitwise-parity risk are recorded in
`docs/ACR_IMPLEMENTATION_DESIGN.md`; all protected populations remain
untouched according to `docs/ACR_SPLIT_AND_RESOURCE_AUDIT.md`. The user
authorized Phase A1 on 2026-08-02. A1 accepted the protocol without amendment
and froze formal record schemas, run identities, recovery rules, artifact
policies, and bounded phase estimates in
`docs/ACR_PHASE_A1_RESOURCE_FREEZE.md` and
`configs/acr/phase_a1_freeze.json`. It used no model, GPU, simulator, ACR
population, or manuscript. The user authorized Phase A2 on 2026-08-02. A2
implemented the separate project-owned camera-factorized adapter, ACR
controller/signals/cache, synchronized camera accounting, compact immutable
FR records, deterministic candidate derivation, and frozen statistics. All 133
repository tests pass, including 31 new ACR tests; source/test Ruff, source
mypy, bootstrap validation, and package build pass. No model, GPU, simulator,
ACR outcome, data-derived threshold, or manuscript was used. The user
authorized A3 on 2026-08-02. Its predeclared 12-query synthetic-input matrix
passed every scientific correctness assertion: exact factorized-FR projected
tokens/actions, camera-block isolation, zero scene tower/projector work on
reuse, one fresh wrist path, current-state downstream parity, and fail-closed
shape/dtype/device/context handling. The original attempt is preserved as
technically failed because the pinned loader's temporary checkpoint rewrites
were audited before restoration. The files were restored to their accepted
hashes and a committed CPU-only adjudication passed without any additional
model query. Total use was 12/16 queries, one GPU, zero simulator resets, zero
episodes, no benchmark population, and no download. The user authorized A4 on
2026-08-02. A4 is limited to one upstream-FR pass over the 100 LIBERO-Object
development episodes, the frozen feasibility gate, and two byte-identical
derivations of exactly three candidates. No ACR rollout is authorized in A4.
The run completed 100/100 terminal episodes with 97 successes, every task at
or above 8/10, 1,773 reconciled queries/traces, and zero technical failures.
Both the original CPU analysis and a transparently preserved recovery produced
the same candidate record byte-for-byte. The three eligible candidates are
frozen in `configs/acr/candidates.json`; no ACR method outcome has yet been
observed. Full evidence is in `reports/PHASE_A4_REPORT.md`.
The user authorized A5 on 2026-08-03. A5 first runs exactly the three frozen
candidates on Object states `0-2` and applies the 30/30 success, per-task,
15% scene-reuse, and invariant gates. Object states `3-9` remain unavailable
until the committed analyzer mechanically identifies an advancing candidate.
No ACR outcome had been observed when the A5 preflight was frozen. Stage 1
subsequently completed all 90 episodes with zero technical failures. The three
candidates achieved 29/30, 24/30, and 23/30 success with 26.06%, 47.40%, and
49.44% scene reuse, respectively. Every camera-component invariant passed,
but every candidate failed the exact success and per-task gates. The committed
analyzer selected no advancing candidate, so Stage 2 was not run and A5
stopped negative. A6 is ineligible and was not started. Full evidence is in
`reports/PHASE_A5_REPORT.md`.

The user then approved a new positive-paper planning route. Phase V2-A used
only immutable A4/A5 records and primary sources. It corrected the conservative
failure identity to Object task 6/state 0 and found that its first-reuse
direction-reversal pattern also occurred in ten successful episodes; both more
aggressive candidates succeeded on that same state. No single signal is
therefore treated as causal. The stronger redesign target is measured execution
overhead: the conservative controller reduced visual CUDA time by 11.94% but
increased synchronized query wall time by 31.24% versus matched FR.

`docs/ACR_V2_EXECUTION_PROTOCOL.md` and `configs/acr/v2_freeze.json` freeze
State-Aware Dual-Path ACR. It retains the exact conservative controller, uses
the original upstream two-view path on refresh and a wrist-only path on reuse,
and requires bitwise correctness plus a paired latency gate before any rollout.
No Version 2 implementation, model query, simulator episode, new population,
GPU use, protected outcome, or manuscript edit occurred in V2-A.

The user authorized Phase V2-B on 2026-08-03. The separate episode-scoped
dual-path adapter now calls the exact original two-view method on refresh and
returns its original tensor unchanged; reuse executes only a fresh wrist path
and combines it with the owned cached scene block. It separately accounts for
physical calls and logical camera work, rejects nested/concurrent use, restores
the original method on every episode exit, applies structural checks in the
production path, reserves full projected finite scans for correctness mode,
and always validates the returned action chunk. All 172 repository tests plus
9 TITAN subtests pass; the 14 new V2-B tests pass locally and in TITAN's pinned
CPU environment. Ruff, formatting, mypy, compilation, bootstrap, and package
build checks pass. No GPU/model/simulator/download/outcome/manuscript work was
performed. Full evidence is in `reports/PHASE_V2_B_REPORT.md`.

The user authorized Phase V2-C on 2026-08-03. The exact six-query correctness,
six-query warm-up, and 36-query counterbalanced timing schedule is frozen in
`configs/acr/v2_c_gate.json` and `docs/ACR_V2_C_PREFLIGHT.md`. The bounded
runner and its CPU gate/recovery tests pass as part of 182 repository tests.
V2-C remained limited to 48 model queries, one GPU/process, zero simulator
resets or episodes, 3,600 seconds, 512 MiB, and no download or
protected-population access.

The first immutable V2-C attempt stopped technically after 7/48 queries. All
six correctness checks completed, followed by one upstream-FR warm-up. The
runner then rejected the valid low-level call truth—two SigLIP and two DINOv2
calls, one per camera—because it expected one of each. No timed sample or
method result was accepted. Checkpoint metadata, both pinned source trees, and
GPU 0 restored cleanly. `docs/ACR_V2_C_RECOVERY_PLAN.md` freezes one 41-query
recovery: the remaining five warm-ups and all 36 timed queries. Cumulative use
will be exactly 48/48; the method, counterbalance, and gates are unchanged.

The recovery completed all remaining 41 queries, bringing cumulative V2-C use
to exactly 48/48. All correctness assertions had passed by preserved control
flow, all 36 timed records and work invariants reconciled, and no outlier was
removed. Reuse halved median visual CUDA time (`150.566` to `75.104` ms), but
dual refresh, dual reuse, and fixed-weight expected wall ratios were `1.40338`,
`1.42995`, and `1.41030`; all exceed their frozen limits. V2-C therefore
stopped negative and V2-D is ineligible. Full evidence is in
`reports/PHASE_V2_C_REPORT.md` and
`reports/runtime/acr_v2_c_recovery.json`.

The user then authorized V3-A as a new diagnosis/protocol phase. A deterministic
bound from immutable V2-C evidence shows that perfect zero-overhead removal of
all measured scene-camera visual work at the fixed 26.055% reuse weight could
reduce weighted wall time by at most `1.6202%`, below the unchanged `2%` paper
gate. Merely removing audit hashing cannot establish a viable new mechanism.

Read-only inspection of pinned OpenVLA-OFT revision
`e4287e94541f459edc4feabc4e181f537cd569a8` confirmed that the two-camera
backbone loops sequentially over scene and wrist, invoking SigLIP and DINOv2
twice. `docs/ACR_V3_EXECUTION_PROTOCOL.md` and
`configs/acr/v3_freeze.json` therefore freeze State-Aware Batched Dual-Path
ACR: batch the two ordered cameras within each tower on refresh, retain the
exact conservative controller, and use the established fresh-wrist reuse
path. Batched Full Refresh is a required ablation so generic batching gains
cannot be attributed to camera reuse. The bounded gate requires predeclared
bfloat16 token closeness, bitwise actions, truthful work, at least 2% weighted
wall acceleration versus upstream FR, at least 10% weighted visual reduction,
and no weighted regression versus Batched FR.

V3-A used no implementation, GPU, model query, simulator episode, new outcome,
download, protected population, or manuscript edit. Full evidence is in
`reports/ACR_V3_DIAGNOSIS_REPORT.md` and
`reports/runtime/acr_v3_feasibility.json`.

The user authorized V3-B on 2026-08-04. Separate Batched Full Refresh and
SA-BDP-ACR adapters now implement the frozen scene-then-wrist tower batches,
single combined projection, owned scene cache, and established V2 wrist-only
reuse path. The production boundary excludes audit hashing, serialization,
file I/O, and projected-token scans; both adapters fail closed on structural,
action, cache, nesting, concurrency, and restoration errors. All 206 repository
tests plus 9 subtests pass in TITAN's pinned CPU environment, including 18 new
V3-B tests and six real-PyTorch CPU assertions. No GPU, checkpoint/model query,
simulator, download, benchmark outcome, protected population, or manuscript
change occurred. Full evidence is in `reports/PHASE_V3_B_REPORT.md`.

The user authorized V3-C on 2026-08-04. Its exact eight correctness queries,
eight warm-ups, 48 timed queries, four-path cyclic counterbalance, numerical
tolerance, latency gates, resource caps, and fail-closed recovery identities
are frozen in `configs/acr/v3_c_gate.json` and
`docs/ACR_V3_C_PREFLIGHT.md` before any real-model execution. V3-D remains
unauthorized regardless of the V3-C result.

V3-C subsequently completed exactly 64/64 model queries on one TITAN RTX with
zero simulator episodes. Both deterministic inputs produced token-exact BFR
and V3-refresh outputs and bitwise-identical actions; V3 reuse was bitwise
identical to V2 reuse and the sequential oracle. All six frozen latency gates
passed. At the fixed reuse weight, V3 achieved a `0.966582` wall ratio versus
sequential FR, a `0.997525` wall ratio versus BFR, and a `31.4092%` visual CUDA
reduction. This is the first positive method result, not yet a task-success or
paper-level result. All 216 repository tests plus 9 subtests and the independent
result reconciliation pass. Full evidence is in `reports/PHASE_V3_C_REPORT.md`.

The user authorized V3-D on 2026-08-04. The exact Object tasks `0-9`, states
`3-9`, seed `0` population is now frozen as 70 adjacent BFR/V3 pairs and 140
total attempts. Pair order alternates deterministically, giving each policy 35
first positions. `configs/acr/v3_d_development.json` and
`docs/ACR_V3_D_PREFLIGHT.md` freeze outcome-blind execution, immutable
records, no automatic retry, the historical A4 sequential-FR latency source,
all success/reuse/visual/wall/restoration gates, and the one-GPU, 12-hour,
2-GiB, no-download resource boundary. The runner and independent analyzer are
implemented and all 221 local tests pass. No V3-D simulator outcome has been
opened; the pre-execution checkpoint must merge and synchronize before launch.

The first V3-D BFR start then stopped technically before a completed query or
action execution because the pinned evaluator returned a valid action list to
the adapter's tensor-only default finite checker. The immutable attempt has
zero completed query records and no opened success outcome; checkpoint/source
restoration passed and GPU 1 returned idle. The narrow correction supplies an
explicit list/NumPy finite checker outside timing. A one-time full recovery
preserves the failed start, changes no scientific design or gate, and raises
only the cumulative episode-start allowance from 140 to 141. It is frozen in
`configs/acr/v3_d_recovery.json` and
`docs/ACR_V3_D_TECHNICAL_RECOVERY.md` pending merge/sync/CPU verification.

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
- ACR A5 Stage 1 completed all 90/90 frozen development episodes with zero technical failures
- all three ACR candidates passed the reuse and instrumentation gates but failed the exact success and per-task gates
- ACR Stage 2 and A6 were not run because the frozen analyzer selected no advancing candidate
- ACR Version 2 diagnosis reproduced the A5 failure and latency evidence with a deterministic machine record
- the SA-DP-ACR controller, dual execution paths, independent splits, gates, resources, and stop rules are frozen
- the separate SA-DP-ACR adapter and 14-test CPU matrix are implemented without changing Version 1
- all 172 repository tests plus 9 TITAN subtests and the static/build/bootstrap gates pass
- separate Batched Full Refresh and SA-BDP-ACR V3 adapters are implemented
- all 206 repository tests plus 9 TITAN subtests and six real-PyTorch CPU assertions pass
- V3-C completed exactly 64/64 bounded real-model queries with all correctness and latency gates passing

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
- no ACR candidate is eligible for Stage 2 or independent A6 confirmation
- no SA-DP-ACR rollout began because its completed correctness/latency gate stopped negative
- no SA-BDP-ACR closed-loop rollout or task-success evaluation has begun

## Next authorized action

Merge and synchronize the frozen V3-D technical recovery, verify it in TITAN's
pinned CPU environment, select one idle GPU from aggregate device telemetry,
and execute the unchanged 140-attempt outcome-blind paired Object matrix under
the recovery run ID. Analyze success only after all terminal records exist.
V3-E, Goal, and final-holdout access remain unauthorized. V2-D remains
ineligible and V2-C may not be rerun or reinterpreted.

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
