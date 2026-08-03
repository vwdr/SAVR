# SAVR Milestones

Last updated: 2026-08-02

Exactly one phase may be `IN_PROGRESS`.

| Phase | Status | Completion evidence | Next gate |
|---|---|---|---|
| 0. Establish authoritative state | COMPLETE | Preparation PR #1 and ledger PR #2 merged with explicit approval; GitHub, TITAN, and local `main` synchronized at `9a5bbd5`; manuscript SHA-256 verified as `4a0fe130f1cbc5557f77a518dcb65a703a647b1c4b8091499d8bfd8e10ab6e4f`; worktrees clean | — |
| 1. Environment and storage feasibility | COMPLETE | Installation, dependency locks, imports, and CPU-only OSMesa LIBERO smoke test passed; `reports/PHASE1_REPORT.md`; empty account-path artifact identified and reversed with user authorization | — |
| 2. Unmodified FR reproduction | COMPLETE | Checkpoint fit one TITAN RTX; 50/50 pilot episodes completed with 49 successes; component timing accepted and merged in PR #8; explicit go decision received | — |
| 3. Controller and cache implementation | COMPLETE | Controller, signals, cache, adapter, and immutable records implemented; 29 CPU tests plus Ruff/mypy/package checks passed; accepted in PR #10 | — |
| 4. Correctness and instrumentation | COMPLETE | 44 TITAN tests passed; six-query run completed with exact FR/reuse parity, zero visual reuse calls, fresh proprioception, valid immutable records, and exact checkpoint restoration; accepted and merged in PR #13; explicit Phase 5 go decision received | — |
| 5. Smoke policies and external-baseline feasibility | COMPLETE | 12/12 episodes and 283/283 queries reconciled; all four policies finished; official VLA-Cache technically excluded with pinned evidence; accepted and merged in PR #15 at `5a4046b` | — |
| 6. Calibration and power | STOPPED_NEGATIVE | FR completed 100/100 successes; nine SAVR settings completed 900/900 episodes with no infrastructure errors; best SAVR result was 52/100, so none met the frozen 2-point constraint; negative stop rule applied | User decision on ending or predeclaring a materially more conservative protocol revision |
| 6R-A. Forensic diagnosis | COMPLETE | Existing Phase 6 artifacts reconciled; closed-loop skip overshoot, concealed wrist-camera changes, unsafe early/consecutive reuse, and threshold-margin weaknesses documented in `reports/PHASE6R_A_DIAGNOSIS_REPORT.md` | User approval for Phase 6R-B |
| 6R-B. Redesign and protocol | COMPLETE | Primary-source review completed; training-free SAVR 2.0 semantics, staged calibration, baselines, power, resources, and stop rules frozen in `docs/PHASE6R_PROTOCOL_V1.md` | Phase 6R-C correctness gates |
| 6R-C. Implementation and correctness | COMPLETE | 88 CPU tests pass; first fixture correctly vetoed a gripper transition; predeclared recovery passed 8/8 queries with exact action parity, current proprio, and zero visual calls on reuse | Phase 6R-D Stage 1 |
| 6R-D. Conservative staged calibration | STOPPED_NEGATIVE | 90/90 terminal episodes, zero technical failures; b05 30/30 at 0% skip, b10 29/30 at 6.72% skip, b15 27/30 at 10.57% skip; no candidate advanced | New user-authorized protocol, if any |
| 6R-E. Baselines, selection, and power | NOT_RUN_INELIGIBLE | Phase 6R-D selected no eligible candidate, so the frozen prerequisite failed | Eligible candidate from a future predeclared method |
| 6S-A. Final forensic localization | COMPLETE | Reproducible design-split analysis localized all four first unsafe reuse events and disclosed the wrist-cap grid | Freeze exactly one SAVR3 design |
| 6S-B. SAVR3 protocol | COMPLETE | `docs/PHASE6S_PROTOCOL_V1.md` frozen before implementation or outcomes | SAVR3 correctness gates |
| 6S-C. SAVR3 implementation and correctness | COMPLETE | 102 CPU tests and all changed-file static checks pass; distinct policy/veto, frozen config, and independent gate validator implemented | Frozen states-3-9 validation |
| 6S-D. Fresh development validation | STOPPED_NEGATIVE | 70/70 terminal, 69/70 success, 9/944 reuses (0.95%), zero technical/invariant failures; both success and skip gates missed | Materially different method with new independent evidence |
| 6S-E. Matched comparisons and power | NOT_RUN_INELIGIBLE | SAVR3 did not pass the frozen Phase 6S-D positive gate | Eligible candidate from a future predeclared method |
| ACR-P. Proposal and master protocol | COMPLETE | Negative evidence archived; ACR method proposal and `docs/ACR_EXECUTION_PROTOCOL_V1.md` prepared without ACR outcomes; Phase A0 authorized | — |
| A0. ACR novelty, source, and split audit | COMPLETE | Narrow novelty distinction, pinned camera factorization, 40-task/50-state mapping, historical split ledger, and exact implementation design verified in the A0 audit/report set; no ACR outcome | — |
| A1. ACR protocol/resource acceptance | COMPLETE | Protocol accepted without amendment; schemas, run IDs, recovery rules, artifact policy, and bounded resource estimates frozen; no ACR outcome | — |
| A2. ACR implementation and CPU verification | COMPLETE | Separate adapter/controller/cache/signals, camera accounting, immutable compact records, deterministic candidate/statistical utilities; 133 repository tests including 31 ACR tests plus static/build checks pass; no model/GPU/simulator/outcome | — |
| A3. ACR bounded real-model correctness | COMPLETE | All scientific proofs passed in 12/16 queries; exact token/action parity, camera isolation, reuse/fresh-wrist/current-state, and fail-closed checks; original technical failure preserved; loader changes restored and CPU adjudication passed with no rerun | — |
| A4. ACR development FR/candidate freeze | IN_PROGRESS | User authorization received; exact 100-episode upstream-FR Object population, feasibility gate, resources, recovery, and derivation rules frozen before outcome access | 100 reconciled FR terminals, feasibility pass, and byte-identical candidate derivations |
| A5. ACR staged development | NOT_STARTED | — | Frozen candidates and explicit development authorization |
| A6. ACR independent Goal confirmation | NOT_STARTED | — | One frozen eligible method and explicit confirmation authorization |
| A7. ACR baselines, transfer, and power | NOT_STARTED | — | Positive confirmation and explicit Phase A7 authorization |
| A8. ACR primary four-suite final evaluation | NOT_STARTED | — | Frozen final package and separate final-holdout authorization |
| A9. ACR evidence/manuscript decision | NOT_STARTED | — | Complete final classification |
| 7. Freeze final protocol | NOT_STARTED | — | User approval of `PROTOCOL_V1.md` |
| 8. Final evaluation | NOT_STARTED | — | Complete reconciled final-run registry |
| 9. Ablations and sensitivity | NOT_STARTED | — | Required confirmatory ablations complete |
| 10. Analysis and claim audit | NOT_STARTED | — | Every manuscript claim mapped to evidence |
| 11. Manuscript completion | NOT_STARTED | — | User-approved evidence-based manuscript changes |

## Active milestone

ACR Phase A4 is the only active phase. It may open exactly the 100-episode
LIBERO-Object development FR population and then apply the frozen feasibility
and deterministic candidate-derivation gates. No ACR rollout is authorized.
All A3 scientific proofs passed exactly within 12/16 queries, and the original
technical failure remains immutable with no model-query rerun.
The prior SAVR Phase 6S-D remains stopped negative, Phase 6S-E is ineligible,
and legacy Phase 7 is unauthorized. Initial-state `10-49` / seed `7,17,27`
outcomes remain untouched across all four supported suites.

## Phase 6R-B current checklist

- [x] Record blanket Phase 6R-B through 6R-E authorization.
- [x] Review current primary VLA caching, action-chunking, and dynamic-compute
  evidence.
- [x] Preserve the training-free claim and validated cache boundary.
- [x] Freeze local per-camera, grouped state/action, transition, temporal, and
  online-budget semantics.
- [x] Freeze staged candidate promotion and negative stop rules.
- [x] Freeze comparison, ablation, efficiency, power, and resource rules.
- [x] Reconfirm the official VLA-Cache technical exclusion.
- [x] Publish, merge, and synchronize the Phase 6R-B checkpoint in PR #19 at
  `f19bd9b325eff5ca08ac59b68e731e9be4f36967`.

## Phase 6R-C current checklist

- [x] Implement SAVR 2.0 without changing SAVR 1.0.
- [x] Implement independent local camera, grouped state/action, and transition
  signals.
- [x] Implement warm-up, stable-fresh, isolated-reuse, and prefix-budget rules.
- [x] Preserve immutable full decision records and cache fail-closed behavior.
- [x] Pass all CPU tests, Ruff, mypy, compilation, and diff checks.
- [x] Publish and synchronize the implementation checkpoint in PR #20 at
  `2a2c7226e39c05667770811eaa9f98cd2d4c635c`.
- [x] Preserve the first bounded run and classify its transition veto without
  weakening the controller.
- [x] Freeze a hashed existing-trace recovery within the cumulative query cap.
- [x] Publish and synchronize the recovery-plan checkpoint in PR #21 at
  `fa7a7d04c0ec544066a5eba908cc2fec147dbbde`.
- [x] Pass the bounded recovery correctness matrix.
- [x] Publish, merge, and synchronize the Phase 6R-C evidence checkpoint in
  PR #22 at `59c1d97aa1ca90b9098b01d68a3c5a2f06cf6051`.

## Phase 6R-D current checklist

- [x] Implement exact adjacent-signal distributions and linear quantiles.
- [x] Implement exact temporal and episode-prefix-budget replay.
- [x] Unit-test quantiles, warm-up, isolated reuse, budget, and transition veto.
- [x] Derive and audit `b05`, `b10`, and `b15` from all 100 FR traces.
- [x] Publish and synchronize the frozen Stage 1 configuration in PR #24 at
  `5e577debd5161f2dc0303615b87f64cac795f58d`.
- [x] Run and reconcile the 30-episode-per-candidate Stage 1 safety screen.
- [x] Apply the frozen advancement gates without threshold relaxation.
- [x] Stop Stage 2 because no candidate was eligible.
- [x] Publish, merge, and synchronize the Phase 6R-D evidence checkpoint.

## Phase 6S current checklist

- [x] Analyze only immutable Stage 1 records and preserve prior negatives.
- [x] Disclose the full wrist-threshold design grid and causal limitations.
- [x] Freeze exactly one SAVR3 controller and a one-shot positive gate.
- [x] Implement SAVR3 without changing SAVR2 behavior.
- [x] Pass the complete correctness and changed-file static-check suite.
- [x] Run and reconcile exactly 70 states-`3-9` development episodes.
- [x] Apply the frozen negative gate without tuning or reruns.
- [x] Record that no positive-result approval point was reached.

## Phase 0 remaining checklist

- [x] Verify repository visibility and GitHub refs.
- [x] Review and merge preparation PR #1 with explicit user approval.
- [x] Synchronize GitHub, TITAN, and local `main`.
- [x] Verify manuscript provenance and clean worktrees.
- [x] Create milestone and decision ledgers.
- [x] Merge the ledger PR with explicit user approval.
- [x] Synchronize the resulting `main` commit everywhere and mark Phase 0 complete.

## Phase 1 current checklist

- [x] Verify official upstream requirements and revisions.
- [x] Verify checkpoint/dataset sizes and whether training data are required.
- [x] Measure current project size and project-filesystem free space.
- [x] Prepare a bounded project-local resource proposal.
- [x] Obtain explicit user approval for Phase 1 downloads and installation.
- [x] Install and lock the environment inside `/home/ved/SAVR`.
- [x] Verify imports and CPU-only headless rendering.
- [x] Report actual storage usage.
- [x] Resolve or explicitly accept the account-level LIBERO path uncertainty.
- [x] Obtain user approval for the Phase 1 checkpoint PR.

## Phase 2 current checklist

- [x] Record the candidate combined checkpoint revision and exact remote size.
- [x] Prepare the bounded Phase 2 download/GPU smoke proposal.
- [x] Obtain explicit user approval for the checkpoint download.
- [x] Download and verify the pinned combined checkpoint within approved limits.
- [x] Select an idle GPU using user-authorized aggregate inspection.
- [x] Complete one bounded unmodified Full Refresh smoke episode.
- [x] Prepare the all-Spatial-task Full Refresh pilot proposal.
- [x] Obtain explicit approval for the all-Spatial-task Full Refresh pilot.
- [x] Complete all 50 planned episodes and reconcile the run manifest.
- [x] Pass the predeclared baseline-feasibility threshold.
- [x] Quantify component timing and complete Phase 2 exit evidence.
- [x] Obtain explicit approval and merge the Phase 2B checkpoint PR.
- [x] Obtain an explicit go/no-go decision for the Phase 3 transition.

## Phase 3 current checklist

- [x] Verify the exact pinned OpenVLA-OFT projected-feature boundary.
- [x] Freeze the controller/cache adapter design and exclusions.
- [x] Implement common FR, PR, VOR, and SAVR controllers.
- [x] Implement exact image, state, and action signals.
- [x] Implement context-safe projected-feature caching.
- [x] Implement the no-upstream-edit OpenVLA-OFT adapter.
- [x] Implement immutable query and episode records.
- [x] Pass CPU unit tests and repository validation.
- [x] Obtain explicit approval and merge the Phase 3 checkpoint PR.
- [x] Obtain approval to prepare a bounded Phase 4 proposal.
- [x] Prepare exact Phase 4 parity, resource, integrity, and stop rules.
- [x] Obtain explicit approval for Phase 4 correctness execution.

## Phase 4 current checklist

- [x] Expand controller truth-table and invalid-input tests.
- [x] Add synchronized query/component timing and CPU fake-backend tests.
- [x] Add query schema and interrupted-run recovery validation.
- [x] Freeze and validate the fail-closed six-query runner.
- [x] Select one qualifying idle GPU using aggregate-only samples.
- [x] Complete the six-query real-model matrix with exact parity.
- [x] Verify zero visual calls and fresh proprioception on reuse.
- [x] Validate all immutable records and restore checkpoint metadata exactly.
- [x] Confirm the selected GPU returned to its pre-run aggregate idle state.
- [x] Review and merge the Phase 4 checkpoint PR #13.
- [x] Obtain an explicit go/no-go decision for Phase 5 execution.

## Phase 5 current checklist

- [x] Research and freeze the bounded Phase 5 protocol.
- [x] Implement and CPU-test the fail-closed core-policy smoke runner.
- [x] Select one qualifying idle GPU using aggregate-only samples.
- [x] Complete all 12 core-policy/state episodes.
- [x] Reconcile refresh trajectories, component counts, and immutable records.
- [x] Pin and audit the official VLA-Cache implementation.
- [x] Establish compatibility or document a reproducible technical exclusion.
- [x] Publish the Phase 5 report and checkpoint PR.
- [x] Obtain user acceptance of the Phase 5 checkpoint.

## Phase 6 current checklist

- [x] Obtain authorization for uninterrupted Phase 6 execution.
- [x] Freeze the calibration split, margin, candidate grid, selection rules,
  budget-matching rules, resource limits, and holdout protections.
- [x] Implement and CPU-test immutable FR signal collection and replay.
- [x] Collect and reconcile 100 paired FR calibration episodes.
- [x] Derive and freeze nine SAVR threshold/horizon settings.
- [x] Complete and reconcile all nine 100-pair SAVR settings.
- [x] Apply the frozen primary SAVR selection rule.
- [x] Stop without matched VOR/PR runs because no SAVR candidate was eligible.
- [x] Record that paired power and the final sample size cannot be confirmed
  without an eligible operating point.
- [x] Record that no primary SAVR/VOR/PR configuration can be frozen under the
  current protocol.
- [x] Publish, review, merge, and synchronize the negative Phase 6 checkpoint.

## Phase 6R-A current checklist

- [x] Preserve the original Phase 6 protocol and negative evidence unchanged.
- [x] Reconcile all existing FR/SAVR episodes and query records.
- [x] Quantify offline-to-online skip-rate transfer.
- [x] Analyze reuse timing, camera aggregation, threshold margin, reuse streaks,
  task outcomes, and action-hash divergence.
- [x] Separate direct observations from plausible causal mechanisms.
- [x] Define evidence-backed requirements for SAVR 2.0.
- [x] Confirm no GPU/simulator or final-holdout run was performed.
- [x] Publish, review, merge, and synchronize the Phase 6R-A checkpoint.
