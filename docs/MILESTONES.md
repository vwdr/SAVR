# SAVR Milestones

Last updated: 2026-07-30

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
| 6R-B. Redesign and protocol | NOT_STARTED | — | Frozen SAVR 2.0 protocol |
| 6R-C. Implementation and correctness | NOT_STARTED | — | All redesign correctness gates pass |
| 6R-D. Conservative staged calibration | NOT_STARTED | — | Eligible SAVR 2.0 candidate or predeclared stop |
| 6R-E. Baselines, selection, and power | NOT_STARTED | — | Frozen candidate set and feasible confirmatory design |
| 7. Freeze final protocol | NOT_STARTED | — | User approval of `PROTOCOL_V1.md` |
| 8. Final evaluation | NOT_STARTED | — | Complete reconciled final-run registry |
| 9. Ablations and sensitivity | NOT_STARTED | — | Required confirmatory ablations complete |
| 10. Analysis and claim audit | NOT_STARTED | — | Every manuscript claim mapped to evidence |
| 11. Manuscript completion | NOT_STARTED | — | User-approved evidence-based manuscript changes |

## Active milestone

No phase is currently active. Phase 6R-A is complete and Phase 6R-B awaits
explicit approval. The original Phase 6 negative result remains unchanged.
Phase 7 remains unauthorized, and the initial-state `10-49` / seed `7,17,27`
holdout remains untouched.

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
