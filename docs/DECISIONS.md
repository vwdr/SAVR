# SAVR Decision Log

Last updated: 2026-08-03

## D-001 — University-server safety boundary

- Classification: `DECISION`
- Status: ACTIVE
- Decision: All project writes on TITAN are restricted to `/home/ved/SAVR`. No unrelated university files, processes, environments, services, permissions, or GPU allocations may be inspected or changed.
- Evidence: `AGENTS.md` and `docs/SAVR_EXECUTION_PROTOCOL.md`.
- Approver: User, before repository bootstrap.

## D-002 — Repository privacy and execution workspace

- Classification: `DECISION`
- Status: ACTIVE
- Decision: Keep `vwdr/SAVR` private. Use `/home/ved/SAVR` as the authoritative execution workspace and `/Users/veddwivedi/Documents/savr` as the synchronized local review copy.
- Evidence: Private GitHub visibility verified on 2026-07-29; repository paths are defined in the execution protocol.
- Approver: User, before repository bootstrap.

## D-003 — Scientific status

- Classification: `FACT`
- Status: ACTIVE
- Decision: SAVR is an unvalidated proposal. No empirical performance, latency, success, or efficiency claim is currently supported.
- Evidence: `PROJECT_STATUS.md`; no implementation or experimental result exists.
- Approver: Not applicable.

## D-004 — Candidate research stack

- Classification: `HYPOTHESIS`
- Status: OPEN
- Decision: OpenVLA-OFT with all four LIBERO suites is the leading candidate stack, but compatibility and the projected-visual-feature cache boundary require direct verification.
- Alternatives: Reject or revise the stack if Phase 1 or Phase 2 evidence shows incompatibility, unsafe semantics, or inadequate practical benefit.
- Evidence: `docs/STACK_ASSESSMENT.md`, manuscript, and execution protocol.
- Approver: Formal stack acceptance remains pending.

## D-005 — Preparation PR #1

- Classification: `DECISION`
- Status: COMPLETE
- Decision: Merge PR #1, containing the unchanged manuscript, its checksum validation, and the SAVR execution protocol.
- Evidence: User explicitly approved the merge on 2026-07-29; GitHub merge commit `2ef3f59aa5543e8347d02df0802c5d949997203d`.
- Approver: User.

## D-006 — Phase 1 authorization

- Classification: `BLOCKER`
- Status: COMPLETE
- Decision: The user approved the bounded Phase 1 environment/source installation with up to `11 GiB` transfer and a `25 GiB` project-local disk cap. This approval did not include checkpoints, datasets, model loading, or GPU use.
- Evidence: Sections 11 and 19 of `docs/SAVR_EXECUTION_PROTOCOL.md`.
- Approver: User, 2026-07-29.

## D-007 — Phase 1 resource proposal

- Classification: `DECISION`
- Status: COMPLETE
- Decision: Use a project-local Python 3.10.14 environment with the pinned OpenVLA-OFT/LIBERO dependency family, skip FlashAttention, exclude checkpoints and training datasets from Phase 1, and enforce a `25 GiB` installed/cache cap with at most `11 GiB` network transfer.
- Alternatives: Reduce the cap and attempt a smaller environment; or defer the project if the shared storage budget is unacceptable.
- Evidence: `docs/PHASE1_RESOURCE_ESTIMATE.md` and `reports/PHASE1_REPORT.md`; measured project storage was about `14.70 GiB`. Exact transfer bytes were not directly metered and remain `UNVERIFIED`.
- Approver: User, 2026-07-29.

## D-008 — Candidate four-suite checkpoint

- Classification: `HYPOTHESIS`
- Status: OPEN
- Decision: Prefer the official combined four-suite checkpoint for Phase 2 because it covers all target suites with one `14.84 GiB` model instead of four checkpoints totaling about `59.38 GiB`.
- Alternatives: Use the four task-specific checkpoints only if baseline reproduction or scientific review rejects the combined checkpoint.
- Evidence: Official OpenVLA-OFT LIBERO documentation and Hugging Face repository metadata recorded in `docs/UPSTREAM_PINS.md`.
- Approver: Formal checkpoint approval is deferred to Phase 2.

## D-009 — Phase 1 compatibility pins

- Classification: `DECISION`
- Status: ACTIVE
- Decision: Use NumPy `1.26.4`, robosuite `1.4.1`, MuJoCo `2.3.7`, OpenCV `4.6.0.66`, Gym `0.25.2`, OSMesa from mesalib `24.3.4`, protobuf `4.21.12`, tensorflow-metadata `1.17.2`, and array-record `0.4.1` for the validated Phase 1 environment.
- Evidence: The upstream dependency family required these compatibility corrections before `pip check`, OpenVLA imports, and CPU-only rendering all passed. Exact resolved packages are in `environment/locks/`.
- Approver: Phase 1 implementation evidence; checkpoint review pending.

## D-010 — LIBERO account-path uncertainty

- Classification: `BLOCKER`
- Status: COMPLETE
- Decision: The initial upstream LIBERO import created an empty `/home/ved/.libero` directory before prompting. With explicit user authorization, only that path was inspected, confirmed empty and timestamp-matched to the import, removed with `rmdir`, and verified absent. All LIBERO access remains forced to `/home/ved/SAVR/cache/libero`.
- Evidence: Narrow `stat`/depth-two inspection and empty-directory removal on 2026-07-29; project-local `LIBERO_CONFIG_PATH` controls in the setup and verification scripts.
- Approver: User, 2026-07-29.

## D-011 — Phase 2A resource proposal

- Classification: `DECISION`
- Status: ACTIVE
- Decision: Download only the pinned combined four-suite checkpoint with up to `16 GiB` transfer and `20 GiB` additional project-local storage, then run one unmodified Full Refresh LIBERO-Spatial episode on one user-selected GPU with a `60-minute` cap.
- Alternatives: Approve the checkpoint download but defer GPU execution; reduce the smoke scope further; or stop if shared-resource coordination is unavailable.
- Evidence: `docs/PHASE2_RESOURCE_ESTIMATE.md` and the exact checkpoint metadata in `docs/UPSTREAM_PINS.md`.
- Approver: User approved the download/storage limits and merge on 2026-07-29. GPU execution remains blocked until a permitted GPU ID is explicitly identified without inspecting shared allocations.

## D-012 — Phase 2A checkpoint verification

- Classification: `FACT`
- Status: COMPLETE
- Decision: The pinned combined checkpoint resolved to the expected revision and 25 files totaling `15,939,168,050` bytes. All local files matched their declared sizes, and additional project allocation was about `14.85 GiB`, below the approved `20 GiB` cap.
- Evidence: `reports/PHASE2A_CHECKPOINT_REPORT.md` and the project-local runtime inventory `reports/runtime/phase2_checkpoint.json`.
- Approver: Direct download and verification evidence; no scientific result is implied.

## D-013 — Responsible GPU selection

- Classification: `DECISION`
- Status: COMPLETE
- Decision: After explicit user authorization, inspect only aggregate per-GPU memory/utilization and select GPU 0 because repeated samples showed 0% utilization, 6 MiB used, and 24,018 MiB free. Do not inspect process identities.
- Evidence: Three selection samples, one immediate pre-launch sample, and one post-run sample recorded in `reports/PHASE2A_FR_SMOKE_REPORT.md`.
- Approver: User, 2026-07-29.

## D-014 — Phase 2A Full Refresh feasibility

- Classification: `FACT`
- Status: COMPLETE
- Decision: The pinned combined checkpoint loaded and completed one unmodified Full Refresh LIBERO-Spatial task 0 / initial-state 0 / seed 0 episode on one TITAN RTX. Peak allocated memory was about `14.98 GiB`. This is feasibility evidence only.
- Evidence: `reports/PHASE2A_FR_SMOKE_REPORT.md` and immutable run `results/phase2a-fr-20260729T220204Z` on TITAN.
- Approver: Direct smoke evidence; no paper-level performance claim is approved.

## D-015 — Phase 2B Full Refresh pilot

- Classification: `DECISION`
- Status: COMPLETE
- Decision: Run exactly 50 calibration-split Full Refresh episodes covering all ten LIBERO-Spatial tasks and initial-state IDs `0-4`, with component timing on one responsibly selected idle GPU for at most three hours and two GiB of new artifacts.
- Review threshold: At least `45/50` successes and no task with `0/5`; otherwise stop for discrepancy review. This is a feasibility threshold, not a paper-level hypothesis test.
- Evidence: User approval on 2026-07-29; `docs/PHASE2B_PILOT_PROPOSAL.md`; `reports/PHASE2B_PILOT_REPORT.md`.
- Approver: User, 2026-07-29.

## D-016 — Phase 2B baseline and timing feasibility

- Classification: `FACT`
- Status: COMPLETE
- Decision: The fixed calibration pilot completed `50/50` terminal episodes with `49/50` successes and no runtime errors. Every task achieved at least `4/5`. Steady-state visual backbone plus projector execution was `15.874%` of total query CUDA time and `15.873%` of synchronized query wall time.
- Interpretation: The baseline passed the predeclared feasibility threshold. Complete elimination of measured visual compute would have an optimistic query-time ceiling of about `15.87%` latency reduction or `1.189×` speedup; real SAVR benefit must be lower. Proceeding to bounded implementation/correctness work is scientifically reasonable, but no SAVR performance claim is supported.
- Evidence: `reports/PHASE2B_PILOT_REPORT.md`; immutable TITAN run `results/phase2b-fr-spatial-pilot-v1`; reproducible aggregation by `scripts/analyze_phase2b_pilot.py`.
- Approver: User accepted the checkpoint by approving PR #8 on 2026-07-29.

## D-017 — Phase 2B checkpoint acceptance

- Classification: `DECISION`
- Status: COMPLETE
- Decision: Accept and merge the Phase 2B runner, reproducible analysis, baseline-feasibility evidence, and bounded latency interpretation. This approval does not authorize Phase 3.
- Evidence: User approval on 2026-07-29; GitHub PR #8; merge commit `6060966f50619522b5c7faad3ee5cad8b7493da5`.
- Approver: User, 2026-07-29.

## D-018 — Phase 3 transition and cache boundary

- Classification: `DECISION`
- Status: ACTIVE
- Decision: Proceed with bounded Phase 3 CPU implementation and unit tests. Cache only the output of OpenVLA-OFT `_process_vision_features`, before the current proprioception token is appended. Integrate through a temporary, exception-safe model-instance interceptor; do not edit upstream source or alter weights/action-head logic.
- Exclusions: No GPU/simulator run, parity claim, calibration, download, or Phase 4 work.
- Evidence: User go decision on 2026-07-29; pinned OpenVLA-OFT commit `e4287e94541f459edc4feabc4e181f537cd569a8`; `docs/PHASE3_IMPLEMENTATION_DESIGN.md`.
- Approver: User, 2026-07-29.

## D-019 — Phase 3 implementation evidence

- Classification: `FACT`
- Status: COMPLETE
- Decision: The project-owned FR/PR/VOR/SAVR controllers, exact signal functions, projected-feature cache, exception-safe OpenVLA-OFT adapter, and immutable record store are implemented. All `29` dependency-light tests pass locally and in TITAN's pinned environment; Ruff and mypy pass; the package builds; pinned upstream source remains clean.
- Limitation: Fake-model CPU evidence does not establish real-model action parity, cached-tensor compatibility, GPU timing, or simulator correctness. Those remain Phase 4 gates.
- Evidence: `reports/PHASE3_IMPLEMENTATION_REPORT.md`; `src/savr/`; `tests/unit/`.
- Approver: User accepted the checkpoint by approving PR #10 on 2026-07-29.

## D-020 — Phase 3 checkpoint acceptance

- Classification: `DECISION`
- Status: COMPLETE
- Decision: Accept and merge the Phase 3 controller, signal, cache, adapter, immutable-record, documentation, and CPU-test evidence. This approval does not authorize Phase 4 or GPU/simulator work.
- Evidence: User approval on 2026-07-29; GitHub PR #10; merge commit `ad838095ea2b8a2fe7fadbde253c86d01d4f5300`.
- Approver: User, 2026-07-29.

## D-021 — Phase 4 correctness proposal

- Classification: `DECISION`
- Status: COMPLETE
- Decision: Before any Phase 5 policy smoke, expand CPU truth-table/recovery tests and run at most six pinned real-model correctness queries on one responsibly selected idle GPU. Require bitwise wrapped-FR equality, zero visual calls on reuse, fresh proprioception, synchronized timing, valid immutable records, and exact checkpoint restoration.
- Resource bound: No downloads; one GPU; 45 minutes; six policy queries; one simulator reset and zero rollout episodes; 256 MiB new artifacts.
- Evidence: User explicitly approved the proposal and execution on 2026-07-29; `docs/PHASE4_CORRECTNESS_PROPOSAL.md`; `reports/PHASE4_CORRECTNESS_REPORT.md`.
- Approver: User, 2026-07-29.

## D-022 — Phase 4 correctness evidence

- Classification: `FACT`
- Status: COMPLETE
- Decision: The six-query real-model matrix passed. Wrapped FR actions were bitwise identical to unmodified upstream on two invocations. Real VOR reuse produced actions bitwise identical to the state-B upstream reference, executed zero vision-backbone/projector calls, used fresh normalized state B, and advanced cache age from zero to one.
- Limitation: This is controlled query-level correctness evidence only. It does not establish task success, trajectory safety, calibrated thresholds, latency benefit, or a SAVR performance claim.
- Evidence: Immutable TITAN run `results/phase4-correctness-v1`; `reports/PHASE4_CORRECTNESS_REPORT.md`; runner revision `28d5eb3dd0874279d04f2c0f51e337b27efdeb09`.
- Approver: User accepted the evidence by approving PR #13 on 2026-07-29.

## D-023 — Phase 4 checkpoint acceptance

- Classification: `DECISION`
- Status: COMPLETE
- Decision: Accept and merge the Phase 4 controller tests, timing/record infrastructure, bounded real-model runner, immutable correctness evidence, and scientific limitations. This approval does not authorize Phase 5 proposal execution, policy rollouts, threshold calibration, or performance claims.
- Evidence: User approval on 2026-07-29; GitHub PR #13; merge commit `3e50e6acf1aa6aa33a19566b0c593d2068e1c968`.
- Approver: User, 2026-07-29.

## D-024 — Phase 5 execution authorization

- Classification: `DECISION`
- Status: COMPLETE
- Decision: Execute the bounded Phase 5 core-policy smoke and official VLA-Cache compatibility audit without intermediate approval pauses. Run exactly 12 core episodes on LIBERO-Spatial task 0 / initial-state IDs `0-2` / seed 0, using FR, PR, VOR, and SAVR diagnostic configurations. Pin and test VLA-Cache in isolation or document a reproducible technical exclusion.
- Resource bound: One responsibly selected GPU; two hours and one GiB for the core run; two hours and eight GiB of added project-local storage for external-baseline compatibility; no new checkpoint/dataset; no changes to the validated core environment.
- Claim boundary: Structural feasibility only. No threshold calibration, non-inferiority conclusion, comparative success claim, latency claim, manuscript edit, or Phase 6 work.
- Evidence: User blanket approval on 2026-07-29; `docs/PHASE5_SMOKE_PROTOCOL.md`.
- Approver: User, 2026-07-29.

## D-025 — Phase 5 core-policy smoke evidence

- Classification: `FACT`
- Status: COMPLETE
- Decision: All 12 fixed LIBERO-Spatial task-0 episodes and all 283 policy queries reached complete, reconciled records. FR succeeded on three of three states with 31/31 refreshes. Diagnostic PR completed 42 refreshes and 42 reuses; VOR and SAVR each completed 30 refreshes and 54 reuses. All three reuse policies reached the horizon without success under deliberately aggressive uncalibrated settings.
- Interpretation: All four controller paths are trajectory-operational and their instrumentation is correct. The failure of aggressive diagnostic reuse establishes the need for Phase 6 calibration; it is not a calibrated policy comparison.
- Evidence: `/home/ved/SAVR/results/phase5-core-smoke-v1`; `/home/ved/SAVR/results/phase5-analysis-v1/analysis.json`; `reports/PHASE5_SMOKE_REPORT.md`.
- Approver: Direct immutable evidence; checkpoint review pending.

## D-026 — Official VLA-Cache technical exclusion

- Classification: `FACT`
- Status: COMPLETE
- Decision: Do not run or report the pinned official VLA-Cache evaluator as a valid external comparison. Its LIBERO loop assigns previous images from the just-appended current frames and suppresses explicit episode error status. The exact source and required Transformers fork import successfully in an isolated project-local environment, but the evaluation semantics violate the frozen Phase 5 validity requirements.
- Reconsideration condition: Review and explicitly label a minimal previous-frame/error-propagation correction before any VLA-Cache GPU episode.
- Evidence: `/home/ved/SAVR/results/phase5-vla-cache-compatibility-v1/audit.json`; VLA-Cache `a4909880573868dee2769343d52e793c0341678b`; Transformers `9a90a37acacf453433168db8d7769b7ea3c40c06`.
- Approver: Direct pinned source/import evidence; checkpoint review pending.

## D-027 — Phase 5 account-cache deviation and remediation

- Classification: `FACT`
- Status: COMPLETE
- Decision: The first compatibility setup allowed pip to write build/download cache entries under `/home/ved/.cache/pip`, outside the project boundary. The exact recent files written or updated by that invocation were narrowly inventoried and unlinked, empty leaf directories were removed where safe, and no recent file remained. The setup was corrected to use `/home/ved/SAVR/cache/pip-vla-cache` and disable the account-level version-check cache before retrying.
- Impact: The validated SAVR environment, system software, unrelated university files, processes, services, and GPU allocations were not modified.
- Evidence: Command-level inventory/remediation record in the Phase 5 execution log; corrected `scripts/setup_vla_cache_compatibility.sh`; `reports/PHASE5_SMOKE_REPORT.md`.
- Approver: Safety remediation performed under the user's Phase 5 authorization and accepted with PR #15.

## D-028 — Phase 5 checkpoint acceptance

- Classification: `DECISION`
- Status: COMPLETE
- Decision: Accept and merge the reconciled four-policy smoke evidence, official VLA-Cache technical exclusion, and documented account-cache remediation. The aggressive diagnostic reuse failures remain feasibility evidence only and are not calibrated comparisons.
- Evidence: User approval; GitHub PR #15; merge commit `5a4046b2b689d71e2ef0a54a6b67629180d5cdd3`.
- Approver: User, 2026-07-29.

## D-029 — Phase 6 calibration authorization and frozen design

- Classification: `DECISION`
- Status: ACTIVE
- Decision: Execute Phase 6 without intermediate approval pauses through its final checkpoint. Use LIBERO-Spatial tasks `0-9`, initial states `0-9`, and seed `0`; freeze a `2`-percentage-point absolute non-inferiority margin; evaluate SAVR skip targets `25%`, `50%`, and `75%` crossed with horizons `2`, `4`, and `8`; select mechanically; match VOR/PR within `2` absolute refresh-rate points when feasible; and perform paired 90%-power planning at one-sided alpha `0.025`.
- Resource bound: One responsibly selected GPU, at most `48 GPU-hours`, at most `2 GiB` of new result artifacts, no downloads/training/upstream changes, and no final-holdout execution or inspection.
- Evidence: User blanket approval; `docs/PHASE6_CALIBRATION_PROTOCOL.md`, frozen before Phase 6 outcome collection.
- Approver: User, 2026-07-29.

## D-030 — Phase 6 Full Refresh calibration evidence

- Classification: `FACT`
- Status: COMPLETE
- Decision: The frozen LIBERO-Spatial calibration oracle completed `100/100` paired episodes with `100/100` successes, `1,309` immutable query traces, and zero infrastructure errors. Protected checkpoint metadata was restored exactly.
- Evidence: `/home/ved/SAVR/results/phase6-fr-signals-v1`; `/home/ved/SAVR/results/phase6-savr-thresholds-v1/threshold_derivation.json`; combined trace-input SHA-256 `c1724072a9108a77a7c8cec936f4a7e79239dca68aac75a288ca3d4638de9804`.
- Approver: Direct reconciled evidence; no final inference is implied.

## D-031 — Phase 6 SAVR grid negative result

- Classification: `FACT`
- Status: COMPLETE
- Decision: All nine frozen SAVR settings completed all `100` pairings (`900/900` episodes total) with zero infrastructure errors. No candidate met the frozen `-2`-percentage-point calibration constraint. The least-degrading setting, `savr-s25-h2`, achieved `52/100` successes, a paired difference of `-48` percentage points from FR, and an online skip rate of `34.69%`.
- Interpretation: The FR-derived offline skip targets did not transfer safely to closed-loop trajectories in the tested operating region. This is negative calibration evidence, not a proof that every more-conservative SAVR configuration must fail.
- Evidence: `/home/ved/SAVR/results/phase6-savr-grid-v1`; `/home/ved/SAVR/results/phase6-savr-selection-v1/selection.json`; `reports/PHASE6_CALIBRATION_REPORT.md`.
- Approver: Direct reconciled evidence.

## D-032 — Phase 6 negative-result stop

- Classification: `DECISION`
- Status: ACTIVE
- Decision: Apply the frozen stop rule because no SAVR candidate is eligible. Do not relax thresholds, enlarge the margin, run matched VOR/PR baselines, confirm a final sample size, begin Phase 7, or inspect the final holdout.
- Next decision: Either end the current SAVR formulation or predeclare a materially more conservative protocol revision. The current calibration outcomes must remain visible and the split cannot be relabeled as fresh evidence.
- Evidence: `docs/PHASE6_CALIBRATION_PROTOCOL.md`; D-030; D-031.
- Approver: Mechanically required by the user-approved frozen Phase 6 protocol.

## D-033 — Pursue a controlled SAVR redesign

- Classification: `DECISION`
- Status: ACTIVE
- Decision: Preserve the original negative Phase 6 evidence and pursue a phased SAVR 2.0 redesign intended to support a meaningful positive-results paper. A positive result is an objective, not a guaranteed conclusion. The final holdout remains protected, and each redesign phase requires approval at its beginning.
- Evidence: User decision on 2026-07-30; `docs/PHASE6R_REDESIGN_ROADMAP.md`.
- Approver: User, 2026-07-30.

## D-034 — Phase 6R-A forensic diagnosis authorization

- Classification: `DECISION`
- Status: COMPLETE
- Decision: Diagnose the original Phase 6 failures using existing calibration artifacts only. Do not change the method, run a GPU/simulator episode, inspect the final holdout, or describe exploratory associations as final causal evidence.
- Evidence: User approval on 2026-07-30; `reports/PHASE6R_A_DIAGNOSIS_REPORT.md`.
- Approver: User, 2026-07-30.

## D-035 — Phase 6R-A diagnosis

- Classification: `FACT`
- Status: COMPLETE
- Decision: FR-replay targets underpredicted every online SAVR skip rate; two-camera averaging concealed an individual camera threshold exceedance on 47.77% of best-setting reuse queries, almost entirely from the wrist camera; earlier first reuse and near-threshold decisions were associated with lower success; and all 87 action-comparable best-setting episodes first changed action hash exactly at first reuse.
- Limitation: The original SAVR records lack raw online observations/actions, task-phase annotations, and rollout videos. The analysis identifies redesign requirements but does not prove a contact-level causal mechanism or validate SAVR 2.0.
- Evidence: `results/phase6r-a-diagnosis-v1/diagnosis.json`; `reports/PHASE6R_A_DIAGNOSIS_REPORT.md`.
- Approver: Direct reconciliation of existing immutable Phase 6 evidence.

## D-036 — Phase 6R-A checkpoint acceptance

- Classification: `DECISION`
- Status: COMPLETE
- Decision: Accept and merge the reproducible Phase 6R-A forensic analysis, redesign roadmap, diagnosis report, and SAVR 2.0 requirements. This acceptance does not authorize Phase 6R-B research/design, implementation, GPU rollouts, calibration, or final-holdout access.
- Evidence: User Phase 6R-A authorization; GitHub PR #17; merge commit `5d2f69038b76bf94d94bbabefb92b0aa91df72dc`.
- Approver: User, 2026-07-30.

## D-037 — Blanket authorization for remaining Phase 6R

- Classification: `DECISION`
- Status: ACTIVE
- Decision: Execute Phase 6R-B through Phase 6R-E without additional approval pauses. Continue to announce and audit every phase boundary. Stop for any safety boundary, material scope/resource change, predeclared negative gate, or final-holdout risk. Phase 7 remains unauthorized.
- Evidence: User instruction on 2026-07-31; `docs/PHASE6R_REDESIGN_ROADMAP.md`.
- Approver: User, 2026-07-31.

## D-038 — Freeze SAVR 2.0 design before implementation

- Classification: `DECISION`
- Status: ACTIVE
- Decision: Implement a separate training-free SAVR 2.0 controller using independent local per-camera change, grouped state/action change, a gripper-transition veto, minimum query warm-up, two stable fresh queries, isolated reuse, and hard episode-prefix skip caps of 5%, 10%, and 15%. Use the existing FR traces only for candidate generation; require staged online safety screening and retain the 2-point success margin.
- Alternatives: Learned routing, token-level KV caching, and task-specific thresholds were rejected because they change scope, require training/upstream redesign, or invite calibration overfitting.
- Evidence: `docs/PHASE6R_B_RESEARCH_AND_DESIGN.md`; frozen `docs/PHASE6R_PROTOCOL_V1.md`; Phase 6R-A diagnosis.
- Approver: User's blanket Phase 6 authorization, 2026-07-31.

## D-039 — Keep SAVR 2.0 separate from SAVR 1.0

- Classification: `DECISION`
- Status: ACTIVE
- Decision: Implement SAVR 2.0 as a separate controller and signal path while reusing the validated projected-feature adapter. Preserve all SAVR 1.0 classes and tests unchanged. Run CPU correctness gates before the bounded real-model check and prohibit calibration until both pass.
- Evidence: `src/savr/savr2.py`; `tests/unit/test_savr2_controller.py`; `tests/unit/test_savr2_signals.py`; `scripts/run_phase6r_c_correctness.py`.
- Approver: Frozen Phase 6R protocol and user's blanket Phase 6 authorization, 2026-07-31.

## D-040 — Correct the Phase 6R-C fixture without weakening SAVR 2.0

- Classification: `DECISION`
- Status: ACTIVE
- Decision: Preserve the failed first correctness run, whose real action chunk correctly activated the gripper-transition veto. Do not weaken the veto or reinterpret the run as reuse evidence. Predeclare one recovery using a hashed Phase 6 FR trace, eight additional model queries, no simulator, and the unchanged controller. Cumulative Phase 6R-C usage remains 18/20 queries. Stop before Phase 6R-D if the recovery fails.
- Evidence: `reports/PHASE6R_C_CORRECTNESS_RECOVERY_PLAN.md`; immutable `phase6r-c-correctness-v1` artifacts on TITAN.
- Approver: Frozen Phase 6R protocol and user's blanket Phase 6 authorization, 2026-07-31.

## D-041 — Accept Phase 6R-C correctness evidence

- Classification: `DECISION`
- Status: ACTIVE
- Decision: Accept Phase 6R-C after preserving the first fixture failure and passing the predeclared recovery. The recovery demonstrated exact reuse parity, current proprioception, zero vision-backbone/projector calls on reuse, complete counters/records, and clean checkpoint restoration within 18/20 cumulative queries. This authorizes Phase 6R-D only and is not an online success or efficiency claim.
- Evidence: `reports/PHASE6R_C_CORRECTNESS_REPORT.md`; recovery summary SHA-256 `9b58b58ef11de5f594066bde4d45c3f56548960431b83c48d25715fdf6e46ef9`.
- Approver: Frozen Phase 6R protocol and user's blanket Phase 6 authorization, 2026-07-31.

## D-042 — Derive Phase 6R-D candidates with exact offline replay

- Classification: `DECISION`
- Status: ACTIVE
- Decision: Build all eight adjacent-query score-family distributions from the complete 100-episode Phase 6 FR trace. Apply the shared 0.001 linear-quantile grid and 0.90 safety margin, then replay the exact SAVR 2.0 temporal and hard prefix-budget semantics for 5%, 10%, and 15% caps. Freeze the closest never-over-budget candidate for each cap before Stage 1.
- Evidence: `scripts/derive_phase6r_d_candidates.py`; `tests/unit/test_phase6r_d_derivation.py`; `docs/PHASE6R_PROTOCOL_V1.md` Section 8.
- Approver: Frozen Phase 6R protocol and user's blanket Phase 6 authorization, 2026-07-31.

## D-043 — Freeze Phase 6R-D Stage 1 candidates

- Classification: `DECISION`
- Status: ACTIVE
- Decision: Freeze `savr2-b05`, `savr2-b10`, and `savr2-b15` exactly as recorded in `configs/calibration/phase6r_d_stage1.json`. Their offline skip estimates are 0.00%, 6.88%, and 9.78%. Retain `b05` despite zero expected reuse because the protocol requires every candidate; apply the 2% online-skip advancement gate without exception. Run exactly states 0-2 for Stage 1 and preserve all attempts/traces.
- Evidence: `reports/PHASE6R_D_CANDIDATE_DERIVATION.md`; semantic config SHA-256 `66874e1a2c209ec5809dd1d777de5ce8eeacee63d85e8e4dd1c6f0876bcfc09d`.
- Approver: Frozen Phase 6R protocol and user's blanket Phase 6 authorization, 2026-07-31.

## D-044 — Apply the Phase 6R-D negative stop

- Classification: `DECISION`
- Status: ACTIVE
- Decision: Stop Phase 6R-D before Stage 2. `b05` fails the 2% skip gate, while `b10` and `b15` fail the 30/30 success gate. No thresholds, margins, pairings, or advancement criteria are relaxed. Phase 6R-E is ineligible because no candidate advanced. Preserve all four unsuccessful task episodes as scientific outcomes.
- Evidence: `reports/PHASE6R_D_STAGE1_REPORT.md`; summary SHA-256 `61a0c9ddfb263ba2123da3dd08500260eba6a454bf335f4830022a81c33a9ebe`.
- Approver: Frozen Phase 6R protocol and user's blanket Phase 6 authorization, 2026-07-31.

## D-045 — Execute Phase 6S through the first positive method gate

- Classification: `DECISION`
- Status: ACTIVE
- Decision: Execute the final SAVR redesign without intermediate approval pauses. Preserve every frozen stop rule and the final holdout. Stop and request user approval immediately after the first predeclared positive method result.
- Evidence: User authorization on 2026-07-31; `docs/PHASE6S_PROTOCOL_V1.md`.
- Approver: User, 2026-07-31.

## D-046 — Freeze one disclosed SAVR3 design and validation

- Classification: `DECISION`
- Status: ACTIVE
- Decision: Starting from `savr2-b15`, add a translation-direction-reversal veto and reduce only the wrist threshold to `0.375`. Retain every other SAVR2 rule. Treat states `0-2` as post-hoc design evidence and run SAVR3 once on states `3-9`; require 70/70 success, 7/7 per task, at least 5% online skip, and zero technical or accounting errors.
- Limitation: The design was selected after inspecting Stage 1 failures. States `3-9` are policy-specific fresh validation, not the final holdout. No positive result is guaranteed.
- Evidence: `reports/PHASE6S_A_FORENSIC_REPORT.md`; `docs/PHASE6S_PROTOCOL_V1.md`.
- Approver: User authorization and frozen Phase 6S protocol, 2026-07-31.

## D-047 — Accept SAVR3 correctness and begin frozen validation

- Classification: `DECISION`
- Status: ACTIVE
- Decision: Accept SAVR3 implementation after 102/102 CPU tests and all changed-file static checks pass. Skip an optional additional real-model correctness run because the validated projected-feature adapter is unchanged; enforce its component invariants on every Phase 6S-D query. Run exactly the frozen states-`3-9` configuration without tuning.
- Evidence: `reports/PHASE6S_C_CORRECTNESS_REPORT.md`; config semantic SHA-256 `10b93d3247f6bec35c7419e362627dffef597ddbcd5dd71f9509a6b66bb52289`.
- Approver: Frozen Phase 6S protocol and user authorization, 2026-07-31.

## D-048 — Apply the Phase 6S-D negative stop

- Classification: `DECISION`
- Status: ACTIVE
- Decision: Stop Phase 6S after the complete frozen SAVR3 validation. The run achieved 69/70 successes and 9/944 reuses (0.9534%), so it failed both the exact success gate and the 5% skip gate. Do not tune or rerun SAVR3, search another local threshold, run Phase 6S-E, or inspect the final holdout. No positive-result approval point was reached.
- Integrity: All 70 terminal records and 944 queries reconcile; all nine reuses skipped exactly one backbone and projector call; no technical or invariant error occurred; protected checkpoint hashes were restored.
- Evidence: `reports/PHASE6S_D_VALIDATION_REPORT.md`; analysis SHA-256 `de570a1b79c7e7e50bf5193f5bf2d2f7048c2336abf10c0dd0b460db51f3e789`.
- Approver: Mechanically required by the frozen Phase 6S protocol, 2026-07-31.

## D-049 — Prepare a separate Asymmetric Camera Refresh route

- Classification: `DECISION`
- Status: ACTIVE
- Decision: Preserve all whole-prefix SAVR negative evidence and prepare a
  materially different Asymmetric Camera Refresh proposal and phase-gated
  execution protocol. ACR always refreshes the wrist-camera pathway and may
  reuse only the scene-camera token block. The protocol separates Object
  development, Goal confirmation, and LIBERO-10 transfer from a protected,
  fresh-state four-suite final evaluation. Creating these documents authorizes
  no implementation, GPU work, simulator outcome, final-holdout access, or
  manuscript change.
- Evidence: User instruction on 2026-08-02;
  `docs/NEGATIVE_RESULTS_PAPER_ARCHIVE.md`;
  `docs/ASYMMETRIC_CAMERA_REFRESH_PROPOSAL.md`;
  `docs/ACR_EXECUTION_PROTOCOL_V1.md`.
- Approver: User authorized planning, 2026-08-02. Execution remains subject to
  the protocol's phase gates.

## D-050 — Authorize ACR Phase A0 only

- Classification: `DECISION`
- Status: ACTIVE
- Decision: Execute the ACR novelty, pinned-source, suite/split, resource, and
  implementation-design audit defined as Phase A0 in
  `docs/ACR_EXECUTION_PROTOCOL_V1.md`. Do not implement ACR, run a model, use a
  GPU, launch a simulator, inspect protected outcomes, derive numerical
  thresholds, modify the manuscript, or begin Phase A1/A2 work.
- Evidence: User instruction to proceed with Phase A0 on 2026-08-02.
- Approver: User, 2026-08-02.

## D-051 — Accept the narrowed ACR novelty boundary at Phase A0

- Classification: `DECISION`
- Status: ACTIVE
- Decision: The ACR novelty gate passes only for the complete conjunction of
  temporal scene-camera projected-block reuse, an always-fresh wrist block,
  unchanged scene-first/wrist-second token positions, deterministic
  training-free fail-closed control, and skipped scene encoder/projector work
  in chunked two-view OpenVLA-OFT. Do not claim generic adaptive multi-view
  perception, temporal VLA caching, state-aware efficient inference, or
  asynchronous multimodal VLA novelty.
- Evidence: `docs/ACR_NOVELTY_AUDIT.md`; current full-text/code audit completed
  2026-08-02.
- Approver: Mechanically required by ACR Protocol V1 Phase A0.

## D-052 — Close ACR Phase A0 and stop before A1

- Classification: `DECISION`
- Status: ACTIVE
- Decision: Accept the pinned source boundary, four-suite/50-state mapping,
  consumed-population ledger, absence of ACR outcomes, and exact implementation
  design. Bitwise camera-factorized projector parity remains unproven and must
  be a hard A3 stop gate. Complete A0 without starting A1 or A2.
- Evidence: `docs/ACR_IMPLEMENTATION_DESIGN.md`;
  `docs/ACR_SPLIT_AND_RESOURCE_AUDIT.md`; `reports/PHASE_A0_REPORT.md`.
- Approver: ACR Protocol V1 Phase A0 exit rules, 2026-08-02. Phase A1 still
  requires user authorization.

## D-053 — Authorize ACR Phase A1 only

- Classification: `DECISION`
- Status: ACTIVE
- Decision: Execute the protocol-acceptance and resource-freeze work defined
  as ACR Phase A1. Reconnect to TITAN only for narrowly scoped repository,
  static hardware, and storage verification. Freeze schemas, run identities,
  recovery rules, artifact limits, and estimates from historical measured
  runtimes. Do not implement ACR, load the model, use a GPU, start a simulator,
  access an ACR population, derive thresholds, or modify the manuscript.
- Evidence: User instruction to proceed to Phase A1 on 2026-08-02.
- Approver: User, 2026-08-02.

## D-054 — Accept the ACR Phase A1 freeze and stop before A2

- Classification: `DECISION`
- Status: ACTIVE
- Decision: Accept ACR Protocol V1 without amendment and freeze the formal
  query, episode, and run schemas; deterministic run-ID templates;
  preserve-and-restart recovery rules; artifact policy; and bounded phase
  estimates. Keep every protocol hard cap and protected population unchanged.
  Complete A1 without implementing ACR or starting A2.
- Evidence: `docs/ACR_PHASE_A1_RESOURCE_FREEZE.md`;
  `configs/acr/phase_a1_freeze.json`; `reports/PHASE_A1_REPORT.md`.
- Approver: Mechanically required by ACR Protocol V1 Phase A1. Phase A2 still
  requires user authorization.

## D-055 — Authorize ACR Phase A2 only

- Classification: `DECISION`
- Status: ACTIVE
- Decision: Implement the separate project-owned camera-factorized adapter,
  ACR controller/cache/signals, camera accounting/timing, immutable compact FR
  records, deterministic candidate derivation, and frozen statistical
  utilities. Preserve SAVR1-3 and both pinned upstream trees unchanged. Run
  CPU/static/build tests only. Do not load the checkpoint/model, use a GPU,
  start LIBERO, access an ACR population, derive outcome-dependent thresholds,
  begin A3, or modify the manuscript.
- Evidence: User instruction to proceed on 2026-08-02; A1 exit passed in
  `reports/PHASE_A1_REPORT.md`.
- Approver: User, 2026-08-02.

## D-056 — Close ACR Phase A2 and stop before A3

- Classification: `DECISION`
- Status: ACTIVE
- Decision: Accept the separate project-owned ACR implementation after all
  133 repository tests, source/test Ruff, source mypy, bootstrap validation,
  and package build pass. Preserve the A2 code and evidence without loading
  the real model, selecting a GPU, starting LIBERO, accessing an ACR
  population, or changing the manuscript. Do not begin A3 until the user
  explicitly authorizes the bounded real-model correctness phase.
- Evidence: `reports/PHASE_A2_REPORT.md`; `src/savr/acr/`; `tests/acr/`.
- Approver: Mechanically required by ACR Protocol V1 Phase A2 exit gate,
  2026-08-02.

## D-057 — Authorize bounded ACR Phase A3

- Classification: `DECISION`
- Status: ACTIVE
- Decision: Run the frozen A3 real-model correctness matrix using at most one
  responsibly selected GPU, 16 policy queries, 3,600 wall seconds, and 512 MiB
  of compact artifacts. Use the 12-query deterministic synthetic-input matrix
  in `docs/ACR_PHASE_A3_PREFLIGHT.md`; leave four queries unused as safety
  margin. Start no simulator, consume no benchmark population or outcome,
  perform no download, and modify nothing outside `/home/ved/SAVR`. Any exact
  projected-token or action parity failure stops A3 immediately without a
  tolerance, rerun, or rollout.
- Evidence: User approval on 2026-08-02; A2 completion at `864044d`.
- Approver: User, 2026-08-02.

## D-058 — Accept A3 scientific proofs after transparent technical recovery

- Classification: `DECISION`
- Status: ACTIVE
- Decision: Accept the A3 correctness gate as passed with technical recovery.
  Preserve the original attempt as `failed`; do not overwrite or relabel it.
  The committed runner reached its final checkpoint audit only after every
  exact token/action parity, camera isolation, reuse, current-state, and
  fail-closed assertion returned successfully. Independent immutable records
  confirm action hashes and camera component truth. Restore only the pinned
  loader's two temporary checkpoint rewrites to their accepted hashes and
  adjudicate the preserved attempt CPU-only. No additional model query,
  tolerance, simulator access, or population outcome is permitted.
- Evidence: `reports/PHASE_A3_REPORT.md`;
  `reports/runtime/acr_a3_adjudication.json`; immutable TITAN run
  `results/acr-a3-correctness-none-v01`.
- Approver: Mechanical A3 evidence adjudication under Protocol V1,
  2026-08-02. A4 still requires explicit user authorization.

## D-059 — Authorize the frozen A4 development-FR population

- Classification: `DECISION`
- Status: ACTIVE
- Decision: Open exactly LIBERO-Object tasks `0-9`, states `0-9`, seed `0`
  once under unmodified upstream two-view FR. Apply the Protocol V1 feasibility
  gate before deriving exactly three candidates twice. Permit no ACR rollout,
  retry, threshold change, replacement candidate, download, protected-population
  access, or manuscript edit in A4.
- Evidence: User approval on 2026-08-02; `docs/ACR_PHASE_A4_PREFLIGHT.md`;
  `configs/acr/development_fr.json`; A3 report and merge `7013e71`.
- Approver: User, 2026-08-02.

## D-060 — Accept A4 feasibility and freeze exactly three candidates

- Classification: `DECISION`
- Status: ACTIVE
- Decision: Accept the A4 feasibility gate after 100/100 terminal upstream-FR
  Object episodes, 97 successes, every task at or above 8/10, 1,773
  reconciled queries/traces, and zero technical failures. Freeze only
  `acr-t25-h2-b30`, `acr-t50-h4-b55`, and `acr-t70-h8-b75` from the
  byte-identical deterministic derivations. Preserve both completed CPU
  analyses and disclose that the recovery was launched after an SSH disconnect
  before the original analysis completion became visible. This is not an ACR
  method result. Do not begin A5, alter any candidate, access a protected
  population, or modify the manuscript without new authorization.
- Evidence: `reports/PHASE_A4_REPORT.md`;
  `reports/runtime/acr_a4_analysis.json`; `configs/acr/candidates.json`;
  immutable TITAN run `results/acr-a4-upstream-fr-object-dev00-09-v01`.
- Approver: Mechanical A4 exit gate under Protocol V1, 2026-08-03. A5 still
  requires explicit user authorization.

## D-061 — Authorize frozen A5 staged development

- Classification: `DECISION`
- Status: ACTIVE
- Decision: Run exactly the three A4-frozen SA-ACR candidates first on
  LIBERO-Object tasks `0-9`, states `0-2`, seed `0`, for 30 attempts each.
  Apply every Protocol V1 Stage 1 gate mechanically. Open states `3-9` only
  for candidates that advance, then apply the frozen development eligibility
  and selection rules. Use at most one responsibly selected GPU, 300 total
  attempts, 86,400 seconds, and 2 GiB of A5 artifacts. Permit no download,
  automatic retry, outcome-driven tuning, protected-population access,
  manuscript edit, or Phase A6 work.
- Evidence: User approval on 2026-08-03; `docs/ACR_PHASE_A5_PREFLIGHT.md`;
  `configs/acr/development_a5.json`; A4 report and merge `a4ec6b8`.
- Approver: User, 2026-08-03.

## D-062 — Apply the A5 Stage 1 negative stop

- Classification: `DECISION`
- Status: ACTIVE
- Decision: Stop A5 before Stage 2 because the committed analyzer selected no
  advancing candidate. Preserve the 90/90 terminal episodes and every
  scientific failure. The candidates achieved 29/30, 24/30, and 23/30
  success with 26.06%, 47.40%, and 49.44% scene reuse; all exact component
  invariants passed and no technical failure occurred, but every candidate
  failed the frozen success and per-task gates. Do not tune, rerun, replace,
  or reinterpret a candidate; do not open Object states `3-9`, A6, or any
  final holdout; do not modify the manuscript. A6 is ineligible.
- Evidence: `reports/PHASE_A5_REPORT.md`;
  `reports/runtime/acr_a5_stage1_analysis.json`; immutable TITAN Stage 1 run
  records.
- Approver: Mechanically required by ACR Protocol V1 and the user-authorized
  frozen A5 preflight, 2026-08-03.

## D-063 — Authorize ACR Version 2 diagnosis and planning

- Classification: `DECISION`
- Status: ACTIVE
- Decision: Pursue a scientifically credible positive-paper route using A5 as
  disclosed exploratory development evidence. Diagnose existing records,
  repeat relevant primary-source research, and freeze a materially revised
  execution method and new protocol. Permit no new GPU/model query, simulator
  episode, unopened ACR outcome, protected-population access, or manuscript
  edit during this planning phase.
- Evidence: User approval on 2026-08-03; immutable A4/A5 records.
- Approver: User, 2026-08-03.

## D-064 — Freeze State-Aware Dual-Path ACR Version 2

- Classification: `DECISION`
- Status: ACTIVE
- Decision: Retain the exact `acr-t25-h2-b30` controller and replace only its
  physical execution architecture. Use the original upstream two-view path on
  scene refresh, cache the split projected scene block, and use a wrist-only
  path on scene reuse. Remove redundant intermediate host synchronizations in
  production mode while preserving structural checks, terminal action-finite
  validation, immutable accounting, and bounded full correctness checks. Add
  no post-hoc signal veto. Require bitwise equivalence and a paired latency
  gate before any rollout; use Object states `3-9` only after those gates.
- Rationale: The conservative candidate achieved 29/30 success, 26.06% scene
  reuse, and 11.94% visual CUDA reduction, but was 31.24% slower in query wall
  time. Its task 6/state 0 failure pattern also appeared in successful
  episodes, and both more aggressive candidates succeeded on the same state,
  so a one-case controller patch is unsupported.
- Evidence: `reports/ACR_V2_DIAGNOSIS_REPORT.md`;
  `reports/runtime/acr_v2_diagnosis.json`;
  `docs/ACR_V2_EXECUTION_PROTOCOL.md`; `configs/acr/v2_freeze.json`.
- Approver: Mechanical V2-A freeze under the user-approved positive-paper
  planning route, 2026-08-03. Phase V2-B remains unauthorized.

## D-065 — Authorize and complete ACR Version 2 Phase V2-B

- Classification: `DECISION`
- Status: ACTIVE
- Decision: Implement the frozen SA-DP-ACR execution architecture on CPU only.
  Keep Version 1 unchanged. Require episode-scoped restoration, exact original
  refresh return identity, wrist-only reuse, truthful physical/logical
  accounting, structural fail-closed behavior, production/correctness finite
  modes, terminal action validation, and immutable recovery identities. Stop
  after CPU/static verification; do not use a GPU, model, simulator, rollout,
  new outcome, download, protected population, or manuscript.
- Evidence: User approval on 2026-08-03; `src/savr/acr/dual_path.py`;
  `tests/acr/test_dual_path_adapter.py`; `reports/PHASE_V2_B_REPORT.md`.
- Result: All 172 repository tests plus 9 TITAN subtests pass. All 14 new
  V2-B tests and changed-file static/build/bootstrap gates pass. The 512 MiB
  cap was respected. No scientific outcome was collected.
- Approver: User authorization and mechanical V2-B exit gate, 2026-08-03.
  Phase V2-C remains unauthorized.

## D-066 — Authorize and freeze ACR Version 2 Phase V2-C

- Classification: `DECISION`
- Status: ACTIVE
- Decision: Execute one bounded real-model correctness and paired-latency gate
  for SA-DP-ACR. Use exactly six correctness queries, six untimed warm-ups,
  and 36 timed queries in the frozen counterbalance, totaling 48. Use one
  responsibly selected GPU/process, zero simulator resets or episodes, 3,600
  seconds, 512 MiB, and no download. Require bitwise refresh/reuse parity,
  exact return identity, camera-work truth, restoration, and all three frozen
  median latency ratios. Any failure stops before V2-D.
- Evidence: User approval on 2026-08-03; `configs/acr/v2_c_gate.json`;
  `docs/ACR_V2_C_PREFLIGHT.md`; `scripts/run_acr_v2_c.py`.
- Approver: User, 2026-08-03.
