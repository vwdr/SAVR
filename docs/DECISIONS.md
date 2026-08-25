# SAVR Decision Log

Last updated: 2026-08-25

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

## D-067 — Preserve the V2-C technical stop and freeze one recovery

- Classification: `DECISION`
- Status: ACTIVE
- Decision: Preserve the original attempt as technically failed after 7/48
  queries. Accept no timed sample from it. Record that all six correctness
  assertions completed and the first upstream-FR warm-up reached its
  synchronized component-count check. Correct only the expected low-level call
  truth: two SigLIP/two DINOv2 calls for upstream/dual refresh and one each for
  dual reuse. Run exactly the remaining five warm-ups and 36 timed queries.
  Cumulative use must equal 48/48. Change no method, timing boundary,
  counterbalance, gate, or population. A further failure ends V2-C.
- Evidence: Immutable parent failure SHA-256
  `745a8cff68921190acc6d738c8febf1667de44b3891d683a377e60172e5354ad`;
  `configs/acr/v2_c_recovery.json`; `docs/ACR_V2_C_RECOVERY_PLAN.md`.
- Approver: Mechanical fail-closed recovery under the user-authorized V2-C
  phase, 2026-08-03.

## D-068 — Apply the V2-C negative latency stop

- Classification: `DECISION`
- Status: ACTIVE
- Decision: Stop SA-DP-ACR Version 2 before V2-D. Preserve exactly 48/48
  cumulative queries, all 36 timed records, and the original technical stop.
  Accept that reuse reduces median visual CUDA work by 50.12%, but reject the
  method because refresh, reuse, and weighted wall ratios of 1.40338, 1.42995,
  and 1.41030 fail every frozen latency gate. Do not rerun, retime, delete
  outliers, reinterpret, open Object states `3-9`, open Goal, or access a final
  population under V2.
- Evidence: `reports/PHASE_V2_C_REPORT.md`;
  `reports/runtime/acr_v2_c_recovery.json`; immutable TITAN parent/recovery
  records.
- Approver: Mechanical V2-C stop under the user-authorized frozen protocol,
  2026-08-03. V2-D is ineligible.

## D-069 — Authorize and freeze ACR Version 3 Phase V3-A

- Classification: `DECISION`
- Status: ACTIVE
- Decision: Preserve V2-C as negative and freeze a materially new execution
  route: State-Aware Batched Dual-Path Asymmetric Camera Refresh
  (`SA-BDP-ACR`). Retain the exact `acr-t25-h2-b30` controller, batch ordered
  scene/wrist samples within each vision tower on refresh, and retain
  wrist-only reuse. Require a Batched Full Refresh ablation so batching and
  camera-reuse contributions cannot be conflated. Move evidence hashing,
  serialization, and file I/O outside every synchronized inference boundary.
  Freeze bfloat16 token tolerance, bitwise action parity, a 64-query
  correctness/latency gate, fresh populations, resources, and stop rules
  before implementation. Authorize no V3 implementation, GPU/model query,
  simulator episode, protected outcome, or manuscript edit in V3-A.
- Rationale: At the fixed reuse weight, an optimistic zero-overhead scene skip
  can reduce weighted wall time by at most 1.6202%, below the 2% gate. The
  pinned two-camera source invokes each vision tower sequentially per camera,
  so a refresh-acceleration mechanism is necessary and technically testable.
- Evidence: `reports/ACR_V3_DIAGNOSIS_REPORT.md`;
  `reports/runtime/acr_v3_feasibility.json`;
  `docs/ACR_V3_EXECUTION_PROTOCOL.md`; `configs/acr/v3_freeze.json`.
- Approver: User authorization on 2026-08-04 and mechanical V3-A exit gate.
  Phase V3-B remains unauthorized.

## D-070 — Authorize and complete ACR Version 3 Phase V3-B

- Classification: `DECISION`
- Status: ACTIVE
- Decision: Implement separate Batched Full Refresh and SA-BDP-ACR adapters
  under the frozen V3 method. Require exact scene-then-wrist batching, one
  SigLIP/DINOv2/projector invocation per refresh, no controller/cache in BFR,
  V2-equivalent wrist-only reuse, fail-closed cache/restoration/concurrency,
  production timing without evidence hashing/serialization/file I/O/full
  projected-token scans, immutable identities, and bounded query accounting.
  Use CPU only and stop before V3-C.
- Evidence: User authorization on 2026-08-04;
  `src/savr/acr/batched_dual_path.py`;
  `tests/acr/test_batched_dual_path.py`;
  `reports/PHASE_V3_B_REPORT.md`.
- Result: All 206 repository tests plus 9 TITAN subtests, 18 new V3-B tests,
  and six real-PyTorch CPU assertions pass. No GPU, model query, simulator,
  benchmark outcome, download, protected population, or manuscript was used.
- Approver: User authorization and mechanical V3-B exit gate, 2026-08-04.
  Phase V3-C remains unauthorized.

## D-071 — Authorize and freeze ACR Version 3 Phase V3-C

- Classification: `DECISION`
- Status: ACTIVE
- Decision: Execute one bounded real-model correctness and latency gate for
  sequential FR, Batched FR, V3 refresh, and V3 reuse. Consume exactly eight
  correctness queries, eight untimed warm-ups, and 48 counterbalanced timed
  queries. Require the frozen two-input token tolerance, bitwise refresh
  actions, V2-exact reuse, truthful physical/logical work, restoration, and
  all six latency gates. Use one responsibly selected GPU/process, zero
  simulator resets or episodes, at most 64 queries, 3,600 seconds, 512 MiB,
  and no download. Stop before V3-D regardless of the result.
- Evidence: User approval on 2026-08-04;
  `configs/acr/v3_c_gate.json`; `docs/ACR_V3_C_PREFLIGHT.md`.
- Approver: User, 2026-08-04.

## D-072 — Accept the positive V3-C correctness and latency result

- Classification: `DECISION`
- Status: ACTIVE
- Decision: Accept V3-C as the first predeclared positive method result. All
  64 unique queries completed, both refresh inputs were token-exact, all
  refresh/reuse actions were bitwise correct, physical/logical work
  reconciled, and all six latency gates passed. Preserve every repetition and
  stop before V3-D. This result authorizes no simulator episode, task-success
  claim, protected-population access, or manuscript change.
- Quantitative result: BFR/sequential wall ratio `0.9689796428`; V3
  refresh/BFR `1.0054524993`; V3 reuse/BFR `0.9750279090`; V3
  weighted/sequential `0.9665817654`; V3 weighted/BFR `0.9975253584`; weighted
  visual CUDA reduction `31.40923355%`.
- Evidence: `reports/PHASE_V3_C_REPORT.md`;
  `reports/runtime/acr_v3_c.json`; result semantic SHA-256
  `3f77171fbf42015fb0f6e74c0f5d49c8f58890a64355b2de4348407cef79ab02`.
- Approver: Mechanical V3-C gate under user authorization, 2026-08-04.
  Phase V3-D requires separate authorization.

## D-073 — Authorize and freeze ACR Version 3 Phase V3-D

- Classification: `DECISION`
- Status: ACTIVE
- Decision: Execute exactly 70 Batched Full Refresh and 70 frozen SA-BDP-ACR
  episodes on LIBERO-Object tasks `0-9`, states `3-9`, seed `0`. Pair the two
  policies by task/state and alternate their adjacent order, giving 35 first
  positions each. Remain outcome-blind until all 140 terminal records exist;
  never retry a scientific failure. Use immutable A4 sequential-FR evidence
  only as the system-latency reference. Apply the frozen success, per-task,
  reuse, visual-CUDA, wall-time, work, cache, and restoration gates
  mechanically. Use one responsibly selected GPU/process, at most 140
  attempts, 43,200 seconds, 2 GiB, and no download. Stop before V3-E regardless
  of the result.
- Evidence: User approval on 2026-08-04;
  `configs/acr/v3_d_development.json`; `docs/ACR_V3_D_PREFLIGHT.md`;
  `scripts/run_acr_v3_d.py`; `scripts/analyze_acr_v3_d.py`.
- Approver: User, 2026-08-04.

## D-074 — Preserve the V3-D technical stop and freeze one recovery

- Classification: `DECISION`
- Status: ACTIVE
- Decision: Preserve the first V3-D run as technically failed after one BFR
  episode start, zero completed query records, zero action executions, and no
  opened success outcome. Correct only the runner/action representation
  boundary by supplying a list/NumPy-aware finite checker outside timing.
  Execute the complete unchanged 140-episode matrix under a new immutable run
  ID. Count the preserved start, making the cumulative attempt cap 141, while
  retaining the cumulative 43,200-second and 2-GiB caps. Do not change the
  method, controller, population, order, timing, gates, outcome blindness, or
  scientific no-retry rule. Stop before V3-E.
- Evidence: Immutable TITAN source-run manifest/completion/summary hashes in
  `configs/acr/v3_d_recovery.json`;
  `docs/ACR_V3_D_TECHNICAL_RECOVERY.md`; regression test in
  `tests/acr/test_v3_d_runner_analysis.py`.
- Approver: Narrow technical recovery under the user's continuing V3-D
  authorization and instruction not to pause before a positive method result,
  2026-08-04.

## D-075 — Preserve recovery 1 and freeze V3-D recovery 2

- Classification: `DECISION`
- Status: ACTIVE
- Decision: Preserve recovery 1 after its completed first-pair BFR episode and
  pre-query V3 identity stop. Keep that BFR episode outcome unopened and
  exclude it from official analysis. Correct only context identity wiring:
  BFR uses `batched-full-refresh`; V3 uses the frozen controller identity
  `acr-t25-h2-b30`. Rerun the complete 140-episode matrix under a new immutable
  run ID and one model process. Count all three prior starts, making the
  cumulative cap 143, while retaining the original wall/artifact limits,
  scientific design, gates, outcome blindness, and no-retry rule. Stop before
  V3-E.
- Evidence: Preserved recovery-1 hashes in
  `configs/acr/v3_d_recovery_2.json`;
  `docs/ACR_V3_D_TECHNICAL_RECOVERY_2.md`; identity regression test in
  `tests/acr/test_v3_d_runner_analysis.py`.
- Approver: Narrow technical recovery under continuing user authorization,
  2026-08-04.

## D-076 — Apply the V3-D negative efficiency stop

- Classification: `DECISION`
- Status: ACTIVE
- Decision: Accept the complete 140-episode recovery-2 evidence and stop V3
  before V3-E. Preserve 67/70 success for both BFR and V3, 25.24% V3 reuse,
  zero technical failures, all invariants, and the 0.96043 wall ratio versus
  sequential FR. Reject the positive gate because visual CUDA reduction was
  8.46% rather than at least 10%, and wall ratio versus BFR was 1.00226 rather
  than at most 1.00. Do not rerun, retime, remove samples, relax gates, open
  Goal, or reinterpret the result.
- Evidence: `reports/PHASE_V3_D_REPORT.md`;
  `reports/runtime/acr_v3_d.json`; immutable TITAN recovery-2 records.
- Approver: Mechanical V3-D gate under user authorization, 2026-08-04.

## D-077 — Freeze an evidence-gated ACR Version 4 redesign protocol

- Classification: `DECISION`
- Status: ACTIVE
- Decision: Preserve V3-D as negative and do not relax, rerun, retime, filter,
  or reinterpret either failed gate. Before another rollout, require a
  materially changed V4 method with two separately attributable mechanisms:
  a generic safety-constrained controller targeting at least 35% realized
  scene reuse, and a numerically verified faster fixed-shape single-view
  executor. Require controller-only, executor-only, and complete-method
  ablations. Promote only after a bounded gate demonstrates at least 12%
  visual-CUDA reduction, wall ratio at most 0.98 versus BFR, and wall ratio at
  most 0.95 versus sequential FR while preserving correctness. Reserve Goal
  for independent confirmation and keep all final populations protected.
- Phase policy: V4-A and V4-B are CPU-only. V4-C is capped at 96 model queries
  and zero simulator episodes. V4-D is capped at 200 paired Object-development
  attempts. V4-E is capped at 300 Goal-confirmation attempts and is eligible
  only after V4-D passes. Each phase stops for separate authorization.
- Evidence: `docs/ACR_V4_REDESIGN_PROTOCOL.md`;
  `configs/acr/v4_redesign_freeze.json`; preserved V3-D evidence in
  `reports/PHASE_V3_D_REPORT.md` and `reports/runtime/acr_v3_d.json`.
- Approver: User request to create the required protocol before trying again,
  2026-08-10. This decision authorizes no V4-A work, GPU/model query,
  simulator episode, download, protected outcome, or manuscript edit.

## D-078 — Authorize V4-A and freeze its output-blind preflight

- Classification: `DECISION`
- Status: ACTIVE
- Decision: Authorize CPU-only Phase V4-A diagnosis of already-opened V3-D,
  A4, and A5 evidence. Before candidate outputs, freeze six controller replay
  candidates from three threshold interpolation levels and two generic
  transition policies, all with warm-up 2, horizon 2, and a 40% prefix reuse
  budget. Freeze episode-cluster bootstrap uncertainty, deterministic
  selection, source profiling, executor feasibility, and negative stop rules.
- Executor rule: Evaluate a project-owned complete fixed-shape reuse-query
  compile/CUDA-Graph boundary first, then a wrist-encoder/projector boundary;
  stop if neither can plausibly support the required wall margin. This is a
  source-feasibility decision only, not a performance claim.
- Resources: Zero GPU, model queries, simulator episodes, downloads, new
  outcomes, Goal/final access, production implementation, or manuscript edits;
  at most 512 MiB of new artifacts.
- Evidence: `docs/ACR_V4_A_PREFLIGHT.md` and
  `configs/acr/v4_a_diagnosis_preflight.json`.
- Approver: User, 2026-08-10. V4-B remains unauthorized.

## D-079 — Preserve the V4-A pre-analysis stop and freeze one recovery

- Classification: `DECISION`
- Status: ACTIVE
- Decision: Preserve the first V4-A analyzer invocation as a technical stop
  during V3-D completion-metadata reconciliation, before A4 loading, candidate
  replay, bootstrap, selection, or output creation. Correct only the generic
  verifier's treatment of the intentionally hashless V3-D completion manifest.
  Keep all hashed queries, episodes, and summary verification plus every
  scientific design and gate unchanged. Allow one complete CPU-only recovery
  after merge and synchronization.
- Evidence: `docs/ACR_V4_A_TECHNICAL_RECOVERY.md` and
  `configs/acr/v4_a_recovery.json`; absent
  `results/acr-v4a-diagnosis-v01` at the stop.
- Approver: Mechanical fail-closed recovery under the user-authorized V4-A
  phase, 2026-08-10.

## D-080 — Preserve V4-A recovery 1 and freeze recovery 2

- Classification: `DECISION`
- Status: ACTIVE
- Decision: Preserve recovery 1 after volatile replay/bootstrap computation
  but before result construction, writing, or printing. Candidate values were
  not reported and the output root remained absent. Correct only A5 integrity
  validation by delegating to the original committed A5 analyzer and matching
  its recomputed aggregate record hash to the published Stage-1 analysis.
  Allow one complete CPU-only recovery 2 after merge and synchronization.
- Evidence: `docs/ACR_V4_A_TECHNICAL_RECOVERY_2.md` and
  `configs/acr/v4_a_recovery_2.json`; absent
  `results/acr-v4a-diagnosis-v01` at the stop.
- Approver: Mechanical fail-closed recovery under the user-authorized V4-A
  phase, 2026-08-10.

## D-081 — Apply the V4-A negative mechanism-selection stop

- Classification: `DECISION`
- Status: ACTIVE
- Decision: Accept the complete V4-A recovery-2 evidence and stop Version 4
  before V4-B. All six frozen candidates are ineligible. Preserve that the
  strongest candidate reached 35.76% replay reuse and 12.15% predicted
  visual-CUDA reduction but produced a maximum reuse streak of two rather than
  the frozen maximum of one. Preserve that every direction-reversal-veto
  candidate fell below the 35% reuse and 12% predicted visual-reduction
  targets. Select no controller or executor.
- Integrity rule: Do not post hoc change horizon 2 to horizon 1, relax the
  maximum-streak gate, add or remove a candidate, reinterpret source
  feasibility as measured speed, or advance to implementation. Any future
  route requires a new output-blind protocol and separate authorization.
- Resources: Zero GPU, model query, simulator episode, download, protected
  outcome, or manuscript change.
- Evidence: `reports/PHASE_V4_A_REPORT.md`;
  `reports/runtime/acr_v4_a.json`; semantic SHA-256
  `e7749e524ea39674a31654204dc879002b129fb8dfef6d89e66e89a38a22ffd8`.
- Approver: Mechanical V4-A gate under user authorization, 2026-08-10.

## D-082 — Freeze the research-first V5 isolated-reuse correction

- Classification: `DECISION`
- Status: ACTIVE
- Decision: Preserve V3/V4 and implement a separately versioned
  Isolated-Reuse State-Aware ACR controller. A completed reuse sets an internal
  latch that forces the next completed query to refresh. Require horizon 1 and
  external cache-age/latch agreement as independent fail-closed checks. Clear
  the latch only after a successfully observed refresh.
- Research basis: Primary VLA caching/adaptive-compute work supports temporal
  redundancy, action-context gating, and fresh task-relevant perception;
  corrective/event-triggered work motivates explicitly bounding stale
  intervals. None establishes one-step reuse safety for this stack, so the
  mechanism remains a project hypothesis requiring later evaluation.
- Exclusions: No threshold/replay selection, executor implementation, GPU,
  model query, simulator episode, download, new outcome, protected access, or
  manuscript change. Legacy ACR behavior and all immutable evidence remain
  unchanged.
- Evidence: `docs/ACR_V5_RESEARCH_AUDIT.md`;
  `docs/ACR_V5_ISOLATED_REUSE_PROTOCOL.md`;
  `configs/acr/v5_isolated_reuse_freeze.json`.
- Approver: User instruction to make the correction after thorough research,
  2026-08-10.

## D-083 — Accept the V5 isolated-reuse CPU correction

- Classification: `DECISION`
- Status: ACTIVE
- Decision: Accept the separately versioned IR-SA-ACR implementation as a
  software-correctness checkpoint. It forces one successfully completed scene
  refresh after each reuse, requires horizon 1, cross-checks external cache age
  against its internal latch, rejects forged consecutive reuse, and exposes
  auditable/resettable state. Legacy ACR behavior remains unchanged.
- Evidence: The deterministic verifier completes 128 corrected queries with 51
  reuses, a 0.40 maximum prefix fraction, and maximum streak one; the preserved
  legacy trace reaches streak two. Batched-adapter and adversarial CPU tests
  pass. Machine semantic SHA-256 is
  `7dcde7e8b96ba7fe79f1eed0cd6a73661e0d0977678f3581062902b445f7de2b`.
- Claim boundary: This proves controller semantics only. It does not select a
  threshold or establish task success, reuse rate on benchmark traces, CUDA or
  wall-time efficiency, or a positive paper result.
- Resources: Zero GPU, model query, simulator episode, download, new outcome,
  protected access, or manuscript change.
- Evidence files: `reports/PHASE_V5_A_CORRECTION_REPORT.md`;
  `reports/runtime/acr_v5_cpu_verification.json`;
  `scripts/verify_acr_v5_isolation.py`.
- Approver: Mechanical V5-A CPU gate under the user's research-first
  correction authorization, 2026-08-10.

## D-084 — Formalize IR-SA-ACR and approve the gated next-step roadmap

- Classification: `DECISION`
- Status: ACTIVE
- Decision: Treat `docs/ACR_V5_FORMAL_METHOD_SPECIFICATION.md` as the exact
  prose/mathematical definition of the implemented method; preserve a complete
  file-level provenance ledger; use the manuscript translation guide to prevent
  unsupported claims; and follow the V5-B through V5-H gated roadmap.
- Authorization: The user's instruction to fully document the method and
  approval of logical next steps authorizes preparation of a frozen,
  output-blind V5-B screening protocol. No replay output may be read before the
  freeze. Passing gates, protected-data boundaries, download controls, and
  shared-server safety are not waived.
- Scientific boundary: V5 remains a software-correctness result. Task success,
  benchmark reuse, visual-work reduction, GPU speed, and a positive-paper claim
  remain unmeasured hypotheses.
- Server boundary: Before any GPU is selected, stop for explicit user
  coordination under `AGENTS.md`; never inspect or interfere with unrelated
  university work.
- Evidence: `docs/ACR_V5_FORMAL_METHOD_SPECIFICATION.md`;
  `docs/ACR_V5_IMPLEMENTATION_AND_PROVENANCE_LEDGER.md`;
  `docs/ACR_V5_MANUSCRIPT_TRANSLATION_GUIDE.md`;
  `docs/ACR_V5_GATED_EVALUATION_ROADMAP.md`.
- Approver: User, 2026-08-10.

## D-085 — Record and correct the documentation-sync path deviation

- Classification: `DEVIATION`
- Status: CORRECTED
- Event: During post-merge TITAN verification, the agent briefly directed its
  own generated semantic-verifier output to
  `/tmp/savr-v5-doc-sync-verify.json`, outside the permitted
  `/home/ved/SAVR` boundary.
- Correction: The exact generated file was immediately removed and its absence
  verified. No unrelated file, directory, process, allocation, permission, or
  configuration was inspected or changed.
- Prevention: Future remote verification must stream output or use a path
  beneath `/home/ved/SAVR`; shell redirection to external temporary paths is
  prohibited.
- Evidence: Agent command record and
  `docs/ACR_V5_IMPLEMENTATION_AND_PROVENANCE_LEDGER.md`.
- Recorder: Codex, 2026-08-10.

## D-086 — Freeze V5-B output-blind isolated-reuse screening

- Classification: `DECISION`
- Status: ACTIVE
- Decision: Screen exactly six IR-SA-ACR candidates on the frozen,
  outcome-free A4 Full-Refresh Object traces. Use threshold levels
  `1.0/1.5/2.0`, hard caps `0.35/0.40`, horizon one, the controller-owned
  post-reuse latch, and no direction-reversal veto. Require deterministic
  replay, maximum streak one, cache/gripper integrity, reuse/work margins, and
  select the least permissive eligible candidate.
- Prior-evidence disclosure: The anchors originate from A4/A5 and V4 informed
  the latch correction and exclusion of the direction-reversal primary
  variant. V5-B is development screening, not independent confirmation.
- Input boundary: Exactly 100 episodes and 1,773 trace records with ordered
  path/content SHA-256
  `3ce22a1d1de7d33ed0a6bcdb52b32f42800d732ec93aed0bfed593f1e536b34b`.
  The loader rejects success/failure/reward/timing fields. Goal, reserve, and
  final populations remain sealed.
- Resources: Zero GPU, model query, simulator episode/reset, download, or new
  task outcome; CPU wall cap 1,800 seconds and artifact cap 256 MiB.
- Stop rule: No eligible candidate stops V5 before executor implementation.
  One selected candidate permits only V5-C protocol preparation.
- Evidence: `docs/ACR_V5_B_OUTPUT_BLIND_PREFLIGHT.md`;
  `configs/acr/v5_b_output_blind_preflight.json`;
  `scripts/analyze_acr_v5_b.py`; `scripts/verify_acr_v5_b_result.py`.
- Approver: User, 2026-08-10; frozen before candidate output.

## D-087 — Accept V5-B and select `v5-a100-b40`

- Classification: `DECISION`
- Status: ACTIVE
- Decision: Accept the complete output-blind V5-B screening and mechanically
  select `v5-a100-b40`, the least permissive of three eligible candidates.
- Evidence: 629/1,773 reuses (`0.3547659334`), episode-bootstrap 95% interval
  `[0.3418062250, 0.3654066607]`; logical visual reduction `0.1773829667`,
  interval `[0.1709031125, 0.1827033303]`; maximum streak one; zero prefix-cap
  violation, gripper-transition reuse, isolation mismatch, or invariant
  failure. Both complete replays were byte-identical and the independent
  verifier returned zero errors.
- Selection: Eligible candidates were `v5-a100-b40`, `v5-a150-b40`, and
  `v5-a200-b40`. The frozen rule first minimizes threshold level, selecting
  `v5-a100-b40` without using success outcomes.
- Claim boundary: Positive offline mechanism evidence only. It does not prove
  online task success, measured CUDA reduction, wall-time speed, or a positive
  paper result.
- Resources/protection: Zero GPU/model/simulator/download/new outcome; success
  fields, Goal, reserve, and final populations remained sealed.
- Evidence: `reports/runtime/acr_v5_b.json` (semantic SHA-256
  `8a9f15b818b58ed2868d4b1123a222a4c062507161ab7de911d8d233f3b1efec`);
  `reports/PHASE_V5_B_REPORT.md`.
- Disposition: `ADVANCE_TO_V5_C_PROTOCOL`.
- Approver: Mechanical frozen V5-B gate under user authorization, 2026-08-10.

## D-088 — Freeze the V5-C split-core static executor

- Classification: `DECISION`
- Status: ACTIVE
- Decision: Preserve `v5-a100-b40` unchanged and implement a project-owned
  static-buffer executor with two fixed-shape cores: fresh wrist visual
  encoding/projecting and fresh downstream language-model/action-head
  computation. Host controller/cache checks, input copies, CPU action transfer,
  and NumPy unnormalization remain outside the cores.
- Research basis: CUDA graph replay requires stable arguments/pointers and
  excludes synchronous/dynamic host operations. The pinned `predict_action`
  includes `.cpu().detach().numpy()` and NumPy processing, so whole-function
  capture is rejected. Wrist-only optimization is retained but is unlikely by
  itself to meet the required end-to-end margin.
- Quantitative target: A later GPU phase needs a reuse/BFR wall ratio near
  `0.930989` at the V5-B reuse lower bound to reach weighted/BFR `0.98`. This is
  a feasibility target derived from prior development measurements, not a new
  result.
- Safety: Exact compatibility key; owned stable buffers; non-reentrant
  lifecycle; prelaunch unavailability forces refresh; postlaunch failure
  invalidates cache/executor, does not observe the controller, and cannot retry;
  exception-safe restoration is mandatory.
- Scope: CPU implementation/tests only after merge. No compile/CUDA graph/GPU,
  model, simulator, timing, download, new outcome, upstream modification,
  protected access, or manuscript change.
- Evidence: `docs/ACR_V5_C_EXECUTOR_RESEARCH_AND_DESIGN.md`;
  `docs/ACR_V5_C_CPU_EXECUTOR_PROTOCOL.md`;
  `configs/acr/v5_c_cpu_executor_freeze.json`.
- Approver: User, 2026-08-10; frozen before implementation.

## D-089 — Accept V5-C CPU executor correctness

- Classification: `DECISION`
- Status: ACTIVE
- Decision: Accept the frozen V5-C software contract as implemented and
  mechanically verified. Permit only V5-D protocol preparation; do not infer
  GPU capture feasibility, latency benefit, memory fit, or task success.
- Evidence: Exact eager/static wrist, scene-first combined-token, and
  normalized-action parity; stable owned buffers; all compatibility/lifecycle,
  failure, cache/controller, reset, and restoration gates; 293 local tests;
  deterministic TITAN semantic SHA-256
  `f7a8d11d4574add57caa630c03463375421d9482984478be769f497b1c9d0b66`.
- Evidence files: `reports/PHASE_V5_C_REPORT.md`;
  `reports/runtime/acr_v5_c_cpu_executor_verification.json`;
  `scripts/verify_acr_v5_c_executor.py`.
- Scope: Zero GPU/model/simulator/download/new outcome/protected access;
  manuscript unchanged.
- Disposition: `ADVANCE_ONLY_TO_V5_D_PROTOCOL_PREPARATION`.
- Approver: Mechanical frozen V5-C gate under user authorization, 2026-08-10.

## D-090 — Freeze bounded V5-D real-tensor feasibility protocol

- Classification: `DECISION`
- Status: ACTIVE
- Decision: Preserve `v5-a100-b40` and the V5-C split-core contract. Freeze a
  compiler-first, raw-CUDA-graph technical waterfall with no backend shopping,
  seven correctness queries, eight warm-ups, all 24 balanced four-path
  permutations, 96 timed queries, and a 111-query hard cap.
- Statistical gates: 10,000 paired block bootstraps with seed `20260810`;
  optimized-reuse/BFR wall median at most `0.930988756983`; weighted wall and
  total-CUDA upper 95% at most `0.98`; optimized/eager sequential-CUDA upper
  95% at most `0.96`; weighted visual reduction lower 95% at least `0.10`;
  refresh/BFR upper 95% at most `1.02`; order deviation at most `0.03`.
- Safety: Exact pinned source/checkpoint/environment hashes; no raw fallback
  after correctness begins; one GPU/process; 23 GiB peak and 6 GiB incremental
  reserved-memory caps; no retry, simulator, outcome, download, upstream edit,
  or manuscript change. GPU selection is deferred until explicit user
  coordination after implementation merges.
- Evidence: `docs/ACR_V5_D_RESEARCH_AND_MEASUREMENT_DESIGN.md`;
  `docs/ACR_V5_D_GPU_FEASIBILITY_PROTOCOL.md`;
  `configs/acr/v5_d_gpu_feasibility_freeze.json` (semantic SHA-256
  `f445cf5d1a5ec6877ebea46ccc3883a11a676b38cb33a711ee4b74baf22f53f8`);
  `reports/PHASE_V5_D_PROTOCOL_REPORT.md`.
- Scope used: Zero GPU/model/simulator/download/new outcome/protected access;
  manuscript unchanged. Read-only hashes were checked only inside
  `/home/ved/SAVR`.
- Disposition: `ADVANCE_ONLY_TO_V5_D_BACKEND_IMPLEMENTATION_AFTER_USER_AUTHORIZATION`.
- Approver: User, 2026-08-10; frozen before implementation/output.

## D-091 — Accept V5-D pre-GPU implementation

- Classification: `DECISION`
- Status: ACTIVE
- Decision: Accept the separately implemented real mixed-dtype executor,
  pinned wrist/downstream cores, compiler/raw waterfall, aggregate-only GPU
  selector, exact 111-query runner, paired analyzer, independent verifier, and
  deterministic preflight. Stop before GPU selection.
- Corrections before output: Replace inaccurate copied suffixes for the six
  deterministic input hashes using immutable V3-C truth; freeze semantic
  SHA-256 is now
  `f445cf5d1a5ec6877ebea46ccc3883a11a676b38cb33a711ee4b74baf22f53f8`.
  Add V5-D-only mixed-dtype executors instead of changing validated V5-C code.
  Record implicit capture-end graph instantiation for pinned PyTorch 2.2,
  which lacks a public `CUDAGraph.instantiate()` method.
- Evidence: `docs/ACR_V5_D_BACKEND_IMPLEMENTATION.md`;
  `reports/PHASE_V5_D_IMPLEMENTATION_REPORT.md`;
  `reports/runtime/acr_v5_d_preflight.json` (semantic SHA-256
  `db097ca8cab44d474a65e22888a72da8c4c6e2489a31188abea67c7ed55bff98`).
- Scope used: Zero GPU/model/simulator/new outcome/protected access and zero
  model, dataset, or TITAN download; manuscript unchanged. TITAN inspection
  was read-only and confined to `/home/ved/SAVR`.
- Disposition: `STOP_FOR_EXPLICIT_USER_COORDINATION_BEFORE_GPU_SELECTION`.
- Approver: User approved implementation, 2026-08-10; GPU phase not inferred.

## D-092 — Preserve V5-D v01 as a zero-query technical stop

- Classification: `DECISION`
- Status: ACTIVE
- Decision: Preserve `acr-v5d-real-tensor-feasibility-v01` without retry.
  Classify its non-interactive LIBERO import `EOFError` as a launcher/preflight
  defect before model load, not positive or negative method evidence.
- Evidence: Three aggregate samples selected physical GPU 0 with 6 MiB used
  and 0% utilization. The launch then stopped because the run-local
  `LIBERO_CONFIG_PATH` lacked `config.yaml`, causing LIBERO's first-use prompt.
  Model queries, backend-preparation launches, correctness/warm-up/timed
  records, simulator calls, downloads, and outcomes were all zero.
- Evidence files: `reports/PHASE_V5_D_V01_TECHNICAL_STOP_REPORT.md`;
  `reports/runtime/acr_v5_d_v01_technical_stop.json` (semantic SHA-256
  `edf5872fa818f5806601f52143cb17cec7dd4974e03cc4e2ed43c3d042fb4412`);
  `docs/ACR_V5_D_V02_RECOVERY_PLAN.md`.
- Protection: Source and checkpoint trees remained clean; post-stop selected
  GPU telemetry was 6 MiB used and 0% utilization. No task outcome or
  manuscript content was accessed.
- Disposition: `STOP_NO_RETRY_PREPARE_SEPARATELY_AUTHORIZED_V5D_V02`.
- Approver: User authorized v01 one-GPU entry, 2026-08-10. This decision does
  not infer authorization for v02 implementation or execution.

## D-093 — Accept V5-D v02 pre-GPU recovery implementation

- Classification: `DECISION`
- Status: ACTIVE
- Decision: Accept the new v02 run identity, compact recovery overlay,
  canonical create-once LIBERO configuration, config attestation, and outer
  pre-model zero-query technical-stop envelope. Preserve every v01 scientific
  method, schedule, tolerance, statistical, memory, resource, and claim field.
- Evidence: Resolved experiment semantic SHA-256
  `4ae65dda537a5b6dcdf9abd34d79e0a9d7defee834a2a8cc2f7107a659f36076`;
  deterministic preflight semantic SHA-256
  `d7c3ed40cc9d5760a846cb15c688fa5c776cbac8f243d948376d16e64427a695`;
  closed-stdin TITAN import semantic SHA-256
  `a3ffc574631e8e250ab8021c0f8b99e0bf329a1e82d085499fb8e19747dd3490`;
  329 local tests.
- Protection: TITAN import preflight used closed stdin and an empty
  `CUDA_VISIBLE_DEVICES`; CUDA stayed uninitialized. GPU inspections, model
  loads/queries, simulator instances/resets/episodes, downloads, and outcomes
  were zero. No manuscript change.
- Evidence files: `configs/acr/v5_d_gpu_feasibility_recovery_v02.json`;
  `reports/runtime/acr_v5_d_v02_preflight.json`;
  `reports/runtime/acr_v5_d_v02_import_preflight.json`;
  `reports/PHASE_V5_D_V02_RECOVERY_IMPLEMENTATION_REPORT.md`.
- Disposition: `STOP_FOR_EXPLICIT_USER_COORDINATION_BEFORE_V02_GPU_SELECTION`.
- Approver: User approved v02 correction implementation, 2026-08-10; v02 GPU
  execution is not inferred.

## D-094 — Preserve V5-D v02 as a pre-correctness technical stop

- Classification: `DECISION`
- Status: ACTIVE
- Decision: Preserve `acr-v5d-real-tensor-feasibility-v02` without retry and
  classify it as no method result. The pinned compiler failed on its first
  preparation call because BF16 PTX requires `sm_80` or newer while the
  selected TITAN RTX is `sm_75`. The restoration guard then blocked raw
  fallback because loader files named `.back.<timestamp>` were outside its
  cleanup allowlist.
- Evidence: The model loaded, but full model queries, correctness, warm-up,
  timing, simulator, download, and outcome counts were zero. Rewards and
  success fields were never accessed. Peak allocated/reserved bytes were
  `15768091136`/`16076767232`; post-stop GPU telemetry was 6 MiB and 0%.
- Recovery: All protected checkpoint hashes already matched their frozen
  originals. The two exact duplicate backups were hash-verified, removed, and
  the checkpoint inventory plus SAVR/OpenVLA-OFT/LIBERO trees were verified
  clean.
- Evidence files: `reports/PHASE_V5_D_V02_TECHNICAL_STOP_REPORT.md`;
  `reports/runtime/acr_v5_d_v02_technical_stop.json` (semantic SHA-256
  `0a30bd847bf2e1549c376200e559a23c670b33c0b01215926c90a15704487661`);
  `docs/ACR_V5_D_V03_RECOVERY_PLAN.md`.
- Disposition: `STOP_NO_RETRY_PREPARE_SEPARATELY_AUTHORIZED_V5D_V03`.
- Approver: User authorized v02 one-GPU entry, 2026-08-10. This decision does
  not infer authorization for v03 implementation or execution.

## D-095 — Accept V5-D v03 pre-GPU restoration recovery

- Classification: `DECISION`
- Status: ACTIVE
- Decision: Accept the v03 immutable identity and the exact checkpoint
  restoration helper. Preserve every v02 scientific method, schedule,
  tolerance, statistic, resource cap, backend order, and claim boundary.
- Correction: Permit only verified new protected-name backup files, including
  the observed `.back.YYYYMMDD_HHMMSS` form. Require baseline content for each
  backup, protected-byte restoration, exact final inventory, non-protected
  drift rejection, fail-closed partial cleanup, and idempotent revalidation.
- Evidence: Resolved semantic SHA-256
  `a9447cd385b4229e54cf85ba8fc7e06e4b4d283b9ac5c655e0c5201fb5d3f297`;
  deterministic preflight semantic SHA-256
  `f25d7f2dd743bf0c4fbe8a56420ba94d136c8cfc6b5baaeedf473e5a5a6ab163`;
  import-preflight semantic SHA-256
  `f8b9002d0998345c7f1a423003a180cfc5a406d52f85e1f3ac2d1f60fe22cdc5`;
  curated CPU-verification semantic SHA-256
  `37587ccd329cfc68672ef048a7422fb8825944a2a61260d1631b8532c2efdf95`;
  341 local tests and 341 TITAN tests plus 9 subtests.
- Process correction: The initial TITAN pytest command used pytest's default
  temporary location outside the project boundary. It was replaced by a full
  passing repeat with both temporary roots confined beneath
  `/home/ved/SAVR/results/`. No external path was inspected or manually
  modified after detection; acceptance uses only the compliant repeat.
- Protection: Zero GPU inspection/selection, CUDA initialization, model
  load/query, simulator use, download, outcome access, or manuscript change.
- Evidence files: `configs/acr/v5_d_gpu_feasibility_recovery_v03.json`;
  `reports/runtime/acr_v5_d_v03_preflight.json`;
  `reports/runtime/acr_v5_d_v03_import_preflight.json`;
  `reports/runtime/acr_v5_d_v03_cpu_verification.json`;
  `reports/PHASE_V5_D_V03_RECOVERY_IMPLEMENTATION_REPORT.md`.
- Disposition: `STOP_FOR_EXPLICIT_USER_COORDINATION_BEFORE_V03_GPU_SELECTION`.
- Approver: User approved v03 recovery implementation, 2026-08-10; GPU
  execution is not inferred.

## D-096 — Preserve V5-D v03 as a backend-environment technical stop

- Classification: `DECISION`
- Status: ACTIVE
- Decision: Preserve `acr-v5d-real-tensor-feasibility-v03` without retry.
  Classify it as no method-performance result and as decisive evidence that the
  frozen backends are infeasible on the current TITAN RTX environment.
- Compiler evidence: One preparation launch failed before correctness because
  BF16 PTX requires `sm_80` or newer and TITAN RTX is `sm_75`. Exact checkpoint
  restoration passed and raw transition was correctly authorized.
- Raw evidence: Eight preparation launches then OOMed before correctness at
  `24184212992` allocated and `24937234432` reserved bytes, exceeding the frozen
  23 GiB reservation cap by `241172480` bytes.
- Scientific boundary: Full queries, correctness, schedule warm-ups, timings,
  simulator operations, downloads, and outcomes were zero. The analyzer and
  finalizer were not run. No method claim follows.
- Protection: Both attempts restored the checkpoint exactly; no loader backup
  remains; all three source trees are clean; post-stop GPU telemetry was 6 MiB
  and 0% utilization.
- Evidence files: `reports/PHASE_V5_D_V03_TECHNICAL_STOP_REPORT.md`;
  `reports/runtime/acr_v5_d_v03_technical_stop.json` (semantic SHA-256
  `1016569f642b21266e8f0b75b5906716200055f5d37385c5501b6711f9a6bd54`);
  `docs/ACR_V5_D_V04_ENVIRONMENT_AMENDMENT_PLAN.md`.
- Disposition:
  `STOP_NO_RETRY_PREPARE_SEPARATELY_AUTHORIZED_V5D_V04_ENVIRONMENT_AMENDMENT`.
- Approver: User authorized v03 one-GPU execution, 2026-08-11. This decision
  does not authorize another cluster, environment amendment, or v04 run.

## D-097 — Freeze an isolated same-TITAN V04 graph-pool remediation

- Classification: `DECISION`
- Status: ACTIVE
- Decision: Keep V03 immutable and evaluate one narrow V04 remediation on the
  existing `ssh titan` host: share one PyTorch private pool across the wrist
  and downstream raw CUDA graphs.
- Research basis: PyTorch 2.2 documents pool sharing as safe only when graphs
  replay in capture order and never concurrently. The optimized SAVR executor
  is structurally wrist-then-downstream. V04 also enforces one capture stream,
  one replay stream, exact ordering, pointer identity, nonconcurrency, and
  fail-closed invalidation.
- Frozen boundary: Preserve the compiler-first waterfall, checkpoint, method,
  tensors, 111-query schedule, correctness/timing/statistical gates, and 23 GiB
  cap. No allocator tuning, cap increase, reduced warm-ups, multi-GPU use,
  quantization, offload, threshold change, or backend shopping.
- Isolation correction: A first implementation changed hashed V03 files and
  was rejected by the immutable V03 verifier. V04 was moved to separate
  overlay, backend, adapter, runner, and test modules; V03 files were restored
  byte-for-byte.
- Verification: 347 local tests; two passing GitHub validation jobs; 7 focused
  TITAN tests; deterministic CUDA-free preflight
  `30899e753a50f0d8e293f81f435de56a4f51dccf41511b50316a4051c2719dda`;
  CUDA-hidden pinned import/API preflight
  `b467a4783d8dc67b5a6e445a6099cc12109f222ca3184f0240bec61ef22df019`;
  curated CPU verification
  `60a5c44647ad1d699ee32ddbe2bd64da95bcb020a33145fda6c35c04299105cd`.
- Evidence files: `docs/ACR_V5_D_V04_TITAN_MEMORY_REMEDIATION_PROTOCOL.md`;
  `configs/acr/v5_d_titan_memory_recovery_v04.json`;
  `reports/PHASE_V5_D_V04_MEMORY_REMEDIATION_IMPLEMENTATION_REPORT.md`;
  `reports/runtime/acr_v5_d_v04_preflight.json`;
  `reports/runtime/acr_v5_d_v04_import_preflight.json`;
  `reports/runtime/acr_v5_d_v04_cpu_verification.json`.
- Protection: Zero GPU inspection, CUDA initialization, model query, simulator,
  download, task outcome, or manuscript change.
- Disposition: `STOP_FOR_EXPLICIT_USER_COORDINATION_BEFORE_V04_GPU_SELECTION`.
- Approver: User authorized research, logical planning, and pre-GPU work on
  2026-08-11. GPU selection remains separately coordinated by repository rule.

## D-098 — Preserve V5-D v04 as a pre-raw transition technical stop

- Classification: `DECISION`
- Status: ACTIVE
- Decision: Preserve `acr-v5d-real-tensor-feasibility-v04` without automatic
  retry and classify it as no method-performance or memory-feasibility result.
- Compiler evidence: The model loaded; one preparation launch repeated the
  expected BF16 PTX failure on TITAN RTX `sm_75`; exact restoration passed and
  a fresh-process raw transition was authorized.
- Transition evidence: The raw process stopped before model load because its
  single immediate GPU snapshot showed 6 MiB used but 33% utilization, above
  the frozen 5% threshold. Later aggregate-only telemetry was 6 MiB and 0% on
  the same physical GPU and UUID.
- Scientific boundary: Raw preparation, full queries, correctness, warm-up,
  timings, simulator operations, downloads, and outcomes were zero. Shared-pool
  feasibility was not tested; no positive or negative method claim follows.
- Protection: Checkpoint hashes and inventories were exact, no loader backup
  remained, all three source trees were clean, and no unrelated process or
  allocation was inspected.
- Evidence files: `reports/PHASE_V5_D_V04_TECHNICAL_STOP_REPORT.md`;
  `reports/runtime/acr_v5_d_v04_technical_stop.json` (semantic SHA-256
  `a3515180022df7938b50956851a2ca05b698819da38b387ddc23b54e59769811`).
- Disposition:
  `STOP_NO_AUTOMATIC_RETRY_PREPARE_SEPARATELY_FROZEN_TRANSITION_RECOVERY`.
- Approver: User authorized the one-GPU V04 execution on 2026-08-11. This does
  not permit result shopping or mutation of the immutable V04 run.

## D-099 — Freeze and accept V5-D v05 transition recovery before GPU use

- Classification: `DECISION`
- Status: ACTIVE
- Decision: Keep V04 immutable and create V05 solely to correct the raw
  process's transition revalidation. Discard two seconds, then require exactly
  three selected-GPU aggregate samples five seconds apart; every sample must
  pass the unchanged 512 MiB and 5% limits and exact index/UUID checks.
- Research basis: NVIDIA defines device utilization over a recent sample
  period between one sixth and one second. The two-second discard excludes that
  documented window before the sustained-idle samples begin.
- Frozen boundary: Preserve V04's method, shared-pool backend, compiler-first
  waterfall, checkpoint/tensors, 111-query schedule, correctness/statistical
  gates, 23 GiB cap, and claim limits. No extra sampling window, GPU switch,
  process/allocation inspection, threshold relaxation, or automatic retry.
- Verification: 353 local tests; both CI jobs on PRs #82 and #83; 6 focused
  TITAN tests; 353 TITAN tests plus 9 subtests; deterministic preflight
  `67c641228f406b8048cacf813b52cc66ef9cc6e7249ab99c0512a7d1fc4cf101`;
  CUDA-hidden import preflight
  `0b71455a193e906fd68b05e89d48b72277b91a2554440ea31e0be85bd050fdb2`;
  curated CPU verification
  `7fad244f58140616ef7abebfb8a907b78f156bd58586e5ab096f5a46260c3dab`.
- Corrections: Preserve and exclude the first nested-basetemp setup failure and
  relative-`PYTHONPATH` import-preflight failure. Independent corrected repeats
  passed without GPU visibility or scientific output.
- Evidence files: `docs/ACR_V5_D_V05_TRANSITION_RECOVERY_PROTOCOL.md`;
  `configs/acr/v5_d_transition_recovery_v05.json`;
  `reports/PHASE_V5_D_V05_TRANSITION_RECOVERY_IMPLEMENTATION_REPORT.md`;
  `reports/runtime/acr_v5_d_v05_preflight.json`;
  `reports/runtime/acr_v5_d_v05_import_preflight.json`;
  `reports/runtime/acr_v5_d_v05_cpu_verification.json`.
- Disposition: `STOP_FOR_EXPLICIT_USER_COORDINATION_BEFORE_V05_GPU_SELECTION`.
- Approver: User authorized logical forward progress after V04 on 2026-08-11;
  repository safety rules still require a separate pause before GPU selection.

## D-100 — Preserve V5-D v05 as a shared-pool memory technical stop

- Classification: `DECISION`
- Status: ACTIVE
- Decision: Preserve `acr-v5d-real-tensor-feasibility-v05` without retry and
  classify it as no method-performance result.
- Transition evidence: The new gate passed all three fixed samples at 6 MiB and
  0% with stable index/UUID, establishing that V04's transition rejection was
  corrected without threshold relaxation.
- Compiler evidence: One preparation launch repeated the expected BF16/PTX
  incompatibility on TITAN RTX `sm_75`; exact restoration passed and raw
  transition was authorized.
- Raw evidence: Wrist warm-up/capture completed. Downstream warm-up then OOMed
  on a 14 MiB request before downstream capture. Peak reserved memory was
  24,939,331,584 bytes (23.2266 GiB), 243,269,632 bytes above the 23 GiB cap.
- Mechanistic boundary: The completed capture order contains only `wrist`.
  Shared-pool reuse could not govern downstream capture because downstream
  warm-up exhausted memory first.
- Scientific boundary: Full queries, correctness, schedule warm-ups, timing,
  simulator operations, downloads, and outcomes were zero. No positive or
  negative method claim follows.
- Protection: Both attempts restored exact checkpoint state; no loader backup
  remains; all source trees are clean; post-stop GPU telemetry was 6 MiB/0%; no
  unrelated process or allocation was inspected.
- Evidence files: `reports/PHASE_V5_D_V05_TECHNICAL_STOP_REPORT.md`;
  `reports/runtime/acr_v5_d_v05_technical_stop.json` (semantic SHA-256
  `cb6d9120fc2e6ee69aaa83d677598d21741be8eaf5a3456bc21461d30eb3cc3f`).
- Disposition:
  `STOP_NO_RETRY_RESEARCH_SEPARATELY_FROZEN_PRECAPTURE_WARMUP_OR_COMPATIBLE_HARDWARE`.
- Approver: User explicitly authorized the single V05 GPU attempt on
  2026-08-11. This does not authorize a V05 retry or a new V06 system change.

## D-101 — Preserve V5-D v06 as a pre-capture memory technical stop

- Classification: `DECISION`
- Status: ACTIVE
- Decision: Preserve `acr-v5d-real-tensor-feasibility-v06` without retry and
  classify it as no method-performance result.
- Raw evidence: All wrist pre-capture warm-ups completed. Downstream warm-up
  then OOMed before either capture, peaking at 24,941,428,736 reserved bytes,
  245,366,784 bytes above the unchanged 23 GiB cap.
- Scientific boundary: Full queries, correctness, schedule warm-ups, timings,
  simulator operations, downloads, and outcomes were zero.
- Evidence files: `reports/PHASE_V5_D_V06_TECHNICAL_STOP_REPORT.md` and
  `reports/runtime/acr_v5_d_v06_technical_stop.json` (semantic SHA-256
  `0588f628a118a2f467215c2337bc23452f3b8e98d0b5865c37be0d2892a18edb`).
- Disposition:
  `STOP_NO_RETRY_V06_MEMORY_INFEASIBLE_ON_TITAN_RTX_USE_COMPATIBLE_HIGHER_MEMORY_HARDWARE_OR_SEPARATELY_RESEARCHED_SYSTEM_CHANGE`.
- Approver: User authorized the single V06 GPU attempt on 2026-08-11.

## D-102 — Preserve V5-D v07 as an allocator-recovery technical stop

- Classification: `DECISION`
- Status: ACTIVE
- Decision: Preserve `acr-v5d-real-tensor-feasibility-v07` without retry and
  classify it as no method-performance result.
- Mechanism evidence: The exact `expandable_segments:True` setting was active.
  Peak reserved-minus-allocated memory fell by 457,361,408 bytes versus V06,
  but peak reservation fell by only 4,194,304 bytes because active allocated
  memory increased.
- Raw evidence: Downstream pre-capture warm-up OOMed on a 22 MiB request before
  either capture. Peak reservation was 24,937,234,432 bytes, 241,172,480 bytes
  above the unchanged 23 GiB cap.
- Scientific boundary: Full queries, correctness, schedule warm-ups, timings,
  simulator operations, downloads, success fields, and outcomes were zero.
  This is environment-feasibility evidence only.
- Protection: Both processes restored exact checkpoint state, no loader backup
  remains, the repository is clean, and post-stop GPU telemetry was 6 MiB/0%.
- Evidence files: `reports/PHASE_V5_D_V07_TECHNICAL_STOP_REPORT.md` and
  `reports/runtime/acr_v5_d_v07_technical_stop.json` (semantic SHA-256
  `17c6c68ed075f6848768d81eb158ae1d522b2b670df37d0c1db3ab54439bc8c1`).
- Disposition:
  `STOP_NO_RETRY_V07_ALLOCATOR_IMPROVED_FRAGMENTATION_BUT_MEMORY_INFEASIBLE_ON_TITAN_RTX`.
- Approver: User authorized the single V07 GPU attempt on 2026-08-11. V5-E
  remains unauthorized.

## D-103 — Freeze V5-D V08 official inference-semantics recovery

- Classification: `DECISION`
- Status: ACTIVE
- Decision: Preserve V07 and create V08 with one raw-process change: enter
  `torch.inference_mode()` after transition revalidation and before model
  initialization, retain it through preparation/capture/correctness/timing,
  and restore the prior thread state in `finally`.
- Root cause: The official pinned OpenVLA evaluator uses inference mode. The
  custom V5-D path used `model.eval()` without disabling gradient tracking.
  Pinned CPU evidence showed repeated static copies retained a growing
  `CopyBackwards` graph (`10 -> 16 -> 22` nodes); inference mode retained none.
- Frozen boundary: Preserve V07's allocator, model/checkpoint, tensors, raw
  backend, lifecycle, 111-query schedule, gates, 23 GiB cap, and claim limits.
  No decoder pruning, cache/output change, quantization, offload, sharding,
  reduced warm-up, cap increase, simulator, or outcome access.
- Verification: 372 local tests; both PR #91 checks; six focused TITAN tests;
  372 TITAN tests plus nine subtests; deterministic preflight semantic SHA-256
  `6c4c6dbfaf01549c2fb58ba330feb952dfc402064ea4d7e2fca81d4ea1782503`;
  CPU mechanism semantic SHA-256
  `154cc8cf8005f90ee3df7669c823f53048d9b3c2f95662538a2810e6faf7eff5`.
- Protection: Zero GPU inspection/selection, CUDA initialization, model query,
  simulator, download, task outcome, protected population, or manuscript edit.
- Evidence: `docs/ACR_V5_D_V08_INFERENCE_SEMANTICS_PROTOCOL.md` and
  `reports/PHASE_V5_D_V08_INFERENCE_RECOVERY_IMPLEMENTATION_REPORT.md`.
- Disposition: `STOP_FOR_EXPLICIT_USER_COORDINATION_BEFORE_V08_GPU_SELECTION`.
- Approver: User approved research and logical forward progress on 2026-08-11;
  GPU selection remains separately coordinated under repository safety rules.

## D-104 — Authorize one frozen V5-D V08 GPU attempt

- Classification: `DECISION`
- Status: ACTIVE
- Decision: Seal the user's explicit 2026-08-11 approval for one V08
  aggregate-only GPU selection and one fail-closed execution with at most 111
  frozen model queries.
- Unchanged boundary: Preserve the V08 method, inference-semantics correction,
  allocator, model/checkpoint, tensors, warm-up/capture lifecycle, schedule,
  gates, 23 GiB cap, and claim limits exactly.
- Exclusions: No automatic retry, simulator, protected outcome, final
  population, manuscript edit, V5-E advance, process-identity inspection, or
  unrelated server interference is authorized.
- Required disposition: Preserve and reconcile the single attempt whether it
  completes or stops technically. Any new attempt requires a separately frozen
  protocol and new authorization.
- Approver: User explicitly replied `approve` after being asked to authorize
  V08 GPU selection and its single fail-closed execution.

## D-105 — Preserve V5-D V08 as memory recovery plus capture technical stop

- Classification: `DECISION`
- Status: ACTIVE
- Decision: Preserve V08 without retry. Its exact inference state reduced peak
  reservation from V07's 23.2246 GiB to 15.2773 GiB, confirming that the prior
  preparation-memory blocker was caused by missing inference semantics.
- Technical stop: Both frozen pre-capture warm-up stages and the wrist graph
  capture completed. The downstream graph capture then reported an operation
  failure caused by a previous CUDA capture error. The immutable record does
  not isolate the originating kernel or operation.
- Scientific boundary: Zero full queries, correctness records, scheduled
  warm-ups, timings, simulator operations, success fields, or outcomes exist.
  V08 is mechanism evidence, not a method-performance result.
- Protection: Checkpoint bytes and inference thread state were restored, no
  loader backup remains, the repository is clean, and selected GPU 0 returned
  to 6 MiB/0% aggregate telemetry.
- Evidence: `reports/PHASE_V5_D_V08_TECHNICAL_STOP_REPORT.md` and
  `reports/runtime/acr_v5_d_v08_technical_stop.json` (semantic SHA-256
  `3572abf107ad1b0ef10557e27c66b3d5ad1d967a5f82b633c572bef907d16d98`).
- Disposition:
  `STOP_NO_RETRY_V08_MEMORY_RECOVERED_SECOND_GRAPH_CAPTURE_TECHNICAL_FAILURE`.
  A new attempt requires separately researched and frozen capture correction;
  V5-E remains unauthorized.

## D-106 — Freeze V5-D V09 default-allocator recovery

- Classification: `DECISION`
- Status: ACTIVE
- Decision: Preserve V08 and create V09 with one raw-process correction:
  remove `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and require the
  default native allocator. Record both environment absence and PyTorch's
  observed allocator backend.
- Rationale: V08's inference correction left 7.7227 GiB below the unchanged
  cap, so V07's experimental memory workaround is no longer necessary. The
  remaining failure occurred at the second shared-pool capture, and PyTorch's
  primary documentation/issues support testing allocator reversion before any
  broader model or graph-body modification.
- Frozen boundary: Preserve V08's inference lifecycle, model/checkpoint,
  tensors, warm-ups, graph bodies, stream/shared pool/order, 111 queries,
  gates, cap, restoration, and exclusions.
- Rejected confounds: No relaxed capture mode, cache/hidden-output change,
  combined graph, quantization, offload, sharding, reduced warm-up, cap change,
  simulator, outcome access, or manuscript edit.
- Verification: Six focused and 378 full local tests, both PR #95 validation
  jobs, six focused TITAN tests, 378 TITAN tests plus nine subtests, Ruff, and
  deterministic preflight semantic SHA-256
  `fa9596664fb831b46c75a6738e312277a0bf4a13bc38279e072abee3479750b2`
  pass. TITAN attested PyTorch `2.2.0+cu118`, absent allocator override,
  backend `native`, zero visible GPUs, and uninitialized CUDA.
- Evidence: `reports/runtime/acr_v5_d_v09_pre_gpu_verification.json` (semantic
  SHA-256
  `92123fdc19a0ac5de703e82340edd15ac0c859e6b607549fa28d8cee6e48d4ec`).
- Disposition: `STOP_BEFORE_V09_GPU_INSPECTION_OR_SELECTION`.
- Approver: User explicitly approved the logical next step after V08 evidence
  reconciliation. This approval does not authorize a V09 GPU attempt.

## D-107 — Authorize one frozen V5-D V09 GPU attempt

- Classification: `DECISION`
- Status: ACTIVE
- Decision: Seal the user's explicit 2026-08-12 approval for one V09
  aggregate-only GPU selection and one fail-closed execution with at most 111
  frozen model queries.
- Unchanged boundary: Preserve the V09 default-allocator correction, V08
  inference semantics, model/checkpoint, tensors, warm-up/capture lifecycle,
  graph bodies, schedule, gates, 23 GiB cap, and claim limits exactly.
- Exclusions: No automatic retry, simulator, protected outcome, final
  population, manuscript edit, V5-E advance, process-identity inspection, or
  unrelated server interference is authorized.
- Required disposition: Preserve and reconcile the single attempt whether it
  completes or stops technically. A new attempt requires a separate identity
  and authorization.
- Approver: User explicitly replied `approve` at the V09 pre-GPU checkpoint.

## D-108 — Preserve V5-D V09 as rejected allocator hypothesis and technical stop

- Classification: `DECISION`
- Status: ACTIVE
- Decision: Preserve V09 without retry. The raw process authenticated an absent
  allocator override and PyTorch's default `native` backend, but failed at the
  same downstream second-graph capture boundary as V08.
- Hypothesis result: Default-allocator reversion did not correct the failure.
  Peak reservation was 15.3848 GiB, 0.1074 GiB above V08 and 7.6152 GiB below
  the unchanged cap. Memory capacity remains solved; allocator choice is ruled
  out as the proposed correction on this pinned stack.
- Scientific boundary: Zero full queries, correctness records, scheduled
  warm-ups, timings, simulator operations, success fields, or outcomes exist.
  V09 is a technical hypothesis result, not a method-performance result.
- Protection: Checkpoint bytes and inference thread state were restored, no
  loader backup remains, the repository is clean, and selected GPU 0 returned
  to 6 MiB/0% aggregate telemetry.
- Evidence: `reports/PHASE_V5_D_V09_TECHNICAL_STOP_REPORT.md` and
  `reports/runtime/acr_v5_d_v09_technical_stop.json` (semantic SHA-256
  `2113acaad46550b26da8bbfcfe25de4e78312e55e7047974d5b555dd88316209`).
- Disposition:
  `STOP_NO_RETRY_V09_DEFAULT_ALLOCATOR_HYPOTHESIS_REJECTED_SECOND_GRAPH_CAPTURE_TECHNICAL_FAILURE`.
  Any further attempt requires a separately researched and frozen
  capture-architecture correction; V5-E remains unauthorized.

## D-109 — Freeze V5-D V10 downstream-only graph recovery protocol

- Classification: `DECISION`
- Status: ACTIVE
- Decision: Replace the failed two-graph raw backend with a hybrid executor:
  run the unchanged wrist core eagerly into owned static buffers, materialize
  scene-first combined tokens eagerly, and capture/replay only the unchanged
  downstream action core as one CUDA graph.
- Causal basis: V08 and V09 both completed wrist capture and failed at the
  downstream second-capture boundary despite different allocators and more
  than 7.6 GiB of cap headroom. Downstream as the first and only capture
  distinguishes a two-capture lifecycle defect from a downstream-body defect.
- Efficiency basis: Twelve V3-C timed reuse queries place the median derived
  downstream CUDA portion at 1076.0382 ms of 1151.7416 ms total, or 93.43%.
  The hybrid therefore targets the dominant work while preserving measurable
  eager wrist and graphed downstream components.
- Frozen boundary: Preserve `v5-a100-b40`, scene-cache semantics, model,
  checkpoint, tensors, default native allocator, inference mode, downstream
  graph body, correctness tolerances, 111-query schedule, all statistical
  gates, 23 GiB cap, restoration, and protected-data exclusions.
- Fail-closed distinction: A downstream-only capture failure rejects V10 and
  stops without graph-body edits, relaxed mode, allocator changes, or retry.
  Capture success must still pass exact correctness and every unchanged
  efficiency/integrity gate.
- Evidence: `docs/ACR_V5_D_V10_DOWNSTREAM_ONLY_GRAPH_PROTOCOL.md` and
  `configs/acr/v5_d_downstream_only_graph_recovery_v10.json` (semantic SHA-256
  `38f7ae4e65f2881c6331db6bed4614f1eafd73a23b073e297e4570b5683b2904`).
- Authorization: On 2026-08-13, the user explicitly approved implementation,
  local/CI testing, and CUDA-hidden TITAN verification. GPU inspection or
  selection, CUDA initialization, model loading/querying, simulator access,
  V5-E, protected outcomes, and manuscript changes remain unauthorized.
- Disposition: `STOP_BEFORE_V10_GPU_INSPECTION_OR_SELECTION`.

## D-110 — Accept V5-D V10 pre-GPU implementation

- Date: 2026-08-13
- Decision: Accept the isolated V10 eager-wrist plus single downstream CUDA
  graph implementation at the pre-GPU checkpoint.
- Verification: 13 focused tests, all 391 repository tests, Ruff/diff checks,
  two byte-identical 15-gate preflights, both PR #100 validation jobs, and
  TITAN CUDA-hidden reproduction at merged revision
  `f587d13b4f439fd075dc53e890641115b6abce1e` passed.
- Resource attestation: zero GPU inspection/selection, CUDA initialization,
  model query, simulator episode, download, task outcome, protected-outcome
  access, or manuscript change.
- Evidence: `reports/PHASE_V5_D_V10_DOWNSTREAM_ONLY_IMPLEMENTATION_REPORT.md`
  and `reports/runtime/acr_v5_d_v10_pre_gpu_verification.json`.
- Authorization: No V10 GPU attempt is authorized by this acceptance.
- Disposition: `STOP_BEFORE_V10_GPU_INSPECTION_OR_SELECTION`.

## D-111 — Authorize one frozen V5-D V10 GPU attempt

- Date: 2026-08-13
- Classification: `DECISION`
- Status: ACTIVE
- Decision: Seal the user's explicit approval for one V10 aggregate-only GPU
  selection and one fail-closed execution with at most 111 frozen model
  queries.
- Configuration: Resolved semantic SHA-256
  `c5a59260858004dd86c8f71d399bde116a1e10ddc585bc0cea79f59fef042911`;
  recovery semantic SHA-256
  `ba68b4dc0c46390b9be37dfdb2e759ea3c6598356b3fd7e30bb203d86ef7799f`.
- Unchanged boundary: Preserve the eager-wrist plus single downstream graph,
  V09 inference semantics and default allocator, model/checkpoint, tensors,
  schedule, gates, 23 GiB cap, and claim limits exactly.
- Exclusions: No automatic retry, simulator, protected outcome, final
  population, manuscript edit, V5-E advance, process-identity inspection, or
  unrelated server interference is authorized.
- Required disposition: Preserve and reconcile the single attempt whether it
  completes or stops technically. Any further attempt requires a new identity
  and authorization.
- Approver: User explicitly wrote `Approve the single frozen V10 GPU attempt`.

## D-112 — Preserve V5-D V10 as rejected capture-architecture hypothesis

- Date: 2026-08-13
- Classification: `DECISION`
- Status: ACTIVE
- Decision: Preserve V10 without retry. The downstream core invalidated its
  first and only CUDA graph capture after both eager warm-up stages; no wrist
  graph, retained prior graph, second capture, or shared pool existed.
- Hypothesis result: Removing the two-capture lifecycle did not resolve the
  failure. V10 therefore rejects that recovery hypothesis and leaves a
  downstream-body or broader pinned-runtime capture incompatibility.
- Scientific boundary: Zero full model queries, correctness records,
  scheduled warm-ups, timings, simulator operations, success fields, or task
  outcomes exist. This is not an ACR performance result.
- Integrity: Peak reservation was 15.3848 GiB, 7.6152 GiB below the frozen
  cap; checkpoint bytes and inference state were restored; loader backups were
  removed; GPU 0 returned to 6 MiB/0%; and the source tree remained clean.
- Evidence: `reports/PHASE_V5_D_V10_TECHNICAL_STOP_REPORT.md` and
  `reports/runtime/acr_v5_d_v10_technical_stop.json` (semantic SHA-256
  `fd72f7bed0869820e885d707264a12e4bbf3a4d97e89b8de2c3eed0a84a856d0`).
- Disposition:
  `STOP_NO_RETRY_V10_TWO_CAPTURE_HYPOTHESIS_REJECTED_DOWNSTREAM_ONLY_CAPTURE_TECHNICAL_FAILURE`.
  V5-E remains ineligible.

## D-113 — Select on-policy counterfactual cache routing for positive-direction research

- Date: 2026-08-24
- Classification: `DECISION`
- Status: SUPERSEDED_BY_D-114
- Decision: Select On-Policy Counterfactual Cache Routing (OPCCR) as the next
  research direction after auditing the completed SAVR/ACR evidence, the pinned
  OpenVLA-OFT architecture, VLA-Cache, and the closest 2025--2026 efficiency
  methods.
- Scientific rationale: The original experiments exposed cache-induced
  closed-loop distribution shift, while the timing pilot located 83.7% of query
  CUDA time downstream of the visual encoder. OPCCR therefore trains a
  reject-capable router on cache-induced trajectories using paired current-state
  full-refresh/cached outputs, and applies it to downstream visual-KV reuse.
- Novelty boundary: Do not claim novelty for KV caching, camera asymmetry,
  DAgger, confidence gating, or rejection individually. The provisional
  contribution is their specific use in on-policy counterfactual supervision
  for a fail-closed VLA cache intervention.
- Evidence boundary: This is a research decision, not a positive result. No
  OPCCR implementation, GPU operation, simulator episode, model download, or
  outcome was produced.
- Execution boundary: Proceed only through the stop-fast R1--R6 gates in
  `docs/ON_POLICY_CACHE_ROUTING_PROPOSAL.md`. R1 is CPU/local only; R2 requires
  an accepted resource estimate and explicit authorization for one frozen GPU
  microbenchmark.
- Evidence: `docs/POSITIVE_RESULTS_DIRECTION_AUDIT.md` and
  `docs/ON_POLICY_CACHE_ROUTING_PROPOSAL.md`.

## D-114 — Supersede OPCCR and select BRACE for gated feasibility research

- Date: 2026-08-24
- Classification: `DECISION`
- Status: SUPERSEDED_IN_PART_BY_D-115
- Decision: Supersede OPCCR before implementation. Its immediate paired-action
  disagreement label is not sufficiently defensible after newer evidence showed
  silent cache-collapse failure and near-chance early action-level detectors.
- Replacement: Select Branch-Rollout Adaptive Cache Execution (BRACE) for
  feasibility research. BRACE reconstructs cache-induced states by resetting to
  the same published initial state and replaying the exact action prefix, then
  applies randomized FR/cache treatments and labels their paired closed-loop
  outcome effect.
- Simulator correction: A flattened MuJoCo state is not a complete branch
  snapshot because controller goals, observable clocks/caches, episode counters,
  wrapper state, action queues, and VLA cache provenance live outside it. Direct
  mid-episode `set_state()` alone is prohibited as evidence.
- Novelty boundary: The provisional contribution is outcome-supervised
  pre-inference selection among frozen VLA cache profiles using replay-verified
  paired interventions on cache-induced trajectories. Do not claim novelty for
  caching, counterfactual simulation, DAgger, failure detection, or adaptive
  compute separately.
- Feasibility boundary: Competitive positive-paper plausibility is judged at
  30--45% before gates and approximately 55--65% conditional on passing physical
  acceleration, replay determinism, label prevalence, and held-out
  predictability gates. These are judgment ranges, not statistical estimates;
  they were reduced after AC2-VLA was added as a close action-aware adaptive-
  computation baseline.
- Evidence boundary: No BRACE implementation, model load, GPU operation,
  simulator outcome, or protected population was produced.
- Next gate: BRACE-B1 CPU/simulator transcript and replay-equivalence harness
  only.
- Evidence: `docs/BRACE_RESEARCH_AND_FEASIBILITY_AUDIT.md` and
  `docs/BRACE_EXECUTION_PROTOCOL_V1.md`.

## D-115 — Adopt BRACE Protocol V2 after adversarial re-audit

- Date: 2026-08-24
- Classification: `DECISION`
- Status: SUPERSEDED_BY_D-116
- Decision: Preserve BRACE as the selected research direction, but supersede
  Protocol V1 before implementation. Re-audit found four material defects in
  V1: a single-query treatment that omitted cumulative reuse, dependence on a
  self-harvested accelerated attention gate, router inputs that did not expose
  mixed token-source ages, and omission of an actuation-slack refresh baseline.
- Redesign: Protocol V2 evaluates bounded clean-provenance cache contracts over
  one, two, and four accelerated queries after a full-refresh anchor. Every
  reused token records its actual source query and age; treatment effects use
  intent-to-treat assignment; and actuation-slack refresh is a mandatory
  executable comparator.
- Novelty boundary: The proposed contribution is outcome-supervised selection
  among bounded, clean-provenance VLA cache contracts using replay-verified
  closed-loop interventions. Do not claim novelty for cache reuse, adaptive
  refresh, confidence gates, counterfactual rollouts, or token-age tracking in
  isolation.
- Feasibility boundary: Positive-paper plausibility is judged at 25--40%
  before gates and approximately 50--60% conditional on passing B1--B5. These
  are calibrated judgment ranges, not statistical estimates or results.
- Evidence boundary: No BRACE implementation, model load, GPU operation,
  simulator outcome, or protected evaluation was produced by the redesign or
  review.
- Authorization: BRACE-B1 is the only next eligible phase and is
  CPU/simulator-only. B2 is conditional on B1 acceptance. B3 and all later
  model/GPU/outcome phases require separate authorization.
- Evidence: `docs/BRACE_EXECUTION_PROTOCOL_V2.md` and
  `docs/BRACE_PROTOCOL_V2_REVIEW.md`.

## D-116 — Freeze BRACE Protocol V2.1 after exhaustive red-team review

- Date: 2026-08-24
- Classification: `DECISION`
- Status: ACTIVE
- Decision: Supersede Protocol V2 before implementation and adopt V2.1 as the
  only executable BRACE plan. The additional audit found an invalid released
  VLA-Cache evaluation path, omitted exact-stack competitors, an unsupported
  pointwise-risk interpretation, incomplete multi-contract overlap and
  sampling rules, brittle profile selection, insufficient control semantics,
  and incomplete power, cost, and sealed-analysis verification.
- Redesign: Require a pinned faithfully corrected VLA-Cache reproduction;
  same-stack VLA-ADP, VLA-Pruner, and SpecPrune-VLA preflights; a bounded
  outcome-blind profile grid; known assignment and inclusion probabilities;
  zero unexplained duplicate-arm terminal discordance; episode-grouped
  empirical risk control; sequential label/cost gates; and independently
  verified sealed analysis.
- Novelty boundary: The provisional contribution is the combined use of
  replay-verified paired terminal intervention effects to select bounded,
  clean-provenance VLA cache contracts on cache-induced trajectories. Caching,
  action-aware pruning/routing, adaptive refresh, selective classification,
  and counterfactual simulation are not claimed individually.
- Feasibility boundary: Competitive positive-paper plausibility is judged at
  15--25% before gates and approximately 40--55% conditional on B1--B5 all
  passing. These are calibrated research judgments, not statistical estimates
  or results. The method is worth stop-fast feasibility testing, not reliance
  on a promised positive outcome.
- Evidence boundary: No BRACE implementation, download, model load, GPU
  operation, policy outcome, or protected evaluation was produced by V2.1 or
  its review.
- Authorization: BRACE-B1 is the only next eligible phase and is
  CPU/simulator-only. Every later phase remains separately gated by V2.1.
- Evidence: `docs/BRACE_EXECUTION_PROTOCOL_V2_1.md` and
  `docs/BRACE_PROTOCOL_V2_1_RED_TEAM_REVIEW.md`.

## D-117 — Adopt the BRACE Formal Method Specification V1

- Date: 2026-08-25
- Classification: `DECISION`
- Status: ACTIVE_METHOD_SPECIFICATION
- Decision: Adopt `docs/BRACE_FORMAL_METHOD_SPECIFICATION_V1.md` as the
  implementation definition subordinate to Protocol V2.1. It formalizes the
  cache operator, profile state, paired estimand, router, training objective,
  joint calibration, timing, algorithms, and module/test contracts.
- Formal-audit corrections: Require nested layerwise reuse sets; runtime-derived
  multimodal token spans; parity-verified sidecar anchor attention; exact
  historical-anchor reconstruction; outcome-blind robot/action drift envelopes
  because OpenVLA-OFT attention is bidirectional; a <=10k-parameter router; and
  calibration of the complete fastest-accepted-contract policy.
- Additional audit corrections: Complete multimodal source records replace
  image-only provenance; exact-source context envelopes cover mixed K/V; and
  an aborted experimental assignment uses FR for its remaining fixed horizon
  so its causal treatment does not silently change.
- Scientific boundary: Static pixels do not imply context-invariant visual K/V,
  and terminal-effect supervision is not assumed predictable. These are B2--B5
  hypotheses and stop conditions. No architecture-level argument changes the
  15--25% pre-gate positive-paper assessment.
- Evidence boundary: Documentation and read-only source/literature inspection
  only; no implementation, model, GPU, simulator outcome, protected data, or
  server operation occurred.
- Authorization: B1 remains the only eligible task. The formal specification
  does not authorize B2 or later work.
- Evidence: `docs/BRACE_FORMAL_METHOD_SPECIFICATION_V1.md` and
  `docs/BRACE_FORMAL_METHOD_AUDIT_V1.md`.

## D-118 — Authorize and freeze the BRACE-B1 replay-equivalence attempt

- Date: 2026-08-25
- Classification: `DECISION`
- Status: COMPLETE_EXECUTED
- Decision: The user authorized BRACE-B1. Freeze one CPU/OSMesa attempt with
  three scripted LIBERO Spatial scenarios (free motion, contact, and gripper
  transition), twelve actions per scenario, and replay prefixes at actions 3,
  6, and 10.
- Acceptance boundary: Each prefix must reconstruct twice in a fresh
  environment and match the recorded complete snapshot plus the next-step
  probe. Modified-prefix and direct-simulator-state-only negative controls must
  be rejected. The transcript hash chain, configuration hash, pinned LIBERO
  revision, repository revision, and resource accounting must reconcile.
- Resource boundary: CUDA is hidden; model and policy queries, downloads, and
  GPU inspection are prohibited. The attempt is capped at 30 environment
  instances, 240 simulator steps, 1,800 seconds, and 256 MiB of artifacts.
- Implementation evidence: The side-effect-free transcript validator, frozen
  runner, and adversarial tests pass local unit, static, compilation, and
  bootstrap checks. Configuration semantic SHA-256 is
  `37952f345bcaecceffe4695fecea8d48c2b67ed19d19222d64fd29eeda1dc04c`.
- Authorization boundary: B1 authorization does not authorize B2, model
  loading, GPU work, learned routing, cache installation, or policy outcomes.
- Evidence: `configs/brace/b1_replay_v1.json`, `scripts/run_brace_b1.py`,
  `src/savr/brace/b1.py`, and `tests/brace/test_b1.py`.

## D-119 — Accept BRACE-B1 and stop before BRACE-B2

- Date: 2026-08-25
- Classification: `DECISION`
- Status: COMPLETE_ACCEPTED
- Decision: Accept the single frozen `brace-b1-replay-v01` attempt. All gates
  passed: 3/3 transcripts validated, 18/18 fresh prefix replays and 9/9 probe
  transitions matched, and all 3 modified-prefix plus all 3 direct-state-only
  controls were rejected.
- Resource reconciliation: 27 environment instances, 198 simulator steps,
  284.40 seconds, and 701,560 artifact bytes remained within their frozen
  caps. CUDA was hidden; there were no model or policy queries, downloads, GPU
  inspections, or writes outside `/home/ved/SAVR`.
- Scientific interpretation: B1 establishes only the replay infrastructure
  hypothesis H1. It is not evidence of cache correctness, speed, learned risk
  prediction, closed-loop reliability, or a positive paper result.
- Authorization boundary: Stop before B2. B2 is protocol-eligible but still
  requires a separate user decision.
- Evidence: `reports/BRACE_B1_REPORT.md` and
  `reports/runtime/brace_b1_replay_v01.json`.

## D-120 — Authorize and freeze BRACE-B2 correctness verification

- Date: 2026-08-25
- Classification: `DECISION`
- Status: COMPLETE_EXECUTED
- Decision: The user authorized B2. Freeze one CPU/synthetic verification of
  the isolated baseline correction, both pinned DynamicCache stacks, runtime
  sequence maps, complete provenance, profile/contract semantics, synthetic
  sidecar attention, immutable intent records, and current comparator sources.
- Source boundary: Pinned public method source may be cloned only below
  `/home/ved/SAVR/third_party`. No model, checkpoint, dataset, external package,
  or unbounded asset download is authorized.
- Comparator boundary: Preserve VLA-ADP's overlay requirement, VLA-Pruner's
  isolated 4.47.0 stack, SpecPrune-VLA's missing top-level license, and Gated
  VLA-Cache's lack of discoverable official code as explicit dispositions.
  Do not call a matched reproduction official.
- Resource boundary: CUDA hidden; zero model queries, policy outcomes, or
  simulator steps; at most 1 GiB of source, 16 MiB of evidence, and 1,800
  seconds. B3 remains unauthorized.
- Configuration: `configs/brace/b2_correctness_v1.json`, semantic SHA-256
  `220c0b6b15253e0ecbea34010258d580cecb90f459d2d55da8bb10c01760285a`.
- Evidence: `docs/BRACE_B2_IMPLEMENTATION_FREEZE.md` and the B2 implementation
  and tests in `src/savr/brace/` and `tests/brace/`.

## D-121 — Preserve B2 v01 technical stop and freeze narrow v02 recovery

- Date: 2026-08-25
- Classification: `DECISION`
- Status: COMPLETE_EXECUTED
- Decision: Preserve `brace-b2-correctness-v01` as an immutable no-result
  technical stop and permit one distinct CPU-only v02 recovery under the
  existing B2 authorization.
- Diagnosis: The v01 checker compared restored cache state with a tensor that
  the cache retained by reference and later mutated. It also required the core
  Transformers 4.40.1 cache source to contain the separate VLA-Cache 4.47.0
  position-update customization. Direct tests showed independent cloning on
  both stacks; the defects were in these two checker oracles.
- Correction: Compare restoration with an immutable pre-mutation clone; test
  BRACE's position-preserving update on both stacks; identify fork-specific
  source behavior separately.
- Scientific boundary: No BRACE algorithm, contract/profile grid, comparator,
  resource cap, B3 proposal, or acceptance gate changed. V01 made zero model
  queries, policy observations, simulator steps, or GPU operations and is not
  a method result.
- Configuration: `configs/brace/b2_correctness_v2.json`, semantic SHA-256
  `04370384312f4474fd5488fa3b4dad1559d3d7916564501c802fdec8214906bd`.
- Authorization boundary: One v02 CPU/synthetic attempt only. B3 remains
  unauthorized.

## D-122 — Accept BRACE-B2 with comparator dispositions and stop before B3

- Date: 2026-08-25
- Classification: `DECISION`
- Status: COMPLETE_ACCEPTED_WITH_DISPOSITIONS
- Decision: Accept `brace-b2-correctness-v02`. Independent reconciliation
  confirmed all 13 gates, the authenticated summary, and 23 passing server
  tests across both pinned DynamicCache stacks.
- Correctness result: Independent clone, failure-safe restoration,
  absolute-position update, corrected evaluator, runtime mapping, synthetic
  sidecar, provenance, contract/profile, immutable-intent, isolation, and
  resource checks passed.
- Comparator dispositions: VLA-ADP requires a core-stack overlay; VLA-Pruner
  requires its isolated customized stack; SpecPrune-VLA remains blocked by its
  missing advertised top-level license; Gated VLA-Cache remains paper-only
  without discoverable official code.
- Resource reconciliation: 132,931,536 source bytes and 5.15 seconds; CUDA was
  hidden and uninitialized; zero model queries, policy observations, and
  simulator steps.
- Scientific interpretation: This accepts software substrate correctness
  only. It is not evidence of real-model parity, speed, reliability, or a
  positive-paper result.
- Authorization boundary: Stop before B3. The bounded B3 proposal remains at
  most 480 balanced real-model queries, one separately authorized GPU, zero
  simulator outcomes, and 23 GiB peak memory.
- Evidence: `reports/BRACE_B2_REPORT.md`,
  `reports/runtime/brace_b2_v01_technical_stop.json`, and
  `reports/runtime/brace_b2_v02.json`.

## D-123 — Authorize and freeze one bounded BRACE-B3 physical attempt

- Date: 2026-08-25
- Classification: `DECISION`
- Status: IN_PROGRESS
- Decision: The user authorized B3. Freeze one fail-closed real-model attempt
  comprising optimized core FR, synchronized cache-stack P0, corrected
  VLA-Cache, four clean P1/P2 profiles, the official VLA-ADP component, and
  official VLA-Pruner timing.
- Scientific boundary: B3 tests physical action parity, complete-cycle timing,
  cache/provenance invariants, memory, mandatory comparator dispositions, and
  real control-window slack. It uses deterministic model inputs and zero
  simulator outcomes; it cannot establish task reliability or a positive
  paper result.
- Resource boundary: 388 planned model queries under a hard cap of 420; one
  aggregate-idle GPU and one model process at a time; strictly less than
  23 GiB peak memory; zero downloads, protected outcomes, or automatic retry;
  three hours and 1 GiB of project-local artifacts.
- Comparator boundary: Corrected VLA-Cache receives required timing.
  VLA-ADP's episode-coupled dynamic controller receives reviewed component
  timing. A CUDA-hidden import preflight found that pinned VLA-Pruner imports
  an absent `experiments/robot/vla_cache_utils.py`; it receives the protocol-
  permitted reviewed technical exclusion, and its unused 32-query allocation
  cannot be reassigned. SpecPrune-VLA remains license-blocked. Gated VLA-Cache remains excluded
  because no official code was found and its discrete-token confidence gate
  is not defined for the pinned continuous L1 head.
- Advance rule: All parity, provenance, memory, disposition, and accounting
  gates must pass, and at least one clean profile must reduce accelerated-query
  wall time by 10% and amortized complete-cycle wall time by 8%. Stop before
  B4 in every case; B4 is not authorized.
- Configuration: `configs/brace/b3_physical_v1.json`, semantic SHA-256
  `30825fd30eb2c0566b30564740a212c51e09dc60f7e9c4bc7315b24b369dea11`.
- Evidence freeze: `docs/BRACE_B3_EXECUTION_FREEZE.md`.

## D-124 — Preserve BRACE-B3 v01 pre-model technical stop

- Date: 2026-08-25
- Classification: `DECISION`
- Status: COMPLETE_TECHNICAL_STOP_NO_RESULT
- Decision: Preserve the sealed v01 launch and stop before any corrected
  attempt. The runner's source-tree guard rejected Git's quoted display form
  for preserved `tmp/` paths containing spaces.
- Resource reconciliation: The aggregate-only selector recorded three idle
  samples and selected GPU 0. No worker started; model loads, model queries,
  simulator outcomes, protected outcome access, and downloads were all zero.
- Correction: Parse NUL-delimited porcelain-v1 records so paths remain raw for
  exact allowlist evaluation. No method, profile, gate, comparator, or resource
  boundary changed.
- Authorization boundary: This is not a scientific result. Automatic retry is
  prohibited, v02 recovery is not authorized, and B4 remains unauthorized.
- Evidence: `reports/BRACE_B3_V01_TECHNICAL_STOP_REPORT.md` and
  `reports/runtime/brace_b3_v01_technical_stop.json`.

## D-125 — Authorize narrow BRACE-B3 v02 recovery

- Date: 2026-08-25
- Classification: `DECISION`
- Status: IN_PROGRESS
- Decision: The user explicitly authorized one v02 recovery after reviewing
  the zero-query v01 diagnosis.
- Sole correction: Replace human-quoted Git porcelain parsing with
  NUL-delimited porcelain-v1 parsing and use a distinct immutable run identity.
- Frozen invariants: Method logic, profiles, thresholds, timing design, parity,
  comparator dispositions, query allocations/caps, memory limit, outcome
  boundary, and acceptance gates are unchanged from v01.
- Resolved configuration semantic SHA-256:
  `45dbf458e014bbd6314e12de92256859f57c716a03c25e9091a2286c09f8e925`.
- Authorization boundary: One v02 attempt, no automatic retry, and stop before
  B4 in every case.
- Evidence freeze: `docs/BRACE_B3_V02_RECOVERY_FREEZE.md`.

## D-126 — Preserve BRACE-B3 v02 pre-query technical stop

- Date: 2026-08-25
- Classification: `DECISION`
- Status: COMPLETE_TECHNICAL_STOP_NO_RESULT
- Decision: Preserve v02 and do not analyze or retry it. The model loaded, but
  the deterministic input fixture indexed normalization metadata with
  `libero_object` instead of the official evaluator-resolved
  `libero_object_no_noops` key.
- Resource reconciliation: Zero completed model queries or methods; the full
  22-query core-FR allocation is conservatively charged. Peak aggregate GPU
  memory was 15,275 MiB during loading. Simulator outcomes and protected
  outcome access were zero.
- Restoration: The three protected checkpoint files exactly match their frozen
  SHA-256 values and no loader backup remains.
- Correction: Read and validate `cfg.unnorm_key` after official model
  initialization. A synthetic regression covers the exact suffix alias.
- Scientific boundary: This is no result for or against BRACE. V03 recovery is
  not authorized, and B4 remains unauthorized.
- Evidence: `reports/BRACE_B3_V02_TECHNICAL_STOP_REPORT.md` and
  `reports/runtime/brace_b3_v02_technical_stop.json`.

## D-127 — Authorize narrow BRACE-B3 v03 recovery

- Date: 2026-08-25
- Classification: `DECISION`
- Status: IN_PROGRESS
- Decision: The user authorized continuation after reviewing the exact v02
  diagnosis and the limits of the correction guarantee.
- Sole correction: The deterministic proprioception fixture uses the official
  evaluator-resolved normalization key rather than a hardcoded unsuffixed key.
- Additional verification: Reconcile every private cache-worker model helper,
  tensor layout, action slice, DynamicCache return, and official cache tuple
  against pinned sources before launch.
- Frozen invariants: No method, profile, threshold, timing, parity, comparator,
  resource, outcome, or acceptance-gate change.
- Resolved configuration semantic SHA-256:
  `8d6922e797432fcfd079a1c61fa1071ec19b6338dbe8a21178a1b9b6bb701ef9`.
- Authorization boundary: One v03 attempt, no automatic retry, stop before B4.
- Evidence freeze: `docs/BRACE_B3_V03_RECOVERY_FREEZE.md`.

## D-128 — Preserve BRACE-B3 v03 autograd-memory technical stop

- Date: 2026-08-25
- Classification: `DECISION`
- Status: COMPLETE_TECHNICAL_STOP_NO_RESULT
- Decision: Preserve v03 without scientific analysis. Core FR completed 22
  queries, but the first custom cache query retained an autograd graph because
  it lacked official inference's no-gradient context. Aggregate GPU memory
  reached 24,017 MiB and the 23 GiB guard terminated the worker.
- Accounting: 22 completed queries, zero completed cache queries, one partial
  cache attempt, and 324 queries conservatively charged. No simulator outcome
  or protected outcome access occurred.
- Interpretation: The observed memory includes training activations and is not
  a valid cache/BRACE feasibility result.
- Correction: Run the complete custom preparation and inference path under
  `torch.inference_mode()` and assert gradients are disabled.
- Restoration: Protected checkpoint hashes match the frozen baseline exactly.
- Authorization boundary: No automatic retry; v04 and B4 are unauthorized.
- Evidence: `reports/BRACE_B3_V03_TECHNICAL_STOP_REPORT.md` and
  `reports/runtime/brace_b3_v03_technical_stop.json`.

## D-129 — Authorize BRACE-B3 v04 inference-mode recovery

- Date: 2026-08-25
- Classification: `DECISION`
- Status: IN_PROGRESS
- Decision: The user authorized continuation after the exact v03 diagnosis.
- Sole correction: Decorate the entire custom cache path with
  `torch.inference_mode()` and reject enabled gradients at runtime.
- Backend verification: The pinned cache model uses `LlamaSdpaAttention` and
  the exact SDPA primitive captured by the frozen sidecar.
- Frozen invariants: Every method, profile, threshold, timing, parity,
  comparator, resource, outcome, and scientific acceptance setting is
  unchanged.
- Resolved configuration semantic SHA-256:
  `21b44336aac5db0f62131449a0f1eb2eaf2ce1c19bde118b4af577c390d9fc86`.
- Authorization boundary: One v04 attempt, no automatic retry, stop before B4.
- Evidence freeze: `docs/BRACE_B3_V04_RECOVERY_FREEZE.md`.

## D-130 — Preserve BRACE-B3 v04 profile-gate technical stop

- Date: 2026-08-25
- Classification: `DECISION`
- Status: COMPLETE_TECHNICAL_STOP_NO_RESULT
- Decision: Preserve v04 without scientific analysis. The cache stack remained
  below the memory gate and progressed through 43 queries, then the first
  profile gate called binary `torch.maximum` with one stacked operand.
- Accounting: 22 core-FR plus 43 reconstructed cache completions; 324 queries
  conservatively charged. Peak aggregate memory was 18,419 MiB. No simulator
  outcomes or protected fields were used.
- Interpretation: V04 confirms the inference-mode memory correction but is not
  a complete BRACE timing/parity result because the cache worker has no terminal
  artifact and no profile reuse query completed.
- Correction: Use two tensor operands and add a real-tensor patch-score test.
- Restoration: Protected checkpoint hashes match the frozen baseline exactly.
- Authorization boundary: No automatic retry; v05 and B4 are unauthorized.
- Evidence: `reports/BRACE_B3_V04_TECHNICAL_STOP_REPORT.md` and
  `reports/runtime/brace_b3_v04_technical_stop.json`.
