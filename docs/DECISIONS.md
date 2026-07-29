# SAVR Decision Log

Last updated: 2026-07-29

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
