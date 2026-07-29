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
- Status: OPEN
- Decision: No environment installation, checkpoint/dataset download, simulator run, or GPU use may begin until Phase 0 closes and the user approves the Phase 1 resource estimate.
- Evidence: Sections 11 and 19 of `docs/SAVR_EXECUTION_PROTOCOL.md`.
- Approver: User approval required.
