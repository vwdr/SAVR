# SAVR Milestones

Last updated: 2026-07-29

Exactly one phase may be `IN_PROGRESS`.

| Phase | Status | Completion evidence | Next gate |
|---|---|---|---|
| 0. Establish authoritative state | COMPLETE | Preparation PR #1 and ledger PR #2 merged with explicit approval; GitHub, TITAN, and local `main` synchronized at `9a5bbd5`; manuscript SHA-256 verified as `4a0fe130f1cbc5557f77a518dcb65a703a647b1c4b8091499d8bfd8e10ab6e4f`; worktrees clean | — |
| 1. Environment and storage feasibility | IN_PROGRESS | Installation, dependency locks, imports, and CPU-only OSMesa LIBERO smoke test passed; `reports/PHASE1_REPORT.md` records evidence and one unresolved boundary audit item | User review/acceptance of the Phase 1 checkpoint PR and boundary-audit decision |
| 2. Unmodified FR reproduction | NOT_STARTED | — | User-approved GPU ID and FR reproduction evidence |
| 3. Controller and cache implementation | NOT_STARTED | — | Unit-tested implementation |
| 4. Correctness and instrumentation | NOT_STARTED | — | FR parity and logging audit |
| 5. Smoke policies and external-baseline feasibility | NOT_STARTED | — | Core-policy smoke completion and VLA-Cache decision |
| 6. Calibration and power | NOT_STARTED | — | Frozen configurations, sample size, and margin approval |
| 7. Freeze final protocol | NOT_STARTED | — | User approval of `PROTOCOL_V1.md` |
| 8. Final evaluation | NOT_STARTED | — | Complete reconciled final-run registry |
| 9. Ablations and sensitivity | NOT_STARTED | — | Required confirmatory ablations complete |
| 10. Analysis and claim audit | NOT_STARTED | — | Every manuscript claim mapped to evidence |
| 11. Manuscript completion | NOT_STARTED | — | User-approved evidence-based manuscript changes |

## Active milestone

Phase 1 is the only active phase. Its approved environment installation and one CPU-only simulator smoke test are complete. Checkpoint/dataset downloads, model loading, GPU work, policy implementation, and experiments remain unauthorized.

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
- [ ] Resolve or explicitly accept the account-level LIBERO path uncertainty.
- [ ] Obtain user approval and merge the Phase 1 checkpoint PR.
