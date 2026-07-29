# SAVR Milestones

Last updated: 2026-07-29

Exactly one phase may be `IN_PROGRESS`.

| Phase | Status | Completion evidence | Next gate |
|---|---|---|---|
| 0. Establish authoritative state | IN_PROGRESS | Private GitHub repository verified; preparation PR #1 merged; GitHub, TITAN, and local `main` synchronized at `2ef3f59`; manuscript SHA-256 verified as `4a0fe130f1cbc5557f77a518dcb65a703a647b1c4b8091499d8bfd8e10ab6e4f`; worktrees clean before this ledger branch | Review and approve the Phase 0 ledger PR |
| 1. Environment and storage feasibility | NOT_STARTED | — | User approval for estimated downloads and project-local installation |
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

Phase 0 is the only active phase. No Phase 1 installation, download, simulation, or GPU work is authorized.

## Phase 0 remaining checklist

- [x] Verify repository visibility and GitHub refs.
- [x] Review and merge preparation PR #1 with explicit user approval.
- [x] Synchronize GitHub, TITAN, and local `main`.
- [x] Verify manuscript provenance and clean worktrees.
- [x] Create milestone and decision ledgers.
- [ ] Merge the ledger PR with explicit user approval.
- [ ] Synchronize the resulting `main` commit everywhere and mark Phase 0 complete.
