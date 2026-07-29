# Next Codex Handoff: Compatibility and FR Smoke Test

Do not execute this handoff until the user approves it and supplies the manuscript source.

## Goal

Establish a reproducible, project-local OpenVLA-OFT + LIBERO environment and run the smallest possible Full Refresh correctness smoke test. Do not implement PR, VOR, or SAVR yet.

## Mandatory boundary

Follow `AGENTS.md`. Work only inside `/home/ved/SAVR`. Do not use `sudo`, inspect unrelated server content or processes, change shared configuration, or launch work on more than one user-approved GPU.

## Required sequence

1. Read `AGENTS.md`, `PROJECT_STATUS.md`, the supplied manuscript, and `docs/EXPERIMENT_PLAN.md`.
2. Reconcile the manuscript’s formal SAVR signals/notation with the repository plan; document discrepancies.
3. Estimate checkpoint, environment, cache, and dataset storage before any download; obtain approval.
4. Coordinate a single GPU with the user. Do not infer availability by inspecting other users' processes.
5. Create a project-local environment under `/home/ved/SAVR`; pin versions and upstream commits.
6. Install only project-local dependencies.
7. Verify headless LIBERO rendering without loading a VLA.
8. Load one approved OpenVLA-OFT LIBERO checkpoint.
9. Run one FR-only episode using fixed task, initial state, and seed.
10. Save a conforming manifest, episode record, raw timings, and sanitized logs.
11. Add a test showing the FR wrapper refreshes on every step and resets cleanly.
12. Stop and report compatibility, peak memory, storage used, exact changes, and blockers.

## Acceptance criteria

- no changes outside `/home/ved/SAVR`
- no system-wide installation
- exactly one user-approved GPU used
- reproducible project-local environment lock
- one complete FR smoke-test record
- no PR/VOR/SAVR implementation
- no large or multi-task experiment
