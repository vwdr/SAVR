# SAVR Agent Instructions

These instructions apply to the entire repository.

## Critical TITAN server boundary

- The only permitted writable server path is `/home/ved/SAVR`.
- Do not inspect, modify, move, rename, delete, copy, or change permissions on unrelated university files or directories.
- Do not inspect or interfere with unrelated processes, jobs, containers, environments, services, network configuration, mounts, users, or GPU allocations.
- Never use `sudo`.
- Do not make system-wide installations or configuration changes.
- Do not terminate or reprioritize any process.
- Do not run broad cleanup, recursive search, or recursive destructive commands outside `/home/ved/SAVR`.
- Do not select `/home/ved` or another broad directory as the workspace.
- If a necessary action would leave `/home/ved/SAVR`, change server state outside it, or create uncertainty about shared resources, stop and ask the user.
- Bootstrap diagnostics must be read-only and narrowly scoped. Static GPU identity/capacity queries are allowed; process or allocation inspection is not.
- Any future GPU workload must use at most one explicitly selected GPU unless the user separately authorizes more. Before choosing a GPU, stop for user coordination rather than inspecting or claiming a shared allocation.

## Scientific integrity

- SAVR is currently a proposal, not a validated method.
- Do not invent results, measurements, citations, implementation status, or supported claims.
- Clearly distinguish hypotheses, plans, observations, and measured evidence.
- Preserve provenance: record commands, configuration, code revision, model/checkpoint identity, benchmark version, seeds, hardware, and timestamps for every experiment.
- Treat task success as the primary safety/performance constraint; efficiency gains are not sufficient if success degrades beyond a predeclared tolerance.
- Do not modify the manuscript or project-provided reference files unless the user explicitly requests it.

## Implementation discipline

- Work in small, reviewable stages.
- Before implementation, confirm the exact base VLA interface, visual-feature boundary, cache lifetime, robot-state representation, and action-history representation.
- Implement FR first as the correctness oracle, then PR, VOR, and SAVR behind one common policy interface.
- Keep threshold selection separate from final evaluation.
- Use deterministic seeds where supported and log sources of nondeterminism.
- Add tests before large runs. Smoke tests must precede benchmark experiments.
- Do not download models, checkpoints, or datasets without explicit approval and a storage estimate.
- Do not launch training or large experiments during bootstrap.

## Git and reporting

- Keep the GitHub repository private unless the user explicitly changes that decision.
- Never commit credentials, tokens, private keys, large checkpoints, datasets, raw caches, or unreviewed generated artifacts.
- Preserve unrelated user changes.
- At each handoff, report exactly what changed, what was verified, what remains blocked, and whether anything outside `/home/ved/SAVR` was modified.
