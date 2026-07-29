# SAVR

SAVR (State-Aware Visual Refresh) is a proposed training-free inference wrapper for vision-language-action (VLA) policies. It aims to reuse visual features when doing so is safe and to refresh them when visual, robot-state, action-history, or maximum-horizon signals indicate that cached features may be stale.

## Current phase

This repository is in **Phase 3: controller and cache implementation**. The
unmodified Full Refresh calibration pilot is complete, and the project-owned
CPU implementation is under review. No SAVR performance claim is supported
until later correctness, calibration, and final-evaluation phases.

Planned comparison policies:

- Full Refresh (FR)
- Periodic Refresh (PR)
- Visual-Only Refresh (VOR)
- State-Aware Visual Refresh (SAVR)

Planned primary outcomes:

- task success
- end-to-end and policy latency
- refresh/skipped-refresh rates
- compute or memory proxies
- failure cases

## Repository map

- `AGENTS.md`: mandatory operating and server-safety rules
- `PROJECT_STATUS.md`: authoritative scientific and implementation status
- `docs/EXPERIMENT_PLAN.md`: staged experimental protocol
- `docs/SAVR_EXECUTION_PROTOCOL.md`: authoritative execution, verification, and anti-hallucination guide
- `docs/STACK_ASSESSMENT.md`: initial OpenVLA-OFT + LIBERO feasibility assessment
- `docs/RESULTS_SCHEMA.md`: logging contract
- `docs/PHASE3_IMPLEMENTATION_DESIGN.md`: frozen controller/cache boundary
- `src/savr/`: controller, signal, cache, immutable-record, and adapter code
- `tests/unit/`: dependency-light CPU correctness tests
- `schemas/`: machine-readable run and episode schemas
- `scripts/`: staged diagnostics, pilot runners, analysis, and validation
- `references/`: project-provided literature material
- `manuscript/`: original LaTeX manuscript and provenance note

## Validation

```bash
python3 scripts/validate_bootstrap.py
python3 -m unittest discover -s tests -v
```

Do not install dependencies or launch experiments until the bootstrap checkpoint is accepted and the next handoff is approved.
GPU, simulator, parity, calibration, and performance experiments remain gated
by `docs/SAVR_EXECUTION_PROTOCOL.md`.
