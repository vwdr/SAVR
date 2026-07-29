#!/usr/bin/env python3
"""Validate the documentation-only SAVR bootstrap without third-party packages."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_FILES = (
    ".github/workflows/ci.yml",
    ".gitignore",
    "AGENTS.md",
    "PROJECT_STATUS.md",
    "README.md",
    "docs/CODEX_HANDOFF_IMPLEMENTATION.md",
    "docs/DECISIONS.md",
    "docs/EXPERIMENT_PLAN.md",
    "docs/MILESTONES.md",
    "docs/PHASE1_RESOURCE_ESTIMATE.md",
    "docs/PHASE2_RESOURCE_ESTIMATE.md",
    "docs/PHASE2B_PILOT_PROPOSAL.md",
    "docs/PHASE3_IMPLEMENTATION_DESIGN.md",
    "docs/RESULTS_SCHEMA.md",
    "docs/SAVR_EXECUTION_PROTOCOL.md",
    "docs/STACK_ASSESSMENT.md",
    "docs/UPSTREAM_PINS.md",
    "environment/README.md",
    "environment/libero-config.yaml",
    "environment/locks/.gitkeep",
    "environment/locks/conda-linux-64-explicit.txt",
    "environment/locks/pip-freeze.txt",
    "environment/phase1-conda.yml",
    "manuscript/README.md",
    "manuscript/State-Aware Visual Refresh for Efficient VLA Inference.tex",
    "pyproject.toml",
    "references/efficiency_papers.txt",
    "reports/BOOTSTRAP_REPORT.md",
    "reports/PHASE1_REPORT.md",
    "reports/PHASE2A_CHECKPOINT_REPORT.md",
    "reports/PHASE2A_FR_SMOKE_REPORT.md",
    "reports/PHASE2B_PILOT_REPORT.md",
    "reports/PHASE3_IMPLEMENTATION_REPORT.md",
    "reports/titan_bootstrap_diagnostics.json",
    "schemas/episode_result.schema.json",
    "schemas/run_manifest.schema.json",
    "scripts/setup_phase1_environment.sh",
    "scripts/analyze_phase2b_pilot.py",
    "scripts/download_phase2_checkpoint.py",
    "scripts/run_phase2a_fr_smoke.py",
    "scripts/run_phase2b_fr_pilot.py",
    "scripts/verify_phase1_environment.py",
    "src/savr/__init__.py",
    "src/savr/cache.py",
    "src/savr/controllers.py",
    "src/savr/integration/__init__.py",
    "src/savr/integration/openvla_oft.py",
    "src/savr/logging.py",
    "src/savr/signals.py",
    "tests/unit/test_cache_and_logging.py",
    "tests/unit/test_controllers.py",
    "tests/unit/test_openvla_adapter.py",
    "tests/unit/test_signals.py",
)


def main() -> int:
    missing = [path for path in REQUIRED_FILES if not (ROOT / path).is_file()]
    if missing:
        raise SystemExit(f"Missing required files: {', '.join(missing)}")

    for relative in ("schemas/episode_result.schema.json", "schemas/run_manifest.schema.json"):
        path = ROOT / relative
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("type") != "object" or not data.get("required"):
            raise SystemExit(f"Schema lacks object/required contract: {relative}")

    agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    required_safety_phrases = (
        "/home/ved/SAVR",
        "Never use `sudo`",
        "Do not inspect or interfere with unrelated processes",
        "at most one explicitly selected GPU",
    )
    absent = [phrase for phrase in required_safety_phrases if phrase not in agents]
    if absent:
        raise SystemExit(f"AGENTS.md missing safety clauses: {absent}")

    print(f"SAVR bootstrap validation passed ({len(REQUIRED_FILES)} required files).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
