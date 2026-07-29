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
    "docs/EXPERIMENT_PLAN.md",
    "docs/RESULTS_SCHEMA.md",
    "docs/SAVR_EXECUTION_PROTOCOL.md",
    "docs/STACK_ASSESSMENT.md",
    "docs/UPSTREAM_PINS.md",
    "manuscript/README.md",
    "manuscript/State-Aware Visual Refresh for Efficient VLA Inference.tex",
    "pyproject.toml",
    "references/efficiency_papers.txt",
    "reports/BOOTSTRAP_REPORT.md",
    "reports/titan_bootstrap_diagnostics.json",
    "schemas/episode_result.schema.json",
    "schemas/run_manifest.schema.json",
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
