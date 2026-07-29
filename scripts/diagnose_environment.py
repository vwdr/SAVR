#!/usr/bin/env python3
"""Collect bounded, read-only bootstrap facts without inspecting processes."""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str]) -> dict[str, object]:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
        return {
            "command": command,
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"command": command, "error": str(exc)}


def module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def collect() -> dict[str, object]:
    commands = {
        "git": ["git", "--version"],
        "python": ["python3", "--version"],
        "nvcc": ["nvcc", "--version"],
        "gpu_static": [
            "nvidia-smi",
            "--query-gpu=index,name,driver_version,memory.total,compute_cap",
            "--format=csv,noheader",
        ],
        "project_disk": ["df", "-Pk", str(PROJECT_ROOT)],
    }
    return {
        "collected_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": str(PROJECT_ROOT),
        "platform": platform.platform(),
        "executables": {
            name: shutil.which(name)
            for name in ("codex", "git", "gh", "python3", "nvcc", "nvidia-smi")
        },
        "commands": {name: run(command) for name, command in commands.items()},
        "python_modules_present": {
            name: module_available(name)
            for name in ("torch", "transformers", "numpy", "libero", "robosuite", "mujoco")
        },
        "safety": {
            "processes_inspected": False,
            "gpu_allocations_inspected": False,
            "gpu_workload_launched": False,
            "files_outside_project_modified": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = json.dumps(collect(), indent=2, sort_keys=True) + "\n"
    if args.output:
        output = args.output.resolve()
        if PROJECT_ROOT not in output.parents:
            raise SystemExit("Output must remain inside /home/ved/SAVR")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
