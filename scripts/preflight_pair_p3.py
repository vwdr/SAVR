#!/usr/bin/env python3
"""Fail-closed, CPU-only preflight for the single PAIR P3 physical run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ROOT = Path("/home/ved/SAVR")
DEFAULT_CONFIG = Path("configs/pair/p3_physical_v1.json")
DEFAULT_OUTPUT = Path("reports/pair_p3/preflight.json")


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def semantic_sha256(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("semantic_sha256", None)
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def git_revision(path: Path) -> str:
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if ROOT != EXPECTED_ROOT:
        raise SystemExit(f"P3 preflight refuses to run outside {EXPECTED_ROOT}")
    if os.environ.get("CUDA_VISIBLE_DEVICES", ""):
        raise RuntimeError("P3 preflight requires CUDA to be hidden")

    sys.path.insert(0, str(ROOT / "src"))
    from savr.pair.p3 import planned_query_count, semantic_sha256 as pair_hash, validate_config

    config = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    inputs = json.loads((ROOT / config["input_manifest"]).read_text(encoding="utf-8"))
    if inputs.get("semantic_sha256") != pair_hash(inputs):
        raise RuntimeError("P3 input-manifest semantic hash mismatch")
    validate_config(config, input_count=len(inputs["inputs"]))
    if planned_query_count(config, len(inputs["inputs"])) != 688:
        raise RuntimeError("P3 frozen query count changed")

    p0 = json.loads((ROOT / "configs/pair/p0_freeze_v1.json").read_text(encoding="utf-8"))
    p2 = json.loads((ROOT / "reports/PAIR_P2_SEMANTIC_MANIFEST.json").read_text(encoding="utf-8"))
    if p2.get("semantic_sha256") != semantic_sha256(p2) or p2.get("decision") != "PASS":
        raise RuntimeError("P2 pass evidence is invalid")
    revisions = p0["authenticated_revisions"]
    observed_revisions = {
        "project_git": git_revision(ROOT),
        "vla_cache_git": git_revision(ROOT / "third_party/vla-cache"),
    }
    for name, observed in observed_revisions.items():
        if observed != revisions[name]:
            raise RuntimeError(f"P3 authenticated revision mismatch: {name}")

    checkpoint = ROOT / config["model"]["checkpoint_relative"]
    for relative in ("config.json", "configuration_prismatic.py", "modeling_prismatic.py"):
        if not (checkpoint / relative).is_file():
            raise RuntimeError(f"P3 checkpoint file is missing: {relative}")

    data_root = (ROOT / inputs["data_root_relative"]).resolve()
    if not data_root.is_relative_to(ROOT):
        raise RuntimeError("P3 data root escaped the project")
    for identity in inputs["inputs"]:
        source = (data_root / identity["source_path"]).resolve()
        if not source.is_relative_to(data_root) or not source.is_file():
            raise RuntimeError("P3 frozen input source is missing or escaped the data root")

    worker_text = (ROOT / "scripts/run_pair_p3_worker.py").read_text(encoding="utf-8")
    analyzer_text = (ROOT / "scripts/analyze_pair_p3.py").read_text(encoding="utf-8")
    libero_config = ROOT / "configs/pair/libero_runtime/config.yaml"
    if not libero_config.is_file():
        raise RuntimeError("P3 project-local LIBERO configuration is missing")
    for line in libero_config.read_text(encoding="utf-8").splitlines():
        _, raw_path = line.split(":", 1)
        if not Path(raw_path.strip()).resolve().is_relative_to(ROOT):
            raise RuntimeError("P3 LIBERO configuration escaped the project")
    forbidden_worker_patterns = {
        "expert action dataset": r"group\s*\[\s*[\"']actions[\"']\s*\]",
        "terminal dataset": r"group\s*\[\s*[\"']dones[\"']\s*\]",
        "automatic retry": r"\bfor\s+attempt\b|\bretry\s*\(",
        "simulator construction": r"\b(get_libero_env|OffScreenRenderEnv)\s*\(",
    }
    for label, pattern in forbidden_worker_patterns.items():
        if re.search(pattern, worker_text):
            raise RuntimeError(f"P3 worker contains forbidden {label} path")
    if "validate_action_record" not in analyzer_text or "reject_protected_fields" not in analyzer_text:
        raise RuntimeError("P3 analyzer protection guards are missing")
    if 'os.environ["LIBERO_CONFIG_PATH"] = str(project_libero_config)' not in worker_text:
        raise RuntimeError("P3 project-local LIBERO guard is missing")
    openvla_text = (ROOT / "src/savr/pair/p3_openvla.py").read_text(encoding="utf-8")
    if (
        '"output_hidden_states": True' not in openvla_text
        or "output.hidden_states[-1]" not in openvla_text
        or "output.last_hidden_state" in openvla_text
    ):
        raise RuntimeError("P3 pinned CausalLM output contract is invalid")

    result_root = ROOT / "results" / config["run_id"]
    if result_root.exists():
        raise RuntimeError("P3 immutable result root already exists")
    if (ROOT / args.output).exists():
        raise RuntimeError("P3 immutable preflight evidence already exists")

    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment["PYTHONPATH"] = ".:src"
    command = [sys.executable, "-m", "pytest", "-q", "tests/pair", "tests/brace"]
    process = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    summary = process.stdout.strip().splitlines()[-1] if process.stdout.strip() else ""
    if process.returncode != 0:
        raise RuntimeError(f"P3 CPU regression gate failed: {summary}")
    passed = sum(int(value) for value in re.findall(r"(\d+) passed", summary))
    if passed < 70:
        raise RuntimeError("P3 focused regression count is unexpectedly small")

    report = {
        "schema_version": "pair-p3-preflight-v1",
        "run_id": config["run_id"],
        "status": "passed",
        "configuration_semantic_sha256": config["semantic_sha256"],
        "input_manifest_semantic_sha256": inputs["semantic_sha256"],
        "p2_semantic_sha256": p2["semantic_sha256"],
        "revisions": observed_revisions,
        "planned_model_queries": 688,
        "result_root_absent": True,
        "cuda_hidden": True,
        "model_loaded": False,
        "checkpoint_contents_loaded": False,
        "expert_actions_accessed": False,
        "terminal_outcomes_accessed": False,
        "simulator_used": False,
        "network_accessed": False,
        "tests": {"passed": passed, "summary": summary},
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    report["semantic_sha256"] = semantic_sha256(report)
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(canonical_bytes(report) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"status": "passed", "tests": passed, "queries": 688}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
