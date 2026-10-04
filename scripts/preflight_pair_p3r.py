#!/usr/bin/env python3
"""CUDA-hidden fail-closed preflight for the single PAIR P3R run."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
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


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def semantic_sha256(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("semantic_sha256", None)
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if ROOT != EXPECTED_ROOT:
        raise SystemExit(f"P3R preflight refuses to run outside {EXPECTED_ROOT}")
    if os.environ.get("CUDA_VISIBLE_DEVICES", ""):
        raise RuntimeError("P3R preflight requires CUDA to be hidden")

    worker_path = ROOT / "scripts/run_pair_p3r_worker.py"
    analyzer_path = ROOT / "scripts/analyze_pair_p3r.py"
    spec = importlib.util.spec_from_file_location("pair_p3r_worker_preflight", worker_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("P3R worker module cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    config = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    inputs = json.loads((ROOT / config["input_manifest"]).read_text(encoding="utf-8"))
    module.validate_p3r_config(config, input_count=len(inputs["inputs"]))
    if inputs.get("semantic_sha256") != semantic_sha256(inputs):
        raise RuntimeError("P3R input-manifest semantic hash mismatch")
    parent = ROOT / config["parent_p3_analysis"]
    if file_sha256(parent) != config["parent_p3_analysis_sha256"]:
        raise RuntimeError("P3R parent P3 analysis hash mismatch")
    parent_analysis = json.loads(parent.read_text(encoding="utf-8"))
    if parent_analysis.get("status") != "scientific_stop" or parent_analysis.get(
        "passing_points"
    ):
        raise RuntimeError("P3R parent decision is not the frozen P3 scientific stop")

    p0 = json.loads((ROOT / "configs/pair/p0_freeze_v1.json").read_text(encoding="utf-8"))
    revisions = p0["authenticated_revisions"]
    observed_revisions = {
        "project_git": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        "vla_cache_git": subprocess.check_output(
            ["git", "-C", str(ROOT / "third_party/vla-cache"), "rev-parse", "HEAD"],
            text=True,
        ).strip(),
    }
    for name, value in observed_revisions.items():
        if value != revisions[name]:
            raise RuntimeError(f"P3R authenticated revision mismatch: {name}")

    model_config = json.loads((ROOT / config["model_config"]).read_text(encoding="utf-8"))
    checkpoint = ROOT / model_config["model"]["checkpoint_relative"]
    for relative in ("config.json", "configuration_prismatic.py", "modeling_prismatic.py"):
        if not (checkpoint / relative).is_file():
            raise RuntimeError(f"P3R checkpoint file is missing: {relative}")
    data_root = (ROOT / inputs["data_root_relative"]).resolve()
    if not data_root.is_relative_to(ROOT):
        raise RuntimeError("P3R data root escaped the project")
    for identity in inputs["inputs"]:
        source = (data_root / identity["source_path"]).resolve()
        if not source.is_relative_to(data_root) or not source.is_file():
            raise RuntimeError("P3R frozen input source is missing or escaped the data root")

    worker_text = worker_path.read_text(encoding="utf-8")
    analyzer_text = analyzer_path.read_text(encoding="utf-8")
    helper_text = (ROOT / "src/savr/pair/p3_openvla.py").read_text(encoding="utf-8")
    forbidden = {
        "expert action dataset": r"group\s*\[\s*[\"']actions[\"']\s*\]",
        "terminal dataset": r"group\s*\[\s*[\"']dones[\"']\s*\]",
        "automatic retry": r"\bfor\s+attempt\b|\bretry\s*\(",
        "simulator construction": r"\b(get_libero_env|OffScreenRenderEnv)\s*\(",
    }
    for label, pattern in forbidden.items():
        if re.search(pattern, worker_text) or re.search(pattern, analyzer_text):
            raise RuntimeError(f"P3R implementation contains forbidden {label} path")
    for required in (
        "ordered_tile_profile_vectorized(",
        "ordered_tile_profile(",
        "torch.equal(legacy[0], vectorized[0])",
        "equivalence_checks != 8",
        "ledger.require_complete()",
    ):
        if required not in worker_text:
            raise RuntimeError("P3R worker equivalence or accounting guard is missing")
    if "def _vectorized_camera_scores(" not in helper_text or "_batched_change_score(" not in helper_text:
        raise RuntimeError("P3R vectorized helper is missing")
    for required in (
        '"legacy_vectorized_exact_equivalence": equivalence_pass',
        '"maximum_total_overhead_fraction"',
        '"minimum_net_saving_fraction"',
        '"authorized": False',
    ):
        if required not in analyzer_text:
            raise RuntimeError("P3R frozen analyzer gate is missing")

    runtime_import_passed = False
    if config.get("technical_recovery") is not None:
        source = ROOT / "third_party/vla-cache/src/openvla-oft"
        import_code = (
            "import sys;"
            f"sys.path.insert(0,{str(source)!r});"
            "import seaborn,torch;"
            "from experiments.robot.libero import run_libero_eval;"
            "assert not torch.cuda.is_initialized()"
        )
        import_environment = dict(os.environ)
        import_environment["CUDA_VISIBLE_DEVICES"] = ""
        import_environment["LIBERO_CONFIG_PATH"] = str(
            ROOT / "configs/pair/libero_runtime"
        )
        import_environment["PYTHONPATH"] = f"{ROOT}:{ROOT / 'src'}"
        imported = subprocess.run(
            [sys.executable, "-c", import_code],
            cwd=ROOT,
            env=import_environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        if imported.returncode != 0:
            tail = imported.stdout.strip().splitlines()[-1] if imported.stdout.strip() else ""
            raise RuntimeError(f"P3R authenticated runtime import failed: {tail}")
        runtime_import_passed = True

    result_root = ROOT / "results" / config["run_id"]
    output = ROOT / args.output
    if result_root.exists() or output.exists():
        raise RuntimeError("P3R immutable output already exists")

    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment["PYTHONPATH"] = ".:src"
    process = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "tests/pair", "tests/brace"],
        cwd=ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    summary = process.stdout.strip().splitlines()[-1] if process.stdout.strip() else ""
    if process.returncode != 0:
        raise RuntimeError(f"P3R regression gate failed: {summary}")
    passed = sum(int(value) for value in re.findall(r"(\d+) passed", summary))
    if passed < 87:
        raise RuntimeError("P3R focused regression count is unexpectedly small")

    report = {
        "schema_version": "pair-p3r-preflight-v1",
        "run_id": config["run_id"],
        "status": "passed",
        "configuration_semantic_sha256": config["semantic_sha256"],
        "input_manifest_semantic_sha256": inputs["semantic_sha256"],
        "parent_p3_analysis_sha256": config["parent_p3_analysis_sha256"],
        "revisions": observed_revisions,
        "planned_model_queries": 210,
        "result_root_absent": True,
        "cuda_hidden": True,
        "model_loaded": False,
        "checkpoint_contents_loaded": False,
        "demonstration_contents_accessed": False,
        "expert_actions_accessed": False,
        "terminal_outcomes_accessed": False,
        "simulator_used": False,
        "network_accessed": False,
        "runtime_python": str(Path(sys.executable).resolve()),
        "runtime_import_passed": runtime_import_passed,
        "tests": {"passed": passed, "summary": summary},
        "worker_sha256": file_sha256(worker_path),
        "analyzer_sha256": file_sha256(analyzer_path),
        "vectorized_helper_sha256": file_sha256(ROOT / "src/savr/pair/p3_openvla.py"),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    report["semantic_sha256"] = semantic_sha256(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(canonical_bytes(report) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"status": "passed", "tests": passed, "queries": 210}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
