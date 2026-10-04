#!/usr/bin/env python3
"""CUDA-hidden preflight for C1H S6 Recovery 01 qualification and attempt."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from typing import Any


ROOT = Path("/home/ved/SAVR")
CONFIG = ROOT / "configs/cac/c1h_headroom_s6_v02_recovery01.json"


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def semantic_sha256(value: dict[str, Any]) -> str:
    body = dict(value)
    body.pop("semantic_sha256", None)
    return hashlib.sha256(canonical_bytes(body)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_worker() -> Any:
    path = ROOT / "scripts/run_cac_c1h_worker.py"
    spec = importlib.util.spec_from_file_location("cac_c1h_recovery01_preflight", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load C1H worker")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def qualification_passed(path: Path) -> bool:
    if not path.is_file():
        return False
    value = json.loads(path.read_text(encoding="utf-8"))
    controls = value.get("technical_controls", {})
    restoration = value.get("checkpoint_restoration", {})
    return bool(
        value.get("semantic_sha256") == semantic_sha256(value)
        and value.get("complete") is True
        and value.get("passed") is True
        and value.get("mode") == "outcome_free_pre_episode_qualification"
        and value.get("model_queries") == 4
        and value.get("terminal_episode_attempts") == 0
        and value.get("terminal_outcomes_opened") is False
        and controls.get("queries") == 4
        and controls.get("independent_official_oracle") is True
        and controls.get("observation_isolated") is True
        and controls.get("d62_anchor_and_reuse") is True
        and max(
            float(controls.get("official_hidden_max_abs", 1.0)),
            float(controls.get("official_normalized_action_max_abs", 1.0)),
            float(controls.get("official_helper_max_abs", 1.0)),
        ) <= 1e-6
        and all(
            restoration.get(key) is True
            for key in ("protected_bytes_restored", "backup_cleanup_complete", "inventory_equal")
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("qualification", "attempt"), required=True)
    args = parser.parse_args()
    if Path.cwd().resolve() != ROOT:
        raise SystemExit(f"C1H Recovery 01 preflight must start in {ROOT}")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    checks: dict[str, bool] = {}
    checks["semantic_hash"] = config.get("semantic_sha256") == semantic_sha256(config)
    checks["schema"] = config.get("schema_version") == "cac-c1h-corrected-s6-recovery01-v1"
    checks["authorization"] = config.get("authorization") == {
        "c1h_authorized": True,
        "c2_authorized": False,
        "terminal_outcomes": True,
        "locked_state_ids_10_49": False,
    }
    checks["source_stop"] = config.get("technical_recovery", {}).get(
        "source_run_id"
    ) == "cac-c1h-headroom-s6-v01"
    checks["corrected_substrate"] = config.get("cache_substrate", {}).get(
        "profile"
    ) == "D62_BAL_PT1_S4C_V1"
    checks["runtime"] = Path(sys.executable).resolve() == Path(
        "/home/ved/SAVR/envs/vla-cache-compat/bin/python"
    ).resolve()
    checks["storage"] = shutil.disk_usage(ROOT).free >= int(
        config["forecast"]["project_disk_margin_required_bytes"]
    )
    population = ROOT / config["population_manifest"]
    rows = [json.loads(line) for line in population.read_text(encoding="utf-8").splitlines() if line]
    checks["population"] = (
        len(rows) == 2240
        and sum(row.get("population") == "headroom_stage1" for row in rows) == 120
        and sum(row.get("population") == "headroom_extension" for row in rows) == 120
    )
    checks["caps"] = config.get("resource_caps") == {
        "gpu_count": 1,
        "model_processes": 1,
        "episode_attempts": 480,
        "model_queries": 20000,
        "wall_seconds": 36000,
        "artifact_bytes": 1073741824,
        "peak_gpu_memory_mib_strict_max": 23552,
        "downloads": 0,
        "automatic_retry": False,
    }
    checks["advance"] = config.get("advance") == {
        "next_phase": "C2", "authorized": False, "stop_before_next_phase": True
    }
    checks["worker_contract"] = False
    try:
        load_worker().validate_config(config)
        checks["worker_contract"] = True
    except BaseException:
        checks["worker_contract"] = False
    qualification_root = ROOT / config["qualification_output_root"]
    attempt_root = ROOT / config["output_root"]
    if args.mode == "qualification":
        checks["qualification_output_absent"] = not qualification_root.exists()
        checks["attempt_output_absent"] = not attempt_root.exists()
    else:
        checks["qualification_passed"] = qualification_passed(
            qualification_root / "worker_summary.json"
        )
        checks["attempt_output_absent"] = not attempt_root.exists()
    ready = all(checks.values())
    print(json.dumps({
        "ready": ready,
        "mode": args.mode,
        "checks": checks,
        "gpu_selected": False,
        "terminal_outcomes_opened": False,
        "writes_performed": False,
        "next_phase_authorized": False,
    }, sort_keys=True))
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())

