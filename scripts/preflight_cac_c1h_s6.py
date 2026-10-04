#!/usr/bin/env python3
"""CUDA-hidden authenticated preflight for corrected-substrate CAC C1H S6."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from typing import Any


ROOT = Path("/home/ved/SAVR")
CONFIG = ROOT / "configs/cac/c1h_headroom_s6_v01.json"


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
    spec = importlib.util.spec_from_file_location("cac_c1h_worker_s6_preflight", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load C1H worker")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    if Path.cwd().resolve() != ROOT:
        raise SystemExit(f"C1H S6 preflight must start in {ROOT}")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    checks: dict[str, bool] = {}
    checks["semantic_hash"] = config.get("semantic_sha256") == semantic_sha256(config)
    checks["schema"] = config.get("schema_version") == "cac-c1h-corrected-s6-v1"
    checks["authorization"] = config.get("authorization") == {
        "c1h_authorized": True,
        "c2_authorized": False,
        "terminal_outcomes": True,
        "locked_state_ids_10_49": False,
    }
    checks["corrected_substrate"] = config.get("cache_substrate") == {
        "profile": "D62_BAL_PT1_S4C_V1",
        "historical_profile_id": "D62_BAL_PT1",
        "historical_profile_values_reused": True,
        "action_readout_corrected": True,
        "compact_positions_explicitly_mapped": True,
        "maximum_cached_query_intervals": 4,
        "reset_rule": "complete dense reset after at most four cached query intervals",
        "actions_per_query": 8,
    }
    checks["output_absent"] = not (ROOT / config["output_root"]).exists()
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
        and {int(row["initial_state_id"]) for row in rows if row.get("population") == "headroom_stage1"}
        == {0, 1, 2}
        and {int(row["initial_state_id"]) for row in rows if row.get("population") == "headroom_extension"}
        == {3, 4, 5}
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
    ready = all(checks.values())
    print(json.dumps({
        "ready": ready,
        "checks": checks,
        "stage1_terminal_episodes": 240,
        "maximum_terminal_episodes": 480,
        "maximum_model_queries": 20000,
        "gpu_selected": False,
        "terminal_outcomes_opened": False,
        "writes_performed": False,
        "next_phase_authorized": False,
    }, sort_keys=True))
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
