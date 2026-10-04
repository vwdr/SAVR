#!/usr/bin/env python3
"""CUDA-hidden authenticated preflight for CAC C1 S5 requalification."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from typing import Any


ROOT = Path("/home/ved/SAVR")
CONFIG = ROOT / "configs/cac/c1_s5_requalification_v01.json"


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


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path.name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    if Path.cwd().resolve() != ROOT:
        raise SystemExit(f"S5 preflight must start in {ROOT}")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    checks: dict[str, bool] = {}
    checks["semantic_hash"] = config.get("semantic_sha256") == semantic_sha256(config)
    checks["schema"] = config.get("schema_version") == "cac-c1-requalification-s5-v1"
    checks["authorization"] = config.get("authorization") == {
        "s5_c1_requalification_authorized": True,
        "source": "User approved S5/CAC C1 requalification on 2026-09-01",
        "simulator": False,
        "terminal_outcomes": False,
        "expert_actions": False,
        "training": False,
        "automatic_retry": False,
    }
    checks["authenticated_files"] = all(
        (ROOT / relative).is_file() and file_sha256(ROOT / relative) == expected
        for relative, expected in config["authenticated_files"].items()
    )
    s4_path = ROOT / config["s4_parent"]["summary"]
    s4 = json.loads(s4_path.read_text(encoding="utf-8"))
    checks["s4_parent"] = (
        file_sha256(s4_path) == config["s4_parent"]["summary_sha256"]
        and s4.get("semantic_sha256") == semantic_sha256(s4)
        and s4.get("semantic_sha256") == config["s4_parent"]["semantic_sha256"]
        and s4.get("complete") is True
        and s4.get("passed") is True
        and s4.get("corrected_substrate_id") == "D62_BAL_PT1_S4C_V1"
        and s4.get("model_calls") == 37
    )
    s3_path = ROOT / config["s3_parent"]["summary"]
    s3 = json.loads(s3_path.read_text(encoding="utf-8"))
    checks["s3_parent"] = (
        file_sha256(s3_path) == config["s3_parent"]["summary_sha256"]
        and s3.get("semantic_sha256") == semantic_sha256(s3)
        and s3.get("semantic_sha256") == config["s3_parent"]["semantic_sha256"]
        and s3.get("complete") is True
        and s3.get("passed") is True
        and s3.get("model_calls") == 32
        and s3.get("comparison_count") == 344
    )
    checks["substrate_contract"] = config.get("substrate") == {
        "corrected_id": "D62_BAL_PT1_S4C_V1",
        "historical_profile_id": "D62_BAL_PT1",
        "historical_profile_values_reused": True,
        "s4_selection_materially_changed": False,
        "s4_selection_changed_observations": 0,
        "action_readout_corrected": True,
        "compact_positions_explicitly_mapped": True,
    }
    checks["official_parity_contract"] = config.get("official_parity") == {
        "independent_released_evaluator": True,
        "reversible_action_head_capture": True,
        "observation_index": 2,
        "hidden_tolerance": 0.000001,
        "normalized_action_tolerance": 0.000001,
        "unnormalized_action_tolerance": 0.000001,
        "raw_values_persisted": False,
        "must_pass_before_internal_controls": True,
    }
    r05 = load_module(
        ROOT / "scripts/run_openvla_semantic_parity_recovery05.py", "s5_preflight_r05"
    )
    r04 = r05.load_recovery04()
    r03 = r04.load_recovery03()
    r02 = r03.load_recovery02()
    r01 = r02.load_recovery01()
    checks["qualified_official_loader_guard"] = (
        r02.function_source_sha256(r01.install_official_loader_guard)
        == r02.QUALIFIED_LOADER_GUARD_SHA256
    )
    worker = (ROOT / "scripts/run_cac_c1_worker.py").read_text(encoding="utf-8")
    checks["official_source_and_oracle"] = (
        'source = ROOT / "third_party/openvla-oft"' in worker
        and "OfficialBoundaryCapture" in worker
        and '"official_parity"' in worker
        and "-57:-1" not in worker
    )
    schedule = config["schedule"]
    checks["call_accounting"] = (
        int(schedule["official_parity_calls"]) == 2
        and sum(
            int(schedule[key])
            for key in (
                "official_parity_calls",
                "warmup_calls",
                "sidecar_and_allfresh_controls",
                "hook_control_calls",
                "recursive_repeat_and_reset_calls",
                "isolation_calls",
                "timing_calls",
            )
        ) == 97
        and int(schedule["timing_calls"]) == 64
        and int(schedule["total_calls"]) == 97
    )
    model_config = json.loads((ROOT / config["model_config"]).read_text(encoding="utf-8"))
    checkpoint = ROOT / model_config["model"]["checkpoint_relative"]
    checks["checkpoint_metadata"] = all(
        (checkpoint / name).is_file()
        and file_sha256(checkpoint / name) == expected
        for name, expected in config["checkpoint_metadata_sha256"].items()
    )
    protected = ("config.json", "configuration_prismatic.py", "modeling_prismatic.py")
    checks["checkpoint_no_stale_backups"] = not any(
        path.name.startswith(tuple(f"{name}.back." for name in protected))
        or path.name.startswith(tuple(f"{name}.backup" for name in protected))
        or path.name in {f"{name}.bak" for name in protected}
        for path in checkpoint.iterdir()
    )
    manifest = json.loads((ROOT / config["input_manifest"]).read_text(encoding="utf-8"))
    data_root = ROOT / manifest["data_root_relative"]
    checks["selected_observation_sources"] = all(
        file_sha256(data_root / row["source_path"]) == row["source_sha256"]
        for row in (manifest["inputs"][2], manifest["inputs"][3])
    )
    checks["output_absent"] = not (ROOT / config["output_root"]).exists()
    checks["storage"] = shutil.disk_usage(ROOT).free >= 8 * 1024**3
    checks["runtime"] = Path(sys.executable).resolve() == Path(
        "/home/ved/SAVR/envs/vla-cache-compat/bin/python"
    ).resolve()
    ready = all(checks.values())
    print(json.dumps({
        "ready": ready,
        "checks": checks,
        "planned_model_calls": 97,
        "model_call_hard_cap": 160,
        "gpu_selected": False,
        "simulator_outcomes": 0,
        "next_phase_authorized": False,
        "writes_performed": False,
    }, sort_keys=True))
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
