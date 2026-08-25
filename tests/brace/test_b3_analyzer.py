from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def action() -> dict[str, object]:
    return {"values": [[0.0] * 7 for _ in range(8)]}


def load_analyzer():
    spec = importlib.util.spec_from_file_location(
        "brace_b3_analyzer_test_module", ROOT / "scripts/analyze_brace_b3.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_analyzer_accepts_a_complete_synthetic_frozen_population(tmp_path, monkeypatch):
    cfg = json.loads((ROOT / "configs/brace/b3_physical_v1.json").read_text())
    config_path = tmp_path / "configs/brace/b3_physical_v1.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(json.dumps(cfg))
    run_path = tmp_path / "results" / cfg["run_id"]
    workers_path = run_path / "workers"
    workers_path.mkdir(parents=True)
    source_revision = "a" * 40
    common = {
        "status": "completed",
        "configuration_semantic_sha256": cfg["semantic_sha256"],
        "source_revision": source_revision,
        "peak_reserved_bytes": 1024,
    }
    actions = {label: action() for label in ("input-a", "input-b", "input-c")}
    core = {**common, "queries": 22, "action_references": actions}
    profiles = {}
    for profile in cfg["profiles"]:
        cycles = []
        for horizon in cfg["measurement"]["horizons"]:
            for repetition in range(6):
                accelerated = [
                    {"wall_ms": 80.0, "action_parity": {"passed": True}}
                    for _ in range(horizon)
                ]
                cycles.append(
                    {
                        "horizon": horizon,
                        "repetition": repetition,
                        "cycle_wall_ms": 80.0 * (horizon + 1),
                        "accelerated": accelerated,
                    }
                )
        profiles[profile["profile_id"]] = {"cycles": cycles}
    cache = {
        **common,
        "queries": 302,
        "p0_action_references": actions,
        "p0_timings": [{"wall_ms": 100.0} for _ in range(16)],
        "profiles": profiles,
        "corrected_vla_cache": {"all_action_parity": True},
        "cache_provenance_reset_invariants": True,
        "warm_profile_parity": [{"passed": True} for _ in cfg["profiles"]],
        "sidecar_parity": {"passed": True},
    }
    adp = {
        **common,
        "queries": 32,
        "timing_validity": "component_timing_only_episode_coupled_dynamic_controller_excluded",
        "timings": [{"wall_ms": 95.0}],
    }
    pruner = {
        **common,
        "queries": 32,
        "timing_validity": "official_temporal_semantic_action_timing",
        "timings": [{"wall_ms": 90.0}],
    }
    for name, worker in {
        "core_fr": core,
        "cache_suite": cache,
        "vla_adp": adp,
        "vla_pruner": pruner,
    }.items():
        (workers_path / f"{name}.json").write_text(json.dumps(worker))
    run = {
        "source_revision": source_revision,
        "planned_queries": 388,
        "queries": 388,
        "peak_aggregate_gpu_memory_used_mib": 1,
    }
    run["semantic_sha256"] = hashlib.sha256(canonical_bytes(run)).hexdigest()
    (run_path / "run_summary.json").write_text(json.dumps(run))

    analyzer = load_analyzer()
    monkeypatch.setattr(analyzer, "ROOT", tmp_path)
    monkeypatch.setattr(analyzer, "EXPECTED_ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["analyze_brace_b3.py"])
    assert analyzer.main() == 0
    analysis = json.loads((run_path / "analysis.json").read_text())
    assert analysis["status"] == "accepted"
    assert all(analysis["gates"].values())
    assert set(analysis["passing_profiles"]) == set(profiles)
    assert analysis["b4_authorized"] is False
    assert "No simulator outcomes" in analysis["interpretation_boundary"]
