from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def load(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def config() -> dict:
    return json.loads(
        (ROOT / "configs/acr/v5_d_gpu_feasibility_freeze.json").read_text(encoding="utf-8")
    )


def synthetic_run(freeze: dict, *, optimized_wall=79.5) -> dict:
    values = {
        "batched-fr": (100.0, 90.0, 30.0, 90.0),
        "v5-refresh": (100.0, 90.0, 30.0, 90.0),
        "eager-reuse": (90.0, 80.0, 20.0, 80.0),
        "optimized-reuse": (optimized_wall, 70.0, 15.0, 70.0),
    }
    queries = []
    for label in freeze["correctness"]["labels"]:
        queries.append(
            {
                "query_index": len(queries),
                "label": label,
                "kind": "correctness",
                "passed": True,
                "checks": {"parity": True, "work": True, "pointers": True},
            }
        )
    for path in freeze["timing"]["paths"]:
        for repetition in range(2):
            wall, total, visual, sequential = values[path]
            queries.append(
                {
                    "query_index": len(queries),
                    "label": f"warmup-{path}-{repetition:02d}",
                    "kind": "warmup",
                    "path": path,
                    "timing": {
                        "wall_ms": wall,
                        "total_cuda_ms": total,
                        "visual_cuda_ms": visual,
                        "sequential_cuda_ms": sequential,
                    },
                }
            )
    for block, order in enumerate(freeze["timing"]["permutations"]):
        for position, path in enumerate(order):
            wall, total, visual, sequential = values[path]
            queries.append(
                {
                    "query_index": len(queries),
                    "label": f"timed-{block:02d}-{position}-{path}",
                    "kind": "timed",
                    "block": block,
                    "position": position,
                    "path": path,
                    "input_label": "input-a" if block % 2 == 0 else "input-b",
                    "timing": {
                        "wall_ms": wall,
                        "total_cuda_ms": total,
                        "visual_cuda_ms": visual,
                        "sequential_cuda_ms": sequential,
                    },
                }
            )
    run = {
        "schema_version": "acr.v5d-run.v1",
        "run_id": freeze["run_id"],
        "status": "completed",
        "configuration_semantic_sha256": freeze["semantic_sha256"],
        "backend": "torch-compile",
        "queries": queries,
        "correctness_pass": True,
        "memory_pass": True,
        "work_pass": True,
        "lifecycle_pass": True,
        "restoration_pass": True,
        "resource_pass": True,
    }
    analyze = load("analyze_acr_v5_d")
    run["semantic_sha256"] = analyze.semantic_sha256(run)
    return run


def test_analysis_is_deterministic_positive_and_independently_verified() -> None:
    freeze = config()
    run = synthetic_run(freeze)
    analyzer = load("analyze_acr_v5_d")
    verifier = load("verify_acr_v5_d")
    first = analyzer.analyze(freeze, run)
    second = analyzer.analyze(freeze, run)
    assert first == second
    assert first["passed"] is True
    assert first["failed_gates"] == []
    assert first["maximum_ordering_deviation"] == 0.0
    assert first["timed_blocks"] == 24 and first["timed_queries"] == 96
    assert verifier.verify(freeze, run, first) == []


def test_analysis_fails_closed_on_negative_efficiency() -> None:
    freeze = config()
    run = synthetic_run(freeze, optimized_wall=99.0)
    analyzer = load("analyze_acr_v5_d")
    verifier = load("verify_acr_v5_d")
    result = analyzer.analyze(freeze, run)
    assert result["passed"] is False
    assert result["disposition"] == "STOP_BEFORE_SIMULATOR"
    assert "optimized_reuse_wall" in result["failed_gates"]
    assert verifier.verify(freeze, run, result) == []
