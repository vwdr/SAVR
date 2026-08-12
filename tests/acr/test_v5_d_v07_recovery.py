from __future__ import annotations

import json
from pathlib import Path

import pytest

from savr.acr.v5_d_runtime import semantic_sha256
from savr.acr.v5_d_v06_runtime import load_v06
from savr.acr.v5_d_v07_runtime import ALLOCATOR, V06_MEASURED_MEMORY, load_v07


ROOT = Path(__file__).resolve().parents[2]


def test_v07_changes_only_allocator_identity_and_authorization() -> None:
    v06 = load_v06(ROOT)
    v07 = load_v07(ROOT)
    changed = {
        "schema_version",
        "status",
        "authorized_at",
        "authorized_scope",
        "protocol",
        "run_id",
        "allocator",
        "recovery_v07",
        "current_authorization",
        "advance_only_to",
        "semantic_sha256",
    }
    assert {key: value for key, value in v07.items() if key not in changed} == {
        key: value for key, value in v06.items() if key not in changed
    }
    assert v07["allocator"] == ALLOCATOR
    assert v07["recovery_v07"]["v06_measured_memory"] == V06_MEASURED_MEMORY
    assert v07["memory"]["peak_reserved_bytes_max"] == 23 * 1024**3


def test_v07_launcher_scopes_exact_setting_to_raw_only() -> None:
    source = (ROOT / "scripts/launch_acr_v5_d_v07.sh").read_text(encoding="utf-8")
    compiler = source.index("scripts/run_acr_v5_d_v07.py --backend torch-compile")
    allocator = source.index("PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
    raw = source.index("scripts/run_acr_v5_d_v07.py --backend raw-cudagraph")
    assert compiler < allocator < raw
    assert "empty_cache" not in source
    assert "max_split_size_mb" not in source
    assert "cudaMallocAsync" not in source


def test_v07_run_rejects_missing_or_inherited_allocator(monkeypatch) -> None:
    import scripts.run_acr_v5_d_v07 as run

    monkeypatch.setattr(run, "requested_backend", lambda: "raw-cudagraph")
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    with pytest.raises(SystemExit, match="lacks the exact"):
        run.main()

    monkeypatch.setattr(run, "requested_backend", lambda: "torch-compile")
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    with pytest.raises(SystemExit, match="compiler process"):
        run.main()


def test_v07_configuration_and_v06_stop_are_authentic() -> None:
    config = load_v07(ROOT)
    stop = json.loads(
        (ROOT / "reports/runtime/acr_v5_d_v06_technical_stop.json").read_text(encoding="utf-8")
    )
    assert config["semantic_sha256"] == semantic_sha256(config)
    assert stop["semantic_sha256"] == semantic_sha256(stop)
    assert config["recovery_v07"]["v06_technical_stop_semantic_sha256"] == stop["semantic_sha256"]


def test_v07_authorization_is_one_attempt_without_simulator() -> None:
    authorization = load_v07(ROOT)["current_authorization"]
    assert authorization["gpu_selection"] is True
    assert authorization["model_queries_max"] == 111
    assert authorization["automatic_retry"] is False
    assert authorization["simulator_use"] is False
    assert authorization["protected_outcome_access"] is False
    assert authorization["manuscript_changes"] is False
