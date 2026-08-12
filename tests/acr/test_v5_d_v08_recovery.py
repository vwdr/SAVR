from __future__ import annotations

import json
from pathlib import Path

import pytest

from savr.acr.v5_d_runtime import semantic_sha256
from savr.acr.v5_d_v07_runtime import load_v07
from savr.acr.v5_d_v08_inference import V08InferenceLifecycle
from savr.acr.v5_d_v08_runtime import INFERENCE_SEMANTICS, V07_MEASURED_MEMORY, load_v08


ROOT = Path(__file__).resolve().parents[2]


class Guard:
    def __init__(self, torch) -> None:
        self.torch = torch

    def __enter__(self):
        self.torch.inference = True
        self.torch.grad = False

    def __exit__(self, *_):
        self.torch.inference = False
        self.torch.grad = True


class FakeTorch:
    def __init__(self) -> None:
        self.grad = True
        self.inference = False

    def is_grad_enabled(self):
        return self.grad

    def is_inference_mode_enabled(self):
        return self.inference

    def inference_mode(self):
        return Guard(self)


def test_v08_changes_only_inference_identity_and_authorization() -> None:
    v07 = load_v07(ROOT)
    v08 = load_v08(ROOT)
    changed = {
        "schema_version",
        "status",
        "authorized_at",
        "authorized_scope",
        "protocol",
        "run_id",
        "inference_semantics",
        "recovery_v08",
        "current_authorization",
        "advance_only_to",
        "semantic_sha256",
    }
    assert {key: value for key, value in v08.items() if key not in changed} == {
        key: value for key, value in v07.items() if key not in changed
    }
    assert v08["inference_semantics"] == INFERENCE_SEMANTICS
    assert v08["recovery_v08"]["v07_measured_memory"] == V07_MEASURED_MEMORY
    assert v08["allocator"] == v07["allocator"]
    assert v08["memory"] == v07["memory"]


def test_v08_inference_lifecycle_enters_after_transition_and_restores() -> None:
    torch = FakeTorch()
    calls = []
    lifecycle = V08InferenceLifecycle(
        torch_module=torch,
        transition=lambda physical: calls.append(physical) or {"index": int(physical)},
    )
    assert lifecycle.snapshot_and_enter("0") == {"index": 0}
    assert calls == ["0"]
    assert lifecycle.active_attestation() == {
        "entered": True,
        "grad_enabled": False,
        "inference_mode_enabled": True,
    }
    lifecycle.close()
    assert lifecycle.lifecycle_record()["restored"] is True
    assert torch.grad is True and torch.inference is False


def test_v08_rejects_inherited_inference_context() -> None:
    torch = FakeTorch()
    torch.inference = True
    lifecycle = V08InferenceLifecycle(torch_module=torch, transition=lambda _: {"index": 0})
    with pytest.raises(RuntimeError, match="inherited"):
        lifecycle.snapshot_and_enter("0")


def test_v08_launcher_preserves_v07_allocator_and_scopes_inference_to_raw() -> None:
    source = (ROOT / "scripts/launch_acr_v5_d_v08.sh").read_text(encoding="utf-8")
    compiler = source.index("scripts/run_acr_v5_d_v08.py --backend torch-compile")
    allocator = source.index("PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
    raw = source.index("scripts/run_acr_v5_d_v08.py --backend raw-cudagraph")
    assert compiler < allocator < raw
    assert "empty_cache" not in source
    assert "max_split_size_mb" not in source
    runner = (ROOT / "scripts/run_acr_v5_d_v08.py").read_text(encoding="utf-8")
    assert "V08InferenceLifecycle" in runner
    assert "runner._run_attempt = guarded_attempt" in runner


def test_v08_configuration_and_v07_stop_are_authentic() -> None:
    config = load_v08(ROOT)
    stop = json.loads(
        (ROOT / "reports/runtime/acr_v5_d_v07_technical_stop.json").read_text(encoding="utf-8")
    )
    assert config["semantic_sha256"] == semantic_sha256(config)
    assert stop["semantic_sha256"] == semantic_sha256(stop)
    assert config["recovery_v08"]["v07_technical_stop_semantic_sha256"] == stop["semantic_sha256"]


def test_v08_authorizes_exactly_one_frozen_gpu_attempt() -> None:
    authorization = load_v08(ROOT)["current_authorization"]
    assert authorization["gpu_selection"] is True
    assert authorization["model_queries_max"] == 111
    assert authorization["automatic_retry"] is False
    assert authorization["simulator_use"] is False
    assert authorization["protected_outcome_access"] is False
    assert authorization["manuscript_changes"] is False
