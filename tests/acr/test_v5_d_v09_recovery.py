from __future__ import annotations

import json
from pathlib import Path

from savr.acr.v5_d_runtime import semantic_sha256
from savr.acr.v5_d_v08_runtime import load_v08
from savr.acr.v5_d_v09_runtime import DEFAULT_ALLOCATOR, V08_MEASURED_EVIDENCE, load_v09


ROOT = Path(__file__).resolve().parents[2]


def test_v09_changes_only_identity_allocator_and_authorization() -> None:
    v08 = load_v08(ROOT)
    v09 = load_v09(ROOT)
    changed = {
        "schema_version",
        "status",
        "authorized_at",
        "authorized_scope",
        "protocol",
        "run_id",
        "allocator",
        "recovery_v09",
        "current_authorization",
        "advance_only_to",
        "semantic_sha256",
    }
    assert {key: value for key, value in v09.items() if key not in changed} == {
        key: value for key, value in v08.items() if key not in changed
    }
    assert v09["raw_cuda_graph"] == v08["raw_cuda_graph"]
    assert v09["pre_capture_warmup"] == v08["pre_capture_warmup"]
    assert v09["inference_semantics"] == v08["inference_semantics"]
    assert v09["memory"] == v08["memory"]


def test_v09_default_allocator_contract_is_exact() -> None:
    config = load_v09(ROOT)
    assert config["allocator"] == DEFAULT_ALLOCATOR
    assert DEFAULT_ALLOCATOR["exact_value"] is None
    assert DEFAULT_ALLOCATOR["must_be_unset_before_torch_import"] is True
    assert DEFAULT_ALLOCATOR["backend"] == "native-default"
    assert DEFAULT_ALLOCATOR["experimental"] is False
    assert DEFAULT_ALLOCATOR["empty_cache_calls"] == 0
    assert DEFAULT_ALLOCATOR["automatic_retries"] == 0


def test_v09_launcher_removes_only_the_v08_allocator_override() -> None:
    source = (ROOT / "scripts/launch_acr_v5_d_v09.sh").read_text(encoding="utf-8")
    assert source.index("--backend torch-compile") < source.index("--backend raw-cudagraph")
    assert "expandable_segments:True" not in source
    assert "PYTORCH_CUDA_ALLOC_CONF=" not in source
    assert "Refusing inherited allocator configuration" in source
    assert "empty_cache" not in source
    assert "max_split_size_mb" not in source
    assert "cudaMallocAsync" not in source


def test_v09_runner_requires_default_allocator_and_preserves_inference_guard() -> None:
    source = (ROOT / "scripts/run_acr_v5_d_v09.py").read_text(encoding="utf-8")
    adapter = (ROOT / "src/savr/acr/v5_d_v09_adapter.py").read_text(encoding="utf-8")
    assert 'os.environ.get("PYTORCH_CUDA_ALLOC_CONF") is not None' in source
    assert 'observed_backend != "native"' in adapter
    assert '"observed_backend": observed_backend' in adapter
    assert "V08InferenceLifecycle" in source
    assert "runner._run_attempt = guarded_attempt" in source
    assert "lifecycle.close()" in source
    assert "inference-lifecycle" in source


def test_v09_configuration_and_v08_stop_are_authentic() -> None:
    config = load_v09(ROOT)
    stop = json.loads(
        (ROOT / "reports/runtime/acr_v5_d_v08_technical_stop.json").read_text(
            encoding="utf-8"
        )
    )
    assert config["semantic_sha256"] == semantic_sha256(config)
    assert stop["semantic_sha256"] == semantic_sha256(stop)
    assert config["recovery_v09"]["v08_technical_stop_semantic_sha256"] == stop[
        "semantic_sha256"
    ]
    assert config["recovery_v09"]["v08_measured_evidence"] == V08_MEASURED_EVIDENCE


def test_v09_authorizes_exactly_one_frozen_gpu_attempt() -> None:
    authorization = load_v09(ROOT)["current_authorization"]
    assert authorization["gpu_selection"] is True
    assert authorization["model_queries_max"] == 111
    assert authorization["automatic_retry"] is False
    assert authorization["simulator_use"] is False
    assert authorization["protected_outcome_access"] is False
    assert authorization["manuscript_changes"] is False
