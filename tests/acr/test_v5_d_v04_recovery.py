from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from savr.acr.v5_d_runtime import load_v5_d_freeze
from savr.acr.v5_d_v04_runtime import load_v04, resolve_v04


ROOT = Path(__file__).resolve().parents[2]


def test_v04_changes_only_the_frozen_backend_memory_contract() -> None:
    v03 = load_v5_d_freeze(ROOT)
    v04 = load_v04(ROOT)
    unchanged = (
        "selected_method",
        "pinned_stack",
        "environment_hashes",
        "checkpoint_hashes",
        "upstream_source_hashes",
        "identities",
        "backend_waterfall",
        "inputs",
        "tensor_contract",
        "correctness",
        "timing",
        "analysis",
        "gates",
        "gpu_selection",
        "resource_caps",
        "recovery",
        "recovery_v02",
        "recovery_v03",
    )
    assert all(v04[key] == v03[key] for key in unchanged)
    assert v04["run_id"] == "acr-v5d-real-tensor-feasibility-v04"
    assert v04["raw_cuda_graph"]["shared_private_pool"] is True
    assert v04["memory"]["peak_reserved_bytes_max"] == 23 * 1024**3
    assert v04["memory"]["raise_cap"] is False


def test_v04_rejects_unhashed_or_out_of_scope_changes() -> None:
    v03 = load_v5_d_freeze(ROOT)
    recovery = json.loads(
        (ROOT / "configs/acr/v5_d_titan_memory_recovery_v04.json").read_text()
    )
    tampered = deepcopy(recovery)
    tampered["memory"]["raise_cap"] = True
    with pytest.raises(ValueError, match="semantic hash"):
        resolve_v04(v03, tampered)
