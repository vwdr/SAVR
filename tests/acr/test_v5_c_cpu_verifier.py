from __future__ import annotations

import importlib.util
import inspect
import json
from pathlib import Path

from savr.acr.reuse_executor import StaticBufferReuseExecutor


ROOT = Path(__file__).resolve().parents[2]


def load_verifier():
    path = ROOT / "scripts/verify_acr_v5_c_executor.py"
    spec = importlib.util.spec_from_file_location("verify_acr_v5_c_executor", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v5_c_verifier_is_deterministic_and_complete():
    verifier = load_verifier()
    first = verifier.verify()
    second = verifier.verify()
    assert first == second
    assert first["verified"] is True
    assert first["completed_queries"] == {"reference": 3, "static": 3}
    assert first["core_calls"]["static"] == {"scene": 0, "wrist": 3, "downstream": 3}
    assert all(first["checks"].values())
    assert first["controller_trace"]["maximum_reuse_streak"] == 1
    assert first["controller_trace"]["maximum_prefix_reuse_fraction"] <= 0.40
    assert first["legacy_separation"]["verified"] is True
    assert len(first["compatibility_key_sha256"]) == 64
    assert len(first["semantic_sha256"]) == 64
    published = json.loads(
        (ROOT / "reports/runtime/acr_v5_c_cpu_executor_verification.json").read_text(
            encoding="utf-8"
        )
    )
    assert published == first


def test_static_hot_path_contains_no_audit_or_host_side_effects():
    source = inspect.getsource(StaticBufferReuseExecutor._execute)
    forbidden = (
        "hashlib",
        "json",
        "open(",
        "Path(",
        "numpy",
        ".cpu(",
        "synchronize",
        "all_finite",
        "tolist",
    )
    assert not any(token in source for token in forbidden)
