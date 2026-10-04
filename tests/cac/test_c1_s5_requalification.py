from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def semantic_sha256(value):
    body = dict(value)
    body.pop("semantic_sha256", None)
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def test_s5_worker_uses_independent_official_oracle_before_controls() -> None:
    source = (ROOT / "scripts/run_cac_c1_worker.py").read_text(encoding="utf-8")
    assert 'source = ROOT / "third_party/openvla-oft"' in source
    assert "install_official_loader_guard" in source
    assert "OfficialBoundaryCapture" in source
    assert source.index("official_parity = None") < source.index("# Model warmup")
    for boundary in (
        "hidden_max_abs",
        "normalized_action_max_abs",
        "unnormalized_action_max_abs",
        "observation_isolated",
    ):
        assert boundary in source
    assert "-57:-1" not in source


def test_s5_preserves_all_c1_controls_and_adds_two_oracle_calls() -> None:
    source = (ROOT / "scripts/run_cac_c1_worker.py").read_text(encoding="utf-8")
    for required in (
        "sidecar_action_equal",
        "allfresh_max_abs",
        "head_reproduction_max_abs",
        "run_recursive_cycle",
        "reset_contract_exact",
        "cache_clone_disjoint",
        "gross_headroom",
        "zero_initialized_adapter_bypass",
    ):
        assert required in source
    assert "expected_planned_calls = 97 if s5_requalification else 95" in source


def test_s5_config_is_frozen_outcome_free_and_stops_before_c1h() -> None:
    path = ROOT / "configs/cac/c1_s5_requalification_v01.json"
    if not path.exists():
        return
    config = json.loads(path.read_text(encoding="utf-8"))
    assert config["semantic_sha256"] == semantic_sha256(config)
    assert config["schedule"]["official_parity_calls"] == 2
    assert config["schedule"]["total_calls"] == 97
    assert config["resource_caps"]["planned_model_calls"] == 97
    assert config["authorization"]["simulator"] is False
    assert config["authorization"]["terminal_outcomes"] is False
    assert config["authorization"]["expert_actions"] is False
    assert config["advance"] == {
        "next_phase": "C1H",
        "authorized": False,
        "stop_before_next_phase": True,
    }


def test_s5_launcher_has_one_attempt_no_retry_or_cleanup() -> None:
    source = (ROOT / "scripts/launch_cac_c1_s5.sh").read_text(encoding="utf-8")
    assert source.count("run_cac_c1_worker.py") == 1
    assert source.count("preflight_cac_c1_s5.py") == 1
    assert "while " not in source and "retry" not in source.lower()
    assert "rm " not in source and "sudo" not in source
