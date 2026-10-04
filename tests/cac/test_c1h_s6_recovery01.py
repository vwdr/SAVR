from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]


class FakeTensor:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape


def load_worker():
    path = ROOT / "scripts/run_cac_c1h_worker.py"
    spec = importlib.util.spec_from_file_location("cac_c1h_worker_recovery_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = ROOT
    return module


def test_captured_policy_output_propagates_both_required_tensors() -> None:
    worker = load_worker()
    hidden = FakeTensor((1, 56, 4096))
    normalized = FakeTensor((1, 8, 7))
    output = worker.build_policy_query_output(
        np,
        {
            "actions": np.zeros((8, 7), dtype=np.float32),
            "cache": object(),
            "action_hidden": hidden,
            "base_action": normalized,
        },
        object(),
        None,
        require_cac=True,
    )
    assert output["action_hidden"] is hidden
    assert output["base_action"] is normalized


@pytest.mark.parametrize("missing", ["action_hidden", "base_action"])
def test_captured_policy_output_rejects_each_missing_tensor(missing: str) -> None:
    worker = load_worker()
    result = {
        "actions": np.zeros((8, 7), dtype=np.float32),
        "cache": object(),
        "action_hidden": FakeTensor((1, 56, 4096)),
        "base_action": FakeTensor((1, 8, 7)),
    }
    del result[missing]
    with pytest.raises(worker.TechnicalStop, match="contract changed"):
        worker.build_policy_query_output(
            np, result, object(), None, require_cac=True
        )


def test_recovery_config_preserves_frozen_scientific_and_resource_boundaries() -> None:
    config = json.loads(
        (ROOT / "configs/cac/c1h_headroom_s6_v02_recovery01.json").read_text()
    )
    assert config["schema_version"] == "cac-c1h-corrected-s6-recovery01-v1"
    assert config["populations"]["episodes_per_stage"] == 240
    assert config["cache_substrate"]["profile"] == "D62_BAL_PT1_S4C_V1"
    assert config["resource_caps"]["episode_attempts"] == 480
    assert config["resource_caps"]["model_queries"] == 20000
    assert config["resource_caps"]["automatic_retry"] is False
    assert config["advance"] == {
        "next_phase": "C2", "authorized": False, "stop_before_next_phase": True
    }


def test_worker_enforces_passing_qualification_before_recovery_attempt() -> None:
    source = (ROOT / "scripts/run_cac_c1h_worker.py").read_text()
    for required in (
        "--qualification-only",
        "outcome_free_pre_episode_qualification",
        "terminal_episode_attempts",
        "C1H Recovery 01 outcome-free qualification did not pass",
        "build_policy_query_output",
    ):
        assert required in source

