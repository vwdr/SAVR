from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_libero_environment_helpers_are_imported_from_libero_utils() -> None:
    tree = ast.parse((ROOT / "scripts/run_cac_c1h_worker.py").read_text(encoding="utf-8"))
    imports = {
        (node.module, alias.name)
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    for name in ("get_libero_dummy_action", "get_libero_env"):
        assert ("experiments.robot.libero.libero_utils", name) in imports
        assert ("experiments.robot.robot_utils", name) not in imports


def load_worker():
    path = ROOT / "scripts/run_cac_c1h_worker.py"
    spec = importlib.util.spec_from_file_location("cac_c1h_worker_for_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = ROOT
    return module


def test_recovery_config_is_explicitly_authorized_and_fully_authenticates() -> None:
    worker = load_worker()
    config = json.loads(
        (ROOT / "configs/cac/c1h_headroom_v03_recovery02.json").read_text(encoding="utf-8")
    )
    assert config["authorization"] == {
        "c1h_authorized": True,
        "c2_authorized": False,
        "terminal_outcomes": True,
        "locked_state_ids_10_49": False,
    }
    assert config["technical_recovery"]["source_run_id"] == (
        "cac-c1h-headroom-v02-recovery01"
    )
    assert config["technical_recovery"]["source_model_queries"] == 1
    assert config["query_accounting"]["technical_control_queries"] == 4
    # Historical Recovery 02 must remain bound to the exact pre-correction code.
    # The S6 requalification intentionally changes that code path, so replaying
    # the old config must now fail closed instead of silently adopting S6.
    with pytest.raises(worker.TechnicalStop, match="frozen code changed"):
        worker.validate_config(config)


def test_runner_contains_the_frozen_pre_episode_safety_controls() -> None:
    source = (ROOT / "scripts/run_cac_c1h_worker.py").read_text(encoding="utf-8")
    for required in (
        "official_helper_max_abs",
        "restore_checkpoint_exact",
        "AggregateMemorySampler",
        "outcomes_opened_only_after_exact_stage_completion",
        '"HF_HUB_OFFLINE": "1"',
        "cache = tracker = anchor_salience = None",
        'cloned["prev_images"]',
        "unpack_official_action_result(np, official_result)",
        '"d62_anchor_and_reuse": True',
        'output_root / "technical_traceback.log"',
    ):
        assert required in source


def test_official_control_contract_isolated_from_in_place_upstream_mutation() -> None:
    worker = load_worker()
    observation = {
        "full_image": np.zeros((4, 4, 3), dtype=np.uint8),
        "wrist_image": np.ones((4, 4, 3), dtype=np.uint8),
        "state": np.arange(8, dtype=np.float32),
    }
    original_state = observation["state"].copy()
    cloned = worker.official_control_observation(np, observation)
    cloned["state"][:] = -1
    assert np.array_equal(observation["state"], original_state)
    assert np.array_equal(cloned["prev_images"][0], observation["full_image"])
    assert np.array_equal(cloned["prev_images"][1], observation["wrist_image"])
    assert not np.shares_memory(cloned["prev_images"][0], observation["full_image"])


def test_official_four_item_return_contract_is_validated_before_use() -> None:
    worker = load_worker()
    cache = object()
    actions, observed_cache, images, metrics = worker.unpack_official_action_result(
        np,
        ([np.zeros(7, dtype=np.float32) for _ in range(8)], cache, ["scene", "wrist"], {}),
    )
    assert actions.shape == (8, 7)
    assert observed_cache is cache
    assert images == ["scene", "wrist"] and metrics == {}
    with pytest.raises(worker.TechnicalStop, match="return contract"):
        worker.unpack_official_action_result(np, np.zeros((8, 7), dtype=np.float32))


def test_recovery_launcher_is_bounded_and_uses_the_exact_recovery_identity() -> None:
    source = (ROOT / "scripts/launch_cac_c1h_recovery02.sh").read_text(encoding="utf-8")
    for required in (
        "set -euo pipefail",
        "c1h_headroom_v03_recovery02.json",
        "cac-c1h-headroom-v03-recovery02",
        "used_mib <= 1024",
        "utilization_percent <= 5",
        "timeout --signal=TERM --kill-after=60 36120",
        'CUDA_VISIBLE_DEVICES="${GPU_ID}"',
        'CAC_PHYSICAL_GPU_ID="${GPU_ID}"',
    ):
        assert required in source
