import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_analyzer():
    spec = importlib.util.spec_from_file_location("analyze_cac_c7", ROOT / "scripts/analyze_cac_c7.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_freezer():
    spec = importlib.util.spec_from_file_location("freeze_cac_c0", ROOT / "scripts/freeze_cac_c0.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_c0_parameter_and_storage_ledgers_are_exact():
    config = json.loads((ROOT / "configs/cac/c0_freeze_v1.json").read_text())
    full = config["full_adapter"]
    assert sum(full["parameter_ledger"].values()) == full["parameter_count"] == 5_070_599
    pooled = config["pooled_candidates"]
    assert sum(pooled["summary_definition"].values()) == pooled["summary_input_dimensions"] == 3_521
    assert (3_521 + 1) * 512 + (512 + 1) * 7 == pooled["mlp"]["parameter_count"]
    assert (3_521 + 1) * 7 == pooled["ridge"]["parameter_count"]
    record = config["representation"]["aligned_record_bytes"] + config["representation"]["sidecar_bytes_per_record_cap"]
    assert 4_200 * record == config["resource_caps"]["c2_bytes"]
    assert (13_200 + 2_880) * record == config["resource_caps"]["c3_and_c7_feature_bytes"]


def test_c0_call_ledgers_are_exact():
    config = json.loads((ROOT / "configs/cac/c0_freeze_v1.json").read_text())
    schedules = config["contract_schedules"]
    weighted = 40 * sum(h + 2 for h in (1, 2, 4))
    assert schedules["c2"]["model_calls"] == 30 * weighted + 240 * 2 + 3 * weighted
    assert schedules["c3"]["model_calls"] == (50 + 20 + 40) * weighted
    assert schedules["c7_locked"]["model_calls"] == 24 * weighted


def test_fixed_analyzer_rejects_incomplete_population():
    analyzer = load_analyzer()
    rows = [{"suite": "libero_spatial", "task_id": "task", "initial_state_id": 10, "policy": "dense", "success": True}]
    try:
        analyzer.validate_episode_population(rows)
    except RuntimeError:
        pass
    else:
        raise AssertionError("incomplete C7 population was accepted")


def test_holm_is_step_down_not_independent_thresholding():
    analyzer = load_analyzer()
    assert analyzer.holm_pass([0.01, 0.03, 0.20]) == [True, False, False]


def test_transition_shortfall_is_reported_only_when_unavailable():
    freezer = load_freezer()
    unavailable = ("libero_goal", "no_transition_task")
    available = ("libero_spatial", "transition_task")
    shortfalls = freezer.validate_transition_minimum(
        [unavailable, available],
        {unavailable: 0, available: 8},
        {unavailable: {"total": 0}, available: {"total": 20}},
    )
    assert len(shortfalls) == 1
    assert shortfalls[0]["task_id"] == "no_transition_task"
    assert shortfalls[0]["eligible_transition_contracts"] == 0


def test_transition_selector_still_fails_when_examples_exist():
    freezer = load_freezer()
    task = ("libero_spatial", "selector_bug_task")
    try:
        freezer.validate_transition_minimum([task], {task: 7}, {task: {"total": 8}})
    except RuntimeError:
        pass
    else:
        raise AssertionError("selector failure was incorrectly treated as data unavailability")
