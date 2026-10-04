from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def worker_module():
    path = ROOT / "scripts/run_pair_p3r_worker.py"
    spec = importlib.util.spec_from_file_location("pair_p3r_worker", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_p3r_freeze_schedule_and_query_accounting_are_exact():
    module = worker_module()
    config = json.loads((ROOT / "configs/pair/p3r_vectorized_v1.json").read_text())
    module.validate_p3r_config(config, input_count=8)
    schedule = module.p3r_schedule(config, 8)
    assert len(schedule) == 24
    assert len({block.block_id for block in schedule}) == 24
    assert {block.profile_id for block in schedule} == {"D59_BAL_PT1", "D62_BAL_PT1"}
    assert {block.horizon for block in schedule} == {2, 4}
    assert sum(2 * (1 + block.horizon) for block in schedule) == 192
    assert config["measurement"]["planned_model_queries"] == 210
