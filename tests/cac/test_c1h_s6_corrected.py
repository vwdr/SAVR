from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_s6_protocol_preserves_the_frozen_scientific_design() -> None:
    protocol = (ROOT / "docs/CAC_C1H_S6_CORRECTED_SUBSTRATE_PROTOCOL_V1.md").read_text()
    for required in (
        "240 terminal episodes",
        "existing ambiguity",
        "D62_BAL_PT1_S4C_V1",
        "does not authorize C2",
        "no automatic retry",
    ):
        assert required in protocol


def test_s6_worker_uses_qualified_official_and_corrected_paths() -> None:
    source = (ROOT / "scripts/run_cac_c1h_worker.py").read_text()
    for required in (
        "install_qualified_official_loader",
        'source = ROOT / "third_party/openvla-oft"',
        "OfficialBoundaryCapture",
        "official_hidden_max_abs",
        "official_normalized_action_max_abs",
        "observation_isolated",
        "D62_BAL_PT1_S4C_V1",
    ):
        assert required in source
    tree = ast.parse(source)
    assert tree is not None


def test_s6_config_freezes_population_gate_resources_and_stop_boundary() -> None:
    config = json.loads((ROOT / "configs/cac/c1h_headroom_s6_v01.json").read_text())
    assert config["cache_substrate"]["profile"] == "D62_BAL_PT1_S4C_V1"
    assert config["populations"]["episodes_per_stage"] == 240
    assert config["resource_caps"]["episode_attempts"] == 480
    assert config["resource_caps"]["model_queries"] == 20000
    assert config["resource_caps"]["automatic_retry"] is False
    assert config["advance"] == {
        "next_phase": "C2", "authorized": False, "stop_before_next_phase": True
    }
    assert config["gate_h"]["stage1_proceed"].startswith("dense>=0.75")


def test_s6_launcher_runs_cuda_hidden_preflight_before_gpu_selection() -> None:
    source = (ROOT / "scripts/launch_cac_c1h_s6.sh").read_text()
    assert source.index('CUDA_VISIBLE_DEVICES=""') < source.index("nvidia-smi -i")
    for required in (
        "set -euo pipefail",
        "used_mib <= 1024",
        "utilization_percent <= 5",
        "timeout --signal=TERM --kill-after=60 36120",
        "cac-c1h-headroom-s6-v01",
    ):
        assert required in source
