from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType


ROOT = Path(__file__).resolve().parents[2]


def load_script(name: str) -> ModuleType:
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RUNNER = load_script("run_acr_v3_d")
ANALYZER = load_script("analyze_acr_v3_d")


def test_frozen_config_and_schedule_are_exact() -> None:
    config = json.loads((ROOT / "configs/acr/v3_d_development.json").read_text())
    RUNNER.validate_frozen_config(config)
    schedule = RUNNER.schedule()
    assert len(schedule) == len(set(schedule)) == 140
    assert (
        sum(
            RUNNER.policy_order(task, state)[0] == RUNNER.POLICIES[0]
            for task in range(10)
            for state in range(3, 10)
        )
        == 35
    )
    assert (
        sum(
            RUNNER.policy_order(task, state)[0] == RUNNER.POLICIES[1]
            for task in range(10)
            for state in range(3, 10)
        )
        == 35
    )
    assert schedule[:4] == (
        (0, 3, "batched-fr"),
        (0, 3, "sa-bdp-acr-t25-h2-b30-v01"),
        (0, 4, "sa-bdp-acr-t25-h2-b30-v01"),
        (0, 4, "batched-fr"),
    )


def passing_metrics() -> dict[str, object]:
    return {
        "policies": {
            RUNNER.POLICIES[0]: {
                "terminal_episodes": 70,
                "technical_failures": 0,
                "successes": 68,
                "per_task_successes": {str(task): 7 for task in range(10)},
            },
            RUNNER.POLICIES[1]: {
                "terminal_episodes": 70,
                "technical_failures": 0,
                "successes": 66,
                "per_task_successes": {str(task): 6 for task in range(10)},
                "scene_reuse_rate": 0.2,
            },
        },
        "comparisons": {
            "v3_visual_cuda_reduction_vs_bfr": 0.1,
            "v3_query_wall_ratio_vs_sequential_fr": 0.98,
            "v3_query_wall_ratio_vs_bfr": 1.0,
        },
        "all_invariants_pass": True,
    }


def test_gate_boundaries_pass_exactly() -> None:
    assert ANALYZER.evaluate_gate(passing_metrics()) == (True, [])


def test_every_gate_fails_closed() -> None:
    mutators = (
        lambda value: value["policies"][RUNNER.POLICIES[0]].update(terminal_episodes=69),
        lambda value: value["policies"][RUNNER.POLICIES[1]].update(technical_failures=1),
        lambda value: value["policies"][RUNNER.POLICIES[1]].update(successes=65),
        lambda value: value["policies"][RUNNER.POLICIES[1]]["per_task_successes"].update({"3": 5}),
        lambda value: value["policies"][RUNNER.POLICIES[1]].update(scene_reuse_rate=0.199999),
        lambda value: value["comparisons"].update(v3_visual_cuda_reduction_vs_bfr=0.099999),
        lambda value: value["comparisons"].update(v3_query_wall_ratio_vs_sequential_fr=0.980001),
        lambda value: value["comparisons"].update(v3_query_wall_ratio_vs_bfr=1.000001),
        lambda value: value.update(all_invariants_pass=False),
    )
    import copy

    for mutate in mutators:
        value = copy.deepcopy(passing_metrics())
        mutate(value)
        assert ANALYZER.evaluate_gate(value)[0] is False


def test_query_count_mapping_is_logically_exact() -> None:
    base = {
        "camera_work": {
            "logical_scene_backbone_calls": 1,
            "logical_scene_projector_calls": 1,
            "logical_wrist_backbone_calls": 1,
            "logical_wrist_projector_calls": 1,
            "downstream_calls": 1,
        },
        "decision": {"scene_refresh": True},
    }
    counts = RUNNER.query_counts(base)
    assert counts["scene_refreshes"] == counts["scene_siglip_calls"] == 1
    assert counts["scene_reuses"] == 0
    base["decision"]["scene_refresh"] = False
    base["camera_work"]["logical_scene_backbone_calls"] = 0
    base["camera_work"]["logical_scene_projector_calls"] = 0
    counts = RUNNER.query_counts(base)
    assert counts["scene_reuses"] == 1
    assert counts["scene_siglip_calls"] == counts["scene_projector_calls"] == 0


def test_visual_cuda_uses_only_frozen_visual_components() -> None:
    from types import SimpleNamespace

    result = SimpleNamespace(
        device_timing=SimpleNamespace(
            component_device_ms={
                "batched.siglip": 1.0,
                "batched.dinov2": 2.0,
                "batched.projector": 3.0,
                "downstream": 100.0,
            }
        )
    )
    assert RUNNER.visual_cuda_ms(result) == 6.0
