from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from savr.acr.records import AttemptIdentity, validate_record
from savr.acr.types import ACRConfiguration, ACRContext, ACRPolicy


ROOT = Path(__file__).resolve().parents[2]


def load_script(name: str) -> ModuleType:
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RUNNER = load_script("run_acr_a5")
ANALYZER = load_script("analyze_acr_a5")


def test_frozen_a5_configuration_matches_canonical_candidates() -> None:
    config = json.loads((ROOT / "configs/acr/development_a5.json").read_text())
    candidates = json.loads((ROOT / "configs/acr/candidates.json").read_text())
    RUNNER.validate_frozen_config(config, candidates)
    assert RUNNER.file_sha256(ROOT / "configs/acr/candidates.json") == (
        "8f1503e4f579df9a0a4026b178492566b4ad206830886dfba2418852574567a4"
    )
    assert config["stages"]["stage1"]["attempts_per_candidate"] * 3 == 90
    assert config["resource_caps"]["episode_attempts"] == 300
    assert RUNNER.ATTEMPT_INDEX == 1


@pytest.mark.parametrize("candidate_id", RUNNER.CANDIDATE_IDS)
def test_frozen_run_ids(candidate_id: str) -> None:
    assert RUNNER.run_id_for("stage1", candidate_id) == (
        f"acr-a5-sa-acr-object-stage1-{candidate_id}-v01"
    )
    assert RUNNER.run_id_for("stage2", candidate_id) == (
        f"acr-a5-sa-acr-object-stage2-{candidate_id}-v01"
    )


def stage1_summary() -> dict[str, object]:
    return {
        "terminal_episodes": 30,
        "successes": 30,
        "per_task_successes": {str(task): 3 for task in range(10)},
        "scene_reuse_rate": 0.15,
        "technical_failures": 0,
    }


def test_stage1_gate_is_exact_and_fail_closed() -> None:
    summary = stage1_summary()
    assert ANALYZER.stage1_pass(summary) == (True, [])
    for field, value in (
        ("terminal_episodes", 29),
        ("successes", 29),
        ("scene_reuse_rate", 0.149999),
        ("technical_failures", 1),
    ):
        changed = {**summary, field: value}
        assert ANALYZER.stage1_pass(changed)[0] is False
    changed = {**summary, "per_task_successes": {**summary["per_task_successes"], "4": 2}}
    assert ANALYZER.stage1_pass(changed)[0] is False


def development_candidate() -> dict[str, object]:
    return {
        "terminal_episodes": 100,
        "successes": 95,
        "per_task_successes": {str(task): 9 for task in range(10)},
        "scene_reuse_rate": 0.40,
        "visual_cuda_reduction": 0.10,
        "technical_failures": 0,
    }


def test_development_gate_boundaries() -> None:
    fr = {"successes": 97, "per_task_successes": {str(task): 10 for task in range(10)}}
    candidate = development_candidate()
    assert ANALYZER.development_eligibility(candidate, fr) == (True, [])
    changed = {**candidate, "successes": 94}
    assert ANALYZER.development_eligibility(changed, fr)[0] is False
    changed = {**candidate, "scene_reuse_rate": 0.39999}
    assert ANALYZER.development_eligibility(changed, fr)[0] is False
    changed = {**candidate, "visual_cuda_reduction": 0.09999}
    assert ANALYZER.development_eligibility(changed, fr)[0] is False
    changed = {
        **candidate,
        "per_task_successes": {**candidate["per_task_successes"], "3": 8},
    }
    assert ANALYZER.development_eligibility(changed, fr)[0] is False


def test_selection_uses_one_point_success_pool_then_frozen_ties() -> None:
    eligible = [
        {
            "candidate_id": "acr-t25-h2-b30",
            "success_difference_vs_fr": -1,
            "query_latency_reduction": 0.02,
            "visual_cuda_reduction": 0.11,
        },
        {
            "candidate_id": "acr-t50-h4-b55",
            "success_difference_vs_fr": -2,
            "query_latency_reduction": 0.03,
            "visual_cuda_reduction": 0.20,
        },
        {
            "candidate_id": "acr-t70-h8-b75",
            "success_difference_vs_fr": -3,
            "query_latency_reduction": 0.50,
            "visual_cuda_reduction": 0.50,
        },
    ]
    horizons = {"acr-t25-h2-b30": 2, "acr-t50-h4-b55": 4, "acr-t70-h8-b75": 8}
    assert ANALYZER.select_candidate(eligible, horizons) == "acr-t50-h4-b55"
    assert ANALYZER.select_candidate([], horizons) is None


def test_point_reduction_and_stage2_lock(tmp_path: Path) -> None:
    episodes = [{"timing": {"steady_visual_cuda_ms": [2.0, 4.0]}}]
    assert ANALYZER.point_from_episode_arrays(episodes, "steady_visual_cuda_ms") == 3.0
    assert ANALYZER.reduction(2.0, 4.0) == 0.5
    with pytest.raises(RuntimeError, match="immutable Stage 1 analysis"):
        RUNNER.require_stage2_eligibility(tmp_path, "acr-t25-h2-b30")


def test_a5_query_record_satisfies_frozen_schema() -> None:
    run_id = RUNNER.run_id_for("stage1", "acr-t25-h2-b30")
    identity = AttemptIdentity(run_id, "sa-acr", "libero-object", 0, 0, 0, 1)
    configuration = ACRConfiguration(
        "acr-t25-h2-b30",
        ACRPolicy.SA_ACR,
        scene_threshold=0.25,
        translation_threshold=0.5,
        horizon=2,
        hard_reuse_cap=0.3,
    )
    context = ACRContext(
        episode_id=identity.episode_id,
        attempt_id=identity.value,
        task_id="00",
        instruction_sha256="a" * 64,
        checkpoint_id="checkpoint",
        upstream_revision="upstream",
        configuration_id=configuration.configuration_id,
        controller_version=configuration.controller_version,
        preprocessing_id="preprocess",
        action_head_id="action-head",
        dtype="torch.bfloat16",
        device="cuda:0",
    )
    decision = SimpleNamespace(
        query_index=2,
        refresh=False,
        reasons=(),
        cache_age_before=0,
        reference_query_index=1,
        scene_score=0.1,
        translation_score=0.1,
        gripper_transition_veto=False,
        completed_reuses_before=0,
        completed_queries_before=2,
        patch_scores=(0.1,) * 64,
        translation_direction_reversals=(False, False, False),
    )
    work = SimpleNamespace(
        scene_siglip_calls=0,
        scene_dinov2_calls=0,
        scene_projector_calls=0,
        wrist_siglip_calls=1,
        wrist_dinov2_calls=1,
        wrist_projector_calls=1,
        downstream_calls=1,
        component_wall_ms={"scene.cache-load": 0.1, "camera-block-concat": 0.1},
    )
    result = SimpleNamespace(
        decision=decision,
        work=work,
        device_timing=SimpleNamespace(
            component_device_ms={
                "scene.cache-load": 0.01,
                "wrist.siglip": 1.0,
                "wrist.dinov2": 1.0,
                "wrist.projector": 1.0,
            },
            total_device_ms=10.0,
        ),
        controller_wall_ms=0.2,
        query_wall_ms=12.0,
        scene_image_sha256="b" * 64,
        wrist_image_sha256="c" * 64,
        proprio_sha256="d" * 64,
    )
    record = RUNNER.query_record(
        run_id=run_id,
        identity=identity,
        result=result,
        configuration=configuration,
        context=context,
        raw_state=(0.0,) * 8,
        normalized_state=(0.0,) * 8,
        actions=[[0.0] * 7] * 8,
        visual_token_count=512,
        model_dtype="torch.bfloat16",
        model_device="cuda:0",
        configuration_sha256="e" * 64,
        savr_revision="f" * 40,
        environment_step=10,
        np=__import__("numpy"),
    )
    schema = json.loads((ROOT / "schemas/acr_query.schema.json").read_text())
    validate_record(record, schema)
    assert record["timing"]["scene_visual_cuda_ms"] == 0.0
    assert record["timing"]["total_visual_cuda_ms"] == 3.0
