from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from savr.acr.records import AttemptIdentity, encode_float_sequence, validate_record
from savr.acr.signals import audit_sha256


ROOT = Path(__file__).resolve().parents[2]


def load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RUNNER = load("run_acr_a4_fr", "scripts/run_acr_a4_fr.py")
ANALYZER = load("analyze_acr_a4", "scripts/analyze_acr_a4.py")


def test_a4_frozen_scope_and_config():
    config = json.loads((ROOT / "configs/acr/development_fr.json").read_text())
    RUNNER.validate_config(config)
    assert RUNNER.ATTEMPT_CAP == 100
    assert RUNNER.WALL_CAP_SECONDS == 28_800
    assert RUNNER.ARTIFACT_CAP_BYTES == 1024**3
    assert RUNNER.TASK_IDS == RUNNER.INITIAL_STATE_IDS == tuple(range(10))


def test_a4_config_rejects_population_or_retry_mutation():
    config = json.loads((ROOT / "configs/acr/development_fr.json").read_text())
    config["initial_state_ids"] = list(range(9))
    with pytest.raises(ValueError, match="initial_state_ids"):
        RUNNER.validate_config(config)
    config = json.loads((ROOT / "configs/acr/development_fr.json").read_text())
    config["recovery"]["automatic_episode_retry"] = True
    with pytest.raises(ValueError, match="retry"):
        RUNNER.validate_config(config)


def test_a4_query_builder_matches_frozen_schema():
    identity = AttemptIdentity(RUNNER.RUN_ID, RUNNER.POLICY, "libero-object", 0, 0, 0, 0)
    timing = SimpleNamespace(
        component_device_ms={
            "vision_backbone": 2.0,
            "visual_projector": 1.0,
            "language_model": 4.0,
            "action_head": 1.0,
        },
        total_device_ms=8.0,
        wall_ms=9.0,
    )
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    actions = np.zeros((8, 7), dtype=np.float32)
    record = RUNNER.build_query_record(
        identity=identity,
        query_index=0,
        environment_step=10,
        global_query_index=0,
        scene_image=image,
        wrist_image=image,
        raw_state=(0.0,) * 8,
        normalized_state=(0.0,) * 8,
        scene_representation=(0.0,) * 3072,
        patch_scores=None,
        scene_score=None,
        translation_score=None,
        gripper_transition_veto=None,
        direction_reversal=None,
        actions=actions,
        timing=timing,
        model_dtype="torch.bfloat16",
        model_device="cuda:0",
        visual_token_count=512,
        configuration_sha256="0" * 64,
        savr_revision="0" * 40,
        context_sha256="0" * 64,
        audit_sha256=audit_sha256,
    )
    schema = json.loads((ROOT / "schemas/acr_query.schema.json").read_text())
    validate_record(record, schema)
    assert record["camera_work"]["scene_siglip_calls"] == 1
    assert record["timing"]["total_visual_cuda_ms"] == 3.0


def test_a4_companion_trace_round_trips_into_candidate_input(tmp_path):
    identity = AttemptIdentity(RUNNER.RUN_ID, RUNNER.POLICY, "libero-object", 0, 0, 0, 0)
    trace = RUNNER.build_trace_record(
        identity=identity,
        query_index=0,
        scene_representation=(0.0,) * 3072,
        normalized_eef_position=(0.0, 0.0, 0.0),
        actions=np.zeros((8, 7), dtype=np.float32),
        gripper_transition_veto=None,
        direction_reversals=None,
        upstream_component_invocations={
            "vision_backbone": 1,
            "visual_projector": 1,
            "language_model": 1,
            "action_head": 1,
        },
        encode_float_sequence=encode_float_sequence,
        np=np,
    )
    path = tmp_path / identity.query_id(0) / "trace"
    path.mkdir(parents=True)
    (path / "record.json").write_text(json.dumps(trace))
    loaded = ANALYZER.load_trace(tmp_path)
    assert len(loaded) == 1
    assert loaded[0].episode_id == identity.episode_id
    assert loaded[0].scene_representation == (0.0,) * 3072
    assert loaded[0].action_chunk == (0.0,) * 56


def episode(task: int, state: int, success: bool = True, status: str = "completed"):
    return {"task_id": task, "initial_state_id": state, "success": success, "status": status}


def test_a4_feasibility_gate_is_exact():
    records = [episode(task, state) for task in range(10) for state in range(10)]
    assert ANALYZER.feasibility(records)["passed"] is True
    records[0]["success"] = False
    records[1]["success"] = False
    records[2]["success"] = False
    assert ANALYZER.feasibility(records)["passed"] is False
    records = [episode(task, state) for task in range(10) for state in range(10)]
    records[0]["status"] = "failed"
    records[0]["success"] = None
    assert ANALYZER.feasibility(records)["passed"] is False
