from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from savr.brace.b1 import (
    B1ValidationError,
    freeze_transcript,
    validate_reconstruction,
    validate_transcript,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def numeric(*values: float):
    return {"kind": "numeric", "shape": [len(values)], "dtype": "float64", "values": list(values)}


def snapshot(value: float, *, image: str = "a" * 64, queue_index: int = 1):
    return {
        "sim_state": numeric(value, value + 1),
        "observations": {"robot0_eef_pos": numeric(value, 0, 0)},
        "images": {"agentview_image": {"kind": "exact", "sha256": image}},
        "counters": {"timestep": queue_index, "done": False},
        "queue": {"next_action_index": queue_index},
    }


def transcript():
    actions = ([0, 0, 0, 0, 0, 0, -1], [0.1, 0, 0, 0, 0, 0, -1])
    return freeze_transcript(
        metadata={"suite": "libero_spatial", "task_id": 0},
        initial_snapshot=snapshot(0, queue_index=0),
        actions=actions,
        step_snapshots=(snapshot(1), snapshot(2, queue_index=2)),
    )


def test_transcript_is_deterministic_complete_and_tamper_evident():
    first = transcript()
    second = transcript()
    assert first == second
    validate_transcript(first)

    incomplete = copy.deepcopy(first)
    incomplete["steps"].pop()
    with pytest.raises(B1ValidationError, match="incomplete"):
        validate_transcript(incomplete)

    mutated = copy.deepcopy(first)
    mutated["steps"][0]["action"][0] = 0.5
    with pytest.raises(B1ValidationError, match="action hash"):
        validate_transcript(mutated)


def test_reconstruction_accepts_tolerance_and_rejects_modified_prefix():
    reference = snapshot(1.0)
    within = snapshot(1.0 + 1e-10)
    reference["snapshot_sha256"] = "a" * 64
    within["snapshot_sha256"] = "b" * 64
    assert validate_reconstruction(
        reference,
        within,
        restoration_mode="env_step_prefix",
        absolute_tolerance=1e-9,
    ).accepted

    changed = snapshot(1.1)
    changed["snapshot_sha256"] = "c" * 64
    verdict = validate_reconstruction(
        reference,
        changed,
        restoration_mode="env_step_prefix",
        absolute_tolerance=1e-9,
    )
    assert not verdict.accepted
    assert any(item.endswith(":values") for item in verdict.mismatches)


def test_direct_state_restoration_is_rejected_even_if_arrays_match():
    reference = snapshot(1.0)
    verdict = validate_reconstruction(
        reference,
        copy.deepcopy(reference),
        restoration_mode="direct_state_only",
        absolute_tolerance=0,
    )
    assert not verdict.accepted
    assert verdict.mismatches[0] == "restoration_mode:not-env-step-prefix"


def test_exact_image_and_counter_fields_do_not_use_float_tolerance():
    reference = snapshot(1.0)
    changed_image = snapshot(1.0, image="b" * 64)
    changed_counter = snapshot(1.0, queue_index=2)
    for candidate in (changed_image, changed_counter):
        verdict = validate_reconstruction(
            reference,
            candidate,
            restoration_mode="env_step_prefix",
            absolute_tolerance=1,
        )
        assert not verdict.accepted


def test_frozen_configuration_and_runner_preserve_b1_boundary():
    config = json.loads((REPOSITORY_ROOT / "configs/brace/b1_replay_v1.json").read_text())
    supplied = config.pop("semantic_sha256")
    from savr.brace.b1 import semantic_sha256

    assert semantic_sha256(config) == supplied
    assert len(config["scenarios"]) == 3
    assert all(len(item["actions"]) == 12 for item in config["scenarios"])
    assert config["resource_caps"] == {
        "cuda_visible": False,
        "model_queries": 0,
        "policy_outcomes": 0,
        "environment_instances": 30,
        "simulator_steps": 240,
        "wall_seconds": 1800,
        "artifact_bytes": 268435456,
        "downloads_allowed": False,
    }
    runner = (REPOSITORY_ROOT / "scripts/run_brace_b1.py").read_text()
    assert 'CUDA_VISIBLE_DEVICES": ""' in runner
    assert "initialize_model" not in runner
    assert "nvidia-smi" not in runner
