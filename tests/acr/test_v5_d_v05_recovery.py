from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from savr.acr.v5_d_runtime import semantic_sha256
from savr.acr.v5_d_v04_runtime import load_v04
from savr.acr.v5_d_v05_runtime import load_v05
from savr.acr.v5_d_v05_transition import (
    V05TransitionEligibilityError,
    V05TransitionSampler,
)


ROOT = Path(__file__).resolve().parents[2]


def sample(*, memory: int = 6, utilization: int = 0, index: int = 0, uuid: str = "gpu-0"):
    return {
        "index": index,
        "uuid": uuid,
        "name": "NVIDIA TITAN RTX",
        "driver_version": "test",
        "memory_total_mib": 24576,
        "memory_used_mib": memory,
        "utilization_percent": utilization,
        "recorded_at_utc": "2026-08-11T00:00:00+00:00",
    }


def make_sampler(values):
    queue = [deepcopy(value) for value in values]
    sleeps = []
    records = []
    config = load_v05(ROOT)

    def snapshot(_physical_id):
        return queue.pop(0)

    sampler = V05TransitionSampler(
        run_id=config["run_id"],
        rule=config["transition_revalidation"],
        expected_index=0,
        expected_uuid="gpu-0",
        snapshot=snapshot,
        sleep=sleeps.append,
        write_once=records.append,
    )
    return sampler, queue, sleeps, records


def test_v05_changes_only_transition_control_and_identity() -> None:
    v04 = load_v04(ROOT)
    v05 = load_v05(ROOT)
    changed = {
        "schema_version",
        "status",
        "authorized_at",
        "authorized_scope",
        "protocol",
        "run_id",
        "transition_revalidation",
        "recovery_v05",
        "current_authorization",
        "advance_only_to",
        "semantic_sha256",
    }
    assert {key: value for key, value in v05.items() if key not in changed} == {
        key: value for key, value in v04.items() if key not in changed
    }
    assert v05["run_id"] == "acr-v5d-real-tensor-feasibility-v05"
    assert v05["recovery_v05"]["v04_technical_stop_semantic_sha256"] == (
        "a3515180022df7938b50956851a2ca05b698819da38b387ddc23b54e59769811"
    )
    assert v05["memory"]["peak_reserved_bytes_max"] == 23 * 1024**3


def test_transition_requires_exact_window_and_caches_passing_result() -> None:
    sampler, queue, sleeps, records = make_sampler([sample(), sample(), sample()])
    result = sampler("0")
    assert result == sample()
    assert sleeps == [2.0, 5.0, 5.0]
    assert queue == []
    assert len(records) == 1
    assert records[0]["all_samples_passed"] is True
    assert records[0]["semantic_sha256"] == semantic_sha256(records[0])
    assert sampler("0") == result
    assert sleeps == [2.0, 5.0, 5.0]
    assert len(records) == 1


@pytest.mark.parametrize(
    "values",
    [
        [sample(utilization=6), sample(), sample()],
        [sample(), sample(memory=513), sample()],
        [sample(), sample(), sample(uuid="other")],
    ],
)
def test_transition_fails_closed_without_another_window(values) -> None:
    sampler, queue, sleeps, records = make_sampler(values)
    with pytest.raises(V05TransitionEligibilityError, match="eligibility window failed"):
        sampler("0")
    assert queue == []
    assert sleeps == [2.0, 5.0, 5.0]
    assert len(records) == 1
    assert records[0]["status"] == "technical-stop"
    assert records[0]["all_samples_passed"] is False
    with pytest.raises(V05TransitionEligibilityError, match="eligibility window failed"):
        sampler("0")
    assert sleeps == [2.0, 5.0, 5.0]
    assert len(records) == 1


def test_transition_rejects_physical_index_before_sampling() -> None:
    sampler, queue, sleeps, records = make_sampler([sample(), sample(), sample()])
    with pytest.raises(V05TransitionEligibilityError, match="physical GPU index changed"):
        sampler("1")
    assert len(queue) == 3
    assert sleeps == []
    assert records == []
