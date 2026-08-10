#!/usr/bin/env python3
"""Dependency-free mechanical verification of the V5 isolated-reuse contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from savr.acr.controller import ACRController
from savr.acr.isolated_controller import (
    ISOLATED_CONTROLLER_VERSION,
    IsolatedACRController,
)
from savr.acr.types import ACRConfiguration, ACRContext, ACRPolicy, SceneDecision


ROOT = Path(__file__).resolve().parents[1]


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scene() -> tuple[float, ...]:
    return (0.0,) * (32 * 32)


def action() -> tuple[float, ...]:
    return (0.0,) * 56


def isolated_configuration(*, reuse_cap: float = 0.4) -> ACRConfiguration:
    return ACRConfiguration(
        configuration_id="ir-sa-acr-verification",
        policy=ACRPolicy.SA_ACR,
        scene_threshold=1.0,
        translation_threshold=1.0,
        horizon=1,
        hard_reuse_cap=reuse_cap,
        controller_version=ISOLATED_CONTROLLER_VERSION,
    )


def context(configuration: ACRConfiguration) -> ACRContext:
    return ACRContext(
        episode_id="verification",
        attempt_id="verification",
        task_id="synthetic",
        instruction_sha256="0" * 64,
        checkpoint_id="none",
        upstream_revision="none",
        configuration_id=configuration.configuration_id,
        controller_version=configuration.controller_version,
        preprocessing_id="synthetic",
        action_head_id="synthetic",
        dtype="float32",
        device="cpu",
        patch_count=1,
    )


def complete(
    controller: ACRController,
    *,
    cache_available: bool,
    cache_age: int,
) -> SceneDecision:
    decision = controller.decide(
        scene_representation=scene(),
        normalized_eef_position=(0.0, 0.0, 0.0),
        cache_available=cache_available,
        cache_age=cache_age,
    )
    controller.observe(
        decision=decision,
        scene_representation=scene(),
        normalized_eef_position=(0.0, 0.0, 0.0),
        action_chunk=action(),
    )
    return decision


def run_trace(controller: ACRController, count: int) -> dict[str, Any]:
    cache_available = False
    cache_age = 0
    streak = maximum_streak = reuses = 0
    maximum_prefix_fraction = 0.0
    decisions: list[bool] = []
    for _ in range(count):
        decision = complete(
            controller,
            cache_available=cache_available,
            cache_age=cache_age,
        )
        reused = not decision.refresh
        decisions.append(reused)
        streak = streak + 1 if reused else 0
        maximum_streak = max(maximum_streak, streak)
        reuses += int(reused)
        maximum_prefix_fraction = max(maximum_prefix_fraction, reuses / controller.query_index)
        cache_available = True
        cache_age = 0 if decision.refresh else cache_age + 1
    return {
        "decisions_reuse": decisions,
        "maximum_prefix_reuse_fraction": maximum_prefix_fraction,
        "maximum_reuse_streak": maximum_streak,
        "queries": count,
        "reuses": reuses,
    }


def verify() -> dict[str, Any]:
    isolated_config = isolated_configuration()
    isolated = IsolatedACRController(isolated_config)
    isolated.reset(context(isolated_config))
    isolated_trace = run_trace(isolated, 128)
    assert isolated_trace["maximum_reuse_streak"] == 1
    assert isolated_trace["maximum_prefix_reuse_fraction"] <= 0.4

    mismatch = IsolatedACRController(isolated_configuration(reuse_cap=0.75))
    mismatch.reset(context(mismatch.configuration))
    complete(mismatch, cache_available=False, cache_age=0)
    complete(mismatch, cache_available=True, cache_age=0)
    assert not complete(mismatch, cache_available=True, cache_age=0).refresh
    mismatch_decision = mismatch.decide(
        scene_representation=scene(),
        normalized_eef_position=(0.0, 0.0, 0.0),
        cache_available=True,
        cache_age=0,
    )
    assert mismatch_decision.refresh
    assert "post-reuse-refresh" in mismatch_decision.reasons
    assert "isolation-state-mismatch" in mismatch_decision.reasons
    assert mismatch.refresh_required_after_reuse

    legacy_config = ACRConfiguration(
        configuration_id="legacy-verification",
        policy=ACRPolicy.SA_ACR,
        scene_threshold=1.0,
        translation_threshold=1.0,
        horizon=2,
        hard_reuse_cap=0.75,
    )
    legacy = ACRController(legacy_config)
    legacy.reset(context(legacy_config))
    legacy_trace = run_trace(legacy, 12)
    assert legacy_trace["maximum_reuse_streak"] == 2

    record: dict[str, Any] = {
        "schema_version": "acr.v5-isolated-reuse-cpu-verification.v1",
        "verified": True,
        "isolated_trace": {
            key: value for key, value in isolated_trace.items() if key != "decisions_reuse"
        },
        "legacy_trace": {
            key: value for key, value in legacy_trace.items() if key != "decisions_reuse"
        },
        "mismatch_reasons": list(mismatch_decision.reasons),
        "source_sha256": {
            "freeze": file_sha256(ROOT / "configs/acr/v5_isolated_reuse_freeze.json"),
            "isolated_controller": file_sha256(ROOT / "src/savr/acr/isolated_controller.py"),
            "types": file_sha256(ROOT / "src/savr/acr/types.py"),
        },
        "resources": {
            "downloads": 0,
            "gpu_count": 0,
            "model_queries": 0,
            "new_outcomes": 0,
            "simulator_episodes": 0,
        },
    }
    record["semantic_sha256"] = hashlib.sha256(canonical_bytes(record)).hexdigest()
    return record


def main() -> int:
    print(json.dumps(verify(), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
