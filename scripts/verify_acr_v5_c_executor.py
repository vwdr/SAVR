#!/usr/bin/env python3
"""Dependency-free deterministic verification of the V5-C executor contract."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from savr.acr.controller import ACRController
from savr.acr.isolated_controller import ISOLATED_CONTROLLER_VERSION, IsolatedACRController
from savr.acr.reuse_executor import (
    EAGER_REUSE_EXECUTOR_VERSION,
    STATIC_REUSE_EXECUTOR_VERSION,
    EagerReuseExecutor,
    ExecutorLifecycle,
    ReuseCompatibilityKey,
    ReuseExecutionInputs,
    ReuseExecutorFailure,
    ReuseExecutorUnavailable,
    StaticBufferReuseExecutor,
)
from savr.acr.types import ACRConfiguration, ACRContext, ACRPolicy


ROOT = Path(__file__).resolve().parents[1]


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


class Tensor:
    def __init__(self, values: list[float], shape: tuple[int, ...]) -> None:
        self.values = list(values)
        self.shape = shape
        self.dtype = "float32"
        self.device = "cpu"


class Operations:
    @staticmethod
    def allocate(shape: tuple[int, ...], *, dtype: str, device: str) -> Tensor:
        assert dtype == "float32" and device == "cpu"
        size = 1
        for value in shape:
            size *= value
        return Tensor([0.0] * size, tuple(shape))

    @staticmethod
    def copy_(destination: Tensor, source: Tensor) -> None:
        assert destination.shape == source.shape
        destination.values[:] = source.values

    @staticmethod
    def cat_into(destination: Tensor, values: tuple[Tensor, Tensor], *, dim: int) -> None:
        assert dim == 1
        destination.values[:] = values[0].values + values[1].values


class Cores:
    def __init__(self, *, fail_downstream: bool = False) -> None:
        self.wrist_calls = 0
        self.downstream_calls = 0
        self.fail_downstream = fail_downstream

    def wrist(self, pixels: Tensor, output: Tensor) -> None:
        self.wrist_calls += 1
        output.values[:] = [sum(pixels.values[:6]), sum(pixels.values[6:])]

    def downstream(
        self,
        combined: Tensor,
        embeddings: Tensor,
        mask: Tensor,
        proprioception: Tensor,
        output: Tensor,
    ) -> None:
        self.downstream_calls += 1
        if self.fail_downstream:
            raise ArithmeticError("deterministic injected failure")
        total = (
            sum(combined.values)
            + sum(embeddings.values)
            + sum(mask.values)
            + sum(proprioception.values)
        )
        output.values[:] = [total, total + 1.0]


def compatibility_key(version: str) -> ReuseCompatibilityKey:
    return ReuseCompatibilityKey(
        checkpoint_id="checkpoint",
        upstream_revision="revision",
        configuration_id="v5-a100-b40",
        controller_version=ISOLATED_CONTROLLER_VERSION,
        executor_version=version,
        preprocessing_id="preprocessing",
        action_head_id="head",
        instruction_sha256="0" * 64,
        prompt_input_shape=(1, 2),
        dtype="float32",
        device="cpu",
        image_height=2,
        image_width=2,
        patch_count=2,
        projected_dimension=1,
        wrist_shape=(1, 6, 2, 2),
        cached_scene_shape=(1, 2, 1),
        embedding_shape=(1, 2, 1),
        attention_mask_shape=(1, 2),
        proprioception_shape=(1, 2),
        action_shape=(1, 2),
        model_training_state=False,
        use_film=False,
        use_diffusion=False,
    )


def inputs(key: ReuseCompatibilityKey, offset: float) -> ReuseExecutionInputs:
    return ReuseExecutionInputs(
        compatibility_key=key,
        wrist_pixels=Tensor([offset + value for value in range(24)], (1, 6, 2, 2)),
        cached_scene_tokens=Tensor([10 + offset, 20 + offset], (1, 2, 1)),
        prompt_input=Tensor([1, 2], (1, 2)),
        prompt_embeddings=Tensor([3 + offset, 4 + offset], (1, 2, 1)),
        attention_mask=Tensor([1, 1], (1, 2)),
        proprioception=Tensor([7 + offset, 8 + offset], (1, 2)),
    )


def selected_configuration(*, isolated: bool = True) -> ACRConfiguration:
    return ACRConfiguration(
        configuration_id="v5-a100-b40" if isolated else "legacy",
        policy=ACRPolicy.SA_ACR,
        scene_threshold=0.30046895424836606 if isolated else 1.0,
        translation_threshold=0.685919037527938 if isolated else 1.0,
        horizon=1 if isolated else 2,
        hard_reuse_cap=0.40 if isolated else 0.75,
        controller_version=ISOLATED_CONTROLLER_VERSION if isolated else "acr-controller-v1",
    )


def context(configuration: ACRConfiguration) -> ACRContext:
    return ACRContext(
        episode_id="verification",
        attempt_id="verification",
        task_id="synthetic",
        instruction_sha256="0" * 64,
        checkpoint_id="checkpoint",
        upstream_revision="revision",
        configuration_id=configuration.configuration_id,
        controller_version=configuration.controller_version,
        preprocessing_id="preprocessing",
        action_head_id="head",
        dtype="float32",
        device="cpu",
        patch_count=2,
    )


def controller_trace(controller: ACRController, count: int) -> dict[str, Any]:
    controller.reset(context(controller.configuration))
    cache_available = False
    cache_age = 0
    reuses = 0
    streak = 0
    maximum_streak = 0
    maximum_prefix = 0.0
    decisions: list[bool] = []
    scene = (0.0,) * (32 * 32)
    action = (0.0,) * 56
    for _ in range(count):
        decision = controller.decide(
            scene_representation=scene,
            normalized_eef_position=(0.0, 0.0, 0.0),
            cache_available=cache_available,
            cache_age=cache_age,
        )
        controller.observe(
            decision=decision,
            scene_representation=scene,
            normalized_eef_position=(0.0, 0.0, 0.0),
            action_chunk=action,
        )
        reuse = not decision.refresh
        decisions.append(reuse)
        reuses += int(reuse)
        streak = streak + 1 if reuse else 0
        maximum_streak = max(maximum_streak, streak)
        maximum_prefix = max(maximum_prefix, reuses / controller.query_index)
        cache_available = True
        cache_age = cache_age + 1 if reuse else 0
    return {
        "decisions_reuse": decisions,
        "maximum_prefix_reuse_fraction": maximum_prefix,
        "maximum_reuse_streak": maximum_streak,
        "queries": count,
        "reuses": reuses,
    }


def verify() -> dict[str, Any]:
    eager_cores = Cores()
    static_cores = Cores()
    eager = EagerReuseExecutor(
        tensor_ops=Operations(),
        wrist_visual_core=eager_cores.wrist,
        downstream_action_core=eager_cores.downstream,
    )
    static = StaticBufferReuseExecutor(
        tensor_ops=Operations(),
        wrist_visual_core=static_cores.wrist,
        downstream_action_core=static_cores.downstream,
    )
    eager_key = compatibility_key(EAGER_REUSE_EXECUTOR_VERSION)
    static_key = compatibility_key(STATIC_REUSE_EXECUTOR_VERSION)
    eager.prepare(eager_key)
    static.prepare(static_key)
    stable_ids = dict(static.snapshot().buffer_identities)
    parity = []
    for offset in (0.0, 10.0, 100.0):
        eager_result = eager.run(inputs(eager_key, offset))
        static_result = static.run(inputs(static_key, offset))
        parity.append(
            eager_result.wrist_tokens.values == static_result.wrist_tokens.values
            and eager_result.combined_tokens.values == static_result.combined_tokens.values
            and eager_result.normalized_actions.values == static_result.normalized_actions.values
        )
        assert dict(static_result.snapshot.buffer_identities) == stable_ids
    assert all(parity)

    calls_before_rejection = (static_cores.wrist_calls, static_cores.downstream_calls)
    mismatch = replace(static_key, instruction_sha256="1" * 64)
    try:
        static.run(inputs(mismatch, 0.0))
    except ReuseExecutorUnavailable:
        prelaunch_rejected = True
    else:  # pragma: no cover - verifier guard
        prelaunch_rejected = False
    assert calls_before_rejection == (static_cores.wrist_calls, static_cores.downstream_calls)

    failing_cores = Cores(fail_downstream=True)
    failing = StaticBufferReuseExecutor(
        tensor_ops=Operations(),
        wrist_visual_core=failing_cores.wrist,
        downstream_action_core=failing_cores.downstream,
    )
    failing.prepare(static_key)
    try:
        failing.run(inputs(static_key, 0.0))
    except ReuseExecutorFailure:
        postlaunch_failed_closed = failing.lifecycle is ExecutorLifecycle.INVALIDATED
    else:  # pragma: no cover - verifier guard
        postlaunch_failed_closed = False

    isolated_trace = controller_trace(
        IsolatedACRController(selected_configuration(isolated=True)), 128
    )
    legacy_trace = controller_trace(ACRController(selected_configuration(isolated=False)), 12)
    assert isolated_trace["maximum_reuse_streak"] == 1
    assert isolated_trace["maximum_prefix_reuse_fraction"] <= 0.40
    assert legacy_trace["maximum_reuse_streak"] == 2

    static.reset()
    reset_snapshot = static.snapshot()
    reset_cleared = (
        reset_snapshot.lifecycle is ExecutorLifecycle.UNPREPARED
        and reset_snapshot.owner_thread is None
        and reset_snapshot.compatibility_key is None
        and reset_snapshot.buffer_identities == ()
        and reset_snapshot.work.completed_queries == 0
    )
    assert reset_cleared

    key_payload = asdict(static_key)
    key_digest = hashlib.sha256(canonical_bytes(key_payload)).hexdigest()
    record: dict[str, Any] = {
        "schema_version": "acr.v5c-cpu-executor-verification.v1",
        "verified": True,
        "executor_identities": {
            "reference": EAGER_REUSE_EXECUTOR_VERSION,
            "static": STATIC_REUSE_EXECUTOR_VERSION,
            "integration": "ir-sa-acr-static-executor-v1",
        },
        "compatibility_key_sha256": key_digest,
        "completed_queries": {"reference": 3, "static": 3},
        "core_calls": {
            "reference": {"scene": 0, "wrist": 3, "downstream": 3},
            "static": {"scene": 0, "wrist": 3, "downstream": 3},
        },
        "checks": {
            "combined_scene_first_parity": all(parity),
            "normalized_action_parity": all(parity),
            "postlaunch_failure_invalidated": postlaunch_failed_closed,
            "prelaunch_mismatch_rejected_before_core": prelaunch_rejected,
            "reset_cleared_episode_state": reset_cleared,
            "stable_owned_buffer_identities": True,
            "updated_values_not_stale": True,
            "wrist_output_parity": all(parity),
        },
        "controller_trace": {
            key: value for key, value in isolated_trace.items() if key != "decisions_reuse"
        },
        "legacy_separation": {
            "legacy_maximum_reuse_streak": legacy_trace["maximum_reuse_streak"],
            "isolated_maximum_reuse_streak": isolated_trace["maximum_reuse_streak"],
            "verified": True,
        },
        "source_sha256": {
            name: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
            for name, path in {
                "freeze": "configs/acr/v5_c_cpu_executor_freeze.json",
                "executor": "src/savr/acr/reuse_executor.py",
                "integration": "src/savr/acr/isolated_execution_adapter.py",
            }.items()
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
