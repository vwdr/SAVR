from __future__ import annotations

from dataclasses import replace

import pytest

from savr.acr.cache import SceneTokenCache
from savr.acr.isolated_controller import ISOLATED_CONTROLLER_VERSION, IsolatedACRController
from savr.acr.isolated_execution_adapter import (
    ISOLATED_EXECUTION_ADAPTER_VERSION,
    EpisodeMethodPatch,
    IsolatedReuseExecutionAdapter,
    RefreshExecutionResult,
)
from savr.acr.reuse_executor import (
    STATIC_REUSE_EXECUTOR_VERSION,
    ExecutorLifecycle,
    ReuseCompatibilityKey,
    ReuseExecutionInputs,
    ReuseExecutorFailure,
    StaticBufferReuseExecutor,
)
from savr.acr.types import ACRConfiguration, ACRContext, ACRPolicy


class Tensor:
    def __init__(self, values, shape, *, dtype="float32", device="cpu"):
        self.values = [float(value) for value in values]
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = device

    def detach(self):
        return self

    def clone(self):
        return Tensor(self.values, self.shape, dtype=self.dtype, device=self.device)


class Ops:
    @staticmethod
    def allocate(shape, *, dtype, device):
        size = 1
        for value in shape:
            size *= value
        return Tensor([0.0] * size, shape, dtype=dtype, device=device)

    @staticmethod
    def copy_(destination, source):
        if destination.shape != source.shape:
            raise ValueError("copy shape mismatch")
        destination.values[:] = source.values

    @staticmethod
    def cat_into(destination, values, *, dim):
        assert dim == 1
        destination.values[:] = values[0].values + values[1].values


class Cores:
    def __init__(self):
        self.wrist_calls = 0
        self.downstream_calls = 0
        self.fail_downstream = False

    def wrist(self, pixels, output):
        self.wrist_calls += 1
        output.values[:] = [sum(pixels.values[:12]), sum(pixels.values[12:])]

    def downstream(self, combined, embeddings, mask, proprioception, output):
        self.downstream_calls += 1
        if self.fail_downstream:
            raise ArithmeticError("injected downstream failure")
        total = (
            sum(combined.values)
            + sum(embeddings.values)
            + sum(mask.values)
            + sum(proprioception.values)
        )
        output.values[:] = [value for _ in range(8) for value in (total, total, total, 0, 0, 0, 0)]


def configuration():
    return ACRConfiguration(
        configuration_id="v5-a100-b40",
        policy=ACRPolicy.SA_ACR,
        scene_threshold=0.30046895424836606,
        translation_threshold=0.685919037527938,
        horizon=1,
        hard_reuse_cap=0.40,
        controller_version=ISOLATED_CONTROLLER_VERSION,
    )


def context():
    return ACRContext(
        episode_id="episode",
        attempt_id="attempt",
        task_id="task",
        instruction_sha256="0" * 64,
        checkpoint_id="checkpoint",
        upstream_revision="revision",
        configuration_id="v5-a100-b40",
        controller_version=ISOLATED_CONTROLLER_VERSION,
        preprocessing_id="preprocessing",
        action_head_id="head",
        dtype="float32",
        device="cpu",
        patch_count=2,
    )


def key():
    return ReuseCompatibilityKey(
        checkpoint_id="checkpoint",
        upstream_revision="revision",
        configuration_id="v5-a100-b40",
        controller_version=ISOLATED_CONTROLLER_VERSION,
        executor_version=STATIC_REUSE_EXECUTOR_VERSION,
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
        proprioception_shape=(1, 8),
        action_shape=(1, 56),
        model_training_state=False,
        use_film=False,
        use_diffusion=False,
    )


def reuse_inputs(compatibility_key=None, *, offset=0.0):
    return ReuseExecutionInputs(
        compatibility_key=compatibility_key or key(),
        wrist_pixels=Tensor([offset + index for index in range(24)], (1, 6, 2, 2)),
        cached_scene_tokens=Tensor((999, 999), (1, 2, 1)),
        prompt_input=Tensor((1, 2), (1, 2)),
        prompt_embeddings=Tensor((3 + offset, 4 + offset), (1, 2, 1)),
        attention_mask=Tensor((1, 1), (1, 2)),
        proprioception=Tensor((0,) * 8, (1, 8)),
    )


def make_adapter():
    cores = Cores()
    executor = StaticBufferReuseExecutor(
        tensor_ops=Ops(),
        wrist_visual_core=cores.wrist,
        downstream_action_core=cores.downstream,
    )
    adapter = IsolatedReuseExecutionAdapter(
        controller=IsolatedACRController(configuration()),
        cache=SceneTokenCache(),
        executor=executor,
        reuse_value_builder=lambda result: tuple(result.normalized_actions.values),
        reuse_action_getter=lambda result: tuple(result.normalized_actions.values),
    )
    return adapter, cores


SCENE = [[0.0, 0.0], [0.0, 0.0]]
STATE = (0.0,) * 8
Q01 = (-1.0,) * 8
Q99 = (1.0,) * 8
ACTION = (0.0,) * 56


class Refresh:
    def __init__(self):
        self.calls = []

    def __call__(self, decision):
        self.calls.append(decision)
        return RefreshExecutionResult(
            value=("refresh", decision.query_index),
            scene_tokens=Tensor((10 + decision.query_index, 20 + decision.query_index), (1, 2, 1)),
            action_chunk=ACTION,
        )


def run(adapter, refresh, *, current_inputs=None):
    return adapter.run_query(
        scene_image=SCENE,
        state=STATE,
        state_q01=Q01,
        state_q99=Q99,
        reuse_inputs=current_inputs or reuse_inputs(),
        refresh_query=refresh,
    )


def test_exact_selected_integration_identity_and_isolated_trace():
    adapter, cores = make_adapter()
    refresh = Refresh()
    assert adapter.version == ISOLATED_EXECUTION_ADAPTER_VERSION
    with adapter.episode(context(), key()):
        results = [
            run(adapter, refresh, current_inputs=reuse_inputs(offset=index)) for index in range(9)
        ]
        assert [result.decision.refresh for result in results] == [
            True,
            True,
            False,
            True,
            False,
            True,
            True,
            False,
            True,
        ]
        reuse_results = [result for result in results if not result.decision.refresh]
        assert all(result.cache_event == "reuse" for result in reuse_results)
        assert all(result.reuse_execution is not None for result in reuse_results)
        assert cores.wrist_calls == cores.downstream_calls == 3
        assert adapter.executor.snapshot().work.scene_core_calls == 0
        assert (
            max(
                sum(not item.decision.refresh for item in results[: index + 1]) / (index + 1)
                for index in range(len(results))
            )
            <= 0.40
        )
        assert "post-reuse-refresh" in results[3].decision.reasons
    assert adapter.executor.lifecycle is ExecutorLifecycle.UNPREPARED
    assert adapter.cache.entry is None


def test_prelaunch_key_mismatch_forces_refresh_without_observing_reuse():
    adapter, cores = make_adapter()
    refresh = Refresh()
    mismatched = replace(key(), instruction_sha256="1" * 64)
    with adapter.episode(context(), key()):
        run(adapter, refresh)
        run(adapter, refresh)
        result = run(adapter, refresh, current_inputs=reuse_inputs(mismatched))
        assert result.decision.refresh
        assert "executor-unavailable" in result.decision.reasons
        assert result.cache_event == "forced-refresh"
        assert adapter.controller.snapshot().completed_reuses == 0
        assert adapter.executor.snapshot().work.prelaunch_rejections == 1
        assert cores.wrist_calls == cores.downstream_calls == 0


def test_missing_or_incompatible_cache_forces_refresh_before_executor():
    adapter, cores = make_adapter()
    refresh = Refresh()
    with adapter.episode(context(), key()):
        run(adapter, refresh)
        run(adapter, refresh)
        adapter.cache.invalidate()
        result = run(adapter, refresh)
        assert result.decision.refresh
        assert "cache" in result.decision.reasons
        assert cores.wrist_calls == cores.downstream_calls == 0


def test_postlaunch_failure_invalidates_cache_and_does_not_observe_or_retry():
    adapter, cores = make_adapter()
    refresh = Refresh()
    with adapter.episode(context(), key()):
        run(adapter, refresh)
        run(adapter, refresh)
        before = adapter.controller.snapshot()
        cores.fail_downstream = True
        with pytest.raises(ReuseExecutorFailure, match="injected downstream failure"):
            run(adapter, refresh)
        after = adapter.controller.snapshot()
        assert after.query_index == before.query_index
        assert after.completed_queries == before.completed_queries
        assert adapter.cache.entry is None
        assert adapter.executor.lifecycle is ExecutorLifecycle.INVALIDATED
        assert len(refresh.calls) == 2
        assert cores.wrist_calls == cores.downstream_calls == 1
        assert adapter.last_failure.reason == "executor-failure"
        assert adapter.last_failure.controller_observed is False
        assert adapter.last_failure.retry_attempted is False


def test_episode_method_restoration_is_exception_safe():
    class Model:
        def boundary(self):
            return "original"

    model = Model()
    adapter, cores = make_adapter()
    refresh = Refresh()
    assert "boundary" not in vars(model)
    with pytest.raises(ReuseExecutorFailure):
        with adapter.episode(
            context(),
            key(),
            patches=(EpisodeMethodPatch(model, "boundary", lambda: "patched"),),
        ):
            assert model.boundary() == "patched"
            run(adapter, refresh)
            run(adapter, refresh)
            cores.fail_downstream = True
            run(adapter, refresh)
    assert "boundary" not in vars(model)
    assert model.boundary() == "original"
    assert adapter.executor.lifecycle is ExecutorLifecycle.UNPREPARED


def test_unselected_controller_is_rejected():
    wrong = replace(configuration(), hard_reuse_cap=0.41)
    executor = StaticBufferReuseExecutor(
        tensor_ops=Ops(),
        wrist_visual_core=Cores().wrist,
        downstream_action_core=Cores().downstream,
    )
    with pytest.raises(ValueError, match="exact V5-B selected controller"):
        IsolatedReuseExecutionAdapter(
            controller=IsolatedACRController(wrong),
            cache=SceneTokenCache(),
            executor=executor,
        )
