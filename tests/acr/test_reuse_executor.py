from __future__ import annotations

import threading
from dataclasses import replace

import pytest

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


class Tensor:
    def __init__(self, values, shape, *, dtype="float32", device="cpu"):
        self.values = [float(value) for value in values]
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = device
        if self.size != len(self.values):
            raise ValueError("data/shape mismatch")

    @property
    def size(self):
        result = 1
        for value in self.shape:
            result *= value
        return result


class Ops:
    def __init__(self):
        self.allocations = 0

    def allocate(self, shape, *, dtype, device):
        self.allocations += 1
        size = 1
        for value in shape:
            size *= value
        return Tensor([0.0] * size, shape, dtype=dtype, device=device)

    @staticmethod
    def copy_(destination, source):
        if (
            destination.shape != source.shape
            or destination.dtype != source.dtype
            or destination.device != source.device
        ):
            raise ValueError("copy metadata mismatch")
        destination.values[:] = source.values

    @staticmethod
    def cat_into(destination, values, *, dim):
        if dim != 1 or len(values) != 2:
            raise ValueError("only scene-first token concatenation is supported")
        scene, wrist = values
        expected = (1, scene.shape[1] + wrist.shape[1], scene.shape[2])
        if destination.shape != expected:
            raise ValueError("concatenation shape mismatch")
        destination.values[:] = scene.values + wrist.values


class Cores:
    def __init__(self):
        self.scene_calls = 0
        self.wrist_calls = 0
        self.downstream_calls = 0
        self.fail_wrist = False
        self.fail_downstream = False
        self.on_wrist = None

    def wrist(self, pixels, output):
        self.wrist_calls += 1
        if self.on_wrist is not None:
            self.on_wrist()
        if self.fail_wrist:
            raise ArithmeticError("wrist failure")
        output.values[:] = [sum(pixels.values[:6]), sum(pixels.values[6:])]

    def downstream(self, combined, embeddings, mask, proprioception, output):
        self.downstream_calls += 1
        if self.fail_downstream:
            raise ArithmeticError("downstream failure")
        proprio = 0.0 if proprioception is None else sum(proprioception.values)
        current = sum(combined.values) + sum(embeddings.values) + sum(mask.values) + proprio
        output.values[:] = [current, current + 1.0]


def key(version=STATIC_REUSE_EXECUTOR_VERSION):
    return ReuseCompatibilityKey(
        checkpoint_id="checkpoint",
        upstream_revision="revision",
        configuration_id="v5-a100-b40",
        controller_version="acr-isolated-controller-v1",
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


def tensor(values, shape):
    return Tensor(values, shape)


def inputs(compatibility_key, *, offset=0.0, proprio=(7.0, 8.0)):
    return ReuseExecutionInputs(
        compatibility_key=compatibility_key,
        wrist_pixels=tensor([offset + value for value in range(24)], (1, 6, 2, 2)),
        cached_scene_tokens=tensor((10 + offset, 20 + offset), (1, 2, 1)),
        prompt_input=tensor((1, 2), (1, 2)),
        prompt_embeddings=tensor((3 + offset, 4 + offset), (1, 2, 1)),
        attention_mask=tensor((1, 1), (1, 2)),
        proprioception=tensor(proprio, (1, 2)),
    )


def executors():
    eager_cores = Cores()
    static_cores = Cores()
    eager = EagerReuseExecutor(
        tensor_ops=Ops(),
        wrist_visual_core=eager_cores.wrist,
        downstream_action_core=eager_cores.downstream,
    )
    static = StaticBufferReuseExecutor(
        tensor_ops=Ops(),
        wrist_visual_core=static_cores.wrist,
        downstream_action_core=static_cores.downstream,
    )
    eager.prepare(key(EAGER_REUSE_EXECUTOR_VERSION))
    static.prepare(key())
    return eager, eager_cores, static, static_cores


def test_frozen_identities_lifecycle_and_exact_reference_parity():
    eager, eager_cores, static, static_cores = executors()
    eager_result = eager.run(inputs(key(EAGER_REUSE_EXECUTOR_VERSION)))
    static_result = static.run(inputs(key()))
    assert eager.executor_version == EAGER_REUSE_EXECUTOR_VERSION
    assert static.executor_version == STATIC_REUSE_EXECUTOR_VERSION
    assert static_result.combined_tokens.values == eager_result.combined_tokens.values
    assert static_result.wrist_tokens.values == eager_result.wrist_tokens.values
    assert static_result.normalized_actions.values == eager_result.normalized_actions.values
    assert static_result.combined_tokens.values[:2] == [10.0, 20.0]
    assert eager_cores.scene_calls == static_cores.scene_calls == 0
    assert (static_cores.wrist_calls, static_cores.downstream_calls) == (1, 1)
    assert static_result.snapshot.lifecycle is ExecutorLifecycle.PREPARED
    assert static_result.snapshot.work.completed_queries == 1


def test_static_buffers_stay_owned_and_current_values_are_not_stale():
    _, _, static, _ = executors()
    first = static.run(inputs(key()))
    first_ids = dict(first.snapshot.buffer_identities)
    first_actions = tuple(first.normalized_actions.values)
    second_inputs = inputs(key(), offset=100.0, proprio=(70.0, 80.0))
    second = static.run(second_inputs)
    assert dict(second.snapshot.buffer_identities) == first_ids
    assert id(second_inputs.wrist_pixels) != first_ids["wrist_pixels"]
    assert tuple(second.normalized_actions.values) != first_actions
    assert second.combined_tokens.values[:2] == [110.0, 120.0]
    assert second.normalized_actions.values[0] > first_actions[0]
    assert second.snapshot.work.completed_queries == 2


def test_current_prompt_and_proprioception_each_change_current_action():
    _, _, static, _ = executors()
    baseline = tuple(static.run(inputs(key())).normalized_actions.values)
    changed_proprio = tuple(
        static.run(inputs(key(), proprio=(70.0, 80.0))).normalized_actions.values
    )
    changed_prompt_inputs = inputs(key())
    changed_prompt_inputs.prompt_embeddings.values[:] = [30.0, 40.0]
    changed_prompt = tuple(static.run(changed_prompt_inputs).normalized_actions.values)
    assert baseline != changed_proprio
    assert baseline != changed_prompt


def test_every_compatibility_field_participates_in_prelaunch_equality():
    assert ReuseCompatibilityKey.field_names() == (
        "checkpoint_id",
        "upstream_revision",
        "configuration_id",
        "controller_version",
        "executor_version",
        "preprocessing_id",
        "action_head_id",
        "instruction_sha256",
        "prompt_input_shape",
        "dtype",
        "device",
        "image_height",
        "image_width",
        "patch_count",
        "projected_dimension",
        "wrist_shape",
        "cached_scene_shape",
        "embedding_shape",
        "attention_mask_shape",
        "proprioception_shape",
        "action_shape",
        "model_training_state",
        "use_film",
        "use_diffusion",
    )
    for field_name in ReuseCompatibilityKey.field_names():
        cores = Cores()
        static = StaticBufferReuseExecutor(
            tensor_ops=Ops(),
            wrist_visual_core=cores.wrist,
            downstream_action_core=cores.downstream,
        )
        original = key()
        static.prepare(original)
        changed = replace(original)
        current = getattr(changed, field_name)
        if isinstance(current, bool):
            replacement = not current
        elif isinstance(current, int):
            replacement = current + 1
        elif isinstance(current, tuple):
            replacement = (*current, 1)
        else:
            replacement = current + "-changed"
        object.__setattr__(changed, field_name, replacement)
        with pytest.raises(ReuseExecutorUnavailable, match="compatibility key mismatch"):
            static.run(inputs(changed))
        assert cores.wrist_calls == cores.downstream_calls == 0
        assert static.lifecycle is ExecutorLifecycle.PREPARED


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"model_training_state": True}, "evaluation mode"),
        ({"use_film": True}, "FiLM"),
        ({"use_diffusion": True}, "diffusion"),
        ({"wrist_shape": (1, 6, 2, -1)}, "positive dimensions"),
    ],
)
def test_unsupported_modes_and_dynamic_shapes_fail_closed(changes, message):
    with pytest.raises(ValueError, match=message):
        replace(key(), **changes)


def test_invalid_runtime_metadata_rejects_before_core_launch():
    _, _, static, cores = executors()
    invalid = inputs(key())
    invalid.wrist_pixels.shape = (1, 6, 2, 3)
    with pytest.raises(ReuseExecutorUnavailable, match="metadata differs"):
        static.run(invalid)
    assert cores.wrist_calls == cores.downstream_calls == 0
    assert static.snapshot().work.prelaunch_rejections == 1


def test_mutated_owned_output_metadata_rejects_next_run_before_launch():
    _, _, static, cores = executors()
    result = static.run(inputs(key()))
    result.wrist_tokens.shape = (1, 3, 1)
    with pytest.raises(ReuseExecutorUnavailable, match="metadata changed"):
        static.run(inputs(key()))
    assert cores.wrist_calls == cores.downstream_calls == 1
    assert static.lifecycle is ExecutorLifecycle.PREPARED


def test_preparation_failure_is_atomic_and_invalidates():
    class FailingOps(Ops):
        def allocate(self, shape, *, dtype, device):
            raise MemoryError("allocation failed")

    executor = StaticBufferReuseExecutor(
        tensor_ops=FailingOps(),
        wrist_visual_core=Cores().wrist,
        downstream_action_core=Cores().downstream,
    )
    with pytest.raises(ReuseExecutorUnavailable, match="preparation failed"):
        executor.prepare(key())
    snapshot = executor.snapshot()
    assert snapshot.lifecycle is ExecutorLifecycle.INVALIDATED
    assert snapshot.compatibility_key is None
    assert snapshot.owner_thread is None
    assert snapshot.buffer_identities == ()


def test_postlaunch_failure_invalidates_without_downstream_retry():
    _, _, static, cores = executors()
    cores.fail_downstream = True
    with pytest.raises(ReuseExecutorFailure, match="downstream failure"):
        static.run(inputs(key()))
    assert cores.wrist_calls == cores.downstream_calls == 1
    assert static.lifecycle is ExecutorLifecycle.INVALIDATED
    assert static.snapshot().work.completed_queries == 0
    assert static.snapshot().work.postlaunch_failures == 1
    with pytest.raises(ReuseExecutorUnavailable, match="INVALIDATED"):
        static.run(inputs(key()))
    assert cores.downstream_calls == 1


def test_nested_reset_and_cross_thread_use_are_rejected():
    _, _, static, cores = executors()
    nested_errors = []

    def check_active_guards():
        for operation in (lambda: static.run(inputs(key())), static.reset):
            try:
                operation()
            except ReuseExecutorUnavailable as error:
                nested_errors.append(str(error))

    cores.on_wrist = check_active_guards
    static.run(inputs(key()))
    assert len(nested_errors) == 2
    assert any("ACTIVE" in message for message in nested_errors)
    assert any("reset" in message for message in nested_errors)

    errors = []
    thread = threading.Thread(
        target=lambda: errors.append(
            pytest.raises(ReuseExecutorUnavailable, static.run, inputs(key()))
        )
    )
    thread.start()
    thread.join()
    assert len(errors) == 1


def test_reset_clears_bindings_buffers_and_counters():
    _, _, static, _ = executors()
    static.run(inputs(key()))
    static.reset()
    snapshot = static.snapshot()
    assert snapshot.lifecycle is ExecutorLifecycle.UNPREPARED
    assert snapshot.owner_thread is None
    assert snapshot.compatibility_key is None
    assert snapshot.buffer_identities == ()
    assert snapshot.work.completed_queries == 0
