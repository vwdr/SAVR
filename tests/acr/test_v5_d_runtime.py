from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from savr.acr.reuse_executor import ReuseCompatibilityKey, ReuseExecutionInputs
from savr.acr.v5_d_runtime import (
    BackendKind,
    BackendWaterfall,
    FrozenQueryLedger,
    MemorySnapshot,
    MethodPatch,
    MethodRestorationGuard,
    ResourceEnvelope,
    TechnicalReason,
    V5DProtocolViolation,
    V5DResourceExceeded,
    V5DStaticBufferReuseExecutor,
    frozen_query_schedule,
    validate_v5_d_freeze,
)


ROOT = Path(__file__).resolve().parents[2]


def config() -> dict:
    return json.loads(
        (ROOT / "configs/acr/v5_d_gpu_feasibility_freeze.json").read_text(encoding="utf-8")
    )


def test_freeze_and_exact_query_ledger() -> None:
    freeze = config()
    validate_v5_d_freeze(freeze)
    schedule = frozen_query_schedule(freeze)
    assert len(schedule) == 111
    assert [item.kind for item in schedule[:7]] == ["correctness"] * 7
    assert [item.kind for item in schedule[7:15]] == ["warmup"] * 8
    assert [item.kind for item in schedule[15:]] == ["timed"] * 96
    assert sum(item.position == 0 for item in schedule if item.kind == "timed") == 24
    ledger = FrozenQueryLedger(freeze)
    with pytest.raises(V5DProtocolViolation, match="expected"):
        ledger.consume(schedule[1].label)
    for identity in schedule:
        assert ledger.consume(identity.label) == identity
    ledger.require_complete()
    with pytest.raises(V5DResourceExceeded, match="cap exhausted"):
        ledger.consume("extra")


def test_waterfall_requires_compile_then_fresh_raw_before_output() -> None:
    waterfall = BackendWaterfall(config())
    with pytest.raises(V5DProtocolViolation, match="compile first"):
        waterfall.begin(BackendKind.RAW_CUDAGRAPH, process_token="raw-1")
    waterfall.begin(BackendKind.TORCH_COMPILE, process_token="compile-1")
    waterfall.record_preparation_launch("compile-wrist-first")
    assert waterfall.technical_failure(
        TechnicalReason.FULL_GRAPH_CAPTURE_ERROR, "unsupported graph"
    )
    with pytest.raises(V5DProtocolViolation, match="fresh process"):
        waterfall.begin(BackendKind.RAW_CUDAGRAPH, process_token="compile-1")
    waterfall.begin(BackendKind.RAW_CUDAGRAPH, process_token="raw-1")
    waterfall.record_preparation_launch("raw-wrist-warmup-0")
    waterfall.begin_correctness()
    waterfall.record_correctness()
    assert not waterfall.technical_failure(
        TechnicalReason.FULL_GRAPH_CAPTURE_ERROR, "raw replay failed"
    )


def test_compile_failure_after_correctness_cannot_unlock_raw() -> None:
    waterfall = BackendWaterfall(config())
    waterfall.begin(BackendKind.TORCH_COMPILE, process_token="compile")
    waterfall.begin_correctness()
    waterfall.record_correctness()
    permitted = waterfall.technical_failure(
        TechnicalReason.STATIC_KEY_RECOMPILE, "recompile after parity output"
    )
    assert permitted is False
    with pytest.raises(V5DProtocolViolation, match="not permitted"):
        waterfall.begin(BackendKind.RAW_CUDAGRAPH, process_token="raw")


def test_preparation_launch_cap_is_cumulative_across_waterfall() -> None:
    waterfall = BackendWaterfall(config())
    waterfall.begin(BackendKind.TORCH_COMPILE, process_token="compile")
    for index in range(12):
        waterfall.record_preparation_launch(f"compile-{index}")
    assert waterfall.technical_failure(
        TechnicalReason.COMPILER_CONSTRUCTION_OR_FIRST_CALL_ERROR, "compile failed"
    )
    waterfall.begin(BackendKind.RAW_CUDAGRAPH, process_token="raw")
    for index in range(12):
        waterfall.record_preparation_launch(f"raw-{index}")
    with pytest.raises(V5DResourceExceeded, match="launch cap"):
        waterfall.record_preparation_launch("raw-over-cap")


def test_resource_envelope_stops_wall_artifact_and_memory_growth() -> None:
    now = [0.0]
    artifacts = [0]
    envelope = ResourceEnvelope(config(), clock=lambda: now[0], artifact_bytes=lambda: artifacts[0])
    gib = 1024**3
    envelope.set_eager_baseline(MemorySnapshot(15 * gib, 16 * gib))
    envelope.observe_memory(MemorySnapshot(16 * gib, 22 * gib))
    with pytest.raises(V5DResourceExceeded, match="incremental"):
        envelope.observe_memory(MemorySnapshot(16 * gib, 22 * gib + 1))
    artifacts[0] = 1024**3 + 1
    with pytest.raises(V5DResourceExceeded, match="artifact"):
        envelope.check_host_resources()
    artifacts[0] = 0
    now[0] = 7200.1
    with pytest.raises(V5DResourceExceeded, match="wall-time"):
        envelope.check_host_resources()


def test_method_guard_restores_absent_and_existing_bindings_on_error() -> None:
    class Target:
        def existing(self):
            return "class"

    target = Target()
    target.existing = lambda: "instance"
    original = target.existing
    with pytest.raises(RuntimeError, match="boom"):
        with MethodRestorationGuard(
            (
                MethodPatch(target, "existing", lambda: "patched"),
                MethodPatch(target, "temporary", lambda: "temporary"),
            )
        ):
            assert target.existing() == "patched"
            assert target.temporary() == "temporary"
            raise RuntimeError("boom")
    assert target.existing is original
    assert "temporary" not in vars(target)


class Tensor:
    def __init__(self, values, shape, *, dtype="torch.bfloat16", device="cuda:0"):
        self.values = list(values)
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = device


class Ops:
    def allocate(self, shape, *, dtype, device):
        size = 1
        for value in shape:
            size *= value
        return Tensor([0] * size, shape, dtype=dtype, device=device)

    @staticmethod
    def copy_(destination, source):
        destination.values[:] = source.values

    @staticmethod
    def cat_into(destination, values, *, dim):
        assert dim == 1
        destination.values[:] = values[0].values + values[1].values


def key() -> ReuseCompatibilityKey:
    return ReuseCompatibilityKey(
        checkpoint_id="checkpoint",
        upstream_revision="revision",
        configuration_id="v5-a100-b40",
        controller_version="acr-isolated-controller-v1",
        executor_version="acr-reuse-executor-static-v1",
        preprocessing_id="preprocessing",
        action_head_id="head",
        instruction_sha256="0" * 64,
        prompt_input_shape=(1, 2),
        dtype="torch.bfloat16",
        device="cuda:0",
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


def mixed_inputs(offset=0) -> ReuseExecutionInputs:
    contract = key()
    return ReuseExecutionInputs(
        compatibility_key=contract,
        wrist_pixels=Tensor(range(offset, offset + 24), contract.wrist_shape),
        cached_scene_tokens=Tensor((10 + offset, 20 + offset), contract.cached_scene_shape),
        prompt_input=Tensor((1, 2), (1, 2), dtype="torch.int64"),
        prompt_embeddings=Tensor((3 + offset, 4 + offset), contract.embedding_shape),
        attention_mask=Tensor((1, 1), (1, 2), dtype="torch.bool"),
        proprioception=Tensor((7 + offset, 8 + offset), contract.proprioception_shape),
    )


def test_v5_d_executor_preserves_mixed_dtypes_and_current_a_b_a_values() -> None:
    def wrist(pixels, output):
        output.values[:] = [sum(pixels.values[:12]), sum(pixels.values[12:])]

    def downstream(combined, embeddings, mask, proprio, output):
        total = sum(combined.values + embeddings.values + mask.values + proprio.values)
        output.values[:] = [total, total + 1]

    executor = V5DStaticBufferReuseExecutor(
        prompt_input_dtype="torch.int64",
        attention_mask_dtype="torch.bool",
        tensor_ops=Ops(),
        wrist_visual_core=wrist,
        downstream_action_core=downstream,
    )
    executor.prepare(key())
    buffer_ids = dict(executor.snapshot().buffer_identities)
    first = tuple(executor.run(mixed_inputs(0)).normalized_actions.values)
    second = tuple(executor.run(mixed_inputs(100)).normalized_actions.values)
    third = tuple(executor.run(mixed_inputs(0)).normalized_actions.values)
    assert first != second and first == third
    assert dict(executor.snapshot().buffer_identities) == buffer_ids
    buffers = executor.owned_buffers_for_backend_preparation()
    assert buffers["prompt_input"].dtype == "torch.int64"
    assert buffers["attention_mask"].dtype == "torch.bool"
    invalid = mixed_inputs()
    invalid = replace(
        invalid,
        attention_mask=Tensor((1, 1), (1, 2), dtype="torch.bfloat16"),
    )
    with pytest.raises(Exception, match="metadata differs"):
        executor.run(invalid)
