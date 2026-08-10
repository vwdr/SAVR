from __future__ import annotations

import inspect
import math
import threading
from dataclasses import replace

import pytest

from savr.acr.batched_dual_path import (
    BatchedDualPathOpenVLAAdapter,
    BatchedFullRefreshAdapter,
    ModelQueryBudget,
)
from savr.acr.cache import SceneCacheEntry
from savr.acr.controller import ACRController
from savr.acr.dual_path import DualPathOpenVLAAdapter
from savr.acr.isolated_controller import (
    ISOLATED_CONTROLLER_VERSION,
    IsolatedACRController,
)
from savr.acr.openvla_oft import TorchTensorOperations
from savr.acr.records import AttemptIdentity, ImmutableRecordStore, semantic_sha256
from savr.acr.types import ACRConfiguration, ACRContext, ACRPolicy


class FakeTensor:
    def __init__(self, values, shape, *, dtype="float32", device="cpu"):
        self.values = tuple(float(value) for value in values)
        self.shape = tuple(int(value) for value in shape)
        self.dtype = dtype
        self.device = device
        self.requires_grad = True
        if math.prod(self.shape) != len(self.values):
            raise ValueError("Fake tensor data/shape mismatch")

    def detach(self):
        return self

    def clone(self):
        result = FakeTensor(self.values, self.shape, dtype=self.dtype, device=self.device)
        result.requires_grad = self.requires_grad
        return result

    def requires_grad_(self, value):
        self.requires_grad = value
        return self


class FakeOps:
    def __init__(self):
        self.finite_checks = []
        self.reshape_inputs = []

    @staticmethod
    def split(value, sections, *, dim):
        if dim != 1 or sum(sections) != value.shape[1]:
            raise ValueError("Fake split supports only dimension one")
        batch_size = value.shape[0]
        inner = math.prod(value.shape[2:])
        outputs = []
        offset = 0
        for section in sections:
            data = []
            for batch in range(batch_size):
                base = batch * value.shape[1] * inner
                data.extend(value.values[base + offset * inner : base + (offset + section) * inner])
            outputs.append(
                FakeTensor(
                    data,
                    (batch_size, section, *value.shape[2:]),
                    dtype=value.dtype,
                    device=value.device,
                )
            )
            offset += section
        return tuple(outputs)

    @staticmethod
    def cat(values, *, dim):
        first = values[0]
        if dim == 0:
            if any(value.shape[1:] != first.shape[1:] for value in values):
                raise ValueError("Fake batch concatenation requires matching samples")
            return FakeTensor(
                tuple(item for value in values for item in value.values),
                (sum(value.shape[0] for value in values), *first.shape[1:]),
                dtype=first.dtype,
                device=first.device,
            )
        if dim == 1:
            if first.shape[0] != 1:
                raise ValueError("Fake token concatenation is batch-one only")
            return FakeTensor(
                tuple(item for value in values for item in value.values),
                (1, sum(value.shape[1] for value in values), first.shape[2]),
                dtype=first.dtype,
                device=first.device,
            )
        if dim == 2:
            batch_size, patches, _ = first.shape
            data = []
            for batch in range(batch_size):
                for patch in range(patches):
                    for value in values:
                        width = value.shape[2]
                        start = (batch * patches + patch) * width
                        data.extend(value.values[start : start + width])
            return FakeTensor(
                data,
                (batch_size, patches, sum(value.shape[2] for value in values)),
                dtype=first.dtype,
                device=first.device,
            )
        raise ValueError("Unsupported fake concatenation")

    def reshape(self, value, shape):
        if math.prod(shape) != len(value.values):
            raise ValueError("Fake reshape changes element count")
        self.reshape_inputs.append((value.shape, tuple(shape), value.values))
        return FakeTensor(value.values, shape, dtype=value.dtype, device=value.device)

    def all_finite(self, value):
        self.finite_checks.append(value)
        values = value.values if hasattr(value, "values") else value
        return all(math.isfinite(float(item)) for item in values)


class FakeBackbone:
    use_fused_vision_backbone = True

    def __init__(self):
        self.patch_count = 2
        self.number_of_images = 2
        self.siglip_inputs = []
        self.dinov2_inputs = []

    def get_num_images_in_input(self):
        return self.number_of_images

    def get_num_patches(self):
        return self.patch_count

    @staticmethod
    def _features(value, offset):
        sample_width = math.prod(value.shape[1:])
        data = []
        for batch in range(value.shape[0]):
            sample = value.values[batch * sample_width : (batch + 1) * sample_width]
            mean = sum(sample) / len(sample) + offset
            data.extend((mean, mean))
        return FakeTensor(
            data,
            (value.shape[0], 2, 1),
            dtype=value.dtype,
            device=value.device,
        )

    def featurizer(self, value):
        self.siglip_inputs.append(value)
        return self._features(value, 0)

    def fused_featurizer(self, value):
        self.dinov2_inputs.append(value)
        return self._features(value, 10)


class FakeModel:
    def __init__(self, ops):
        self.ops = ops
        self.vision_backbone = FakeBackbone()
        self.projector_inputs = []
        self.original_refresh_calls = 0
        self.last_tokens = None
        self.last_state = None

    def projector(self, value):
        self.projector_inputs.append(value)
        projected = tuple(item for item in value.values for _ in range(2))
        return FakeTensor(
            projected,
            (value.shape[0], value.shape[1], 4),
            dtype=value.dtype,
            device=value.device,
        )

    def _camera(self, value):
        regular, fused = self.ops.split(value, (3, 3), dim=1)
        return self.ops.cat(
            (
                self.vision_backbone.featurizer(regular),
                self.vision_backbone.fused_featurizer(fused),
            ),
            dim=2,
        )

    def _process_vision_features(self, pixel_values, language_embeddings, use_film=False):
        assert language_embeddings is not None and not use_film
        self.original_refresh_calls += 1
        scene, wrist = self.ops.split(pixel_values, (6, 6), dim=1)
        features = self.ops.cat((self._camera(scene), self._camera(wrist)), dim=1)
        return self.projector(features)

    def predict(self, pixel_values, language, state, *, actions=None, recurse=None):
        if recurse is not None:
            recurse()
        self.last_tokens = self._process_vision_features(pixel_values, language)
        self.last_state = tuple(state)
        return tuple(0.0 for _ in range(56)) if actions is None else actions


def configuration():
    return ACRConfiguration(
        "sa-bdp-acr-t25-h2-b30-v01",
        ACRPolicy.SA_ACR,
        scene_threshold=0.2476380718954248,
        translation_threshold=0.5479944908411765,
        horizon=2,
        hard_reuse_cap=0.30,
    )


def context(*, configuration_id=None, **changes):
    value = ACRContext(
        episode_id="episode",
        attempt_id="attempt",
        task_id="task",
        instruction_sha256="0" * 64,
        checkpoint_id="checkpoint",
        upstream_revision="revision",
        configuration_id=configuration_id or configuration().configuration_id,
        controller_version="acr-controller-v1",
        preprocessing_id="preprocessing",
        action_head_id="head",
        dtype="float32",
        device="cpu",
        patch_count=2,
    )
    return replace(value, **changes)


def pixels(scene=1.0, wrist=2.0, *, channels=12, dtype="float32", device="cpu"):
    scene_channels = min(channels, 6)
    wrist_channels = max(channels - scene_channels, 0)
    values = (scene,) * (scene_channels * 4) + (wrist,) * (wrist_channels * 4)
    return FakeTensor(values, (1, channels, 2, 2), dtype=dtype, device=device)


def language(*, dtype="float32", device="cpu"):
    return FakeTensor((0.0,) * 20, (1, 5, 4), dtype=dtype, device=device)


def make_bfr(*, correctness_mode=False, observer=None):
    ops = FakeOps()
    model = FakeModel(ops)
    adapter = BatchedFullRefreshAdapter(
        model=model,
        tensor_ops=ops,
        correctness_mode=correctness_mode,
        projected_tokens_observer=observer,
    )
    return model, adapter, ops


def make_v3(*, correctness_mode=False, observer=None, action_finite_checker=None):
    ops = FakeOps()
    model = FakeModel(ops)
    adapter = BatchedDualPathOpenVLAAdapter(
        model=model,
        controller=ACRController(configuration()),
        tensor_ops=ops,
        correctness_mode=correctness_mode,
        projected_tokens_observer=observer,
        action_finite_checker=action_finite_checker,
    )
    return model, adapter, ops


def make_isolated_v5():
    ops = FakeOps()
    model = FakeModel(ops)
    config = ACRConfiguration(
        "v5-a100-b40",
        ACRPolicy.SA_ACR,
        scene_threshold=0.30046895424836606,
        translation_threshold=0.685919037527938,
        horizon=1,
        hard_reuse_cap=0.40,
        controller_version=ISOLATED_CONTROLLER_VERSION,
    )
    adapter = BatchedDualPathOpenVLAAdapter(
        model=model,
        controller=IsolatedACRController(config),
        tensor_ops=ops,
    )
    return model, adapter


def run_bfr(
    model,
    adapter,
    *,
    pixel_values=None,
    lang=None,
    actions=None,
    recurse=None,
):
    values = pixels() if pixel_values is None else pixel_values
    embeddings = language() if lang is None else lang
    return adapter.run_query(
        lambda: model.predict(
            values,
            embeddings,
            (0.0,) * 8,
            actions=actions,
            recurse=recurse,
        )
    )


def run_v3(
    model,
    adapter,
    *,
    scene=1.0,
    wrist=2.0,
    state=(0.0,) * 8,
    actions=None,
    pixel_values=None,
    lang=None,
    recurse=None,
):
    values = pixels(scene, wrist) if pixel_values is None else pixel_values
    embeddings = language() if lang is None else lang
    return adapter.run_query(
        query=lambda: model.predict(
            values,
            embeddings,
            state,
            actions=actions,
            recurse=recurse,
        ),
        scene_image=[[scene] * 32 for _ in range(32)],
        wrist_image=[[wrist] * 32 for _ in range(32)],
        state=state,
        state_q01=(0.0,) * 8,
        state_q99=(1.0,) * 8,
    )


def test_batched_fr_preserves_exact_scene_wrist_order_and_upstream_tokens():
    oracle_ops = FakeOps()
    oracle = FakeModel(oracle_ops)
    values = pixels(scene=1, wrist=2)
    expected = oracle._process_vision_features(values, language())
    model, adapter, ops = make_bfr()
    with adapter.episode(context(configuration_id="batched-fr-v01")):
        result = run_bfr(model, adapter, pixel_values=values)
    assert model.last_tokens.values == expected.values
    assert model.last_tokens.shape == (1, 4, 4)
    assert model.vision_backbone.siglip_inputs[0].values[:12] == (1.0,) * 12
    assert model.vision_backbone.siglip_inputs[0].values[12:] == (2.0,) * 12
    assert ops.reshape_inputs[0][0:2] == ((2, 2, 2), (1, 4, 2))
    assert result.work.mode == "batched-fr"
    result.work.validate()


def test_bfr_has_no_controller_or_cache_and_calls_each_component_once():
    model, adapter, _ = make_bfr()
    assert not hasattr(adapter, "controller")
    assert not hasattr(adapter, "cache")
    with adapter.episode(context(configuration_id="batched-fr-v01")):
        result = run_bfr(model, adapter)
    assert len(model.vision_backbone.siglip_inputs) == 1
    assert len(model.vision_backbone.dinov2_inputs) == 1
    assert len(model.projector_inputs) == 1
    assert (
        result.work.physical_siglip_calls,
        result.work.physical_dinov2_calls,
        result.work.physical_projector_calls,
    ) == (1, 1, 1)


def test_v3_refresh_batches_once_and_owns_the_scene_cache():
    observed = []
    model, adapter, _ = make_v3(observer=observed.append)
    with adapter.episode(context()):
        result = run_v3(model, adapter)
        assert result.decision.refresh
        assert result.cache_event == "refresh"
        assert result.work.mode == "v3-refresh"
        result.work.validate()
        assert adapter.cache.entry is not None
        assert adapter.cache.entry.tokens.values == model.last_tokens.values[:8]
        assert adapter.cache.entry.tokens.requires_grad is False
        assert observed == [model.last_tokens]
    assert adapter.cache.entry is None
    assert len(model.vision_backbone.siglip_inputs) == 1
    assert len(model.vision_backbone.dinov2_inputs) == 1


def test_v3_reuse_is_exactly_the_established_v2_wrist_path():
    v2_ops = FakeOps()
    v2_model = FakeModel(v2_ops)
    v2_config = replace(configuration(), configuration_id="sa-dp-acr-t25-h2-b30-v01")
    v2 = DualPathOpenVLAAdapter(
        model=v2_model,
        controller=ACRController(v2_config),
        tensor_ops=v2_ops,
    )
    v3_model, v3, _ = make_v3()
    with v2.episode(context(configuration_id=v2_config.configuration_id)), v3.episode(context()):
        v2_results = []
        v3_results = []
        for index in range(4):
            wrist = float(index + 2)
            v2_results.append(run_v3(v2_model, v2, wrist=wrist))
            v3_results.append(run_v3(v3_model, v3, wrist=wrist))
        assert not v2_results[-1].decision.refresh
        assert not v3_results[-1].decision.refresh
        assert v3_model.last_tokens.values == v2_model.last_tokens.values
        assert v3_results[-1].value == v2_results[-1].value
        assert v3_results[-1].work.mode == "v3-reuse"
        v3_results[-1].work.validate()


def test_isolated_v5_controller_runs_through_batched_adapter_without_consecutive_reuse():
    model, adapter = make_isolated_v5()
    isolated_context = context(
        configuration_id="v5-a100-b40",
        controller_version=ISOLATED_CONTROLLER_VERSION,
    )
    with adapter.episode(isolated_context):
        results = [run_v3(model, adapter, wrist=float(index + 2)) for index in range(5)]
    assert [result.decision.refresh for result in results] == [True, True, False, True, False]
    assert [result.work.mode for result in results] == [
        "v3-refresh",
        "v3-refresh",
        "v3-reuse",
        "v3-refresh",
        "v3-reuse",
    ]
    assert "post-reuse-refresh" in results[3].decision.reasons


def test_v3_cache_mismatch_forces_batched_refresh():
    model, adapter, _ = make_v3()
    with adapter.episode(context()):
        for index in range(3):
            run_v3(model, adapter, wrist=float(index + 2))
        assert adapter.cache.entry is not None
        entry = adapter.cache.entry
        adapter.cache._entry = SceneCacheEntry(
            context=entry.context,
            tokens=entry.tokens,
            metadata=replace(entry.metadata, dtype="float16"),
            refresh_query_index=entry.refresh_query_index,
        )
        result = run_v3(model, adapter, wrist=5)
        assert result.decision.refresh
        assert result.cache_event == "forced-refresh"
        assert result.work.mode == "v3-refresh"
    assert len(model.vision_backbone.siglip_inputs) == 4


@pytest.mark.parametrize(
    ("pixel_values", "lang", "message"),
    [
        (pixels(channels=11), language(), "pixel tensor shape"),
        (pixels(dtype="float16"), language(), "Pixel tensor dtype/device"),
        (pixels(), language(dtype="float16"), "Language embeddings dtype/device"),
    ],
)
def test_v3_structural_failures_stop_closed(pixel_values, lang, message):
    model, adapter, _ = make_v3()
    with adapter.episode(context()):
        for index in range(3):
            run_v3(model, adapter, wrist=float(index + 2))
        with pytest.raises(RuntimeError, match=message):
            run_v3(model, adapter, pixel_values=pixel_values, lang=lang)
        assert adapter.cache.entry is None
        assert adapter.controller.query_index == 3
        assert adapter.last_failure is not None


def test_batched_tower_shape_and_patch_mismatch_stop_closed():
    model, adapter, _ = make_bfr()
    model.vision_backbone.patch_count = 3
    with pytest.raises(RuntimeError, match="patch count"):
        with adapter.episode(context(configuration_id="batched-fr-v01")):
            run_bfr(model, adapter)


def test_bfr_and_v3_restore_after_downstream_exception():
    model, bfr, _ = make_bfr()
    original = FakeModel._process_vision_features
    with pytest.raises(RuntimeError, match="downstream"):
        with bfr.episode(context(configuration_id="batched-fr-v01")):
            run_bfr(
                model,
                bfr,
                recurse=lambda: (_ for _ in ()).throw(RuntimeError("downstream")),
            )
    assert "_process_vision_features" not in vars(model)
    assert FakeModel._process_vision_features is original
    model, v3, _ = make_v3()
    with pytest.raises(RuntimeError, match="downstream"):
        with v3.episode(context()):
            run_v3(
                model,
                v3,
                recurse=lambda: (_ for _ in ()).throw(RuntimeError("downstream")),
            )
    assert "_process_vision_features" not in vars(model)
    assert v3.cache.entry is None


def test_nested_and_concurrent_v3_use_is_rejected():
    model, adapter, _ = make_v3()
    with adapter.episode(context()):
        with pytest.raises(RuntimeError, match="Nested or concurrent"):
            with adapter.episode(context()):
                pass
        with pytest.raises(RuntimeError, match="Nested V3 query"):
            run_v3(model, adapter, recurse=lambda: run_v3(model, adapter))
        errors = []

        def other_thread():
            try:
                run_v3(model, adapter)
            except RuntimeError as error:
                errors.append(str(error))

        thread = threading.Thread(target=other_thread)
        thread.start()
        thread.join()
        assert errors == ["Concurrent V3 query use is prohibited"]


def test_nested_and_concurrent_bfr_use_is_rejected():
    model, adapter, _ = make_bfr()
    bfr_context = context(configuration_id="batched-fr-v01")
    with adapter.episode(bfr_context):
        with pytest.raises(RuntimeError, match="Nested or concurrent"):
            with adapter.episode(bfr_context):
                pass
        with pytest.raises(RuntimeError, match="Nested Batched-FR query"):
            run_bfr(model, adapter, recurse=lambda: run_bfr(model, adapter))
        errors = []

        def other_thread():
            try:
                run_bfr(model, adapter)
            except RuntimeError as error:
                errors.append(str(error))

        thread = threading.Thread(target=other_thread)
        thread.start()
        thread.join()
        assert errors == ["Concurrent Batched-FR query use is prohibited"]


def test_production_has_only_action_finite_check_and_correctness_scans_tokens():
    model, adapter, ops = make_v3(correctness_mode=False)
    with adapter.episode(context()):
        run_v3(model, adapter)
    assert len(ops.finite_checks) == 1
    assert isinstance(ops.finite_checks[0], tuple)
    model, adapter, ops = make_v3(correctness_mode=True)
    with adapter.episode(context()):
        run_v3(model, adapter)
    assert len(ops.finite_checks) == 4
    assert sum(isinstance(value, FakeTensor) for value in ops.finite_checks) == 3


def test_nonfinite_action_invalidates_cache_and_does_not_advance_controller():
    model, adapter, _ = make_v3()
    with adapter.episode(context()):
        with pytest.raises(RuntimeError, match="non-finite"):
            run_v3(model, adapter, actions=(float("nan"),) + (0.0,) * 55)
        assert adapter.cache.entry is None
        assert adapter.controller.query_index == 0
        assert adapter.last_failure is not None
        assert adapter.last_failure.cache_invalidated


def test_hot_paths_have_no_evidence_hashing_serialization_or_file_io():
    source = inspect.getsource(
        __import__("savr.acr.batched_dual_path", fromlist=["batched_dual_path"])
    )
    for forbidden in (
        "audit_sha256",
        "json.dumps",
        "write_text",
        "write_bytes",
        "from pathlib",
        "pathlib.",
    ):
        assert forbidden not in source


def test_model_query_budget_consumes_before_call_and_fails_closed():
    budget = ModelQueryBudget(2)
    assert budget.consume("correctness-00") == 0
    assert budget.consume("timed-00") == 1
    with pytest.raises(RuntimeError, match="exhausted"):
        budget.consume("timed-01")
    with pytest.raises(ValueError, match="unique"):
        ModelQueryBudget(2).consume("")
    assert budget.labels == ("correctness-00", "timed-00")


def test_v3_immutable_identity_and_recovery_are_monotonic(tmp_path):
    identity = AttemptIdentity(
        "acr-v3c-correctness-v01",
        "sa-bdp-acr",
        "synthetic",
        0,
        0,
        0,
        0,
    )
    record = {
        "schema_version": "acr.v3-query.v1",
        "query_id": identity.query_id(0),
        "status": "failed",
        "record_sha256": semantic_sha256({"query_id": identity.query_id(0)}),
    }
    store = ImmutableRecordStore(tmp_path)
    store.write_once(identity.query_id(0), record)
    with pytest.raises(FileExistsError):
        store.write_once(identity.query_id(0), {**record, "status": "completed"})
    pairing = identity.value.rsplit("/attempt-", 1)[0]
    assert store.next_attempt_index(pairing) == 1


def test_torch_operations_expose_the_v3_reshape_contract():
    class Value:
        def __init__(self):
            self.observed = None

        def reshape(self, shape):
            self.observed = shape
            return self

    value = Value()
    result = TorchTensorOperations(object()).reshape(value, (1, 4, 2))
    assert result is value
    assert value.observed == (1, 4, 2)
